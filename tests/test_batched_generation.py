"""Batched generation must produce exactly what one-at-a-time generation did.

The sweep is ~23k generations, so it now decodes in batches. Batching a
decoder-only model is easy to get subtly wrong -- right-padding, a shared
prompt-width slice, or reordering for length-grouping all fail the same way:
every row still returns a plausible string, just the *wrong* row's. Nothing
downstream can detect that, so it is pinned here with a stub model whose output
is a deterministic function of its own input.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vncompress.evaluation import VCCBench, VCCBenchConfig  # noqa: E402

PAD = 0


class StubTokenizer:
    pad_token_id = PAD
    eos_token_id = 99
    chat_template = None  # force _build_prompt_ids' 'raw' path

    def encode(self, text, add_special_tokens=False):
        return [int(t) for t in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        ids = [int(i) for i in (ids.tolist() if hasattr(ids, 'tolist') else ids)]
        if skip_special_tokens:
            ids = [i for i in ids if i != PAD]
        return ' '.join(str(i) for i in ids)


class StubModel:
    """Appends a fingerprint of the row's own input.

    The fingerprint deliberately mixes two things a real decoder-only model
    depends on: the row's real (attended) tokens, and the token sitting at the
    LAST position -- because that is the position generation actually continues
    from. Right-padding a batch leaves a pad token there, which is exactly why
    it corrupts short rows on a real model; a stub that only looked at the
    attention mask would be blind to it and the test would pass either way.
    """

    device = 'cpu'

    def __init__(self, fail=False):
        self.fail = fail
        self.seen_widths = []

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=4, **kw):
        if self.fail:
            raise RuntimeError("CUDA out of memory (simulated)")
        self.seen_widths.append(input_ids.shape[1])
        rows = []
        for row, mask in zip(input_ids, attention_mask if attention_mask is not None
                             else torch.ones_like(input_ids)):
            real = row[mask.bool()].tolist()
            new = ([sum(real), int(row[-1])] * max_new_tokens)[:max_new_tokens]
            rows.append(row.tolist() + new)
        return torch.tensor(rows, dtype=torch.long)


class StubResult:
    def __init__(self, ids):
        self.compressed_ids = ids


class StubSample:
    def __init__(self, i):
        self.query = str(1000 + i)
        self.reference_answer = 'ref'
        self.metadata = {'sample_id': f's{i}'}


def make_pending(lengths):
    """One pending entry per length; token ids are unique per sample."""
    pending = []
    for i, n in enumerate(lengths):
        ids = [i * 100 + k + 1 for k in range(n)]
        pending.append((object(), StubSample(i), StubResult(ids), 'ctx'))
    return pending


def expected_output(pending_entry, max_new_tokens):
    """What the stub must return for this sample, computed independently."""
    _m, sample, result, _ct = pending_entry
    prompt = result.compressed_ids + [int(sample.query)]
    new = ([sum(prompt), prompt[-1]] * max_new_tokens)[:max_new_tokens]
    return ' '.join(str(t) for t in new)


def _bench(batch_size, max_new_tokens=4):
    bench = VCCBench(VCCBenchConfig(
        generation_batch_size=batch_size, max_new_tokens=max_new_tokens,
        prompt_style='raw', do_sample=False,
    ))
    return bench


class TestBatchedGeneration:

    def test_batched_matches_unbatched_sample_for_sample(self):
        lengths = [7, 3, 11, 5, 2, 9, 4, 6, 1, 8]
        pending = make_pending(lengths)
        tok = StubTokenizer()

        single, _e, _t, bs1 = _bench(1)._run_generation(
            StubModel(), tok, pending, None, 'needle_in_haystack', 4.0)
        batched, _e2, _t2, bs8 = _bench(8)._run_generation(
            StubModel(), tok, pending, None, 'needle_in_haystack', 4.0)

        assert bs1 == 1 and bs8 == 8
        assert batched == single, "batched output does not match one-at-a-time"
        for entry, got in zip(pending, batched):
            assert got == expected_output(entry, 4), "output landed on the wrong sample"

    def test_uneven_lengths_are_left_padded_not_right(self):
        """A right-padded batch would fold pad ids into the fingerprint."""
        pending = make_pending([1, 12, 3])
        texts, err = _bench(4)._generate_batch(
            StubModel(), StubTokenizer(),
            [_bench(4)._build_prompt_ids(StubTokenizer(), r.compressed_ids, s.query)
             for _m, s, r, _ct in pending])
        assert err is None
        for entry, got in zip(pending, texts):
            assert got == expected_output(entry, 4)

    def test_length_grouping_does_not_reorder_results(self):
        """Prompts are sorted longest-first internally; results must not be."""
        pending = make_pending([2, 30, 5, 25, 1])
        out, _e, _t, _bs = _bench(2)._run_generation(
            StubModel(), StubTokenizer(), pending, None, 'long_document_qa', 2.0)
        for entry, got in zip(pending, out):
            assert got == expected_output(entry, 4)

    def test_failed_batch_falls_back_to_one_at_a_time(self):
        """An OOM on a batch must not score its neighbours as failures."""
        pending = make_pending([4, 6, 8])
        bench = _bench(8)
        out, errors, _t, _bs = bench._run_generation(
            StubModel(fail=True), StubTokenizer(), pending, None, 'needle_in_haystack', 4.0)
        # Every sample was retried individually; the stub fails there too, so
        # each records its own error rather than silently scoring 0.
        assert all(o is None for o in out)
        assert all(e and 'out of memory' in e for e in errors), errors

    def test_batch_size_is_recorded_for_latency_reading(self):
        bench = _bench(4)
        _o, _e, times, bs = bench._run_generation(
            StubModel(), StubTokenizer(), make_pending([3, 3, 3, 3]), None, 'cross_lingual', 8.0)
        assert bs == 4
        # Amortised, so every member of a batch shares the batch's wall time.
        assert len(set(round(t, 6) for t in times)) == 1

    def test_custom_generation_fn_is_never_batched(self):
        """A caller-supplied hook takes one sample and cannot be batched."""
        seen = []

        def hook(model, tokenizer, compressed_ids=None, query=None, **kw):
            seen.append(list(compressed_ids))
            return 'hooked'

        pending = make_pending([2, 3, 4])
        out, _e, _t, bs = _bench(8)._run_generation(
            StubModel(), StubTokenizer(), pending, hook, 'needle_in_haystack', 4.0)
        assert bs == 1
        assert out == ['hooked'] * 3
        assert seen == [r.compressed_ids for _m, _s, r, _c in pending]
