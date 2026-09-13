"""Tests for vncompress/encoder_compression.py (wave-2 E6).

CPU-only, no model download: a stub encoder (a tiny torch.nn.Module returning
deterministic per-token keep/drop logits) and a stub fast tokenizer exposing
`return_offsets_mapping` are injected via the constructor, so the whole
keep/drop -> char-span -> generator-token pipeline is exercised offline.
"""
import pytest
import torch

from vncompress.compression import CompressionConfig, create_compressor
from vncompress.encoder_compression import EncoderClassifierCompressor


def _is_subsequence(sub, full):
    it = iter(full)
    return all(item in it for item in sub)


class StubEncoderTokenizer:
    """Whitespace tokenizer with real character offsets, mimicking a HF fast
    tokenizer's `return_offsets_mapping`. Ids are 1-based; the word text is
    recoverable so the stub model can score by content."""

    def __init__(self):
        self._vocab = {}
        self._inv = {}

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False, truncation=False, **kw):
        ids, offsets = [], []
        for m in _iter_words(text):
            word = m.group(0)
            tid = self._vocab.setdefault(word, len(self._vocab) + 1)
            self._inv[tid] = word
            ids.append(tid)
            offsets.append((m.start(), m.end()))
        out = {'input_ids': ids}
        if return_offsets_mapping:
            out['offset_mapping'] = offsets
        return out

    def word_of(self, tid):
        return self._inv.get(tid, '')


def _iter_words(text):
    import re
    return re.finditer(r'\S+', text)


class StubEncoder(torch.nn.Module):
    """Returns 2-label logits per input token. Tokens whose decoded word is in
    `keep_words` get a high keep logit; everything else gets a high drop logit.
    A trivial parameter makes `.parameters()`/`.to()`/device detection work."""

    def __init__(self, enc_tok, keep_words):
        super().__init__()
        self.enc_tok = enc_tok
        self.keep_words = set(keep_words)
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        b, s = input_ids.shape
        logits = torch.zeros(b, s, 2)
        for j in range(s):
            word = self.enc_tok.word_of(int(input_ids[0, j]))
            if word in self.keep_words:
                logits[0, j] = torch.tensor([0.0, 5.0])   # keep
            else:
                logits[0, j] = torch.tensor([5.0, 0.0])   # drop
        return type('O', (), {'logits': logits})


class MockTokenizer:
    """Whitespace generation tokenizer (same surface as conftest.MockTokenizer)."""

    def __init__(self):
        self._i2w, self._w2i = {}, {}

    def encode(self, text, add_special_tokens=False, **kw):
        import re
        return [self._reg(w) for w in re.findall(r'\S+', text)]

    def _reg(self, w):
        if w not in self._w2i:
            i = len(self._i2w)
            self._w2i[w] = i
            self._i2w[i] = w
        return self._w2i[w]

    def decode(self, ids, skip_special_tokens=True, **kw):
        if isinstance(ids, int):
            ids = [ids]
        # Trailing space per token so _token_spans (which decodes one id at a
        # time and concatenates) reconstructs separable words -- real HF
        # tokenizers carry the leading/▁ space; the plain mock would not.
        return ''.join(self._i2w.get(i, '') + ' ' for i in ids)


def _make(keep_words, ratio=2.0, keep_boundary=1):
    gen_tok = MockTokenizer()
    enc_tok = StubEncoderTokenizer()
    encoder = StubEncoder(enc_tok, keep_words)
    comp = EncoderClassifierCompressor(
        gen_tok, model=None, device='cpu',
        encoder_model=encoder, encoder_tokenizer=enc_tok,
        config=CompressionConfig(target_ratio=ratio, keep_boundary_tokens=keep_boundary),
    )
    return gen_tok, comp


def test_missing_encoder_raises():
    gen_tok = MockTokenizer()
    comp = EncoderClassifierCompressor(gen_tok, device='cpu')
    with pytest.raises(RuntimeError):
        comp.compress(gen_tok.encode("a b c d e f g h"))


def test_compress_is_valid_and_shorter():
    gen_tok, comp = _make(keep_words={'bbb', 'ddd', 'fff'})
    ids = gen_tok.encode("aaa bbb ccc ddd eee fff ggg hhh")
    result = comp.compress(list(ids))
    assert 0 < result.compressed_length < len(ids)
    assert result.compression_ratio > 1.0
    assert _is_subsequence(result.compressed_ids, ids)


def test_token_order_preserved_and_boundaries_kept():
    gen_tok, comp = _make(keep_words={'ccc', 'ddd'}, keep_boundary=1)
    ids = gen_tok.encode("aaa bbb ccc ddd eee fff ggg hhh")
    result = comp.compress(list(ids))
    assert result.compressed_ids[0] == ids[0]
    assert result.compressed_ids[-1] == ids[-1]
    assert _is_subsequence(result.compressed_ids, ids)


def test_keep_words_are_preferentially_retained():
    # With a tight budget, the encoder's "keep" words should survive over
    # neutral filler in the middle.
    gen_tok, comp = _make(keep_words={'KEEPME'}, ratio=2.0, keep_boundary=1)
    ids = gen_tok.encode("f1 f2 f3 KEEPME f4 f5 f6 f7 f8 f9")
    keep_id = gen_tok.encode("KEEPME")[0]
    result = comp.compress(list(ids))
    assert keep_id in result.compressed_ids


def test_ratio_is_approximately_respected():
    gen_tok, comp = _make(keep_words=set(), ratio=2.0, keep_boundary=1)
    ids = gen_tok.encode(" ".join(f"w{i}" for i in range(20)))
    result = comp.compress(list(ids))
    assert result.compressed_length == pytest.approx(len(ids) / 2, abs=3)


def test_metadata_records_keep_label():
    gen_tok, comp = _make(keep_words={'bbb'})
    result = comp.compress(list(gen_tok.encode("aaa bbb ccc ddd")))
    assert result.metadata['keep_label'] == 1
    assert result.metadata['query_applied'] is False


def test_get_name_includes_tag():
    gen_tok = MockTokenizer()
    comp = EncoderClassifierCompressor(gen_tok, device='cpu', encoder_id='vinai/phobert-base')
    assert 'phobert' in comp.get_name().lower()


def test_lazy_registry_resolves_encoder_class():
    gen_tok = MockTokenizer()
    comp = create_compressor('encoder', gen_tok, model=None, device='cpu', encoder_id='some/id')
    assert isinstance(comp, EncoderClassifierCompressor)
