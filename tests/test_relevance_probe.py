"""Tests for the wave-2 E4 query-relevance probe (vncompress/linguistics.py):
build_relevance_labels and RelevanceConsistencyLoss, plus the vncompress-vi-v2
supervision path in vncompress/training.py.

CPU-only, no model download: build_relevance_labels uses conftest's
MockTokenizer, offset-based labelling uses the local OffsetTokenizer, and the
probe is exercised on a random hidden-state tensor.
"""
import json
import re
from types import SimpleNamespace

import pytest
import torch

from vncompress.linguistics import (
    QUERY_PREFIX_TEMPLATE,
    PhonologicalConsistencyLoss,
    RelevanceConsistencyLoss,
    build_relevance_labels,
    build_relevance_labels_from_offsets,
)
from vncompress.training import (
    RelevanceDataset,
    load_relevance_samples,
    relevance_class_weights,
)


class OffsetTokenizer:
    """Whitespace tokenizer that reports character offsets, standing in for a
    fast HuggingFace tokenizer (conftest's MockTokenizer has no offsets)."""

    is_fast = True
    _WORD_RE = re.compile(r"\S+", re.UNICODE)

    def __init__(self):
        self._to_id = {}
        self._to_word = {}

    def _register(self, word):
        if word not in self._to_id:
            self._to_id[word] = len(self._to_id)
            self._to_word[self._to_id[word]] = word
        return self._to_id[word]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        ids, offsets = [], []
        for match in self._WORD_RE.finditer(text):
            ids.append(self._register(match.group()))
            offsets.append((match.start(), match.end()))
        out = {'input_ids': ids}
        if return_offsets_mapping:
            out['offset_mapping'] = offsets
        return out

    def encode(self, text, add_special_tokens=False, truncation=False, max_length=None, **kwargs):
        ids = self(text)['input_ids']
        return ids[:max_length] if truncation and max_length else ids

    def decode(self, ids, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        return " ".join(self._to_word.get(i, "") for i in ids)


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def qa_row(**overrides):
    context = "Mở đầu bài viết. " + ("đệm " * 200) + "Thủ đô là Hà Nội."
    answer = "Hà Nội"
    start = context.index(answer)
    row = {
        'id': 'qa_1', 'doc_id': 'viquad:Việt_Nam', 'split': 'train',
        'task': 'long_document_qa', 'context': context, 'query': 'Thủ đô là gì?',
        'answer': answer, 'answer_span': [start, start + len(answer)],
    }
    row.update(overrides)
    return row


class TestBuildRelevanceLabels:
    def test_answer_tokens_positive_filler_negative(self, tokenizer):
        # Answer syllables should be labelled 1; unrelated filler 0.
        context = "công ty được thành lập tại Hà Nội vào năm 2010"
        answer = "Hà Nội"
        ids = tokenizer.encode(context)
        labels = build_relevance_labels(tokenizer, ids, answer)
        assert len(labels) == len(ids)
        words = context.split()
        pos = {i for i, w in enumerate(words) if w.lower() in ("hà", "nội")}
        for i, lab in enumerate(labels):
            if i in pos:
                assert lab == 1, f"expected token {words[i]!r} relevant"
        # A clearly-unrelated content word is negative.
        assert labels[words.index("thành")] == 0

    def test_short_tokens_are_ignored(self, tokenizer):
        # Sub-min_token_len pieces -> ignore_index (-100), neither train nor score.
        ids = tokenizer.encode("a bb ccc")
        labels = build_relevance_labels(tokenizer, ids, "ccc", min_token_len=2)
        assert labels[0] == -100          # "a" too short
        assert labels[2] == 1             # "ccc" overlaps answer

    def test_empty_answer_gives_no_positives(self, tokenizer):
        ids = tokenizer.encode("một hai ba bốn")
        labels = build_relevance_labels(tokenizer, ids, "")
        assert all(lab in (0, -100) for lab in labels)


class TestBuildRelevanceLabelsFromOffsets:
    def test_only_tokens_overlapping_the_span_are_positive(self):
        # offsets for "aa bb cc" -> span covers "bb" alone.
        offsets = [(0, 2), (3, 5), (6, 8)]
        assert build_relevance_labels_from_offsets(offsets, (3, 5)) == [0, 1, 0]

    def test_partial_overlap_counts(self):
        # A token straddling the span start is still answer content.
        offsets = [(0, 5), (5, 10)]
        assert build_relevance_labels_from_offsets(offsets, (3, 7)) == [1, 1]

    def test_touching_but_not_overlapping_is_negative(self):
        # Token ends exactly where the span begins: no shared character.
        offsets = [(0, 3), (3, 6)]
        assert build_relevance_labels_from_offsets(offsets, (3, 6)) == [0, 1]

    def test_zero_width_pieces_are_ignored(self):
        offsets = [(0, 0), (0, 2), (2, 2)]
        assert build_relevance_labels_from_offsets(offsets, (0, 2)) == [-100, 1, -100]

    def test_beats_syllable_overlap_on_a_repeated_syllable(self):
        """The reason this function exists: the syllable matcher fires on every
        other occurrence of an answer syllable in the context."""
        tokenizer = OffsetTokenizer()
        context = "đức tính tốt và Đức là nước lớn"
        encoded = tokenizer(context, return_offsets_mapping=True)
        span = (context.index("Đức"), context.index("Đức") + 3)

        exact = build_relevance_labels_from_offsets(encoded['offset_mapping'], span)
        overlap = build_relevance_labels(tokenizer, encoded['input_ids'], "Đức")
        assert sum(1 for label in exact if label == 1) == 1
        assert sum(1 for label in overlap if label == 1) == 2  # "đức tính" caught too


class TestLoadRelevanceSamples:
    def test_reads_v2_jsonl_keeping_query_and_span(self, tmp_path):
        path = write_jsonl(tmp_path / 'qa.jsonl', [qa_row()])
        samples = load_relevance_samples(path)
        assert len(samples) == 1
        assert samples[0]['query'] == 'Thủ đô là gì?'
        assert samples[0]['reference_answer'] == 'Hà Nội'
        start, end = samples[0]['answer_span']
        assert samples[0]['context'][start:end] == 'Hà Nội'

    def test_short_context_extractive_qa_is_kept(self, tmp_path):
        """v2's name for rows under the long-document threshold -- just as
        extractive, and 2,813 of its 6,000 qa rows."""
        path = write_jsonl(tmp_path / 'qa.jsonl', [qa_row(task='short_context_extractive_qa')])
        assert len(load_relevance_samples(path)) == 1

    def test_split_filter(self, tmp_path):
        path = write_jsonl(tmp_path / 'qa.jsonl', [
            qa_row(id='a', split='train'), qa_row(id='b', split='validation')])
        assert len(load_relevance_samples(path, split='train')) == 1
        assert len(load_relevance_samples(path, split='validation')) == 1
        assert len(load_relevance_samples(path)) == 2

    def test_holdout_matches_across_id_prefix_and_spacing(self, tmp_path):
        """`viquad:Hà_Nội` in train and "Hà Nội" in a benchmark are one
        document -- the leak measured in dataset_v2_review.md SS5.3."""
        path = write_jsonl(tmp_path / 'qa.jsonl', [qa_row(doc_id='viquad:Hà_Nội')])
        assert load_relevance_samples(path)
        with pytest.raises(RuntimeError):
            load_relevance_samples(path, holdout_docs=['Hà Nội'])

    def test_a_span_that_does_not_quote_the_answer_is_dropped(self, tmp_path):
        path = write_jsonl(tmp_path / 'qa.jsonl', [qa_row(answer_span=[0, 5])])
        assert load_relevance_samples(path)[0]['answer_span'] is None

    def test_missing_file_raises_instead_of_substituting_demo_data(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_relevance_samples(str(tmp_path / 'absent.jsonl'))

    def test_wrong_shaped_json_raises_instead_of_substituting_demo_data(self, tmp_path):
        """The shipped default was `{metadata, paragraphs}` with no `samples`
        key, which used to silently train the probe on two demo sentences."""
        path = tmp_path / 'training_corpus_v1.json'
        path.write_text(json.dumps({'metadata': {}, 'paragraphs': ['xin chào']}), encoding='utf-8')
        with pytest.raises(RuntimeError):
            load_relevance_samples(str(path))

    def test_empty_data_path_raises(self):
        with pytest.raises(ValueError):
            load_relevance_samples(None)


class TestRelevanceDataset:
    def test_window_keeps_an_answer_far_into_the_context(self):
        """81% of v2 answers start past char 1,024; truncating from the front
        cuts the positive labels off entirely."""
        tokenizer = OffsetTokenizer()
        sample = {
            'context': qa_row()['context'], 'query': '', 'reference_answer': 'Hà Nội',
            'answer_span': tuple(qa_row()['answer_span']), 'task': 'long_document_qa',
            'doc_id': 'd', 'split': 'train',
        }
        dataset = RelevanceDataset([sample], tokenizer, max_length=32)
        assert len(dataset) == 1
        ids, labels = dataset[0]
        assert dataset.stats['span_labelled'] == 1
        positives = tokenizer.decode([i for i, label in zip(ids, labels) if label == 1])
        assert positives == "Hà Nội."

    def test_front_truncation_would_have_lost_it(self):
        """Contrast case: the pre-span path sees only the head of the context."""
        tokenizer = OffsetTokenizer()
        context = qa_row()['context']
        head = tokenizer.encode(context, truncation=True, max_length=32)
        assert "Hà" not in tokenizer.decode(head)

    def test_query_prefix_is_present_but_never_scored(self):
        tokenizer = OffsetTokenizer()
        sample = {
            'context': qa_row()['context'], 'query': 'Thủ đô là gì?',
            'reference_answer': 'Hà Nội', 'answer_span': tuple(qa_row()['answer_span']),
            'task': 'long_document_qa', 'doc_id': 'd', 'split': 'train',
        }
        dataset = RelevanceDataset([sample], tokenizer, max_length=64,
                                   query_template=QUERY_PREFIX_TEMPLATE)
        ids, labels = dataset[0]
        prefix_len = len(tokenizer.encode(QUERY_PREFIX_TEMPLATE.format(query=sample['query'])))
        assert "Thủ" in tokenizer.decode(ids[:prefix_len])
        assert all(label == -100 for label in labels[:prefix_len])
        assert any(label == 1 for label in labels[prefix_len:])

    def test_falls_back_to_overlap_labels_without_a_span(self, tokenizer):
        sample = {
            'context': "công ty được thành lập tại Hà Nội vào năm 2010 " * 3,
            'query': '', 'reference_answer': 'Hà Nội', 'answer_span': None,
            'task': 'long_document_qa', 'doc_id': 'd', 'split': 'train',
        }
        dataset = RelevanceDataset([sample], tokenizer, max_length=64)
        assert dataset.stats['overlap_labelled'] == 1
        assert dataset.stats['span_labelled'] == 0
        assert any(label == 1 for _, labels in dataset.samples for label in labels)

    def test_samples_without_a_positive_label_are_dropped_and_counted(self, tokenizer):
        sample = {
            'context': "một hai ba bốn năm sáu bảy tám chín mười " * 5,
            'query': '', 'reference_answer': 'không_hề_xuất_hiện', 'answer_span': None,
            'task': 'long_document_qa', 'doc_id': 'd', 'split': 'train',
        }
        dataset = RelevanceDataset([sample], tokenizer, max_length=64)
        assert len(dataset) == 0
        assert dataset.stats['dropped_no_positive'] == 1


class TestRelevanceClassWeights:
    def test_weight_is_the_negative_to_positive_ratio(self):
        dataset = SimpleNamespace(samples=[([0] * 10, [1, 1] + [0] * 8)])
        assert relevance_class_weights(dataset) == [1.0, 4.0]

    def test_ignored_labels_do_not_count(self):
        dataset = SimpleNamespace(samples=[([0] * 6, [1, 0, 0, -100, -100, -100])])
        assert relevance_class_weights(dataset) == [1.0, 2.0]

    def test_ratio_is_capped(self):
        dataset = SimpleNamespace(samples=[([0] * 1001, [1] + [0] * 1000)])
        assert relevance_class_weights(dataset, cap=50.0) == [1.0, 50.0]

    def test_no_positives_falls_back_to_neutral(self):
        dataset = SimpleNamespace(samples=[([0] * 4, [0, 0, 0, 0])])
        assert relevance_class_weights(dataset) == [1.0, 1.0]

    def test_weighting_changes_the_loss_but_not_the_checkpoint(self):
        """Weights must not reach state_dict, or models.load_scorer's key and
        dim checks would see a probe it cannot load."""
        torch.manual_seed(0)
        hidden = torch.randn(1, 6, 16)
        labels = torch.tensor([[1, 0, 0, 0, 0, 0]])
        plain = RelevanceConsistencyLoss(hidden_dim=16)
        weighted = RelevanceConsistencyLoss(hidden_dim=16, class_weights=[1.0, 20.0])
        weighted.load_state_dict(plain.state_dict())

        assert plain(hidden, labels).item() != weighted(hidden, labels).item()
        assert set(plain.state_dict()) == set(weighted.state_dict())


class TestRelevanceConsistencyLoss:
    def test_forward_returns_scalar_loss(self):
        probe = RelevanceConsistencyLoss(hidden_dim=32)
        h = torch.randn(2, 5, 32)
        labels = torch.tensor([[1, 0, -100, 1, 0], [0, 1, 1, -100, 0]])
        mask = torch.ones(2, 5)
        loss = probe(h, labels, mask)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_score_importance_range_and_shape(self):
        probe = RelevanceConsistencyLoss(hidden_dim=32)
        h = torch.randn(2, 7, 32)
        scores = probe.score_importance(h)
        assert scores.shape == (2, 7)
        assert float(scores.min()) >= 0.5
        assert float(scores.max()) <= 3.0

    def test_length_mismatch_raises(self):
        probe = RelevanceConsistencyLoss(hidden_dim=16)
        h = torch.randn(1, 4, 16)
        labels = torch.zeros(1, 3, dtype=torch.long)
        try:
            probe(h, labels)
            assert False, "expected ValueError on S mismatch"
        except ValueError:
            pass


class TestInterfaceCompatibleWithToneProbe:
    """The relevance probe must be a drop-in for the tone probe so LACCScorer /
    models.load_scorer consume it through the identical code path."""

    def test_shares_tone_probe_surface(self):
        hidden_dim = 48
        probe = RelevanceConsistencyLoss(hidden_dim=hidden_dim)
        # Attributes LACCScorer / load_scorer read:
        assert hasattr(probe, "tone_classifier")
        assert hasattr(probe, "num_tones")
        assert hasattr(probe, "score_importance")
        assert probe.num_tones == 2
        # First classifier layer is Linear(hidden_dim, ...), as load_scorer's
        # state['tone_classifier.0.weight'].shape[1] dim-check expects.
        assert probe.tone_classifier[0].weight.shape[1] == hidden_dim

    def test_state_dict_keys_match_loader_expectation(self):
        probe = RelevanceConsistencyLoss(hidden_dim=24)
        state = probe.state_dict()
        assert "tone_classifier.0.weight" in state
        assert state["tone_classifier.0.weight"].shape[1] == 24

    def test_same_classifier_shape_as_tone_probe(self):
        # Both build the same Sequential shape (only the final class count differs),
        # so a checkpoint loads into whichever class matches its meta.
        rel = RelevanceConsistencyLoss(hidden_dim=40)
        tone = PhonologicalConsistencyLoss(hidden_dim=40)
        assert type(rel.tone_classifier[0]) is type(tone.tone_classifier[0])
        assert rel.tone_classifier[0].weight.shape == tone.tone_classifier[0].weight.shape
