"""Tests for vncompress/training.py and the model-loading helpers in
vncompress/models.py that training depends on. CPU-only: none of these
exercise an actual training loop (which requires a GPU for the SLM
pipeline) -- they cover dataset/collator construction, the tone-label
pipeline, target-module selection, and the tone probe's scoring formula,
using the same MockTokenizer as the rest of the suite.
"""
from types import SimpleNamespace

import pytest
import torch

from vncompress.linguistics import PhonologicalConsistencyLoss
from vncompress.models import lora_target_modules, resize_embeddings_if_needed
from vncompress.training import (
    SLMCollator,
    ToneDataCollator,
    ToneTrainingDataset,
    VietnameseToneDataset,
    load_training_texts,
)


# ============================================================================
# Training data loading
# ============================================================================


def test_load_training_texts_falls_back_to_demo_corpus():
    # No path, and (in a clean checkout) no data/benchmark/*.json present ->
    # falls back to the small built-in Vietnamese demo corpus.
    texts = load_training_texts(data_path=None)
    assert len(texts) >= 1
    assert all(isinstance(t, str) and t for t in texts)


def test_load_training_texts_reads_paragraphs_shape(tmp_path):
    import json

    path = tmp_path / "corpus.json"
    long_text = "Xin chào các bạn. " * 20  # > 200 chars
    path.write_text(json.dumps({"paragraphs": [{"text": long_text}]}), encoding="utf-8")
    texts = load_training_texts(str(path))
    assert texts == [long_text]


def test_load_training_texts_reads_samples_shape(tmp_path):
    import json

    path = tmp_path / "corpus.json"
    long_text = "Xin chào các bạn. " * 20
    path.write_text(json.dumps({"samples": [{"context": long_text}]}), encoding="utf-8")
    assert load_training_texts(str(path)) == [long_text]


def test_load_training_texts_filters_short_paragraphs(tmp_path):
    import json

    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"paragraphs": [{"text": "quá ngắn"}]}), encoding="utf-8")
    assert load_training_texts(str(path)) == []


# ============================================================================
# LACC model training dataset/collator (vncompress/training.py)
# ============================================================================


class TestToneTrainingDataset:
    def test_builds_one_sample_per_long_enough_text(self, tokenizer):
        texts = ["xin chào các bạn " * 3, "một hai"]  # second is too short (<10 tokens)
        ds = ToneTrainingDataset(texts, tokenizer, max_length=64)
        assert len(ds) == 1

    def test_sample_has_matching_length_fields(self, tokenizer):
        text = "xin chào các bạn hôm nay trời đẹp quá " * 2
        ds = ToneTrainingDataset([text], tokenizer, max_length=64)
        sample = ds[0]
        assert len(sample["input_ids"]) == len(sample["labels"]) == len(sample["tone_labels"]) == sample["length"]

    def test_empty_corpus_gives_empty_dataset(self, tokenizer):
        assert len(ToneTrainingDataset([], tokenizer, max_length=64)) == 0


class TestToneDataCollator:
    def test_pads_batch_to_max_length_in_batch(self, tokenizer):
        texts = ["xin chào các bạn hôm nay trời đẹp quá lắm " * 2, "một hai ba bốn năm sáu bảy tám chín mười"]
        ds = ToneTrainingDataset(texts, tokenizer, max_length=64)
        collator = ToneDataCollator(pad_token_id=0, max_length=64)
        batch = collator([ds[i] for i in range(len(ds))])
        assert batch["input_ids"].shape[0] == len(ds)
        assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape == batch["tone_labels"].shape

    def test_attention_mask_marks_real_tokens(self, tokenizer):
        text = "xin chào các bạn hôm nay trời đẹp quá lắm rồi đấy nhé bạn ơi"
        ds = ToneTrainingDataset([text], tokenizer, max_length=64)
        collator = ToneDataCollator(pad_token_id=0, max_length=64)
        batch = collator([ds[0]])
        assert batch["attention_mask"].sum().item() == ds[0]["length"]


# ============================================================================
# SLM training dataset/collator
# ============================================================================


class TestVietnameseToneDataset:
    def test_builds_ids_and_tones_per_sample(self, tokenizer):
        text = "xin chào các bạn hôm nay trời đẹp quá " * 2
        ds = VietnameseToneDataset([text], tokenizer, max_length=64)
        assert len(ds) == 1
        ids, tones = ds[0]
        assert len(ids) == len(tones)

    def test_short_texts_are_dropped(self, tokenizer):
        assert len(VietnameseToneDataset(["một hai"], tokenizer, max_length=64)) == 0


class TestSLMCollator:
    def test_pads_to_batch_max_width(self, tokenizer):
        ds = VietnameseToneDataset(
            ["xin chào các bạn hôm nay trời đẹp quá lắm luôn " * 2,
             "một hai ba bốn năm sáu bảy tám chín mười"],
            tokenizer, max_length=64,
        )
        batch = SLMCollator(pad_id=0)([ds[i] for i in range(len(ds))])
        assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape == batch["tone_labels"].shape
        assert batch["input_ids"].shape[0] == len(ds)


# ============================================================================
# Model-loading helpers used by both training pipelines
# ============================================================================


class TestResizeEmbeddingsIfNeeded:
    def _model(self, vocab_size, hidden=8):
        embedding = torch.nn.Embedding(vocab_size, hidden)

        class FakeModel:
            def get_input_embeddings(self):
                return embedding

            def get_output_embeddings(self):
                return None

            def resize_token_embeddings(self, n):
                nonlocal embedding
                new_embedding = torch.nn.Embedding(n, hidden)
                with torch.no_grad():
                    new_embedding.weight[:vocab_size] = embedding.weight
                embedding = new_embedding

        return FakeModel()

    def test_no_op_when_sizes_already_match(self):
        model = self._model(vocab_size=10)
        assert resize_embeddings_if_needed(model, tokenizer=list(range(10))) is False

    def test_resizes_and_zeros_new_rows(self):
        model = self._model(vocab_size=10)
        changed = resize_embeddings_if_needed(model, tokenizer=list(range(15)))
        assert changed is True
        weight = model.get_input_embeddings().weight
        assert weight.shape[0] == 15
        assert torch.all(weight[10:] == 0)


class TestLoraTargetModules:
    """The tone-probe trainer must accept a Qwen3-4B base (and stay working
    for GPT-2 SLMs), so the training target can be switched without a code edit."""

    def _model(self, model_type):
        return SimpleNamespace(config=SimpleNamespace(model_type=model_type))

    @pytest.mark.parametrize("mt", ["qwen3", "qwen3_moe", "qwen2", "qwen", "llama"])
    def test_qwen_family_targets_projections(self, mt):
        mods = lora_target_modules(self._model(mt))
        assert "q_proj" in mods and "gate_proj" in mods

    def test_gpt2_targets_conv1d_modules(self):
        assert lora_target_modules(self._model("gpt2")) == ["c_attn", "c_proj", "c_fc"]

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported model_type"):
            lora_target_modules(self._model("mamba"))


# ============================================================================
# Tone probe scoring formula (used both as training loss and as LACC's
# trained-tone-probe inference signal)
# ============================================================================


def test_score_importance_matches_paper_formula():
    # S_tone-model = 1 + max_k softmax(MLP(h))_k, clamped [0.5, 3.0].
    probe = PhonologicalConsistencyLoss(hidden_dim=16, lambda_tone=0.0).eval()
    h = torch.randn(1, 5, 16)
    with torch.no_grad():
        score = probe.score_importance(h)
        logits = probe.tone_classifier(h)
        expected = 1.0 + torch.softmax(logits, dim=-1).max(dim=-1).values
    assert torch.allclose(score, torch.clamp(expected, 0.5, 3.0), atol=1e-5)


def test_tone_classifier_loss_requires_matching_sequence_length():
    probe = PhonologicalConsistencyLoss(hidden_dim=16)
    hidden_states = torch.randn(1, 5, 16)
    tone_labels = torch.zeros(1, 4, dtype=torch.long)  # wrong length
    with pytest.raises(ValueError, match="Sequence length mismatch"):
        probe(hidden_states, tone_labels)


# ============================================================================
# Document-level train/eval split (docs/dataset_pipeline.md §9)
# ============================================================================


def _corpus_file(tmp_path, n_docs=20, paras_per_doc=5):
    import json

    paragraphs = [
        {'source': 'uvw-2026', 'topic_id': f'doc{d}', 'paragraph_index': i,
         'text': f'Đoạn {i} của tài liệu {d}. ' + ('Nội dung tiếng Việt. ' * 15)}
        for d in range(n_docs) for i in range(paras_per_doc)
    ]
    path = tmp_path / 'corpus.json'
    path.write_text(json.dumps({'paragraphs': paragraphs}), encoding='utf-8')
    return str(path)


def test_load_train_eval_texts_splits_by_document_not_by_paragraph(tmp_path):
    """The regression this replaces: `random_split` over paragraphs scattered
    ~65% of UVW articles across both sides, because one article contributes up
    to 162 paragraphs that all share a topic_id."""
    from vncompress.dataset import normalize_file
    from vncompress.training import load_train_eval_texts

    path = _corpus_file(tmp_path, n_docs=20, paras_per_doc=5)
    train_texts, eval_texts, meta = load_train_eval_texts(path)

    assert train_texts and eval_texts
    assert set(train_texts).isdisjoint(set(eval_texts))

    doc_of = {r.context: r.doc_key for r in normalize_file(path)}
    train_docs = {doc_of[t] for t in train_texts}
    eval_docs = {doc_of[t] for t in eval_texts}
    assert train_docs.isdisjoint(eval_docs), 'a document must not straddle the split'
    assert meta['n_train_documents'] == len(train_docs)
    assert meta['n_eval_documents'] == len(eval_docs)


def test_load_train_eval_texts_is_deterministic(tmp_path):
    from vncompress.training import load_train_eval_texts

    path = _corpus_file(tmp_path)
    first = load_train_eval_texts(path)[1]
    second = load_train_eval_texts(path)[1]
    assert first == second


def test_load_train_eval_texts_holds_out_roughly_the_requested_ratio(tmp_path):
    from vncompress.training import load_train_eval_texts

    path = _corpus_file(tmp_path, n_docs=100, paras_per_doc=5)
    train_texts, eval_texts, _ = load_train_eval_texts(path, eval_ratio=0.1)
    assert len(eval_texts) == 50
    assert len(train_texts) == 450


def test_load_train_eval_texts_survives_a_single_document_corpus(tmp_path):
    import json

    from vncompress.training import load_train_eval_texts

    path = tmp_path / 'one_doc.json'
    path.write_text(json.dumps({'paragraphs': [
        {'topic_id': 'only', 'paragraph_index': i, 'text': f'Đoạn {i}. ' + ('Nội dung. ' * 40)}
        for i in range(4)
    ]}), encoding='utf-8')

    train_texts, eval_texts, meta = load_train_eval_texts(str(path))
    assert train_texts and eval_texts, 'validation must still have something to score'
    assert meta.get('degenerate_single_document') is True


def test_relevance_samples_split_by_document(tmp_path):
    """E4 used to train on every long_document_qa / needle_in_haystack sample
    of vcc_bench_v1.json -- the same samples benchmark.py scored."""
    import json

    from vncompress.dataset import normalize_file
    from vncompress.training import load_train_eval_relevance_samples

    samples = [
        {'sample_id': f'doc_qa_{d:04d}_q{q}', 'source': 'wikipedia', 'domain': f'Topic_{d}',
         'task': 'long_document_qa', 'context': f'Ngữ cảnh {d}. ' + ('Câu văn tiếng Việt. ' * 20),
         'query': f'q{q}', 'reference_answer': 'Ngữ cảnh'}
        for d in range(30) for q in range(3)
    ]
    path = tmp_path / 'bench.json'
    path.write_text(json.dumps({'samples': samples}), encoding='utf-8')

    train, eval_, meta = load_train_eval_relevance_samples(str(path))
    assert train and eval_
    doc_of = {r.context: r.doc_key for r in normalize_file(str(path))}
    assert {doc_of[s['context']] for s in train}.isdisjoint({doc_of[s['context']] for s in eval_})
    assert meta['n_eval_documents'] >= 1


def test_relevance_loader_does_not_re_split_a_pre_split_file(tmp_path):
    import json

    from vncompress.training import load_train_eval_relevance_samples

    samples = [{'sample_id': f'needle_{i:04d}', 'task': 'needle_in_haystack',
                'context': f'Ngữ cảnh {i}. ' + ('Nội dung. ' * 40), 'query': 'q',
                'reference_answer': 'Ngữ cảnh'} for i in range(10)]
    path = tmp_path / 'vcc_bench_train.json'
    path.write_text(json.dumps({'metadata': {'split': 'train'}, 'samples': samples}), encoding='utf-8')

    train, eval_, meta = load_train_eval_relevance_samples(str(path))
    assert len(train) == 10, 'a file already declared as one side of a split must be used whole'
    assert eval_ == []
    assert 'pre-split' in meta['split_source']
