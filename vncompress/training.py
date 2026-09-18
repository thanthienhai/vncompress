"""
Training — the tone-aware fine-tuning pipelines behind LACC's model signals.
==============================================================================
Two pipelines, both driven by `train.py`:

  1. LACC model training (`train.py --mode lacc`): fine-tune a (generation-
     sized) causal LM with the phonological consistency auxiliary loss, so
     its own hidden states can later serve as LACCCompressor's same-tokenizer
     tone-probe signal.
  2. SLM / tone-probe training (`train.py --mode slm`): LoRA fine-tune a
     small Vietnamese causal LM (default: chronopt-research/vietnamese-gpt2-base)
     on low-VRAM GPUs, producing the LACCScorer this repo's `lightweight` and
     `full` hardware tiers pair with a generation model via models.load_scorer().

`validate_slm()` (used by `train.py --mode slm --validate`) evaluates a
trained SLM checkpoint's held-out perplexity and tone-probe accuracy -- the
"is this checkpoint any good" step before it's used as a LACC scorer.

Contents: dataset + collator classes, ToneAwareTrainer (manual PyTorch loop
for LACC model training), run_lacc_training(), run_slm_training(),
validate_slm(), and the shared load_training_texts() data loader.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .dataset_build import document_key, external_eval_documents
from .linguistics import (
    QUERY_PREFIX_TEMPLATE,
    TONE_NAME_TO_ID,
    PhonologicalConsistencyLoss,
    RelevanceConsistencyLoss,
    build_relevance_labels,
    build_relevance_labels_from_offsets,
    get_tone_analyzer,
)
from .models import lora_target_modules, resize_embeddings_if_needed


# ============================================================================
# Shared training data
# ============================================================================


def _demo_texts() -> List[str]:
    """Small built-in Vietnamese corpus so training/validation always has
    something to run on, even with no dataset file present."""
    return [
        "Luật Bảo vệ Môi trường năm 2020 quy định về hoạt động bảo vệ môi trường, "
        "quyền, nghĩa vụ và trách nhiệm của cơ quan, tổ chức, cộng đồng dân cư, "
        "hộ gia đình và cá nhân trong hoạt động bảo vệ môi trường. Bảo vệ môi trường "
        "là quyền, nghĩa vụ và trách nhiệm của mọi cơ quan, tổ chức, cộng đồng dân cư, "
        "hộ gia đình và cá nhân." * 5,
        "Thị trường chứng khoán Việt Nam đã có phiên giao dịch tích cực vào ngày "
        "hôm nay khi chỉ số VN-Index tăng 12 điểm, đạt mức 1280 điểm. Khối lượng "
        "giao dịch đạt hơn 1 tỷ cổ phiếu với tổng giá trị giao dịch hơn 25 nghìn "
        "tỷ đồng. Nhóm cổ phiếu ngân hàng và bất động sản dẫn đầu đà tăng trưởng." * 5,
        "Trí tuệ nhân tạo đang phát triển nhanh chóng và có tác động sâu rộng đến "
        "mọi mặt của đời sống xã hội. Các mô hình ngôn ngữ lớn đã đạt được những "
        "tiến bộ vượt bậc trong việc hiểu và sinh văn bản tiếng Việt. Tuy nhiên, "
        "việc xử lý các văn bản tiếng Việt dài vẫn còn nhiều thách thức do đặc điểm "
        "ngôn ngữ đơn lập, có thanh điệu và nhiều từ ghép." * 5,
        "Học máy là một lĩnh vực của trí tuệ nhân tạo liên quan đến việc phát triển "
        "các thuật toán cho phép máy tính học từ dữ liệu. Có ba loại học máy chính: "
        "học có giám sát, học không giám sát và học tăng cường. Trong học có giám sát, "
        "mô hình được huấn luyện trên dữ liệu đã được gán nhãn." * 5,
    ]


MIN_TRAINING_CHARS = 200

# The v2 corpus and this repo's benchmark are both Vietnamese Wikipedia, so they
# overlap by construction. Every training entrypoint holds these documents out
# by default and says so, because a contaminated benchmark is not detectable
# from the score it produces.
DEFAULT_BENCHMARK_HOLDOUT = os.path.join('data', 'benchmark', 'vcc_bench_v2.json')


def resolve_holdout_documents(paths=None, disabled: bool = False) -> Sequence[str]:
    """Document keys a training run must exclude, with the repo's own benchmark
    as the default. Prints what it held out; returns () when disabled."""
    if disabled:
        print("Holdout disabled: training may include benchmark documents.")
        return ()
    paths = list(paths or [])
    if not paths and os.path.exists(DEFAULT_BENCHMARK_HOLDOUT):
        paths = [DEFAULT_BENCHMARK_HOLDOUT]
    if not paths:
        return ()
    keys = sorted(external_eval_documents(paths))
    print(f"Holding out {len(keys)} document(s) used by {', '.join(paths)}")
    return keys


def _load_corpus_jsonl(path, split, holdout_keys, min_chars, lang) -> List[Tuple[str, str]]:
    """vncompress-vi-v2 `corpus.jsonl`: one paragraph per line.

    `split` matters more here than in the old JSON corpus: 6,766 of the 72,301
    rows are the `test` split, and `eval/test.jsonl` is drawn from those same
    documents. Defaulting to `train` makes training on them impossible rather
    than merely discouraged.
    """
    rows, documents, dropped_split, dropped_lang, dropped_holdout = [], set(), 0, 0, 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if split and row.get('split') != split:
                dropped_split += 1
                continue
            if holdout_keys and document_key(row.get('doc_id', '')) in holdout_keys:
                dropped_holdout += 1
                continue
            # 2,410 rows are labelled `en`, and sampling them shows MediaWiki
            # timeline markup ("barset:Presidents from:1789.40 till:1797.17")
            # and English bibliographies, not Vietnamese prose. Neither teaches
            # a Vietnamese LM anything (dataset_v2_review.md SS5.1).
            if lang and row.get('text_lang') != lang:
                dropped_lang += 1
                continue
            text = row.get('text', '')
            if len(text) <= min_chars:
                continue
            key = document_key(row.get('doc_id', ''))
            rows.append((text, key))
            documents.add(key)
    print(f"Training corpus: {len(rows):,} texts over {len(documents):,} documents "
          f"from {path} (split={split!r})")
    if dropped_split or dropped_lang or dropped_holdout:
        print(f"  dropped: {dropped_split:,} other-split, {dropped_lang:,} non-{lang}, "
              f"{dropped_holdout:,} held-out-document")
    return rows


def load_training_texts_with_docs(
    data_path: Optional[str] = None,
    split: Optional[str] = 'train',
    holdout_docs: Iterable[str] = (),
    min_chars: int = MIN_TRAINING_CHARS,
    lang: Optional[str] = 'vi',
) -> List[Tuple[str, str]]:
    """`load_training_texts`, but each text is paired with its document key.

    For callers that split train/validation themselves: corpus.jsonl averages
    ~17 paragraphs per document, so splitting by row puts siblings of the same
    article on both sides and the held-out score reads high for the wrong
    reason. Corpus shapes that carry no doc_id yield an empty key.
    """
    if data_path and os.path.exists(data_path):
        if data_path.endswith('.jsonl'):
            holdout_keys = {document_key(str(d).replace(' ', '_')) for d in holdout_docs if d}
            return _load_corpus_jsonl(data_path, split, holdout_keys, min_chars, lang)
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'paragraphs' in data:
            return [(p['text'], '') for p in data['paragraphs'] if len(p['text']) > min_chars]
        if isinstance(data, dict) and 'samples' in data:
            return [(s.get('context', ''), '') for s in data['samples'] if len(s.get('context', '')) > min_chars]
        if isinstance(data, list):
            return [(item if isinstance(item, str) else item.get('text', item.get('context', '')), '')
                    for item in data]
        return []

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in (
        os.path.join(here, 'data', 'vncompress_vi_v2', 'corpus.jsonl'),
        os.path.join(here, 'data', 'benchmark', 'training_corpus_v1.json'),
        os.path.join(here, 'data', 'benchmark', 'wikipedia_vi_raw.json'),
        os.path.join(here, 'vcc_bench_data', 'training_corpus_v1.json'),
        os.path.join(here, 'vcc_bench_data', 'wikipedia_vi_raw.json'),
    ):
        if os.path.exists(candidate):
            return load_training_texts_with_docs(candidate, split=split, holdout_docs=holdout_docs,
                                                 min_chars=min_chars, lang=lang)
    return [(t, '') for t in _demo_texts()]


def load_training_texts(
    data_path: Optional[str] = None,
    split: Optional[str] = 'train',
    holdout_docs: Iterable[str] = (),
    min_chars: int = MIN_TRAINING_CHARS,
    lang: Optional[str] = 'vi',
) -> List[str]:
    """Load training texts, preferring vncompress-vi-v2's `corpus.jsonl`.

    `.jsonl` is read as the v2 corpus shape and filtered by `split`,
    `holdout_docs` (documents an external benchmark uses) and `lang`. `.json`
    keeps the older shapes: {"paragraphs": [{"text": ...}]},
    {"samples": [{"context": ...}]}, or a flat list of strings/dicts -- those
    carry no split, so the filters do not apply to them.
    """
    return [text for text, _ in load_training_texts_with_docs(
        data_path, split=split, holdout_docs=holdout_docs, min_chars=min_chars, lang=lang)]


# ============================================================================
# 1. LACC model training (generation-sized model + tone auxiliary loss)
# ============================================================================


class ToneTrainingDataset(Dataset):
    """Causal-LM samples with a per-token dominant-tone label, for the
    phonological consistency auxiliary loss."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512, tone_analyzer=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tone_analyzer = tone_analyzer or get_tone_analyzer()
        self.samples = []
        for text in texts:
            encoded = self._encode(text)
            if encoded and len(encoded['input_ids']) >= 10:
                self.samples.append(encoded)

    def _encode(self, text: str) -> Optional[Dict]:
        ids = self.tokenizer.encode(text, max_length=self.max_length, truncation=True, add_special_tokens=True)
        if len(ids) < 10:
            return None
        tone_labels = []
        for tid in ids:
            ts = self.tokenizer.decode([tid]).replace('▁', ' ').replace('Ġ', ' ').strip()
            tn = self.tone_analyzer.get_dominant_tone(ts[:20])
            tone_labels.append(TONE_NAME_TO_ID.get(tn or 'ngang', 0))
        return {'input_ids': ids, 'labels': list(ids), 'tone_labels': tone_labels, 'length': len(ids)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


class ToneDataCollator:
    """Pads a batch of ToneTrainingDataset samples; tone labels match input shape."""

    def __init__(self, pad_token_id: int, max_length: int = 512):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict:
        max_len = min(max(len(s['input_ids']) for s in batch), self.max_length)
        bs = len(batch)
        input_ids = torch.full((bs, max_len), self.pad_token_id, dtype=torch.long)
        attn_mask = torch.zeros(bs, max_len, dtype=torch.long)
        labels = torch.full((bs, max_len), -100, dtype=torch.long)
        tone_labels = torch.zeros(bs, max_len, dtype=torch.long)
        for i, s in enumerate(batch):
            L = min(len(s['input_ids']), max_len)
            input_ids[i, :L] = torch.tensor(s['input_ids'][:L])
            attn_mask[i, :L] = 1
            labels[i, :L] = torch.tensor(s['labels'][:L])
            tl = s['tone_labels'][:L]
            tone_labels[i, :len(tl)] = torch.tensor(tl)
        return {'input_ids': input_ids, 'attention_mask': attn_mask, 'labels': labels, 'tone_labels': tone_labels}


class ToneAwareTrainer:
    """Manual PyTorch training loop with the auxiliary tone loss:
    L_total = L_LM + lambda_tone * L_tone. A raw loop (rather than the
    HuggingFace Trainer) avoids Trainer/PEFT compatibility issues on some
    platforms."""

    def __init__(self, model, tokenizer, tone_criterion, config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.tone_criterion = tone_criterion
        self.config = config
        self.device = next(model.parameters()).device

    def train(self, train_loader, eval_loader=None) -> int:
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        config, model = self.config, self.model
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=config.get('lr', 2e-4), weight_decay=0.01)
        total_steps = len(train_loader) * config.get('epochs', 3)
        scheduler = get_linear_schedule_with_warmup(optimizer, config.get('warmup', 50), total_steps)

        model.train()
        global_step = 0
        for epoch in range(config.get('epochs', 3)):
            print(f"\n--- Epoch {epoch + 1}/{config.get('epochs', 3)} ---")
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attn = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                tone_labels = batch['tone_labels'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels, output_hidden_states=True)
                lm_loss = outputs.loss
                tone_loss = self.tone_criterion(outputs.hidden_states[-1], tone_labels, attn)
                loss = (lm_loss + tone_loss) / config.get('grad_accum', 4)
                loss.backward()

                if (batch_idx + 1) % config.get('grad_accum', 4) == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                if global_step > 0 and global_step % config.get('log_steps', 10) == 0:
                    print(f"  step={global_step} lm={lm_loss.item():.4f} tone={tone_loss.item():.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")
                if config.get('eval_steps') and global_step % config['eval_steps'] == 0 and eval_loader:
                    self._eval(eval_loader)
                if 0 < config.get('max_steps', -1) <= global_step:
                    break
            if 0 < config.get('max_steps', -1) <= global_step:
                break
        return global_step

    def _eval(self, loader):
        self.model.eval()
        total_lm, total_tone, n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attn = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                tone_labels = batch['tone_labels'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attn, labels=labels, output_hidden_states=True)
                total_lm += outputs.loss.item()
                total_tone += self.tone_criterion(outputs.hidden_states[-1], tone_labels, attn).item()
                n += 1
        print(f"  [Eval] lm={total_lm / n:.4f} tone={total_tone / n:.4f}")
        self.model.train()


def run_lacc_training(
    model_name: str = 'Qwen/Qwen2.5-0.5B-Instruct',
    output_dir: str = './models/lacc',
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_length: int = 512,
    lambda_tone: float = 0.1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_qlora: bool = False,
    max_steps: int = -1,
    train_data_path: Optional[str] = None,
    device: str = 'cuda',
    holdout_docs: Iterable[str] = (),
):
    """Fine-tune `model_name` with LoRA + the phonological consistency
    auxiliary loss. Saves `<output_dir>/final` (LoRA adapter + tokenizer) and
    `<output_dir>/tone_probe.pt` (the trained tone classifier, reusable at
    inference time as this model's LACC tone-probe signal)."""
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("=" * 60)
    print("LACC model training")
    print(f"Model: {model_name}  |  LoRA r={lora_r}  |  lambda_tone={lambda_tone}")
    print(f"Epochs: {num_epochs}  |  Batch: {batch_size}  |  LR: {learning_rate}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(trust_remote_code=True, torch_dtype=torch.float16)
    if use_qlora:
        # Quantized loads need a device_map at load time -- see models.py's
        # module docstring for the Windows/CUDA caching_allocator_warmup
        # segfault this can trigger; retry, or use the non-QLoRA path.
        if device == 'cuda' and torch.cuda.is_available():
            model_kwargs['device_map'] = {'': 0}
            model_kwargs['low_cpu_mem_usage'] = False
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device == 'cuda' and torch.cuda.is_available():
            model = model.to('cuda')
    model.config.output_hidden_states = True

    if use_qlora:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    ))
    model.print_trainable_parameters()

    texts = load_training_texts(train_data_path, holdout_docs=holdout_docs)
    dataset = ToneTrainingDataset(texts, tokenizer, max_length=max_length)
    collator = ToneDataCollator(pad_token_id=tokenizer.pad_token_id, max_length=max_length)
    print(f"  {len(dataset)} training samples from {len(texts)} texts")

    split = int(0.9 * len(dataset))
    train_loader = DataLoader(Subset(dataset, range(split)), batch_size=batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(Subset(dataset, range(split, len(dataset))), batch_size=batch_size, collate_fn=collator)

    hidden_dim = model.config.hidden_size
    device_actual = next(model.parameters()).device
    dtype_actual = next(model.parameters()).dtype
    tone_criterion = PhonologicalConsistencyLoss(hidden_dim=hidden_dim, lambda_tone=lambda_tone).to(device=device_actual, dtype=dtype_actual)

    trainer = ToneAwareTrainer(
        model=model, tokenizer=tokenizer, tone_criterion=tone_criterion,
        config=dict(epochs=num_epochs, lr=learning_rate, warmup=50, grad_accum=4, log_steps=10, eval_steps=None, max_steps=max_steps),
    )
    global_step = trainer.train(train_loader, eval_loader)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, 'final'))
    tokenizer.save_pretrained(os.path.join(output_dir, 'final'))
    torch.save(tone_criterion.state_dict(), os.path.join(output_dir, 'tone_probe.pt'))
    print(f"\nSaved: {output_dir}/final + tone_probe.pt  (optimizer steps={global_step})")
    return model, tokenizer, tone_criterion


# ============================================================================
# 2. SLM training (small Vietnamese LM, LoRA, low VRAM)
# ============================================================================


class VietnameseToneDataset(Dataset):
    """Causal-LM samples plus a Vietnamese tone id for every input token."""

    def __init__(self, texts: List[str], tokenizer, max_length: int):
        analyzer = get_tone_analyzer()
        self.samples = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
            if len(ids) < 10:
                continue
            tones = []
            for token_id in ids:
                piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                tone = analyzer.get_dominant_tone(piece.strip())
                tones.append(TONE_NAME_TO_ID.get(tone or 'ngang', 0))
            self.samples.append((ids, tones))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class SLMCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        width = max(len(ids) for ids, _ in batch)
        bsz = len(batch)
        ids = torch.full((bsz, width), self.pad_id, dtype=torch.long)
        labels = torch.full((bsz, width), -100, dtype=torch.long)
        mask = torch.zeros((bsz, width), dtype=torch.long)
        tones = torch.zeros((bsz, width), dtype=torch.long)
        for row, (sample_ids, sample_tones) in enumerate(batch):
            n = len(sample_ids)
            ids[row, :n] = torch.tensor(sample_ids)
            labels[row, :n] = torch.tensor(sample_ids)
            mask[row, :n] = 1
            tones[row, :n] = torch.tensor(sample_tones)
        return {'input_ids': ids, 'labels': labels, 'attention_mask': mask, 'tone_labels': tones}


def run_slm_training(
    model_name: str = 'chronopt-research/vietnamese-gpt2-base',
    output_dir: str = './models/slm',
    train_data_path: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 8,
    max_length: int = 256,
    lr: float = 1e-4,
    lora_r: int = 8,
    lambda_tone: float = 0.1,
    grad_accum: int = 2,
    max_steps: int = -1,
    gradient_checkpointing: bool = True,
    base_dtype: str = 'float32',
    holdout_docs: Iterable[str] = (),
    load_4bit: bool = False,
):
    """LoRA fine-tune a small Vietnamese causal LM with the tone auxiliary
    loss, tuned for low-VRAM GPUs (default settings fit a T4 16GB; pass
    batch_size=1/max_length=128/grad_accum=8 for a 6GB card). Larger bases
    (e.g. Qwen3-4B) can load in bfloat16 or 4-bit NF4 (`load_4bit=True`, QLoRA).

    Saves `<output_dir>/final` (LoRA adapter, tokenizer, and `val_split.json`
    -- the exact held-out split, so `validate_slm()` always scores the same
    data regardless of later --train-data-path/--max-length choices) and
    `<output_dir>/tone_probe.pt` + `tone_probe_meta.json`.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("SLM training requires an NVIDIA CUDA GPU; CPU training is intentionally disabled.")
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)} | model: {model_name}")
    print(f"batch={batch_size}, seq={max_length}, accum={grad_accum}, LoRA r={lora_r}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_4bit:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            device_map={'': 0},
        )
        resize_embeddings_if_needed(model, tokenizer)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
        autocast_dtype = torch.bfloat16
    else:
        weight_dtype = torch.bfloat16 if base_dtype == 'bfloat16' else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=weight_dtype)
        resize_embeddings_if_needed(model, tokenizer)
        model = model.to(device)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        # FP32 base -> FP16 autocast (needs GradScaler); bf16 base -> bf16 autocast (no scaler).
        autocast_dtype = torch.float16 if weight_dtype == torch.float32 else torch.bfloat16

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.05,
        target_modules=lora_target_modules(model), bias='none',
    ))
    model.print_trainable_parameters()

    dataset = VietnameseToneDataset(
        load_training_texts(train_data_path, holdout_docs=holdout_docs), tokenizer, max_length)
    if len(dataset) < 2:
        raise RuntimeError("Need at least two usable texts in the training dataset.")
    train_n = min(max(1, int(len(dataset) * 0.9)), len(dataset) - 1)
    train_ds, val_ds = random_split(dataset, [train_n, len(dataset) - train_n], generator=torch.Generator().manual_seed(42))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=SLMCollator(tokenizer.pad_token_id), pin_memory=True)

    tone_loss = PhonologicalConsistencyLoss(model.config.hidden_size, lambda_tone=lambda_tone).to(device)
    params = [p for p in list(model.parameters()) + list(tone_loss.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    updates_per_epoch = max(1, (len(loader) + grad_accum - 1) // grad_accum)
    planned = max_steps if max_steps > 0 else updates_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, min(10, planned // 10), planned)

    use_scaler = autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    tone_loss.train()
    for _epoch in range(epochs):
        for batch_i, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            window_start = (batch_i // grad_accum) * grad_accum
            window_size = min(window_start + grad_accum, len(loader)) - window_start
            with torch.amp.autocast('cuda', dtype=autocast_dtype):
                out = model(**{k: batch[k] for k in ('input_ids', 'attention_mask', 'labels')}, output_hidden_states=True)
                tl = tone_loss(out.hidden_states[-1], batch['tone_labels'], batch['attention_mask'])
                scaled_loss = (out.loss + tl) / window_size
            scaler.scale(scaled_loss).backward()
            if (batch_i + 1) % grad_accum == 0 or batch_i + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
                if step == 1 or step % 10 == 0:
                    print(f"step={step} lm={out.loss.item():.4f} tone={tl.item():.4f} "
                          f"vram={torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB")
                if 0 < max_steps <= step:
                    break
        if 0 < max_steps <= step:
            break

    final_dir = os.path.join(output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    # save_embedding_layers=False: PEFT turns this on automatically once
    # embeddings are resized; resize_embeddings_if_needed() is deterministic,
    # so the loader (models.load_scorer) reconstructs the extra rows exactly.
    model.save_pretrained(final_dir, save_embedding_layers=False)
    tokenizer.save_pretrained(final_dir)
    torch.save(tone_loss.state_dict(), os.path.join(output_dir, 'tone_probe.pt'))
    with open(os.path.join(output_dir, 'tone_probe_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'base_model': model_name, 'adapter_dir': final_dir, 'hidden_size': model.config.hidden_size,
            'num_tones': tone_loss.num_tones, 'lambda_tone': lambda_tone, 'max_length': max_length,
            'lora_r': lora_r, 'base_dtype': '4bit-nf4' if load_4bit else base_dtype,
        }, f, ensure_ascii=False, indent=2)

    val_samples = [dataset[i] for i in val_ds.indices]
    with open(os.path.join(final_dir, 'val_split.json'), 'w', encoding='utf-8') as f:
        json.dump({'max_length': max_length, 'base_model': model_name, 'samples': val_samples}, f)

    print(f"Saved LoRA adapter + tokenizer: {final_dir}")
    print(f"Saved held-out validation split ({len(val_samples)} texts): {final_dir}/val_split.json")
    print(f"Saved trained tone probe: {output_dir}/tone_probe.pt (+ tone_probe_meta.json) | optimizer steps={step}")


# ============================================================================
# 3. Query-relevance probe training (wave-2 E4)
# ============================================================================
#
# Same frozen-base-plus-probe recipe as the tone probe / probe-control study,
# but the label is answer-relevance (build_relevance_labels) instead of tone.
# The base model is frozen; only the RelevanceConsistencyLoss probe learns, so
# this is cheap (no fine-tuning) -- it teaches a probe to read "is this token
# answer-relevant" off the SLM's hidden states, the signal wave-1 showed LACC
# was missing. The saved relevance_probe.pt loads via models.load_scorer(
# probe_kind='relevance') and plugs into LACC's tone_source='model' path.


# Tasks whose answer is a verbatim span/needle in the context, so span
# supervision means something. `short_context_extractive_qa` is the v2 name for
# rows under LONG_DOCUMENT_MIN_TOKENS; it is just as extractive as the long ones
# and dropping it would throw away 2,813 of vncompress-vi-v2's 6,000 qa rows.
RELEVANCE_TASKS = ('long_document_qa', 'needle_in_haystack', 'short_context_extractive_qa')


def _relevance_row(context, query, answer, answer_span, task, doc_id, split) -> Optional[dict]:
    """One normalized supervision row, or None if it cannot supervise anything."""
    if len(context) <= 100 or not answer:
        return None
    span = None
    if isinstance(answer_span, (list, tuple)) and len(answer_span) == 2:
        start, end = int(answer_span[0]), int(answer_span[1])
        # Trust the corpus only as far as it checks out: a span that does not
        # quote the answer back is a broken label, not a usable one.
        if 0 <= start < end <= len(context) and context[start:end] == answer:
            span = (start, end)
    return {'context': context, 'query': query or '', 'reference_answer': answer,
            'answer_span': span, 'task': task or '', 'doc_id': doc_id or '', 'split': split or ''}


def _load_relevance_jsonl(path, tasks, split, holdout_keys) -> List[dict]:
    """vncompress-vi-v2 `qa.jsonl`: one JSON object per line, carrying the
    verified `answer_span` that makes exact token labels possible."""
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if split and row.get('split') != split:
                continue
            if tasks and row.get('task') not in tasks:
                continue
            if holdout_keys and document_key(row.get('doc_id', '')) in holdout_keys:
                continue
            sample = _relevance_row(row.get('context', ''), row.get('query', ''), row.get('answer', ''),
                                    row.get('answer_span'), row.get('task'), row.get('doc_id'), row.get('split'))
            if sample:
                out.append(sample)
    return out


def _load_relevance_json(path, tasks, split, holdout_keys) -> List[dict]:
    """VCC-Bench-shaped JSON ({"samples": [{context, reference_answer, task}]}).
    No `answer_span` here, so these rows fall back to syllable-overlap labels."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = data.get('samples', data if isinstance(data, list) else [])
    out = []
    for row in records or ():
        if not isinstance(row, dict):
            continue
        if tasks and row.get('task') not in tasks:
            continue
        doc_id = row.get('doc_id') or row.get('title') or ''
        if holdout_keys and document_key(str(doc_id).replace(' ', '_')) in holdout_keys:
            continue
        sample = _relevance_row(row.get('context', ''), row.get('query', '') or row.get('question', ''),
                                row.get('reference_answer', ''), row.get('answer_span'),
                                row.get('task'), doc_id, row.get('split'))
        if sample:
            out.append(sample)
    return out


def load_relevance_samples(
    data_path: Optional[str],
    tasks: Optional[Sequence[str]] = RELEVANCE_TASKS,
    split: Optional[str] = None,
    holdout_docs: Iterable[str] = (),
) -> List[dict]:
    """Load (context, query, answer, answer_span) supervision rows for E4.

    `.jsonl` reads the vncompress-vi-v2 `qa` shape, `.json` the VCC-Bench shape.
    `split` filters on the row's own split field; `holdout_docs` drops documents
    an external benchmark already uses (reduced through `document_key`, so
    "Hà Nội" and `viquad:Hà_Nội` are the same document).

    Raises rather than substituting demo data. The previous version fell back to
    two hardcoded sentences whenever the path did not parse, which is what the
    shipped default did: `training_corpus_v1.json` is `{metadata, paragraphs}`,
    has no `samples` key, and so trained -- and saved -- a probe on 2 rows while
    looking like a normal run.
    """
    if not data_path:
        raise ValueError(
            "Relevance-probe training needs --data-path: a vncompress-vi-v2 qa.jsonl "
            "or a VCC-Bench-shaped training JSON.")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Relevance corpus not found: {data_path}\n"
            "Fetch the dataset first, e.g.\n"
            "  huggingface-cli download anhalu/vncompress-vi-v2 qa.jsonl "
            "--repo-type dataset --local-dir data/vncompress_vi_v2")
    holdout_keys = {document_key(str(d).replace(' ', '_')) for d in holdout_docs if d}
    reader = _load_relevance_jsonl if data_path.endswith('.jsonl') else _load_relevance_json
    samples = reader(data_path, tuple(tasks) if tasks else (), split, holdout_keys)
    if not samples:
        raise RuntimeError(
            f"No usable relevance rows in {data_path} "
            f"(split={split!r}, tasks={tuple(tasks) if tasks else 'any'}, "
            f"{len(holdout_keys)} held-out document(s)). Expected rows with a context "
            "longer than 100 chars and a non-empty answer.")
    return samples


def _span_window(ids, offsets, answer_span, budget: int):
    """Token window of at most `budget` tokens that contains the answer span.

    v2 contexts run to 24,000 characters and 81% of answers start past char
    1,024, so encoding from the beginning and truncating at max_length cuts the
    positive labels off 4,857 of 6,000 rows -- the sample then either drops out
    or trains on whatever syllables happened to collide. Centring the window on
    the span keeps the supervision the corpus actually paid for.
    """
    start_char, end_char = answer_span
    hits = [i for i, (a, b) in enumerate(offsets)
            if b > a and a < end_char and b > start_char]
    if not hits:
        return None
    first, last = hits[0], hits[-1]
    if last - first + 1 >= budget:
        lo, hi = first, first + budget
    else:
        lo = max(0, first - (budget - (last - first + 1)) // 2)
        hi = min(len(ids), lo + budget)
        lo = max(0, hi - budget)
    return list(ids[lo:hi]), list(offsets[lo:hi])


class RelevanceDataset(Dataset):
    """Causal-LM token ids plus per-token answer-relevance labels.

    Labels come from the corpus's verified `answer_span` when there is one
    (exact: a token is positive iff it overlaps the span) and from
    syllable overlap otherwise. With `query_template`, the question is prefixed
    to the context and its tokens are labelled `ignore_index`, so the probe
    reads question-conditioned hidden states but is scored only on the context.
    """

    def __init__(self, samples: List[dict], tokenizer, max_length: int,
                 query_template: Optional[str] = None, ignore_index: int = -100):
        self.samples: List[Tuple[List[int], List[int]]] = []
        self.stats = {'span_labelled': 0, 'overlap_labelled': 0, 'dropped_no_positive': 0,
                      'dropped_too_short': 0, 'dropped_span_untokenizable': 0}
        can_offset = bool(getattr(tokenizer, 'is_fast', False))
        max_prefix = max(16, max_length // 4)

        for sample in samples:
            prefix_ids: List[int] = []
            if query_template and sample.get('query'):
                prefix_ids = tokenizer.encode(query_template.format(query=sample['query']),
                                              add_special_tokens=False)[:max_prefix]
            budget = max_length - len(prefix_ids)
            if budget < 10:
                self.stats['dropped_too_short'] += 1
                continue

            ids, labels = None, None
            if sample.get('answer_span') and can_offset:
                encoded = tokenizer(sample['context'], add_special_tokens=False,
                                    return_offsets_mapping=True)
                window = _span_window(encoded['input_ids'], encoded['offset_mapping'],
                                      sample['answer_span'], budget)
                if window is None:
                    self.stats['dropped_span_untokenizable'] += 1
                else:
                    ids, offsets = window
                    labels = build_relevance_labels_from_offsets(
                        offsets, sample['answer_span'], ignore_index=ignore_index)
                    self.stats['span_labelled'] += 1
            if ids is None:
                ids = tokenizer.encode(sample['context'], add_special_tokens=True,
                                       truncation=True, max_length=budget)
                labels = build_relevance_labels(tokenizer, ids, sample['reference_answer'],
                                                ignore_index=ignore_index)
                self.stats['overlap_labelled'] += 1

            if len(ids) < 10:
                self.stats['dropped_too_short'] += 1
                continue
            if not any(lab == 1 for lab in labels):
                self.stats['dropped_no_positive'] += 1
                continue
            self.samples.append((prefix_ids + ids, [ignore_index] * len(prefix_ids) + labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def relevance_class_weights(dataset, cap: float = 50.0) -> List[float]:
    """Inverse-frequency CE weights [irrelevant, relevant] for a RelevanceDataset.

    The answer span is a small minority of every window, so unweighted training
    converges on "nothing is relevant" -- high accuracy, useless probe. The
    negative class keeps weight 1 and the positive one is scaled by how much
    rarer it is, capped so a freak batch cannot produce an enormous gradient.
    """
    positive = negative = 0
    for _, labels in dataset.samples:
        for label in labels:
            if label == 1:
                positive += 1
            elif label == 0:
                negative += 1
    if not positive:
        return [1.0, 1.0]
    return [1.0, min(negative / positive, cap)]


class RelevanceCollator:
    """Pads a batch; relevance labels are padded with -100 (ignore_index) so
    padding neither trains nor scores the probe."""

    def __init__(self, pad_id: int, ignore_index: int = -100):
        self.pad_id = pad_id
        self.ignore_index = ignore_index

    def __call__(self, batch):
        width = max(len(ids) for ids, _ in batch)
        bsz = len(batch)
        ids = torch.full((bsz, width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((bsz, width), dtype=torch.long)
        labels = torch.full((bsz, width), self.ignore_index, dtype=torch.long)
        for row, (sample_ids, sample_labels) in enumerate(batch):
            n = len(sample_ids)
            ids[row, :n] = torch.tensor(sample_ids)
            mask[row, :n] = 1
            labels[row, :n] = torch.tensor(sample_labels)
        return {'input_ids': ids, 'attention_mask': mask, 'relevance_labels': labels}


def run_relevance_probe_training(
    adapter_dir: Optional[str] = None,
    base_model: str = 'chronopt-research/vietnamese-gpt2-base',
    output_dir: str = './models/relevance',
    train_data_path: Optional[str] = None,
    use_adapter: bool = True,
    epochs: int = 3,
    batch_size: int = 8,
    max_length: int = 512,
    lr: float = 1e-3,
    max_steps: int = -1,
    base_dtype: str = 'float32',
    load_4bit: bool = False,
    seed: int = 42,
    split: Optional[str] = 'train',
    val_split: Optional[str] = 'validation',
    holdout_docs: Iterable[str] = (),
    query_conditioned: bool = True,
    min_samples: int = 32,
    balance_classes: bool = True,
    focal_gamma: float = 0.0,
    class_weight_cap: float = 50.0,
):
    """Train ONLY a query-relevance probe on a frozen SLM's hidden states (E4).

    `adapter_dir` (a LoRA adapter from `train.py --mode slm`) is loaded on top
    of its base when given and `use_adapter=True`; otherwise `base_model` is
    used directly. The base is frozen -- only the RelevanceConsistencyLoss probe
    learns -- so this is cheap. Saves `<output_dir>/relevance_probe.pt` +
    `relevance_probe_meta.json` (probe_kind='relevance'), loadable by
    models.load_scorer(..., probe_kind='relevance')."""
    if not torch.cuda.is_available():
        raise RuntimeError("Relevance-probe training requires an NVIDIA CUDA GPU.")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    device = torch.device('cuda')

    resolved_base = base_model
    if adapter_dir:
        try:
            from peft import PeftConfig
            resolved_base = PeftConfig.from_pretrained(adapter_dir).base_model_name_or_path
        except Exception:
            pass
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_4bit:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            resolved_base,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            device_map={'': 0})
        resize_embeddings_if_needed(model, tokenizer)
    else:
        weight_dtype = torch.bfloat16 if base_dtype == 'bfloat16' else torch.float32
        model = AutoModelForCausalLM.from_pretrained(resolved_base, torch_dtype=weight_dtype)
        resize_embeddings_if_needed(model, tokenizer)
        model = model.to(device)
    if adapter_dir and use_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)

    query_template = QUERY_PREFIX_TEMPLATE if query_conditioned else None
    holdout_docs = tuple(holdout_docs)
    samples = load_relevance_samples(train_data_path, split=split, holdout_docs=holdout_docs)
    dataset = RelevanceDataset(samples, tokenizer, max_length, query_template=query_template)
    if len(dataset) < min_samples:
        raise RuntimeError(
            f"Only {len(dataset)} usable relevance-labelled sample(s) from {train_data_path} "
            f"(split={split!r}); need at least {min_samples}. Label stats: {dataset.stats}. "
            "Lower --min-samples only if you mean to train on that few.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=RelevanceCollator(tokenizer.pad_token_id))
    train_documents = {document_key(s['doc_id']) for s in samples if s.get('doc_id')}
    print(f"  {len(dataset)} relevance-labelled samples over {len(train_documents)} document(s)")
    print(f"  labels: {dataset.stats}")
    print(f"  query-conditioned: {query_conditioned}")

    val_loader = None
    val_dataset = None
    if val_split:
        try:
            val_samples = load_relevance_samples(train_data_path, split=val_split, holdout_docs=holdout_docs)
        except RuntimeError:
            val_samples = []
        if val_samples:
            val_dataset = RelevanceDataset(val_samples, tokenizer, max_length, query_template=query_template)
            if len(val_dataset):
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                        collate_fn=RelevanceCollator(tokenizer.pad_token_id))
                print(f"  {len(val_dataset)} validation samples")
    if val_loader is None:
        print("  WARNING: no validation split -- the probe will be saved without an accuracy number.")

    class_weights = relevance_class_weights(dataset, cap=class_weight_cap) if balance_classes else None
    if class_weights:
        print(f"  class weights [irrelevant, relevant]: "
              f"[{class_weights[0]:.2f}, {class_weights[1]:.2f}] (cap={class_weight_cap})")
    print(f"  focal gamma: {focal_gamma}" + ("  (0 = plain cross-entropy)" if not focal_gamma else ""))
    probe = RelevanceConsistencyLoss(model.config.hidden_size, lambda_relevance=1.0,
                                     class_weights=class_weights,
                                     focal_gamma=focal_gamma).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    step = 0
    for epoch in range(epochs):
        probe.train()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                hs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                           output_hidden_states=True).hidden_states[-1]
            with torch.amp.autocast('cuda', dtype=torch.float16):
                loss = probe(hs.float(), batch['relevance_labels'], batch['attention_mask'])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step == 1 or step % 20 == 0:
                print(f"  epoch={epoch} step={step} relevance_loss={loss.item():.4f}")
            if 0 < max_steps <= step:
                break
        if 0 < max_steps <= step:
            break

    val_metrics = evaluate_relevance_probe(probe, model, val_loader, device) if val_loader else None
    if val_metrics:
        budget_str = ' '.join(
            f"recall@{k.split('_at_')[1]}={v:.4f}"
            for k, v in val_metrics.items() if k.startswith('recall_at_')
        )
        print(f"  validation: PR-AUC={val_metrics['pr_auc']:.4f} {budget_str} "
              f"(argmax F1={val_metrics['f1']:.4f}, positives={val_metrics['support_positive']}/"
              f"{val_metrics['support_total']})")

    os.makedirs(output_dir, exist_ok=True)
    probe_path = os.path.join(output_dir, 'relevance_probe.pt')
    torch.save(probe.state_dict(), probe_path)
    with open(os.path.join(output_dir, 'relevance_probe_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'base_model': resolved_base, 'adapter_dir': adapter_dir, 'hidden_size': model.config.hidden_size,
            'num_classes': probe.num_classes, 'probe_kind': 'relevance', 'max_length': max_length,
            'base_dtype': '4bit-nf4' if load_4bit else base_dtype,
            # Provenance: without these a probe trained on a toy corpus is
            # indistinguishable from a real one once the run's stdout is gone.
            'train_data_path': train_data_path,
            'train_split': split,
            'num_train_samples': len(dataset),
            'num_train_documents': len(train_documents),
            'num_val_samples': len(val_dataset) if val_dataset is not None else 0,
            'label_stats': dataset.stats,
            'holdout_docs': list(holdout_docs),
            'epochs': epochs, 'optimizer_steps': step, 'lr': lr, 'seed': seed,
            # Inference must rebuild this prefix or the probe is read in a
            # format it was never trained in (compression.LACCScorer).
            'query_conditioned': bool(query_conditioned),
            'query_template': query_template,
            'class_weights': class_weights,
            'class_weight_cap': class_weight_cap,
            'focal_gamma': focal_gamma,
            'val_metrics': val_metrics,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved relevance probe: {probe_path} (+ relevance_probe_meta.json) | optimizer steps={step}")
    return probe


@torch.no_grad()
def evaluate_relevance_probe(
    probe, model, loader, device, budget_ratios: Sequence[float] = (2.0, 4.0, 8.0),
) -> Dict[str, float]:
    """Held-out PR-AUC + recall@budget on the RELEVANT class (docs/lacc_coling2027_tasklist.md G0).

    Argmax F1 fixes a threshold (0.5) the compressor never actually uses: LACC
    ranks tokens by score and keeps the top 1/ratio, so accuracy/F1 at a fixed
    threshold does not measure what deployment needs. PR-AUC is threshold-free
    ranking quality; recall@budget is the recall of true-relevant tokens when
    only the top 1/ratio survive, at the same ratios VCC-Bench sweeps (2x/4x/8x)
    -- computed per window (one row = one document's scored window) and pooled,
    since that is the unit the compressor actually ranks within.

    Accuracy/precision/recall/F1 (argmax) are still returned for continuity
    with older reports, but are no longer the headline number: a span covers a
    handful of tokens in a 512-token window, so "everything is irrelevant"
    already scores ~0.98 accuracy.
    """
    from sklearn.metrics import average_precision_score

    probe.eval()
    true_positive = false_positive = false_negative = correct = total = 0
    all_scores: List[float] = []
    all_labels: List[int] = []
    recall_hits = {r: 0 for r in budget_ratios}
    recall_support = {r: 0 for r in budget_ratios}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        hidden = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                       output_hidden_states=True).hidden_states[-1]
        logits = probe.tone_classifier(hidden.float())
        probs = torch.softmax(logits, dim=-1)[..., 1]  # P(relevant)
        predicted = logits.argmax(dim=-1)
        labels = batch['relevance_labels']
        scored = labels != -100

        for row_probs, row_labels, row_scored in zip(probs, labels, scored):
            row_probs = row_probs[row_scored]
            row_labels = row_labels[row_scored]
            if row_labels.numel() == 0:
                continue
            all_scores.extend(row_probs.tolist())
            all_labels.extend(row_labels.tolist())
            n_pos = int((row_labels == 1).sum())
            if n_pos == 0:
                continue
            order = torch.argsort(row_probs, descending=True)
            for ratio in budget_ratios:
                k = max(1, int(row_labels.numel() / ratio))
                kept = order[:k]
                recall_hits[ratio] += int((row_labels[kept] == 1).sum())
                recall_support[ratio] += n_pos

        predicted, labels = predicted[scored], labels[scored]
        correct += int((predicted == labels).sum())
        total += int(labels.numel())
        true_positive += int(((predicted == 1) & (labels == 1)).sum())
        false_positive += int(((predicted == 1) & (labels == 0)).sum())
        false_negative += int(((predicted == 0) & (labels == 1)).sum())

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pr_auc = (
        float(average_precision_score(all_labels, all_scores))
        if all_labels and any(l == 1 for l in all_labels) else 0.0
    )
    recall_at_budget = {
        f'recall_at_{ratio:g}x': (recall_hits[ratio] / recall_support[ratio] if recall_support[ratio] else 0.0)
        for ratio in budget_ratios
    }
    return {
        'pr_auc': pr_auc,
        **recall_at_budget,
        'accuracy': correct / total if total else 0.0,
        'precision': precision, 'recall': recall, 'f1': f1,
        'support_positive': true_positive + false_negative, 'support_total': total,
    }


# ============================================================================
# SLM validation (perplexity + tone-probe accuracy on a trained checkpoint)
# ============================================================================


def _tone_lookup_baseline(validation, tokenizer) -> Dict[str, float]:
    """Accuracy of the training-free predictor that maps a token id to a tone
    by decoding + running the tone analyzer -- exactly how VietnameseToneDataset
    builds its labels. This is the CEILING for any tone predictor reading a
    representation of token i: the label is a deterministic function of token
    i, so this predictor needs no training data at all."""
    analyzer = get_tone_analyzer()
    cache: Dict[int, int] = {}

    def lookup(tid):
        if tid not in cache:
            piece = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            cache[tid] = TONE_NAME_TO_ID.get(analyzer.get_dominant_tone(piece.strip()) or 'ngang', 0)
        return cache[tid]

    total = correct = marked_total = marked_correct = 0
    for ids, tones in validation:
        for tid, label in zip(ids, tones):
            pred = lookup(tid)
            total += 1
            correct += pred == label
            if label != 0:
                marked_total += 1
                marked_correct += pred == label
    return {'all': correct / max(total, 1), 'marked': marked_correct / max(marked_total, 1)}


def _per_class_prf(confusion: torch.Tensor):
    """Precision/recall/F1 per class from a (C, C) [true][pred] matrix."""
    tp = confusion.diag().double()
    support = confusion.sum(1).double()
    predicted = confusion.sum(0).double()
    precision = tp / predicted.clamp(min=1)
    recall = tp / support.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return precision, recall, f1, support


def _split_fingerprint(validation) -> str:
    """Stable hash of a split's exact token-id sequences, so a paired
    comparison can refuse to compare two runs scored on different splits."""
    h = hashlib.sha256()
    for ids, _ in validation:
        h.update(b','.join(str(i).encode() for i in ids))
        h.update(b'|')
    return h.hexdigest()[:16]


def validate_slm(
    adapter_dir: str = './models/slm/final',
    tone_probe_path: str = './models/slm/tone_probe.pt',
    train_data_path: Optional[str] = None,
    max_length: int = 128,
    batch_size: int = 1,
    no_adapter: bool = False,
    dump_per_sample: Optional[str] = None,
    dtype: str = 'float32',
    holdout_docs: Iterable[str] = (),
    load_4bit: bool = False,
) -> Dict:
    """Evaluate a trained SLM checkpoint's held-out LM loss/perplexity and
    Vietnamese tone-probe accuracy (macro-F1, confusion matrix, and the
    training-free lookup baseline / majority-class baseline for honest
    comparison). Returns a dict of the headline numbers; full detail is
    printed. `no_adapter=True` evaluates the raw base model (a fair
    perplexity baseline) and skips tone-probe scoring, since the probe was
    trained jointly with the LoRA-adapted hidden states."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device('cuda')
    config = PeftConfig.from_pretrained(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_4bit:
        from transformers import BitsAndBytesConfig

        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            device_map={'': 0})
        resize_embeddings_if_needed(base, tokenizer)
    else:
        weight_dtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float32
        base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, dtype=weight_dtype)
        resize_embeddings_if_needed(base, tokenizer)  # must match run_slm_training() exactly
        base = base.to(device)

    if no_adapter:
        model, probe = base.eval(), None
        print("[no_adapter] Evaluating the raw base model, no LoRA. Tone accuracy skipped.")
    else:
        model = PeftModel.from_pretrained(base, adapter_dir).eval()
        probe = PhonologicalConsistencyLoss(model.config.hidden_size, lambda_tone=0.0).to(device)
        probe.load_state_dict(torch.load(tone_probe_path, map_location=device, weights_only=True))
        probe.eval()
    model.config.use_cache = False

    val_path = os.path.join(adapter_dir, 'val_split.json')
    if os.path.exists(val_path):
        with open(val_path, encoding='utf-8') as f:
            saved = json.load(f)
        validation = [tuple(s) for s in saved['samples']]
        print(f"Loaded held-out split saved at training time: {len(validation)} texts")
    else:
        print("[WARN] val_split.json not found; rebuilding the split from train_data_path -- "
              "pass the SAME train_data_path/max_length used during training.")
        ds = VietnameseToneDataset(
            load_training_texts(train_data_path, holdout_docs=holdout_docs), tokenizer, max_length)
        if len(ds) < 2:
            raise RuntimeError("Need at least two valid texts.")
        train_n = min(max(1, int(len(ds) * 0.9)), len(ds) - 1)
        _, validation = random_split(ds, [train_n, len(ds) - train_n], generator=torch.Generator().manual_seed(42))
        validation = [tuple(s) for s in validation]
    loader = DataLoader(validation, batch_size=batch_size, collate_fn=SLMCollator(tokenizer.pad_token_id))

    num_tones = probe.num_tones if probe is not None else 0
    confusion = torch.zeros(num_tones, num_tones, dtype=torch.long)
    sample_nll_sums, sample_ntokens = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=probe is not None)
                if probe is not None:
                    logits = probe.tone_classifier(out.hidden_states[-1])
            shift_logits = out.logits[:, :-1, :].float()
            shift_labels = batch['labels'][:, 1:]
            tok_nll = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
                ignore_index=-100, reduction='none',
            ).reshape(shift_labels.shape)
            sample_nll_sums.extend(tok_nll.sum(1).tolist())
            sample_ntokens.extend((shift_labels != -100).sum(1).tolist())
            if probe is not None:
                pred = logits.argmax(-1)
                mask = batch['attention_mask'].bool()
                t = batch['tone_labels'][mask].flatten()
                p = pred[mask].flatten()
                confusion += torch.bincount(t * num_tones + p, minlength=num_tones * num_tones).reshape(num_tones, num_tones).cpu()

    total_tokens = sum(sample_ntokens)
    nll = sum(sample_nll_sums) / max(total_tokens, 1)
    print(f"Validation texts: {len(validation)}")
    print(f"Scored tokens: {total_tokens}")
    print(f"LM validation loss (NLL): {nll:.4f}")
    print(f"Perplexity: {math.exp(min(nll, 20)):.2f}")

    result: Dict = {'nll': nll, 'perplexity': math.exp(min(nll, 20)), 'num_samples': len(validation), 'total_tokens': total_tokens}

    if dump_per_sample:
        os.makedirs(os.path.dirname(os.path.abspath(dump_per_sample)), exist_ok=True)
        with open(dump_per_sample, 'w', encoding='utf-8') as f:
            json.dump({
                'adapter_dir': adapter_dir, 'no_adapter': no_adapter, 'base_model': config.base_model_name_or_path,
                'num_samples': len(validation), 'split_fingerprint': _split_fingerprint(validation),
                'corpus_nll': nll, 'nll_sums': sample_nll_sums, 'n_tokens': sample_ntokens,
            }, f)
        print(f"Wrote per-sample NLL: {dump_per_sample}")

    if probe is None:
        print("Tone accuracy: skipped (no_adapter). Compare the Perplexity line above against a "
              "matching adapter run's Perplexity for a fair before/after comparison.")
        return result

    all_count = int(confusion.sum().item())
    all_correct = int(confusion.diag().sum().item())
    marked_count = int(confusion[1:].sum().item())
    marked_correct = int(confusion.diag()[1:].sum().item())
    print(f"Tone accuracy (all tokens): {all_correct / max(all_count, 1):.2%}")
    print(f"Tone accuracy (marked tones only): {marked_correct / max(marked_count, 1):.2%} ({marked_count} tokens)")
    ngang_tokens = all_count - marked_count
    print(f"Majority-class baseline (always predict ngang): {ngang_tokens / max(all_count, 1):.2%}")
    print("-> Judge tone learning by 'marked tones only' vs this baseline, not 'all tokens'.")

    ceiling = _tone_lookup_baseline(validation, tokenizer)
    print("\nTraining-free token-id lookup (tokenizer + tone analyzer, no training):")
    print(f"  all tokens: {ceiling['all']:.2%} | marked tones only: {ceiling['marked']:.2%}")
    print("-> This is the ceiling, and it is free. The probe measures how much tone information")
    print("   survives in the hidden states, NOT whether tone is predictable.")

    precision, recall, f1, support = _per_class_prf(confusion)
    present = [c for c in range(num_tones) if support[c] > 0]
    macro_f1 = float(f1[present].mean())
    marked_present = [c for c in present if c != 0]
    macro_f1_marked = float(f1[marked_present].mean()) if marked_present else float('nan')
    print(f"\nPer-tone breakdown (macro-F1 all: {macro_f1:.4f} | marked only: {macro_f1_marked:.4f})")
    names = {v: k for k, v in TONE_NAME_TO_ID.items()}
    print(f"  {'tone':<10}{'support':>10}{'precision':>11}{'recall':>9}{'f1':>8}")
    for c in present:
        print(f"  {names.get(c, f'class{c}'):<10}{int(support[c]):>10}{precision[c]:>11.4f}{recall[c]:>9.4f}{f1[c]:>8.4f}")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    header = ''.join(f"{names.get(c, c):>10}" for c in present)
    print(f"  {'true\\pred':<10}{header}")
    for r in present:
        row = ''.join(f"{int(confusion[r][c]):>10}" for c in present)
        print(f"  {names.get(r, r):<10}{row}")

    result.update({
        'tone_accuracy_all': all_correct / max(all_count, 1),
        'tone_accuracy_marked': marked_correct / max(marked_count, 1),
        'macro_f1_all': macro_f1, 'macro_f1_marked': macro_f1_marked,
        'lookup_baseline': ceiling,
    })
    return result
