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
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .linguistics import TONE_NAME_TO_ID, PhonologicalConsistencyLoss, get_tone_analyzer
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


def load_training_texts(data_path: Optional[str] = None) -> List[str]:
    """Load training texts from a JSON file, falling back to the bundled
    VCC-Bench training corpus, then to a small built-in demo corpus.

    Accepted JSON shapes: {"paragraphs": [{"text": ...}, ...]},
    {"samples": [{"context": ...}, ...]}, or a flat list of strings/dicts.
    """
    if data_path and os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'paragraphs' in data:
            return [p['text'] for p in data['paragraphs'] if len(p['text']) > 200]
        if isinstance(data, dict) and 'samples' in data:
            return [s.get('context', '') for s in data['samples'] if len(s.get('context', '')) > 200]
        if isinstance(data, list):
            return [item if isinstance(item, str) else item.get('text', item.get('context', '')) for item in data]
        return []

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in (
        os.path.join(here, 'data', 'benchmark', 'training_corpus_v1.json'),
        os.path.join(here, 'data', 'benchmark', 'wikipedia_vi_raw.json'),
        os.path.join(here, 'vcc_bench_data', 'training_corpus_v1.json'),
        os.path.join(here, 'vcc_bench_data', 'wikipedia_vi_raw.json'),
    ):
        if os.path.exists(candidate):
            return load_training_texts(candidate)
    return _demo_texts()


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

    texts = load_training_texts(train_data_path)
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

    dataset = VietnameseToneDataset(load_training_texts(train_data_path), tokenizer, max_length)
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
        ds = VietnameseToneDataset(load_training_texts(train_data_path), tokenizer, max_length)
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
