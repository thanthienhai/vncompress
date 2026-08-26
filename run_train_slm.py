#!/usr/bin/env python3
"""Memory-conscious tone-aware LoRA training for small Vietnamese Causal LMs.

Default: chronopt-research/vietnamese-gpt2-base (137M parameters).
Tuned for an NVIDIA T4 16 GB: batch=8, sequence length=256, FP16 AMP and
activation checkpointing.  For a 6 GB card use --batch-size 1 --max-length 128
--grad-accum 8.  This is a *new* trainer; run_training.py is left unchanged
for Qwen-family models.

Examples:
  python run_train_slm.py --quick
  python run_train_slm.py --epochs 3
  python run_train_slm.py --batch-size 1 --max-length 128 --grad-accum 8  # 6 GB
  python run_train_slm.py --model VLAI-AIVN/vigpt2-aio --quick
"""
import argparse
import json
import os
import sys
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class VietnameseToneDataset(Dataset):
    """Causal-LM samples plus a Vietnamese tone ID for every input token."""
    def __init__(self, texts, tokenizer, max_length):
        from vncompress.tone_aware import TONE_NAME_TO_ID, get_tone_analyzer
        self.samples = []
        analyzer = get_tone_analyzer()
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True,
                                   truncation=True, max_length=max_length)
            if len(ids) < 10:
                continue
            tones = []
            for token_id in ids:
                piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                tone = analyzer.get_dominant_tone(piece.strip())
                tones.append(TONE_NAME_TO_ID.get(tone or "ngang", 0))
            self.samples.append((ids, tones))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class Collator:
    def __init__(self, pad_id):
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
        return {"input_ids": ids, "labels": labels, "attention_mask": mask,
                "tone_labels": tones}


def resize_embeddings_if_needed(model, tokenizer) -> bool:
    """Grow the embedding matrix to cover every tokenizer id, deterministically.

    Some community checkpoints (e.g. chronopt-research/vietnamese-gpt2-base)
    ship a tokenizer with more ids than the checkpoint has embedding rows --
    here `<|endoftext|>` sits at id 50257 while the matrix has 50257 rows, so
    valid ids stop at 50256. That id is also pad_token_id, so it appears in
    `input_ids` whenever a batch pads, and an un-resized model dies with a CUDA
    `srcIndex < srcSelectDimSize` assertion on nearly every batch.

    The new rows are zeroed rather than left to `resize_token_embeddings`'s
    random mean-resizing, for two reasons: the row is never a prediction
    target (padding is masked to -100) so its value is irrelevant to training,
    and a *deterministic* row means the adapter does not have to ship the whole
    embedding matrix to be reloadable. That matters -- PEFT flips
    `save_embedding_layers` on as soon as embeddings are resized, which grew
    the saved adapter from 4.7 MB to 313 MB of frozen, untrained weights.

    Returns True if a resize happened.
    """
    if len(tokenizer) == model.get_input_embeddings().weight.shape[0]:
        return False
    old_rows = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        model.get_input_embeddings().weight[old_rows:].zero_()
        out = model.get_output_embeddings()
        if out is not None and out.weight.shape[0] >= len(tokenizer):
            out.weight[old_rows:].zero_()
    return True


def load_texts(path: Optional[str]):
    """Reuse the project's VCC-Bench/demo data loader."""
    from run_training import load_training_texts
    return load_training_texts(path)


def target_modules(model):
    """PEFT names for GPT-2-style SLMs; retain a useful Qwen fallback."""
    model_type = getattr(model.config, "model_type", "")
    if model_type in {"gpt2", "gpt_neo", "gptj"}:
        # GPT-2 uses fused QKV Conv1D (c_attn), projection and MLP modules.
        return ["c_attn", "c_proj", "c_fc"]
    if model_type in {"qwen2", "qwen"}:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    raise ValueError(f"Unsupported model_type={model_type!r}. Add its PEFT target modules to target_modules().")


def main():
    ap = argparse.ArgumentParser(description="Train a small Vietnamese CausalLM on low-VRAM GPUs")
    ap.add_argument("--model", default="chronopt-research/vietnamese-gpt2-base")
    ap.add_argument("--output-dir", default="./trained_slm")
    ap.add_argument("--train-data-path")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lambda-tone", type=float, default=0.1)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    ap.add_argument("--quick", action="store_true", help="30 optimizer steps, 1 epoch, 128 tokens")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This low-VRAM trainer requires an NVIDIA CUDA GPU; CPU training is intentionally disabled.")
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    from peft import LoraConfig, TaskType, get_peft_model
    from vncompress.tone_aware import PhonologicalConsistencyLoss

    if args.quick:
        args.epochs, args.max_steps, args.max_length = 1, 30, min(args.max_length, 128)
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} | model: {args.model}")
    print(f"batch={args.batch_size}, seq={args.max_length}, accum={args.grad_accum}, LoRA r={args.lora_r}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Keep master/trainable weights in FP32 for stable GradScaler updates.
    # AMP below still performs CUDA matrix operations in FP16; this 137M model
    # remains comfortably within a T4's 16 GB even with FP32 weights.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    resize_embeddings_if_needed(model, tokenizer)
    model = model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_r * 2,
        lora_dropout=0.05, target_modules=target_modules(model), bias="none",
    ))
    model.print_trainable_parameters()

    dataset = VietnameseToneDataset(load_texts(args.train_data_path), tokenizer, args.max_length)
    if len(dataset) < 2:
        raise RuntimeError("Need at least two usable texts in the training dataset.")
    train_n = max(1, int(len(dataset) * .9))
    train_n = min(train_n, len(dataset) - 1)
    train_ds, val_ds = random_split(dataset, [train_n, len(dataset) - train_n], generator=torch.Generator().manual_seed(42))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=Collator(tokenizer.pad_token_id), pin_memory=True)
    # Keep trainable probe weights in FP32. GradScaler cannot unscale FP16
    # gradients; autocast still runs its matrix multiplications in FP16.
    tone_loss = PhonologicalConsistencyLoss(model.config.hidden_size, lambda_tone=args.lambda_tone).to(device)
    # Include the probe parameters: unlike the old script, tone_probe is trained too.
    params = [p for p in list(model.parameters()) + list(tone_loss.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    updates_per_epoch = max(1, (len(loader) + args.grad_accum - 1) // args.grad_accum)
    planned = args.max_steps if args.max_steps > 0 else updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, min(10, planned // 10), planned)
    scaler = torch.amp.GradScaler("cuda")

    step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train(); tone_loss.train()
    for epoch in range(args.epochs):
        for batch_i, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # Divide by the true number of micro-batches in this accumulation
            # window (not always grad_accum) so the final partial group — and
            # small corpora where len(loader) < grad_accum — are weighted right.
            window_start = (batch_i // args.grad_accum) * args.grad_accum
            window_size = min(window_start + args.grad_accum, len(loader)) - window_start
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(**{k: batch[k] for k in ("input_ids", "attention_mask", "labels")}, output_hidden_states=True)
                tl = tone_loss(out.hidden_states[-1], batch["tone_labels"], batch["attention_mask"])
                loss = out.loss + tl
                scaled_loss = loss / window_size
            scaler.scale(scaled_loss).backward()
            # Also flush a partial accumulation at the end of a small dataset.
            if (batch_i + 1) % args.grad_accum == 0 or batch_i + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                scheduler.step(); step += 1
                if step == 1 or step % 10 == 0:
                    print(f"step={step} lm={out.loss.item():.4f} tone={tl.item():.4f} vram={torch.cuda.max_memory_allocated()/2**30:.2f} GB")
                if args.max_steps > 0 and step >= args.max_steps:
                    break
        if args.max_steps > 0 and step >= args.max_steps:
            break

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    # save_embedding_layers=False: PEFT turns it on automatically once
    # embeddings are resized, which shipped 2 x [vocab, 768] frozen tensors
    # (4.7 MB -> 313 MB) that were never trained. resize_embeddings_if_needed()
    # is deterministic, so the loader reconstructs them exactly.
    model.save_pretrained(final_dir, save_embedding_layers=False)
    tokenizer.save_pretrained(final_dir)
    torch.save(tone_loss.state_dict(), os.path.join(args.output_dir, "tone_probe.pt"))
    # Persist the exact held-out split so evaluate_slm.py scores the same
    # validation set regardless of --train-data-path/--max-length at eval time.
    val_samples = [dataset[i] for i in val_ds.indices]
    with open(os.path.join(final_dir, "val_split.json"), "w", encoding="utf-8") as f:
        json.dump({"max_length": args.max_length, "base_model": args.model,
                   "samples": val_samples}, f)
    print(f"Saved LoRA adapter and tokenizer: {final_dir}")
    print(f"Saved held-out validation split ({len(val_samples)} texts): {final_dir}/val_split.json")
    print(f"Saved trained tone probe: {args.output_dir}/tone_probe.pt | optimizer steps={step}")


if __name__ == "__main__":
    main()
