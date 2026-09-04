#!/usr/bin/env python3
"""Tone-aware LoRA training for Vietnamese Causal LMs (SLM up to a few-B base).

Default: chronopt-research/vietnamese-gpt2-base (137M parameters), FP32 master
weights + FP16 AMP, tuned for a T4 16 GB (batch=8, len=256) or a 6 GB card
(--batch-size 1 --max-length 128 --grad-accum 8).

Larger bases (e.g. Qwen3-4B) load in bfloat16 or 4-bit NF4 (QLoRA) so they fit a
single GPU; the tone probe auto-sizes to the base's hidden dim, and everything
downstream (evaluate_slm.py, the slm_tone_probe compressor) is base-agnostic.

Examples:
  python run_train_slm.py --quick
  python run_train_slm.py --batch-size 1 --max-length 128 --grad-accum 8   # 6 GB, gpt2-base
  # Qwen3-4B, evaluate the tone loss on a strong base:
  python run_train_slm.py --model Qwen/Qwen3-4B --load-4bit \
      --train-data-path vcc_bench_data/training_corpus_v1.json \
      --epochs 2 --batch-size 1 --max-length 512 --grad-accum 16   # ~12-16 GB
  python run_train_slm.py --model Qwen/Qwen3-4B --base-dtype bfloat16 ...   # >=24 GB, no quant
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
    """PEFT target module names by architecture (GPT-2 SLM and Qwen/LLaMA-style)."""
    model_type = getattr(model.config, "model_type", "")
    if model_type in {"gpt2", "gpt_neo", "gptj"}:
        # GPT-2 uses fused QKV Conv1D (c_attn), projection and MLP modules.
        return ["c_attn", "c_proj", "c_fc"]
    # Qwen (qwen/qwen2/qwen3/qwen3_moe) and LLaMA-family share the standard
    # attention + MLP projection names. startswith keeps new Qwen revisions
    # working without another edit here.
    if model_type.startswith("qwen") or model_type in {"llama", "mistral"}:
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
    ap.add_argument("--base-dtype", choices=["float32", "bfloat16"], default="float32",
                    help="Base weight dtype. float32 for a small SLM (default); "
                         "bfloat16 for multi-billion-param bases (e.g. Qwen3-4B) that "
                         "would not fit in float32.")
    ap.add_argument("--load-4bit", action="store_true",
                    help="QLoRA: load the base in 4-bit NF4 so a ~4B model fits a single "
                         "12-16 GB GPU. Overrides --base-dtype for the base weights "
                         "(compute stays bfloat16). NOTE: bitsandbytes needs device_map "
                         "at load, which can segfault on this Windows/CUDA dev box "
                         "(see docs/benchmark.md) -- intended for a Linux/T4-class GPU.")
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

    # Precision: a tiny SLM trains fine with FP32 master weights + FP16 autocast
    # (the original low-VRAM path). A multi-billion-param base (Qwen3-4B) cannot
    # hold FP32 weights on one GPU, so it loads in bfloat16 or 4-bit NF4 (QLoRA)
    # and trains under bfloat16 autocast, which needs no loss scaling.
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            device_map={"": 0},
        )
        resize_embeddings_if_needed(model, tokenizer)
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=not args.no_gradient_checkpointing)
        autocast_dtype = torch.bfloat16
    else:
        base_dtype = torch.bfloat16 if args.base_dtype == "bfloat16" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=base_dtype)
        resize_embeddings_if_needed(model, tokenizer)
        model = model.to(device)
        if not args.no_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        # FP32 base -> FP16 autocast (needs GradScaler); bf16 base -> bf16
        # autocast (no scaler). See the training loop's use_scaler below.
        autocast_dtype = torch.float16 if base_dtype == torch.float32 else torch.bfloat16
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
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
    # GradScaler is only needed for FP16 (bf16 has FP32's exponent range and
    # needs no loss scaling). enabled=False makes every scaler call a no-op, so
    # the same loop below serves both precisions.
    use_scaler = autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

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
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
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
    # Record what the probe belongs to, so the inference-time loader
    # (compressors/slm_tone_probe.py) can verify it is being paired with the
    # right base model / adapter instead of silently loading a mismatched
    # classifier. tone_probe.pt is only a raw state_dict and carries none of
    # this on its own.
    with open(os.path.join(args.output_dir, "tone_probe_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base_model": args.model,
            "adapter_dir": final_dir,
            "hidden_size": model.config.hidden_size,
            "num_tones": tone_loss.num_tones,
            "lambda_tone": args.lambda_tone,
            "max_length": args.max_length,
            "lora_r": args.lora_r,
            "base_dtype": "4bit-nf4" if args.load_4bit else args.base_dtype,
        }, f, ensure_ascii=False, indent=2)
    # Persist the exact held-out split so evaluate_slm.py scores the same
    # validation set regardless of --train-data-path/--max-length at eval time.
    val_samples = [dataset[i] for i in val_ds.indices]
    with open(os.path.join(final_dir, "val_split.json"), "w", encoding="utf-8") as f:
        json.dump({"max_length": args.max_length, "base_model": args.model,
                   "samples": val_samples}, f)
    print(f"Saved LoRA adapter and tokenizer: {final_dir}")
    print(f"Saved held-out validation split ({len(val_samples)} texts): {final_dir}/val_split.json")
    print(f"Saved trained tone probe: {args.output_dir}/tone_probe.pt (+ tone_probe_meta.json) | optimizer steps={step}")


if __name__ == "__main__":
    main()
