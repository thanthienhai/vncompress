#!/usr/bin/env python3
"""Control experiments for the Vietnamese tone probe.

`evaluate_slm.py` reports how well a probe reads tone off the LoRA-adapted
model's hidden states. On its own that number proves nothing: the label is a
deterministic function of the token id (verified: 0 of 13,727 token ids in the
held-out split map to more than one tone), so a zero-training lookup table
scores 100%. Two controls make the probe number interpretable.

  --mode frozen_base   Train the identical probe on the base model's hidden
                       states, LoRA disabled. If this matches the LoRA probe,
                       --lambda-tone contributed nothing and the "tone-aware
                       training" claim does not hold.

  --mode lora          Same probe, LoRA enabled. Reproduces the training-time
                       setup under identical probe hyperparameters so the
                       frozen_base comparison is like-for-like (the shipped
                       tone_probe.pt was trained jointly with the LoRA weights,
                       so comparing it directly to a freshly trained probe
                       would confound "better representation" with "different
                       probe training run").

  --control-task       Hewitt & Liang (2019) control task: replace each token
                       TYPE's label with a random one drawn from the label
                       distribution, then retrain. Probe selectivity =
                       real-task accuracy - control-task accuracy. High
                       selectivity means the probe is reading structure from
                       the representation; near-zero means it is memorizing
                       token identity, which is what a deterministic
                       token->tone label invites.

The model is always frozen; only probe parameters get gradients, so the
model forward runs under no_grad and this is far cheaper than fine-tuning.

Usage:
  python scripts/train_probe_control.py --mode frozen_base
  python scripts/train_probe_control.py --mode lora
  python scripts/train_probe_control.py --mode frozen_base --control-task
  python scripts/train_probe_control.py --mode lora --control-task
"""
import argparse
import json
import os
import random
import sys

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_train_slm import Collator, VietnameseToneDataset, load_texts  # noqa: E402


def build_matching_split(corpus_path, tokenizer, max_length, expected_val):
    """Rebuild train/val exactly as run_train_slm.py did, and verify the val
    half matches the split saved at training time.

    The probe needs the TRAIN half, but val_split.json only stores val. The
    split is deterministic (seed 42) given the same corpus/tokenizer/max_length,
    so it can be reconstructed -- but silently reconstructing a *different*
    split would leak training texts into the probe's evaluation, so this
    verifies rather than assumes.
    """
    dataset = VietnameseToneDataset(load_texts(corpus_path), tokenizer, max_length)
    if len(dataset) < 2:
        raise SystemExit(f"Corpus {corpus_path} yielded <2 usable texts.")
    train_n = min(max(1, int(len(dataset) * .9)), len(dataset) - 1)
    train_ds, val_ds = random_split(
        dataset, [train_n, len(dataset) - train_n],
        generator=torch.Generator().manual_seed(42),
    )
    rebuilt = [tuple(dataset[i]) for i in val_ds.indices]
    if len(rebuilt) != len(expected_val):
        raise SystemExit(
            f"Rebuilt val split has {len(rebuilt)} texts but val_split.json has "
            f"{len(expected_val)}. The corpus or --max-length does not match the "
            f"one used to train this adapter -- pass the right --corpus/--max-length."
        )
    for i, (a, b) in enumerate(zip(rebuilt, expected_val)):
        if list(a[0]) != list(b[0]):
            raise SystemExit(
                f"Rebuilt val split diverges from val_split.json at index {i}. "
                f"The corpus file has changed since this adapter was trained "
                f"(build_training_corpus.py overwrites it) -- cannot safely "
                f"recover the train half."
            )
    return [tuple(dataset[i]) for i in train_ds.indices], rebuilt


def apply_control_labels(samples, num_tones, seed):
    """Hewitt & Liang control task: each token TYPE gets a fixed random label,
    sampled from the empirical label distribution so class priors are kept."""
    rng = random.Random(seed)
    counts = [0] * num_tones
    for ids, tones in samples:
        for t in tones:
            counts[t] += 1
    population = [t for t in range(num_tones) if counts[t] > 0]
    freqs = [counts[t] for t in population]
    mapping = {}
    out = []
    for ids, tones in samples:
        new = []
        for tid in ids:
            if tid not in mapping:
                mapping[tid] = rng.choices(population, weights=freqs, k=1)[0]
            new.append(mapping[tid])
        out.append((list(ids), new))
    return out


@torch.no_grad()
def hidden_states(model, batch):
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                output_hidden_states=True)
    return out.hidden_states[-1]


def evaluate(model, probe, loader, device, num_tones):
    confusion = torch.zeros(num_tones, num_tones, dtype=torch.long)
    probe.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = probe.tone_classifier(hidden_states(model, batch))
            mask = batch["attention_mask"].bool()
            t = batch["tone_labels"][mask].flatten()
            p = logits.argmax(-1)[mask].flatten()
            confusion += torch.bincount(
                t * num_tones + p, minlength=num_tones * num_tones,
            ).reshape(num_tones, num_tones).cpu()
    total = int(confusion.sum())
    correct = int(confusion.diag().sum())
    marked_total = int(confusion[1:].sum())
    marked_correct = int(confusion.diag()[1:].sum())
    tp = confusion.diag().double()
    support = confusion.sum(1).double()
    predicted = confusion.sum(0).double()
    precision = tp / predicted.clamp(min=1)
    recall = tp / support.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    present = [c for c in range(num_tones) if support[c] > 0]
    return {
        "accuracy_all": correct / max(total, 1),
        "accuracy_marked": marked_correct / max(marked_total, 1),
        "macro_f1": float(f1[present].mean()),
        "macro_f1_marked": float(f1[[c for c in present if c != 0]].mean()),
        "tokens": total,
        "marked_tokens": marked_total,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["frozen_base", "lora"], required=True)
    ap.add_argument("--control-task", action="store_true",
                    help="Train on random per-token-type labels (probe selectivity control)")
    ap.add_argument("--adapter-dir", default="./trained_slm/final")
    ap.add_argument("--corpus", default="vcc_bench_data/training_corpus_v1.json")
    ap.add_argument("--max-length", type=int, default=128,
                    help="Must match the value used to train the adapter")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train-samples", type=int, default=None,
                    help="Subsample the train half for speed on small GPUs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", help="Append the result as one JSON line to this file")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                    help="Base weight dtype; bfloat16 for a ~4B base (e.g. Qwen3-4B).")
    ap.add_argument("--load-4bit", action="store_true",
                    help="Load the base in 4-bit NF4 to fit a large base on a smaller GPU.")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vncompress.tone_aware import PhonologicalConsistencyLoss

    device = torch.device("cuda")
    peft_config = PeftConfig.from_pretrained(args.adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            device_map={"": 0})
        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))
    else:
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, dtype=dtype)
        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
    if args.mode == "lora":
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():          # only the probe learns
        p.requires_grad_(False)

    with open(os.path.join(args.adapter_dir, "val_split.json"), encoding="utf-8") as f:
        expected_val = [tuple(s) for s in json.load(f)["samples"]]
    train, validation = build_matching_split(args.corpus, tokenizer, args.max_length, expected_val)
    print(f"[{args.mode}{'/control' if args.control_task else ''}] "
          f"train={len(train)} val={len(validation)} (val verified against val_split.json)")

    hidden_size = model.config.hidden_size
    probe = PhonologicalConsistencyLoss(hidden_size, lambda_tone=1.0).to(device)
    num_tones = probe.num_tones

    if args.control_task:
        # Derive the mapping from train+val together so a token type gets the
        # same random label in both halves -- otherwise the control task is
        # unlearnable by construction and selectivity is meaningless.
        combined = apply_control_labels(list(train) + list(validation), num_tones, args.seed)
        train, validation = combined[:len(train)], combined[len(train):]
    if args.max_train_samples:
        train = train[:args.max_train_samples]

    collate = Collator(tokenizer.pad_token_id)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(validation, batch_size=args.batch_size, collate_fn=collate)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda")
    step = 0
    for epoch in range(args.epochs):
        probe.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = probe(hidden_states(model, batch), batch["tone_labels"], batch["attention_mask"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step == 1 or step % 100 == 0:
                print(f"  epoch={epoch} step={step} probe_loss={loss.item():.4f}")

    result = evaluate(model, probe, val_loader, device, num_tones)
    result.update(mode=args.mode, control_task=args.control_task, epochs=args.epochs,
                  lr=args.lr, batch_size=args.batch_size, train_samples=len(train))
    print(f"\n=== {args.mode}{' / CONTROL TASK' if args.control_task else ''} ===")
    print(f"  accuracy (all tokens)   : {result['accuracy_all']:.2%}")
    print(f"  accuracy (marked tones) : {result['accuracy_marked']:.2%}")
    print(f"  macro-F1                : {result['macro_f1']:.4f}  (marked only: {result['macro_f1_marked']:.4f})")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"  appended to {args.out}")
    print("\nRun all four (frozen_base, lora) x (real, --control-task) and read:")
    print("  lora - frozen_base  = what LoRA/--lambda-tone actually added to the representation")
    print("  real - control_task = probe selectivity (near 0 => the probe is memorizing token ids)")


if __name__ == "__main__":
    main()
