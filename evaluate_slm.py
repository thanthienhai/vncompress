#!/usr/bin/env python3
"""Evaluate the held-out LM loss/perplexity and Vietnamese-tone probe accuracy.

Example:
  python evaluate_slm.py --adapter-dir trained_slm/final --tone-probe trained_slm/tone_probe.pt

Baseline comparison (raw base model, no LoRA) on the exact same held-out
split, to check whether fine-tuning actually helped/hurt perplexity:
  python evaluate_slm.py --adapter-dir trained_slm/final --no-adapter

Paired statistical comparison of the two runs above:
  python evaluate_slm.py --adapter-dir trained_slm/final --tone-probe trained_slm/tone_probe.pt \
      --dump-per-sample results_slm/lora.json
  python evaluate_slm.py --adapter-dir trained_slm/final --no-adapter \
      --dump-per-sample results_slm/base.json
  python scripts/compare_slm_runs.py results_slm/base.json results_slm/lora.json
"""
import argparse
import hashlib
import json
import math
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_train_slm import (
    Collator,
    VietnameseToneDataset,
    load_texts,
    resize_embeddings_if_needed,
)


def tone_lookup_baseline(validation, tokenizer):
    """Accuracy of the training-free predictor that maps a token id to a tone
    by decoding the token and running the tone analyzer -- i.e. exactly how
    VietnameseToneDataset builds its labels.

    This is the *ceiling* for any tone predictor that reads a representation
    of token i, because the label is a deterministic function of token i and
    this predictor needs no training data at all. Report the probe against
    it: a probe scoring below this is losing information that was free.
    """
    from vncompress.tone_aware import TONE_NAME_TO_ID, get_tone_analyzer

    analyzer = get_tone_analyzer()
    cache = {}

    def lookup(tid):
        if tid not in cache:
            piece = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            cache[tid] = TONE_NAME_TO_ID.get(analyzer.get_dominant_tone(piece.strip()) or "ngang", 0)
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
    return {
        "all": correct / max(total, 1),
        "marked": marked_correct / max(marked_total, 1),
    }


def per_class_prf(confusion):
    """Precision/recall/F1 per class from a (C, C) [true][pred] matrix."""
    tp = confusion.diag().double()
    support = confusion.sum(1).double()
    predicted = confusion.sum(0).double()
    precision = tp / predicted.clamp(min=1)
    recall = tp / support.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return precision, recall, f1, support


def split_fingerprint(validation):
    """Stable hash of the exact token-id sequences in this split, so a paired
    comparison can refuse to compare two runs scored on different splits
    (trained_slm/final/val_split.json is overwritten by every training run)."""
    h = hashlib.sha256()
    for ids, _ in validation:
        h.update(b",".join(str(i).encode() for i in ids))
        h.update(b"|")
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", default="./trained_slm/final")
    ap.add_argument("--tone-probe", default="./trained_slm/tone_probe.pt")
    ap.add_argument("--train-data-path")
    ap.add_argument("--max-length", type=int, default=128,
                    help="Must match the max length used during training")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--no-adapter", action="store_true",
                    help="Evaluate the raw base model (skip the LoRA adapter) for a fair "
                         "perplexity baseline. Tone accuracy is skipped in this mode: the "
                         "tone probe was trained jointly with the LoRA-adapted hidden states, "
                         "so it isn't a meaningful readout of the un-adapted base model.")
    ap.add_argument("--dump-per-sample", metavar="PATH",
                    help="Write per-sample NLL to PATH (JSON) for a paired significance "
                         "test against another run -- see scripts/compare_slm_runs.py")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vncompress.tone_aware import PhonologicalConsistencyLoss

    device = torch.device("cuda")
    config = PeftConfig.from_pretrained(args.adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, dtype=torch.float32)
    # Must match run_train_slm.py exactly, or the adapter loads onto a
    # differently-shaped (or differently-initialised) embedding matrix.
    resize_embeddings_if_needed(base, tokenizer)
    base = base.to(device)
    if args.no_adapter:
        model = base.eval()
        probe = None
        print("[--no-adapter] Evaluating the raw base model, no LoRA. Tone accuracy skipped.")
    else:
        model = PeftModel.from_pretrained(base, args.adapter_dir).eval()
        probe = PhonologicalConsistencyLoss(model.config.hidden_size, lambda_tone=0.0).to(device)
        probe.load_state_dict(torch.load(args.tone_probe, map_location=device, weights_only=True))
        probe.eval()
    model.config.use_cache = False

    # Prefer the exact held-out split saved at training time; this makes the
    # evaluation independent of --train-data-path/--max-length mismatches.
    val_path = os.path.join(args.adapter_dir, "val_split.json")
    if os.path.exists(val_path):
        with open(val_path, encoding="utf-8") as f:
            saved = json.load(f)
        validation = [tuple(s) for s in saved["samples"]]
        print(f"Loaded held-out split saved at training time: {len(validation)} texts")
    else:
        print("[WARN] val_split.json not found; rebuilding the split from --train-data-path. "
              "Pass the SAME --train-data-path and --max-length used during training.")
        ds = VietnameseToneDataset(load_texts(args.train_data_path), tokenizer, args.max_length)
        if len(ds) < 2:
            raise RuntimeError("Need at least two valid texts.")
        train_n = min(max(1, int(len(ds) * .9)), len(ds) - 1)
        _, validation = random_split(ds, [train_n, len(ds) - train_n], generator=torch.Generator().manual_seed(42))
        validation = [tuple(s) for s in validation]
    loader = DataLoader(validation, batch_size=args.batch_size, collate_fn=Collator(tokenizer.pad_token_id))

    num_tones = probe.num_tones if probe is not None else 0
    confusion = torch.zeros(num_tones, num_tones, dtype=torch.long)
    sample_nll_sums, sample_ntokens = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            output_hidden_states=probe is not None)
                if probe is not None:
                    logits = probe.tone_classifier(out.hidden_states[-1])
            # Per-sample causal NLL, so the aggregate below is identical to the
            # usual token-weighted mean while also supporting a paired test.
            shift_logits = out.logits[:, :-1, :].float()
            shift_labels = batch["labels"][:, 1:]
            tok_nll = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100, reduction="none",
            ).reshape(shift_labels.shape)
            sample_nll_sums.extend(tok_nll.sum(1).tolist())
            sample_ntokens.extend((shift_labels != -100).sum(1).tolist())
            if probe is not None:
                pred = logits.argmax(-1)
                mask = batch["attention_mask"].bool()
                t = batch["tone_labels"][mask].flatten()
                p = pred[mask].flatten()
                confusion += torch.bincount(
                    t * num_tones + p, minlength=num_tones * num_tones,
                ).reshape(num_tones, num_tones).cpu()

    total_tokens = sum(sample_ntokens)
    nll = sum(sample_nll_sums) / max(total_tokens, 1)
    print(f"Validation texts: {len(validation)}")
    print(f"Scored tokens: {total_tokens}")
    print(f"LM validation loss (NLL): {nll:.4f}")
    print(f"Perplexity: {math.exp(min(nll, 20)):.2f}")

    if args.dump_per_sample:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_per_sample)), exist_ok=True)
        with open(args.dump_per_sample, "w", encoding="utf-8") as f:
            json.dump({
                "adapter_dir": args.adapter_dir,
                "no_adapter": args.no_adapter,
                "base_model": config.base_model_name_or_path,
                "num_samples": len(validation),
                "split_fingerprint": split_fingerprint(validation),
                "corpus_nll": nll,
                "nll_sums": sample_nll_sums,
                "n_tokens": sample_ntokens,
            }, f)
        print(f"Wrote per-sample NLL: {args.dump_per_sample}")

    if probe is None:
        print("Tone accuracy: skipped (--no-adapter). Compare the Perplexity line above "
              "against a matching adapter run's Perplexity for a fair before/after comparison.")
        return

    all_count = int(confusion.sum().item())
    all_correct = int(confusion.diag().sum().item())
    marked_count = int(confusion[1:].sum().item())
    marked_correct = int(confusion.diag()[1:].sum().item())
    print(f"Tone accuracy (all tokens): {all_correct / max(all_count, 1):.2%}")
    print(f"Tone accuracy (marked tones only): {marked_correct / max(marked_count, 1):.2%} ({marked_count} tokens)")
    # The 'all tokens' figure is inflated by the ngang (unmarked) majority.
    # Report the always-predict-ngang baseline so it can be read honestly.
    ngang_tokens = all_count - marked_count
    print(f"Majority-class baseline (always predict ngang): {ngang_tokens / max(all_count, 1):.2%}")
    print("-> Judge tone learning by 'marked tones only' vs this baseline, not by 'all tokens'.")

    # Ceiling: the label is a deterministic function of the token id, so a
    # zero-training lookup built from the tokenizer alone is a real (not
    # oracle) predictor. Anything below it means the probe is lossy.
    ceiling = tone_lookup_baseline(validation, tokenizer)
    print("\nTraining-free token-id lookup (tokenizer + tone analyzer, no training):")
    print(f"  all tokens: {ceiling['all']:.2%} | marked tones only: {ceiling['marked']:.2%}")
    print("-> This is the ceiling, and it is free. The probe measures how much tone")
    print("   information survives in the hidden states, NOT whether tone is predictable.")

    # Accuracy hides per-class collapse when classes are this imbalanced
    # (ngang ~46% of tokens, nga ~3%). Macro-F1 weights every tone equally.
    from vncompress.tone_aware import TONE_NAME_TO_ID
    names = {v: k for k, v in TONE_NAME_TO_ID.items()}
    precision, recall, f1, support = per_class_prf(confusion)
    present = [c for c in range(num_tones) if support[c] > 0]
    macro_f1 = float(f1[present].mean())
    marked_present = [c for c in present if c != 0]
    macro_f1_marked = float(f1[marked_present].mean()) if marked_present else float("nan")
    print(f"\nPer-tone breakdown (macro-F1 all: {macro_f1:.4f} | marked only: {macro_f1_marked:.4f})")
    print(f"  {'tone':<10}{'support':>10}{'precision':>11}{'recall':>9}{'f1':>8}")
    for c in present:
        print(f"  {names.get(c, f'class{c}'):<10}{int(support[c]):>10}"
              f"{precision[c]:>11.4f}{recall[c]:>9.4f}{f1[c]:>8.4f}")
    print("-> Macro-F1 is the honest headline: plain accuracy can stay high while a rare")
    print("   tone (nga) collapses to ~0 recall, and hoi<->nga confusion is the classic")
    print("   Vietnamese failure mode -- read it off the confusion matrix below.")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    header = "".join(f"{names.get(c, c):>10}" for c in present)
    corner = "true\\pred"
    print(f"  {corner:<10}{header}")
    for r in present:
        row = "".join(f"{int(confusion[r][c]):>10}" for c in present)
        print(f"  {names.get(r, r):<10}{row}")


if __name__ == "__main__":
    main()
