#!/usr/bin/env python3
"""Evaluate the held-out LM loss/perplexity and Vietnamese-tone probe accuracy.

Example:
  python evaluate_slm.py --adapter-dir trained_slm/final --tone-probe trained_slm/tone_probe.pt
"""
import argparse
import json
import math
import os
import sys
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_train_slm import Collator, VietnameseToneDataset, load_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", default="./trained_slm/final")
    ap.add_argument("--tone-probe", default="./trained_slm/tone_probe.pt")
    ap.add_argument("--train-data-path")
    ap.add_argument("--max-length", type=int, default=128,
                    help="Must match the max length used during training")
    ap.add_argument("--batch-size", type=int, default=1)
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
    base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, dtype=torch.float32).to(device)
    model = PeftModel.from_pretrained(base, args.adapter_dir).eval()
    model.config.use_cache = False
    probe = PhonologicalConsistencyLoss(model.config.hidden_size, lambda_tone=0.0).to(device)
    probe.load_state_dict(torch.load(args.tone_probe, map_location=device, weights_only=True))
    probe.eval()

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
    loader = DataLoader(validation, batch_size=args.batch_size, collate_fn=Collator(tokenizer.pad_token_id))

    total_nll, valid_tokens, all_correct, all_count = 0.0, 0, 0, 0
    marked_correct, marked_count = 0, 0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            labels=batch["labels"], output_hidden_states=True)
                logits = probe.tone_classifier(out.hidden_states[-1])
            # out.loss is mean causal NLL over non-padding tokens, except first token.
            n = int(batch["attention_mask"][:, 1:].sum().item())
            total_nll += out.loss.float().item() * n
            valid_tokens += n
            pred = logits.argmax(-1)
            mask = batch["attention_mask"].bool()
            all_correct += int(((pred == batch["tone_labels"]) & mask).sum().item())
            all_count += int(mask.sum().item())
            marked = mask & (batch["tone_labels"] != 0)
            marked_correct += int(((pred == batch["tone_labels"]) & marked).sum().item())
            marked_count += int(marked.sum().item())
    nll = total_nll / max(valid_tokens, 1)
    print(f"Validation texts: {len(validation)}")
    print(f"LM validation loss (NLL): {nll:.4f}")
    print(f"Perplexity: {math.exp(min(nll, 20)):.2f}")
    print(f"Tone accuracy (all tokens): {all_correct / max(all_count, 1):.2%}")
    print(f"Tone accuracy (marked tones only): {marked_correct / max(marked_count, 1):.2%} ({marked_count} tokens)")
    # The 'all tokens' figure is inflated by the ngang (unmarked) majority.
    # Report the always-predict-ngang baseline so it can be read honestly.
    ngang_tokens = all_count - marked_count
    print(f"Majority-class baseline (always predict ngang): {ngang_tokens / max(all_count, 1):.2%}")
    print("-> Judge tone learning by 'marked tones only' vs this baseline, not by 'all tokens'.")


if __name__ == "__main__":
    main()
