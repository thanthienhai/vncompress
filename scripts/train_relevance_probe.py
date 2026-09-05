#!/usr/bin/env python3
"""Train the wave-2 E4 query-relevance probe.

Wave 1 refuted tone as a compression signal but showed the *training method*
works. E4 keeps that machinery and swaps the label: instead of predicting a
token's tone (a deterministic function of its id, useless for token selection),
the probe predicts whether a token is RELEVANT to the answer -- the signal LACC
was missing. The base model is frozen; only the RelevanceConsistencyLoss probe
learns, so this is cheap (probe-only, no fine-tuning).

Labels are weak supervision: a context token is positive iff its surface form
overlaps the reference answer (span-overlap, RECOMP/EXIT-style), built from
VCC-Bench's existing (context, reference_answer) pairs -- so no new annotation.

Usage:
  # On the trained SLM adapter (uses its LoRA-adapted hidden states):
  python scripts/train_relevance_probe.py --adapter-dir models/qwen3/final \\
      --data-path data/benchmark/vcc_bench_v2.json --output-dir models/qwen3 --load-4bit

  # Or on a plain base model:
  python scripts/train_relevance_probe.py --base-model Qwen/Qwen3-4B \\
      --data-path data/benchmark/vcc_bench_v2.json --output-dir models/qwen3_relevance

Then A/B it against the tone probe with the SAME SLM (this is the headline E4
measurement -- expect it to reverse the wave-1 A/B, preserving tone worse but
answering better):
  python scripts/verify_tone_probe_e2e.py --scorer-adapter-dir models/qwen3/final \\
      --tone-probe-path models/qwen3/relevance_probe.pt ...
  (load_scorer auto-detects probe_kind='relevance' from relevance_probe_meta.json)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vncompress.training import run_relevance_probe_training  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--adapter-dir', default=None,
                    help="LoRA adapter dir from 'train.py --mode slm' (e.g. models/qwen3/final). "
                         "If omitted, --base-model is used directly.")
    ap.add_argument('--base-model', default='chronopt-research/vietnamese-gpt2-base',
                    help='Base model when --adapter-dir is not given.')
    ap.add_argument('--no-adapter', action='store_true',
                    help='Load only the adapter dir base model (isolates what fine-tuning added).')
    ap.add_argument('--data-path', default='data/benchmark/vcc_bench_v2.json',
                    help='VCC-Bench-shaped JSON with (context, reference_answer, task) samples.')
    ap.add_argument('--output-dir', default='./models/relevance')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-length', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max-steps', type=int, default=-1)
    ap.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    ap.add_argument('--load-4bit', action='store_true',
                    help='Load the base in 4-bit NF4 (fit a large base on a smaller GPU).')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    run_relevance_probe_training(
        adapter_dir=args.adapter_dir,
        base_model=args.base_model,
        output_dir=args.output_dir,
        train_data_path=args.data_path,
        use_adapter=not args.no_adapter,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lr=args.lr,
        max_steps=args.max_steps,
        base_dtype=args.dtype,
        load_4bit=args.load_4bit,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
