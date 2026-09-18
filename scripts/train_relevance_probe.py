#!/usr/bin/env python3
"""Train the wave-2 E4 query-relevance probe.

Wave 1 refuted tone as a compression signal but showed the *training method*
works. E4 keeps that machinery and swaps the label: instead of predicting a
token's tone (a deterministic function of its id, useless for token selection),
the probe predicts whether a token is RELEVANT to the answer -- the signal LACC
was missing. The base model is frozen; only the RelevanceConsistencyLoss probe
learns, so this is cheap (probe-only, no fine-tuning).

Supervision comes from vncompress-vi-v2's `qa.jsonl`, which carries a verified
`answer_span` (exact on 6,000/6,000 rows). A context token is positive iff it
overlaps that span -- not merely iff it repeats one of the answer's syllables,
which fires on every unrelated occurrence of them elsewhere in the article.
Fetch it first:

  huggingface-cli download anhalu/vncompress-vi-v2 qa.jsonl \\
      --repo-type dataset --local-dir data/vncompress_vi_v2

Usage:
  # On the trained SLM adapter (uses its LoRA-adapted hidden states):
  python scripts/train_relevance_probe.py --adapter-dir models/qwen3/final \\
      --output-dir models/qwen3 --load-4bit

  # Or on a plain base model:
  python scripts/train_relevance_probe.py --base-model Qwen/Qwen3-4B \\
      --output-dir models/qwen3_relevance

Never point --data-path at data/benchmark/vcc_bench_v2.json (or v1): that is
the evaluation benchmark (see WAVE2_HANDOFF.md / benchmark.py), and training
the probe on it is a train/test leak. The leak also arrives from the other
direction -- `viquad:Hà_Nội` sits in v2's qa TRAIN split and in 32 of
vcc_bench_v2's 414 samples -- so those documents are held out by default; pass
--no-holdout to keep them and --holdout-docs-from to hold out more.

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

from vncompress.training import (  # noqa: E402
    DEFAULT_BENCHMARK_HOLDOUT,
    resolve_holdout_documents,
    run_relevance_probe_training,
)

DEFAULT_DATA = os.path.join('data', 'vncompress_vi_v2', 'qa.jsonl')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--adapter-dir', default=None,
                    help="LoRA adapter dir from 'train.py --mode slm' (e.g. models/qwen3/final). "
                         "If omitted, --base-model is used directly.")
    ap.add_argument('--base-model', default='chronopt-research/vietnamese-gpt2-base',
                    help='Base model when --adapter-dir is not given.')
    ap.add_argument('--no-adapter', action='store_true',
                    help='Load only the adapter dir base model (isolates what fine-tuning added).')
    ap.add_argument('--data-path', default=DEFAULT_DATA,
                    help='vncompress-vi-v2 qa.jsonl, or a VCC-Bench-shaped TRAINING json. '
                         'Never a benchmark/eval file -- see module docstring.')
    ap.add_argument('--split', default='train', help="Row split to train on (default: train).")
    ap.add_argument('--val-split', default='validation',
                    help="Row split for held-out metrics; '' to skip (default: validation).")
    ap.add_argument('--holdout-docs-from', action='append', default=None, metavar='PATH',
                    help='Drop documents used by this external benchmark. Repeatable. '
                         f'Defaults to {DEFAULT_BENCHMARK_HOLDOUT} when it exists.')
    ap.add_argument('--no-holdout', action='store_true',
                    help='Train on benchmark documents too (contaminates that benchmark).')
    ap.add_argument('--no-query-conditioned', action='store_true',
                    help='Score the context alone, without the question prefix. The probe then '
                         'learns "does this token look like an answer", not query relevance.')
    ap.add_argument('--min-samples', type=int, default=32,
                    help='Refuse to train on fewer usable samples than this (default: 32).')
    ap.add_argument('--no-balance-classes', action='store_true',
                    help='Plain cross-entropy. The answer span is a few percent of each window, '
                         'so this converges on "nothing is relevant" -- high accuracy, no signal.')
    ap.add_argument('--focal-gamma', type=float, default=0.0,
                    help='Focal loss exponent: (1-p_t)^gamma scales each token\'s loss, so easy '
                         'negatives stop dominating the gradient. 2.0 is the usual starting point; '
                         '0.0 (default) is plain cross-entropy, as every probe before 2026-09-18.')
    ap.add_argument('--class-weight-cap', type=float, default=50.0,
                    help='Ceiling on the positive class weight (inverse frequency, ~65x on the v2 '
                         'corpus, so this binds). Lower it to trade recall for precision -- with '
                         'the default cap the probe measured P=4.9%%/R=59%%.')
    ap.add_argument('--output-dir', default='./models/relevance')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-length', type=int, default=512,
                    help='Token window per sample, centred on the answer span (default: 512).')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max-steps', type=int, default=-1)
    ap.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    ap.add_argument('--load-4bit', action='store_true',
                    help='Load the base in 4-bit NF4 (fit a large base on a smaller GPU).')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    holdout = resolve_holdout_documents(args.holdout_docs_from, args.no_holdout)

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
        split=args.split,
        val_split=args.val_split or None,
        holdout_docs=holdout,
        query_conditioned=not args.no_query_conditioned,
        min_samples=args.min_samples,
        balance_classes=not args.no_balance_classes,
        focal_gamma=args.focal_gamma,
        class_weight_cap=args.class_weight_cap,
    )


if __name__ == '__main__':
    main()
