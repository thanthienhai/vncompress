#!/usr/bin/env bash
#
# Train E4 (relevance probe) + E6 (encoder) on the published, clean
# thanthienhai/vncompress-vi-v2, then evaluate everything -- sweep, tone/
# relevance probe A/Bs, and a ppl/tone/morphology ablation -- on VCC-Bench v2.
# This is the "is the proposed training method any good" run, answered on a
# benchmark that was never trained on (leak-check CLEAN, re-verified after
# the data rebuild this session did).
#
# Run this ON THE GPU CLUSTER (pod-test / wherever the wave-2 H100 run lives),
# NOT on the Windows dev box -- see the code-sync step below first.
#
#   ./scripts/run_wave2_train_eval.sh              # everything, GPU 0
#   GPU=1 ./scripts/run_wave2_train_eval.sh         # pick a GPU
#   QUICK=1 ./scripts/run_wave2_train_eval.sh       # smoke-test the wiring first (recommended)
#
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

# --- 0. Before running: sync CODE to the cluster (data is handled below) ---
#
# This box's working tree has fixes the cluster's checkout does not, and none
# of them are committed yet (deliberately left for you to review/commit --
# see `git status`). rsync works regardless of commit state:
#
#   rsync -avz vncompress/ scripts/ tests/ run_pipeline.sh requirements.txt \
#       user@pod-test:/path/to/vncompress/
#
# What's in it: 524 auto-retry (vncompress/api_pool.py), html.unescape fix
# (build_vncompress_vi_v2.py / fetch_sources_v2.py), the `bench` pipeline
# stage + E4's qa_synthetic merge (run_pipeline.sh), openai in
# requirements.txt.

QUICK="${QUICK:-0}"
GPU="${GPU:-0}"
QUICK_FLAG=()
[ "$QUICK" = 1 ] && QUICK_FLAG=(--quick)

# --- 1. Data: pull straight from the published dataset, not through this box.
#
# HF_DATASET overrides run_pipeline.sh's default (anhalu/vncompress-vi-v2).
# DATA_FILES adds qa_synthetic.jsonl -- off by default in run_pipeline.sh
# because the original anhalu dataset doesn't have that file and would 404;
# thanthienhai/vncompress-vi-v2 does.
export HF_DATASET="${HF_DATASET:-thanthienhai/vncompress-vi-v2}"
export DATA_FILES="${DATA_FILES:-corpus.jsonl qa.jsonl qa_synthetic.jsonl}"

echo "=== data + slm + probe + encoder + bench (GPU=$GPU, quick=$QUICK, dataset=$HF_DATASET) ==="
GPU="$GPU" ./run_pipeline.sh --stages data,slm,probe,encoder,bench "${QUICK_FLAG[@]}"

echo ""
echo "=== Done. Results: ==="
echo "  results/bench_wave2/sweep/              -- E1/E2/E5(rule)/E7/E6 vs VCC-Bench v2"
echo "  results/bench_wave2/tone_probe_ab/       -- trained tone probe vs rule (paper limitation #2)"
echo "  results/bench_wave2/relevance_probe_ab/  -- E4 headline: relevance probe vs rule"
echo "  results/bench_wave2/ablation/            -- ppl-only / tone-only / morph-only / combined, isolated"
