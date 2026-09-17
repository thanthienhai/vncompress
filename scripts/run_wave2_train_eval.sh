#!/usr/bin/env bash
#
# Thin convenience wrapper: `run_pipeline.sh` now defaults to
# thanthienhai/vncompress-vi-v2 (the entity-clean, Ha-Noi-held-out rebuild)
# and downloads qa_synthetic.jsonl by default, so this script no longer needs
# to override HF_DATASET/DATA_FILES itself -- it just picks the right stages
# and prints where the results land.
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

# --- Before running: sync CODE to the cluster (data downloads itself now) --
#
# This box's working tree may have fixes the cluster's checkout does not.
# Check `git log` on both sides first -- if the cluster already has this
# commit (or later), skip this. Otherwise:
#
#   git push                                          # from this box, if not already
#   ssh user@pod-test 'cd /path/to/vncompress && git pull'
#
# rsync works too if the cluster checkout isn't a clean git clone:
#   rsync -avz vncompress/ scripts/ tests/ run_pipeline.sh requirements.txt \
#       user@pod-test:/path/to/vncompress/

QUICK="${QUICK:-0}"
GPU="${GPU:-0}"
QUICK_FLAG=()
[ "$QUICK" = 1 ] && QUICK_FLAG=(--quick)

echo "=== data + slm + probe + encoder + bench (GPU=$GPU, quick=$QUICK) ==="
GPU="$GPU" ./run_pipeline.sh --stages data,slm,probe,encoder,bench "${QUICK_FLAG[@]}"

echo ""
echo "=== Done. Results: ==="
echo "  results/bench_wave2/sweep/              -- E1/E2/E5(rule)/E7/E6 vs VCC-Bench v2"
echo "  results/bench_wave2/tone_probe_ab/       -- trained tone probe vs rule (paper limitation #2)"
echo "  results/bench_wave2/relevance_probe_ab/  -- E4 headline: relevance probe vs rule"
echo "  results/bench_wave2/ablation/            -- ppl-only / tone-only / morph-only / combined, isolated"
