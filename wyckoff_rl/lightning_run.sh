#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Wyckoff RL — Lightning AI Studio Setup & Run Script
#
# Usage on Lightning AI Studio (L4 or A100):
#   1. Clone the repo:
#        git clone https://github.com/docdude/ElegantRL.git && cd ElegantRL
#   2. Run setup + training:
#        bash wyckoff_rl/lightning_run.sh [OPTIONS]
#
# Options:
#   --setup-only     Just install dependencies, don't train
#   --model MODEL    Agent: ppo (default), sac, td3
#   --reward REWARD  Reward: pnl (default), log_ret, sharpe, sortino
#   --split N        Single split (default: all 10)
#   --gpu N          GPU device (default: 0)
#   --eval-only      Skip training, just evaluate existing checkpoints
#   --break-step N   Override training steps (default: 2M from config)
#   --regenerate     Regenerate NPZ from SCID file before training
#   --scid PATH      Path to SCID file (default: wyckoff_rl/data/NQZ25-CME.scid)
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Parse arguments ──────────────────────────────────────────────────────
SETUP_ONLY=false
EVAL_ONLY=false
MODEL="ppo"
REWARD="pnl"
SPLIT_ARG=""
GPU=0
BREAK_STEP=""
REGENERATE=false
SCID_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --setup-only)   SETUP_ONLY=true; shift ;;
        --eval-only)    EVAL_ONLY=true; shift ;;
        --model)        MODEL="$2"; shift 2 ;;
        --reward)       REWARD="$2"; shift 2 ;;
        --split)        SPLIT_ARG="--split $2"; shift 2 ;;
        --gpu)          GPU="$2"; shift 2 ;;
        --break-step)   BREAK_STEP="--break-step $2"; shift 2 ;;
        --regenerate)   REGENERATE=true; shift ;;
        --scid)         SCID_PATH="$2"; shift 2 ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Environment Setup ────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  Wyckoff RL — Lightning AI Studio"
echo "═══════════════════════════════════════════════════════════════"
echo "  Project: $PROJECT_DIR"
echo "  Model: $MODEL, Reward: $REWARD"
echo "  GPU: $GPU"

# Check GPU
if command -v nvidia-smi &>/dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
fi

# Install dependencies
echo ""
echo "── Installing dependencies ──"
pip install -q torch numpy matplotlib gymnasium pandas RiskLabAI scikit-learn xgboost scipy numba

# Install ElegantRL in editable mode
pip install -q -e .

# Verify data
DATA_PATH="$PROJECT_DIR/wyckoff_rl/data/wyckoff_nq_selected.npz"
if [[ ! -f "$DATA_PATH" ]] && ! $REGENERATE; then
    echo "ERROR: Data not found at $DATA_PATH"
    echo "  Either copy existing NPZ:"
    echo "    cp /path/to/wyckoff_nq_selected.npz wyckoff_rl/data/"
    echo "  Or regenerate from SCID:"
    echo "    bash wyckoff_rl/lightning_run.sh --regenerate --scid /path/to/CONTRACT.scid"
    exit 1
fi
if [[ -f "$DATA_PATH" ]]; then
    echo "  Data: $DATA_PATH ($(du -h "$DATA_PATH" | cut -f1))"
fi

# ── Regenerate Data from SCID (optional) ─────────────────────────────────
if $REGENERATE; then
    echo ""
    echo "── Regenerating NPZ from SCID ──"
    SCID_ARG=""
    if [[ -n "$SCID_PATH" ]]; then
        SCID_ARG="--scid $SCID_PATH"
    fi
    python -c "
import sys, os
sys.path.insert(0, '$PROJECT_DIR')
from wyckoff_rl.data_pipeline.feature_extraction import run_full_pipeline
from wyckoff_rl.data_pipeline.feature_selection import run_feature_selection
import numpy as np

scid = '${SCID_PATH:-}' or None
result = run_full_pipeline(scid_path=scid)
print(f'Phase 1 complete: {result[\"n_bars\"]:,} bars, {result[\"n_features\"]} features')

# Phase 2: feature selection (consensus MDI/MDA)
tech = result['tech_ary']
close = result['close_ary']
# Use sign of returns as labels for MDI/MDA
rets = np.diff(close[:, 0]) / close[:-1, 0]
labels = (rets > 0).astype(int)
sel = run_feature_selection(tech[1:], labels=labels, method='all', top_k=15)
if 'mdi_selected_ary' in sel:
    selected = sel['mdi_selected_ary']
    names = sel['mdi_selected_names']
    # Prepend the first row back
    selected = np.vstack([tech[0:1, :selected.shape[1]], selected])
else:
    selected = tech
    names = result['feature_names']

npz_out = '$PROJECT_DIR/wyckoff_rl/data/wyckoff_nq_selected.npz'
np.savez_compressed(npz_out, close_ary=close, tech_ary=selected.astype(np.float32),
                    feature_names=np.array(names))
print(f'Saved: {npz_out} ({os.path.getsize(npz_out)/1024/1024:.1f} MB)')
"
    echo "Data regeneration complete."
fi

# Quick sanity check
python -c "
import numpy as np
d = np.load('$DATA_PATH')
print(f'  Bars: {d[\"close_ary\"].shape[0]:,}, Features: {d[\"tech_ary\"].shape[1]}')
print(f'  Close: {d[\"close_ary\"].min():.0f} – {d[\"close_ary\"].max():.0f}')
"

if $SETUP_ONLY; then
    echo ""
    echo "Setup complete. Run training with:"
    echo "  python -m wyckoff_rl.run_train --gpu $GPU --model $MODEL --reward $REWARD"
    exit 0
fi

# ── Training ─────────────────────────────────────────────────────────────
if ! $EVAL_ONLY; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Starting Training"
    echo "═══════════════════════════════════════════════════════════════"

    # Dry run first
    python -m wyckoff_rl.run_train --dry-run --model "$MODEL" --reward "$REWARD"

    # Full training
    python -m wyckoff_rl.run_train \
        --gpu "$GPU" \
        --model "$MODEL" \
        --reward "$REWARD" \
        $SPLIT_ARG \
        $BREAK_STEP

    echo ""
    echo "Training complete."
fi

# ── Find latest results dir ──────────────────────────────────────────────
RESULTS_BASE="$PROJECT_DIR/wyckoff_effort/rl_results"
LATEST_RUN=$(ls -td "$RESULTS_BASE"/*/ 2>/dev/null | head -1)

if [[ -z "$LATEST_RUN" ]]; then
    echo "No results found in $RESULTS_BASE"
    exit 1
fi
echo "Results: $LATEST_RUN"

# ── Checkpoint Evaluation ────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Evaluating Checkpoints vs NQ Buy-and-Hold"
echo "═══════════════════════════════════════════════════════════════"

python -m wyckoff_rl.eval_all_checkpoints \
    --results-dir "$LATEST_RUN" \
    --gpu "$GPU" \
    $SPLIT_ARG

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  DONE"
echo "═══════════════════════════════════════════════════════════════"
echo "Results: $LATEST_RUN"
echo ""
echo "  Checkpoint evaluations: */checkpoint_results.csv"
echo "  Comparison plots:        */checkpoint_comparison.png"
if [[ -z "$SPLIT_ARG" ]]; then
    echo "  Aggregate summary:       aggregate_checkpoint_results.csv"
fi
