"""
Configuration for CPCV / WF / KCV training pipeline.

Centralized settings mirroring FinRL_Crypto's config_main.py structure.
Modify these for your dataset and hardware.
"""

import os
from functools import reduce
import operator as op


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CACHE_DIR = os.path.join(PROJECT_ROOT, "datasets")
ALPACA_NPZ_PATH = os.path.join(DATA_CACHE_DIR, "alpaca_stock_data.numpy.npz")

# Results directory
RESULTS_DIR = os.path.join(PROJECT_ROOT, "train_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

# Total trading days in price_array.npy (rows)
# Set dynamically in scripts by loading the array
TOTAL_DAYS = None  # will be set at runtime


# ─────────────────────────────────────────────────────────────────────────────
# CPCV Settings (Lopez de Prado)
# ─────────────────────────────────────────────────────────────────────────────

N_GROUPS = 5           # N: number of fold groups
K_TEST_GROUPS = 2      # K: test groups per split
EMBARGO_DAYS = 7       # embargo after each test fold boundary

def nCr(n, r):
    """Combinations C(n, r)."""
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom

N_SPLITS = nCr(N_GROUPS, K_TEST_GROUPS)     # C(5,2) = 10
N_PATHS = N_SPLITS * K_TEST_GROUPS // N_GROUPS  # 10*2/5 = 4


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Settings
# ─────────────────────────────────────────────────────────────────────────────

WF_TRAIN_RATIO = 0.7
WF_VAL_RATIO = 0.2
WF_TEST_RATIO = 0.1
WF_GAP_DAYS = 7


# ─────────────────────────────────────────────────────────────────────────────
# K-Fold CV Settings
# ─────────────────────────────────────────────────────────────────────────────

KCV_N_FOLDS = 5


# ─────────────────────────────────────────────────────────────────────────────
# DRL Agent Settings
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_NAME = "ppo"
RANDOM_SEED = 1943
GPU_ID = 0

# PPO defaults (can be overridden by HPO)
DEFAULT_ERL_PARAMS = {
    "net_dims": [128, 64],
    "learning_rate": 1e-4,
    "batch_size": 128,
    "break_step": 1_000_000,
    "gamma": 0.99,
    "lambda_entropy": 0.01,
    "clip_grad_norm": 3.0,
    "repeat_times": 16.0,
    "ratio_clip": 0.25,
    "lambda_gae_adv": 0.95,
    "if_use_v_trace": True,
    "eval_per_step": 5_000,
    "eval_times": 32,
}

# Environment defaults
DEFAULT_ENV_PARAMS = {
    "initial_amount": 1e6,
    "max_stock": 1e2,
    "cost_pct": 1e-3,
    "num_envs": 2048,       # 2**11, empirically tested on 2080 (8GB)
}

# VecNormalize settings
USE_VEC_NORMALIZE = True
VEC_NORMALIZE_KWARGS = {
    "norm_obs": True,
    "norm_reward": True,
    "clip_obs": 10.0,
    "clip_reward": 10.0,
    "gamma": 0.99,
}


# ─────────────────────────────────────────────────────────────────────────────
# HPO Settings (Hydra / Hypersweeper)
# ─────────────────────────────────────────────────────────────────────────────

H_TRIALS = 50           # number of HPO trials
HPO_METRIC = "avg_sharpe"  # metric to optimize


# NOTE: H200 GPU scaling (num_envs=16384, batch_size=4096, break_step=2M)
# is NOT wired into the CPCV pipeline yet. When moving to H200, add --h200
# flag to optimize_cpcv.py + per-fold num_envs scaling in train_split.
# See demo_FinRL_Alpaca_VecEnv_CV.py lines 960-1055 for reference.


def print_config():
    """Print current configuration."""
    print(f"\n{'='*60}")
    print(f"Pipeline Configuration")
    print(f"{'='*60}")
    print(f"  Data:     {ALPACA_NPZ_PATH}")
    print(f"  CPCV:     N={N_GROUPS}, K={K_TEST_GROUPS}, embargo={EMBARGO_DAYS}d")
    print(f"  Splits:   {N_SPLITS} splits, {N_PATHS} paths")
    print(f"  Agent:    {DEFAULT_MODEL_NAME.upper()}")
    print(f"  Seed:     {RANDOM_SEED}")
    print(f"  GPU:      {GPU_ID}")
    print(f"  HPO:      {H_TRIALS} trials")
    print(f"{'='*60}\n")
