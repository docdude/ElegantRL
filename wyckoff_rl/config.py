"""
Wyckoff RL Pipeline — Configuration.

Single-instrument NQ futures environment with Wyckoff features,
trained via Adaptive CPCV using ElegantRL.
"""

import os
from functools import reduce
import operator as op

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "wyckoff_rl", "data")

# Default NPZ — z-score + tanh normalized features
WYCKOFF_NPZ_PATH = os.path.join(DATA_DIR, "wyckoff_nq_normalized.npz")

# Results
RESULTS_DIR = os.path.join(PROJECT_ROOT, "wyckoff_effort", "rl_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CPCV Settings
# ─────────────────────────────────────────────────────────────────────────────

N_GROUPS = 5
K_TEST_GROUPS = 2
EMBARGO_BARS = 500        # in range bars (~500 bars ≈ a few sessions)

def nCr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom

N_SPLITS = nCr(N_GROUPS, K_TEST_GROUPS)     # C(5,2) = 10
N_PATHS = N_SPLITS * K_TEST_GROUPS // N_GROUPS  # 10*2/5 = 4


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive CPCV
# ─────────────────────────────────────────────────────────────────────────────

ADAPTIVE_FEATURE = "ER_Ratio"      # Wyckoff effort-result ratio (best: 34.5% improvement over std CPCV)
ADAPTIVE_SMOOTH_WINDOW = 50       # smoothing window for tech feature (bars)
ADAPTIVE_N_SUBSPLITS = 3
ADAPTIVE_LOWER_Q = 0.25
ADAPTIVE_UPPER_Q = 0.75


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ENV_PARAMS = {
    "initial_amount": 1000.0,       # in NQ points
    "cost_per_trade": 0.5,          # points per side (covers commission + slippage)
    "reward_mode": "pnl",           # "pnl", "log_ret", "sharpe", "sortino"
    "reward_scale": 1.0,
    "num_envs": 256,                # GPU-vectorized parallel episodes (auto-scaled to GPU memory)
    "episode_len": 4096,            # sub-episode length for PPO (None = full data)
}


# ─────────────────────────────────────────────────────────────────────────────
# DRL Agent
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_NAME = "ppo"
RANDOM_SEED = 42
GPU_ID = 0

DEFAULT_ERL_PARAMS = {
    "net_dims": [128, 64],
    "learning_rate": 1e-4,
    "batch_size": 256,
    "break_step": 2_000_000,
    "gamma": 0.99,
    "lambda_entropy": 0.02,
    "clip_grad_norm": 3.0,
    "repeat_times": 16.0,
    "ratio_clip": 0.25,
    "lambda_gae_adv": 0.95,
    "if_use_v_trace": True,
    "eval_per_step": 5_000,
    "eval_times": 32,
}


# ─────────────────────────────────────────────────────────────────────────────
# Reward mode descriptions (for reference)
# ─────────────────────────────────────────────────────────────────────────────
# "pnl"     — Normalized PnL change per step. Simple, stable. Good baseline.
# "log_ret" — Log return of portfolio. Scale-invariant.
# "sharpe"  — Differential Sharpe (Moody & Saffell 1998). Penalizes variance.
# "sortino" — Differential Sortino. Only penalizes downside variance.
#
# The reward is the MOST CRITICAL design choice. All four are implemented.
# Start with "pnl" for stability, graduate to "sharpe" or "sortino" for
# risk-adjusted learning. Compare all 4 in HPO.


def print_config():
    print(f"\n{'='*60}")
    print(f"Wyckoff RL Configuration")
    print(f"{'='*60}")
    print(f"  Data:     {WYCKOFF_NPZ_PATH}")
    print(f"  CPCV:     N={N_GROUPS}, K={K_TEST_GROUPS}, embargo={EMBARGO_BARS} bars")
    print(f"  Splits:   {N_SPLITS} splits, {N_PATHS} paths")
    print(f"  Agent:    {DEFAULT_MODEL_NAME.upper()}")
    print(f"  Reward:   {DEFAULT_ENV_PARAMS['reward_mode']}")
    print(f"  Seed:     {RANDOM_SEED}")
    print(f"  GPU:      {GPU_ID}")
    print(f"{'='*60}\n")
