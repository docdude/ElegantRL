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

# Default NPZ — z-score + tanh normalized features (40pt range bars)
WYCKOFF_NPZ_PATH = os.path.join(
    PROJECT_ROOT, "wyckoff_effort", "pipeline_output", "wyckoff_nq_40pt.npz"
)

# Results
RESULTS_DIR = os.path.join(PROJECT_ROOT, "wyckoff_effort", "rl_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CPCV Settings
# ─────────────────────────────────────────────────────────────────────────────

N_GROUPS = 5
K_TEST_GROUPS = 2
EMBARGO_BARS = 100        # in range bars (reduced for 7.6K bar dataset; ~100 bars ≈ 1 session)

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

ADAPTIVE_FEATURE = "er_ratio"       # Wyckoff effort-result ratio (new 58-feature pipeline naming)
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
    "reward_scale": 2**8,           # 256; targets cumR std ~241 per 1024-step episode (author: keep near 256)
    "num_envs": 256,                # GPU-vectorized parallel episodes (auto-scaled to GPU memory)
    "episode_len": 1024,            # sub-episode length for PPO (~7 sub-episodes in 7.6K bars)
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
    "batch_size": 512,
    "break_step": 2_000_000,
    "gamma": 0.99,
    "lambda_entropy": 0.02,
    "clip_grad_norm": 3.0,
    "repeat_times": 4.0,
    "ratio_clip": 0.25,
    "lambda_gae_adv": 0.95,
    "if_use_v_trace": True,
    "eval_per_step": 50_000,
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
