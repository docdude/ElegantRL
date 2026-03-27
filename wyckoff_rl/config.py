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
EMBARGO_BARS = 200        # in range bars (~200 bars ≈ 2 sessions; prevents leakage across folds)

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
# Instrument Presets
# ─────────────────────────────────────────────────────────────────────────────

INSTRUMENT_PRESETS = {
    "nq": {
        "tick_size": 0.25,
        "tick_value": 5.0,          # $5/tick → $20/point
        "commission": 1.50,         # $/side/contract
        "bar_range": 40.0,          # default range bar size (pts)
        "pnl_norm": 10000.0,        # reward normalization (dense_pnl auto-overrides to 500)
        "slippage_ticks": 1.0,
    },
    "us30": {
        "tick_size": 1.0,
        "tick_value": 1.0,          # $1/point (micro)
        "commission": 1.00,         # $/side/contract (micro)
        "bar_range": 100.0,         # 100pt range bars
        "pnl_norm": 100.0,          # $100/bar-move → O(1) reward
        "slippage_ticks": 1.0,
    },
}

DEFAULT_INSTRUMENT = "nq"


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

# Sliding window & feature selection
from wyckoff_rl.feature_config import SELECTED_INDICES, N_SELECTED_FEATURES, WINDOW_SIZE

DEFAULT_ENV_PARAMS = {
    "initial_amount": 1000.0,       # in NQ points (legacy continuous env)
    "cost_per_trade": 0.5,          # points per side (legacy continuous env)
    "reward_mode": "dense_pnl",       # "dense_pnl", "sparse_exit", "pnl", "log_ret", "sharpe", "sortino"
    "sign_flip": True,                # randomly flip long/short each episode to kill directional bias
    "reward_scale": 1.0,            # NQ discrete env handles its own scaling
    "num_envs": 4096,               # GPU-vectorized parallel episodes (auto-scaled to GPU memory)
    "episode_len": 2048,            # sub-episode length for PPO (~3.7 sub-episodes in 7.6K bars)
    "window_size": WINDOW_SIZE,     # sliding window of bars for temporal context (legacy continuous)
    "feature_indices": SELECTED_INDICES,  # column indices into 61-feature tech_ary (legacy continuous)
    "continuous_sizing": False,     # False={-1,0,+1} binary, True=[-1,+1] continuous
    "trade_reward_weight": 0.5,     # 0.0=bar-only (original), 0.5=adds concentrated trade-close bonus
    # NQ Wyckoff-Weis discrete env params
    "commission": 1.50,             # $ per side per contract (NQ)
    "slippage_ticks": 1.0,          # ticks of slippage per side
    "max_position_size": 2,         # max contracts
}


# ─────────────────────────────────────────────────────────────────────────────
# DRL Agent
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_NAME = "discrete_ppo"
RANDOM_SEED = 42
GPU_ID = 0

DEFAULT_ERL_PARAMS = {
    "net_dims": [256, 128],
    "learning_rate": 2e-4,
    "batch_size": 512,
    "break_step": 2_000_000,
    "gamma": 0.99,
    "lambda_entropy": -0.003, # Entropy bonus; small enough not to overwhelm the reduced reward scales
    "clip_grad_norm": 3.0,
    "repeat_times": 4.0,
    "ratio_clip": 0.25,
    "lambda_gae_adv": 0.95,
    "if_use_v_trace": True,
    "eval_per_step": 50_000,
    "eval_times": 16,
    "loss_weight": 1.0,             # Asymmetric advantage: 1.0=symmetric (standard PPO), 2.0=2x penalty for losses
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
