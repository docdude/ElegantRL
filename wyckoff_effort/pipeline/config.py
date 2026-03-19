"""
Pipeline configuration — paths, feature schema, and defaults.
"""

import os
import sys

# ── Ensure RiskLabAI is importable ───────────────────────────────────────────
_RISKLABAI_ROOT = "/opt/RiskLabAI.py"
if _RISKLABAI_ROOT not in sys.path:
    sys.path.insert(0, _RISKLABAI_ROOT)

# ── Paths ────────────────────────────────────────────────────────────────────

WYCKOFF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WYCKOFF_ROOT, "datasets")
OUTPUT_DIR = os.path.join(WYCKOFF_ROOT, "pipeline_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default SCID path (place your .scid files in wyckoff_effort/datasets/)
DEFAULT_SCID_PATH = os.path.join(DATA_DIR, "NQZ25-CME.scid")

# Default output NPZ
DEFAULT_NPZ_PATH = os.path.join(OUTPUT_DIR, "wyckoff_nq.npz")


# ── Range Bar Settings ───────────────────────────────────────────────────────

RANGE_BAR_SIZE = 40.0  # NQ points per bar (40-point range bars)
RTH_ONLY = False       # True = regular trading hours only


# ── Wyckoff Analyzer Settings ────────────────────────────────────────────────

WYCKOFF_PARAMS = {
    "reversal_pct": 0.5,
    "er_lookback": 20,
    "vol_threshold": 1.5,
    "er_threshold": 0.5,
    "swing_lookback": 20,
    "vol_multiplier": 2.0,
    "phase_lookback": 50,
}


# ── New Feature Engineering Settings (wyckoff_features.py) ──────────────────

REVERSAL_POINTS = 200.0  # ZigZag reversal for Weis Wave (5× bar size)


# ── Feature Schema ───────────────────────────────────────────────────────────
# These are the columns extracted from analyze_wyckoff() output into tech_ary.
# Order matters — it defines the column positions in the NPZ.

WYCKOFF_FEATURE_COLUMNS = [
    # Core microstructure
    "delta_norm",       # normalized bar delta (-1 to +1)
    "cvd_slope",        # CVD trend direction (tanh-scaled)
    "ER_Ratio",         # effort/result ratio
    "VolumeFilter",     # volume vs MA ratio

    # Wave features
    "WaveDir",          # current wave direction (+1/-1)
    "WaveER",           # wave-level effort/result

    # Strength filter
    "VolumeStrength",   # 0-4 discrete strength
    "TimeStrength",     # 0-4 discrete strength

    # Phase classification
    "Phase",            # 0-4 (Accum/Markup/Distrib/Markdown/Uncertain)

    # Event flags (binary)
    "Spring",           # 0/1
    "Upthrust",         # 0/1
    "SellingClimax",    # 0/1
    "BuyingClimax",     # 0/1
    "Absorption",       # 0/1
    "AbsorptionType",   # -1/0/+1

    # Wave comparison (from ww_utils)
    "LargeWave",        # 0/1
    "LargeER",          # 0/1
    "WaveVsSame",       # -1/0/+1
    "WaveVsPrev",       # -1/0/+1
    "ERVsSame",         # -1/0/+1
    "ERVsPrev",         # -1/0/+1
]

N_WYCKOFF_FEATURES = len(WYCKOFF_FEATURE_COLUMNS)

# Old feature count kept for backward compat; new pipeline uses N_NEW_FEATURES
# which is set dynamically by wyckoff_features.build_all_features()


# ── Triple Barrier Settings (for meta-labeling) ─────────────────────────────

TRIPLE_BARRIER_DEFAULTS = {
    "pt_multiplier": 1.5,    # take-profit = multiplier × ATR
    "sl_multiplier": 1.0,    # stop-loss = multiplier × ATR
    "vertical_bars": 50,     # max holding period in bars
    "min_holding": 5,        # minimum bars before exit
    "atr_period": 20,        # ATR lookback
}


# ── Meta-Labeling Settings ───────────────────────────────────────────────────

META_LABEL_DEFAULTS = {
    "n_estimators": 500,
    "max_depth": 5,
    "min_samples_leaf": 20,
    "n_splits": 5,           # CPCV folds
    "n_test_groups": 2,      # CPCV test groups per split
    "embargo_pct": 0.01,     # embargo fraction
}


# ── Feature Selection Settings ───────────────────────────────────────────────

FEATURE_SELECTION_DEFAULTS = {
    "variance_threshold": 0.95,  # PCA cumulative variance to retain
    "kde_bandwidth": 0.01,       # for RMT denoising
}
