"""
Wyckoff event ontology and configuration for the transformer architecture.

Defines the regime labels, event types, and architecture hyperparameters.
"""
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════
# Regime Labels (Layer 2 — simplified regime classification)
# ═══════════════════════════════════════════════════════════════════════

REGIME_LABELS = {
    0: "balance",       # Range-bound: accumulation, distribution, re-accum, re-distrib, unclear
    1: "uptrend",       # Markup: rising structure, demand > supply
    2: "downtrend",     # Markdown: falling structure, supply > demand
}

N_REGIMES = len(REGIME_LABELS)

# Legacy alias (kept for loading old checkpoints)
PHASE_LABELS = REGIME_LABELS
N_PHASES = N_REGIMES

# ═══════════════════════════════════════════════════════════════════════
# Event Labels (Layer 2 — simplified event detection)
# Multi-label: multiple events can be active simultaneously
# ═══════════════════════════════════════════════════════════════════════

EVENT_LABELS = {
    0: "spring_like",       # Price breaks below support, reverses (includes springs, LPS, tests of support)
    1: "upthrust_like",     # Price breaks above resistance, reverses (includes UT, LPSY, tests of resistance)
    2: "absorption_like",   # High effort, little result — indecision / SOT (effort > result, no demand/supply)
    3: "exhaustion_like",   # Climax volume + reversal — SC/BC terminal moves
}

N_EVENTS = len(EVENT_LABELS)

# ═══════════════════════════════════════════════════════════════════════
# Feature Groups (semantic grouping for the encoder)
# Indices into ALL_FEATURES (72 columns in tech_ary)
# ═══════════════════════════════════════════════════════════════════════

# Raw bar features the transformer should learn from
TRANSFORMER_FEATURE_INDICES = [
    # Bar microstructure (10 features)
    0,   # body_ratio
    1,   # upper_wick_ratio
    2,   # lower_wick_ratio
    3,   # close_location
    4,   # delta_ratio
    5,   # vol_vs_ma20
    8,   # duration_norm
    9,   # cvd_slope_fast
    12,  # return_1
    14,  # volatility_20
    # Weis Wave structure (11 features — precomputed wave objects, NO heuristic scores)
    15,  # wave_direction
    16,  # wave_progress
    17,  # wave_displacement_norm
    18,  # wave_vol_cumulative_norm
    19,  # wave_delta_ratio
    20,  # wave_vol_vs_same
    21,  # wave_vol_vs_prev
    22,  # wave_disp_vs_same
    23,  # wave_disp_vs_prev
    29,  # wave_vol_trend_up
    30,  # wave_vol_trend_down
    31,  # wave_shortening_up
    32,  # wave_shortening_down
    # Range / structural context (3 features)
    48,  # pct_in_range
    49,  # range_width_norm
    50,  # bars_in_range
    # Library wave features (3 features — pure wave comparison, no heuristic flags)
    61,  # lib_volume_strength
    62,  # lib_wave_vs_same_dir
    64,  # lib_wave_vs_prev
]

N_TRANSFORMER_FEATURES = len(TRANSFORMER_FEATURE_INDICES)  # 29

# ═══════════════════════════════════════════════════════════════════════
# Architecture Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TransformerConfig:
    # Sequence
    seq_len: int = 64               # Context window (bars)
    n_bar_features: int = N_TRANSFORMER_FEATURES  # Per-bar input dim

    # Encoder
    d_model: int = 32               # Embedding dimension (smaller for small datasets)
    n_heads: int = 4                # Attention heads
    n_layers: int = 2               # Transformer layers (fewer = less overfit risk)
    d_ff: int = 64                  # Feed-forward inner dim
    dropout: float = 0.3            # Dropout rate

    # Position features (appended at policy layer, not encoder)
    n_position_features: int = 8    # [side, size, entry_dist, unreal, real, bars_in, mfe, mae]

    # Action space
    n_actions: int = 6              # HOLD, ENTER_L, ENTER_S, ADD, REDUCE, EXIT

    # Heads
    n_phases: int = N_REGIMES       # Regime classification (balance/uptrend/downtrend)
    n_events: int = N_EVENTS        # Event detection (multi-label, 4 channels)

    # Training
    phase_loss_weight: float = 1.0
    event_loss_weight: float = 0.5
    rl_loss_weight: float = 1.0

    # Episode
    max_episode_bars: int = 512     # Longer episodes for full schematics
    warmup_bars: int = 32           # Bars before first action (encoder needs context)

    # Feature indices (into 72-column tech_ary); None → use TRANSFORMER_FEATURE_INDICES
    feature_indices: list[int] | None = None
