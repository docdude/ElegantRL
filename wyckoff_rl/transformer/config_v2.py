"""
Wyckoff event ontology and configuration for the transformer architecture.

v2 ontology:
- Phase head learns coarse structural regime:
    markdown / accumulation / markup / distribution
- Event head learns stateful schematic events:
    SC/AR/ST/spring/SOS/LPS and BC/AR/ST/UT/SOW/LPSY
    plus failed spring / failed upthrust

Uncertain bars are ignored in the phase loss via PHASE_IGNORE_INDEX.
"""

from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════
# Regime Labels
# ═══════════════════════════════════════════════════════════════════════

PHASE_IGNORE_INDEX = -100

REGIME_LABELS = {
    0: "markdown",
    1: "accumulation",
    2: "markup",
    3: "distribution",
}
N_REGIMES = len(REGIME_LABELS)

# Legacy alias
PHASE_LABELS = REGIME_LABELS
N_PHASES = N_REGIMES


# ═══════════════════════════════════════════════════════════════════════
# Event Labels
# ═══════════════════════════════════════════════════════════════════════

EVENT_LABELS = {
    0: "selling_climax",
    1: "automatic_rally",
    2: "st_support",
    3: "spring",
    4: "sos",
    5: "lps",

    6: "buying_climax",
    7: "automatic_reaction",
    8: "st_resistance",
    9: "upthrust",
    10: "sow",
    11: "lpsy",

    12: "failed_spring",
    13: "failed_upthrust",
}
N_EVENTS = len(EVENT_LABELS)
EVENT_LABEL_TO_INDEX = {v: k for k, v in EVENT_LABELS.items()}


# ═══════════════════════════════════════════════════════════════════════
# Feature Groups
# ═══════════════════════════════════════════════════════════════════════

TRANSFORMER_FEATURE_INDICES = [
    # Bar microstructure
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

    # Weis wave structure
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

    # Range / structural context
    48,  # pct_in_range
    49,  # range_width_norm
    50,  # bars_in_range

    # Library wave comparison features
    61,  # lib_volume_strength
    62,  # lib_wave_vs_same_dir
    64,  # lib_wave_vs_prev
]
N_TRANSFORMER_FEATURES = len(TRANSFORMER_FEATURE_INDICES)


# ═══════════════════════════════════════════════════════════════════════
# Architecture Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TransformerConfig:
    # Sequence
    seq_len: int = 64
    n_bar_features: int = N_TRANSFORMER_FEATURES

    # Encoder
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 96
    dropout: float = 0.38

    # Position features
    n_position_features: int = 8

    # Action space
    n_actions: int = 6

    # Heads
    n_phases: int = N_REGIMES
    n_events: int = N_EVENTS

    # Training
    phase_loss_weight: float = 1.0
    event_loss_weight: float = 0.5
    rl_loss_weight: float = 1.0

    # Episode
    max_episode_bars: int = 512
    warmup_bars: int = 32

    # Feature indices
    feature_indices: list[int] | None = None
