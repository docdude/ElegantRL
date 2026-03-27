"""
Wyckoff event ontology and configuration for the transformer architecture.

Defines the Wyckoff phase labels, event types, and architecture hyperparameters.
"""
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════
# Wyckoff Phase Labels (Layer 2 — regime classification)
# ═══════════════════════════════════════════════════════════════════════

PHASE_LABELS = {
    0: "no_regime",         # Unclear / transitional
    1: "accumulation",      # PS → SC → AR → ST → Spring → SOS → LPS → BU
    2: "re_accumulation",   # Consolidation within markup
    3: "markup",            # Rising wave structure, demand > supply
    4: "distribution",      # PSY → BC → AR → ST → UT → SOW → LPSY
    5: "re_distribution",   # Consolidation within markdown  
    6: "markdown",          # Falling wave structure, supply > demand
}

N_PHASES = len(PHASE_LABELS)

# ═══════════════════════════════════════════════════════════════════════
# Wyckoff Event Labels (Layer 2 — event detection)
# Multi-label: multiple events can be active simultaneously
# ═══════════════════════════════════════════════════════════════════════

EVENT_LABELS = {
    0: "none",              # No significant event
    1: "spring",            # Price breaks below support, reverses, low volume
    2: "upthrust",          # Price breaks above resistance, reverses, low volume
    3: "selling_climax",    # High volume, wide spread, down — SC
    4: "buying_climax",     # High volume, wide spread, up — BC
    5: "test",              # Return to support/resistance on LOW volume
    6: "sos",               # Sign of Strength — up move on expanding volume
    7: "sow",               # Sign of Weakness — down move on expanding volume
    8: "no_demand",         # Up bar on low volume — weakness in disguise
    9: "no_supply",         # Down bar on low volume — demand in disguise
    10: "effort_gt_result", # High vol, small displacement — absorption
    11: "lps",              # Last Point of Support — final test before markup
    12: "lpsy",             # Last Point of Supply — final test before markdown
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
    # Weis Wave structure (15 features — precomputed wave objects)
    15,  # wave_direction
    16,  # wave_progress
    17,  # wave_displacement_norm
    18,  # wave_vol_cumulative_norm
    19,  # wave_delta_ratio
    20,  # wave_vol_vs_same
    21,  # wave_vol_vs_prev
    22,  # wave_disp_vs_same
    23,  # wave_disp_vs_prev
    27,  # demand_score_3wave
    28,  # supply_score_3wave
    29,  # wave_vol_trend_up
    30,  # wave_vol_trend_down
    31,  # wave_shortening_up
    32,  # wave_shortening_down
    # Wyckoff events — raw scores as attention landmarks (6 features)
    35,  # spring_score
    36,  # upthrust_score
    37,  # sc_score
    38,  # bc_score
    39,  # absorption_score
    41,  # stopping_action_score
    # Range / structural context (4 features)
    48,  # pct_in_range
    49,  # range_width_norm
    50,  # bars_in_range
    34,  # large_wave_score
    # Library wave features (6 features)
    61,  # lib_volume_strength
    62,  # lib_wave_vs_same_dir
    64,  # lib_wave_vs_prev
    67,  # lib_pivot_flag
    68,  # lib_exhaust_up
    69,  # lib_exhaust_down
]

N_TRANSFORMER_FEATURES = len(TRANSFORMER_FEATURE_INDICES)  # 41

# ═══════════════════════════════════════════════════════════════════════
# Architecture Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TransformerConfig:
    # Sequence
    seq_len: int = 128              # Context window (bars)
    n_bar_features: int = N_TRANSFORMER_FEATURES  # Per-bar input dim

    # Encoder
    d_model: int = 64               # Embedding dimension
    n_heads: int = 4                # Attention heads
    n_layers: int = 3               # Transformer layers
    d_ff: int = 128                 # Feed-forward inner dim
    dropout: float = 0.3            # Dropout rate (higher for small datasets)

    # Position features (appended at policy layer, not encoder)
    n_position_features: int = 8    # [side, size, entry_dist, unreal, real, bars_in, mfe, mae]

    # Action space
    n_actions: int = 6              # HOLD, ENTER_L, ENTER_S, ADD, REDUCE, EXIT

    # Heads
    n_phases: int = N_PHASES        # Phase classification
    n_events: int = N_EVENTS        # Event detection (multi-label)

    # Training
    phase_loss_weight: float = 1.0
    event_loss_weight: float = 0.5
    rl_loss_weight: float = 1.0

    # Episode
    max_episode_bars: int = 512     # Longer episodes for full schematics
    warmup_bars: int = 32           # Bars before first action (encoder needs context)

    # Feature indices (into 72-column tech_ary); None → use TRANSFORMER_FEATURE_INDICES
    feature_indices: list[int] | None = None
