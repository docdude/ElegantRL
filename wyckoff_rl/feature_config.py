"""
Curated feature selection for Wyckoff sliding-window observation.

Based on feature-return correlation analysis (58 → 33 features):
  - Drops near-zero-signal features (|r| < 0.01 at all horizons)
  - Drops redundant features (|r_pair| > 0.8)
  - Drops degenerate features (2-3 unique values)
  - Drops features that only make sense with single-bar context
    (e.g. bars_since_* decay timers) — the sliding window replaces them
  - Keeps raw event scores that become useful as temporal landmarks in the window

The sliding window lets the network see sequences of these per-bar signals,
replacing hand-crafted temporal compressions (decay timers, phase scores)
with learned temporal pattern recognition.
"""

# All 58 features in the original NPZ, in order
ALL_FEATURES = [
    'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio', 'close_location', 'delta_ratio',
    'vol_vs_ma20', 'vol_vs_ma50', 'er_ratio', 'duration_norm', 'cvd_slope_fast',
    'cvd_slope_slow', 'cvd_divergence', 'return_1', 'return_5', 'volatility_20',
    'wave_direction', 'wave_progress', 'wave_displacement_norm', 'wave_vol_cumulative_norm', 'wave_delta_ratio',
    'wave_vol_vs_same', 'wave_vol_vs_prev', 'wave_disp_vs_same', 'wave_disp_vs_prev', 'wave_er_vs_same',
    'wave_er_vs_prev', 'wave_delta_vs_same', 'demand_score_3wave', 'supply_score_3wave', 'wave_vol_trend_up',
    'wave_vol_trend_down', 'wave_shortening_up', 'wave_shortening_down', 'yellow_bar', 'large_wave_score',
    'spring_score', 'upthrust_score', 'sc_score', 'bc_score', 'absorption_score',
    'absorption_direction', 'stopping_action_score', 'bars_since_spring', 'bars_since_upthrust', 'bars_since_climax',
    'cumulative_absorption', 'event_sequence_bull', 'event_sequence_bear', 'pct_in_range', 'range_width_norm',
    'bars_in_range', 'support_test_count', 'resistance_test_count', 'phase_accum_score', 'phase_markup_score',
    'phase_distrib_score', 'phase_markdown_score', 'trend_4x',
]

# ─── Features to DROP ───────────────────────────────────────────────────────

# Redundant with body_ratio (|r| > 0.92)
# close_location, return_1

# Redundant with vol_vs_ma20 (|r| = 0.889)
# vol_vs_ma50

# Near-zero signal at all horizons (max |r| < 0.01)
# er_ratio, wave_er_vs_same, wave_er_vs_prev, trend_4x, lower_wick_ratio

# Degenerate (2-3 unique values, not useful as continuous signal)
# wave_direction (binary ±1), absorption_direction (ternary)

# Replaced by sliding window temporal context:
# bars_since_spring, bars_since_upthrust, bars_since_climax (decay timers)
# event_sequence_bull, event_sequence_bear (hand-crafted schematic score)
# phase_accum_score, phase_markup_score, phase_distrib_score, phase_markdown_score
# support_test_count, resistance_test_count (low unique values)
# cumulative_absorption (10-bar rolling sum — window captures this directly)

SELECTED_FEATURES = [
    # Block 1: Bar Microstructure (9 of 15)
    'body_ratio',               # Bar character: trend vs reversal
    'upper_wick_ratio',         # Rejection of higher prices
    'delta_ratio',              # Buyer/seller conviction per bar
    'vol_vs_ma20',              # Relative volume (effort)
    'duration_norm',            # Bar pace / urgency
    'cvd_slope_fast',           # Short-term order flow momentum
    'cvd_slope_slow',           # Longer-term order flow trend
    'cvd_divergence',           # Price vs order flow disagreement
    'return_5',                 # 5-bar momentum context
    'volatility_20',            # Regime / contraction detection

    # Block 2: Weis Wave Analysis (14 of 20)
    'wave_progress',            # Where we are in current wave
    'wave_displacement_norm',   # Current wave size vs typical
    'wave_vol_cumulative_norm', # Current wave volume vs typical
    'wave_delta_ratio',         # Directional conviction in wave
    'wave_vol_vs_same',         # Volume comparison (same direction)
    'wave_vol_vs_prev',         # Volume comparison (vs previous)
    'wave_disp_vs_same',        # Displacement comparison (same dir)
    'wave_disp_vs_prev',        # Displacement comparison (vs prev)
    'wave_delta_vs_same',       # Delta ratio change (same dir)
    'demand_score_3wave',       # Multi-wave demand pressure
    'supply_score_3wave',       # Multi-wave supply pressure
    'wave_vol_trend_up',        # Volume trend in up waves (best predictor)
    'wave_vol_trend_down',      # Volume trend in down waves (best predictor)
    'wave_shortening_up',       # Displacement exhaustion (up)
    'wave_shortening_down',     # Displacement exhaustion (down)
    'yellow_bar',               # Cur vol exceeds prev same-dir total
    'large_wave_score',         # Cur vol / avg of last 4 completed

    # Block 3: Wyckoff Events — raw scores as temporal landmarks (5 of 13)
    'spring_score',             # Spring detection (87% zero → landmark in window)
    'upthrust_score',           # Upthrust detection (87% zero → landmark)
    'sc_score',                 # Selling climax score
    'bc_score',                 # Buying climax score
    'absorption_score',         # High vol + small price change
    'stopping_action_score',    # Large wave vol + small displacement

    # Block 4: Range/Context (3 of 10)
    'pct_in_range',             # Price location within structure
    'range_width_norm',         # Range size (context)
    'bars_in_range',            # How developed the range is
]

N_SELECTED_FEATURES = len(SELECTED_FEATURES)  # 33

# Precompute column indices for fast slicing
SELECTED_INDICES = [ALL_FEATURES.index(f) for f in SELECTED_FEATURES]

# Sliding window size (bars of context)
WINDOW_SIZE = 30


def select_features(tech_ary):
    """Select curated features from full 58-feature tech_ary.

    Parameters
    ----------
    tech_ary : np.ndarray, shape (n_bars, 58)

    Returns
    -------
    np.ndarray, shape (n_bars, N_SELECTED_FEATURES)
    """
    return tech_ary[:, SELECTED_INDICES]
