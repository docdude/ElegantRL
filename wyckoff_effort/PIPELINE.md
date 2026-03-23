# Wyckoff Effort — Data Pipeline Reference

## Directory Structure

```
wyckoff_effort/
├── pipeline/              # 5-phase data processing pipeline
├── utils/                 # SCID parsing, Wyckoff analysis, plotting
│   └── models_utils/      # Weis Wave models & custom indicators
├── deep_m_effort/         # Deep-M Effort indicator (visualization)
├── datasets/              # Input SCID files
└── pipeline_output/       # Generated NPZ, Parquet, models
```

---

## Pipeline Flow

```
SCID File (raw ticks, 40-byte records)
    │
    ▼
[scid_parser.py]  ──→  Tick DataFrame (datetime, OHLC, volume, bid/ask)
    │
    ▼
resample_range_bars()  ──→  40pt range bars (~7,635 bars)
    │
    ▼
[wyckoff_analyzer.py]  ──→  Wyckoff features (delta, CVD, waves, events, phase)
    │
    ▼
[wyckoff_features.py]  ──→  58-feature matrix (tech_ary)
    │
    ▼
[NPZ save]  ──→  wyckoff_nq_40pt.npz  (close_ary + tech_ary + feature_names)
    │
    ▼
[feature_selection.py]  ──→  Denoising (RMT Marcenko-Pastur), PCA, MDI/MDA ranking
    │
    ▼
[signal_extraction.py]  ──→  Wyckoff signal events + triple barrier labels
    │
    ▼
[meta_labeling.py]  ──→  P(success) classifier (RF / XGBoost via CPCV)
    │
    ▼
[bet_sizing.py]  ──→  Position sizes from meta-label probabilities
    │
    ▼
[evaluation.py]  ──→  Sharpe, Sortino, Max DD, PBO, PSR
```

---

## Scripts

### Pipeline Module (`pipeline/`)

| Script | Phase | Purpose |
|--------|-------|---------|
| `config.py` | — | Central config: paths, feature columns, `RANGE_BAR_SIZE=40.0` |
| `feature_extraction.py` | 1 | SCID ticks → range bars → Wyckoff analysis → NPZ |
| `feature_selection.py` | 2 | RMT covariance denoising, PCA orthogonalization, feature importance |
| `wyckoff_features.py` | 3 | Extended feature engineering (65 features: microstructure, waves, events, phase) |
| `signal_extraction.py` | 4a | Wyckoff signal events + triple barrier labeling + sample weights |
| `meta_labeling.py` | 4b | Secondary classifier for P(success) per signal |
| `bet_sizing.py` | 4b | Meta-label probabilities → position sizes (CDF / Kelly / tiered) |
| `evaluation.py` | 5 | Backtest evaluation: Sharpe, Sortino, PBO, PSR |
| `test_smoke.py` | Test | Smoke test for all preprocessing phases |

### Utils (`utils/`)

| Script | Purpose |
|--------|---------|
| `scid_parser.py` | Parse Sierra Chart `.scid` binary files (tick-by-tick), `resample_range_bars()` |
| `wyckoff_analyzer.py` | Full Wyckoff analysis: delta, CVD, Weis waves, absorption, Spring/Upthrust/SC/BC, phase |
| `alpaca_client.py` | Alpaca Markets data client (alternative to SCID) |
| `plotter.py` | Deep-M Effort zone visualization (Plotly) |
| `models_utils/ww_models.py` | Data models: ZigZag, Wave config enums/classes |
| `models_utils/ww_utils.py` | Weis Wave core: ZigZag segmentation, wave comparison, yellow bar detection |
| `models_utils/custom_mas.py` | Moving averages: SMA, EMA, WMA, Hull, VIDYA, Kaufman Adaptive |

### Deep-M Effort (`deep_m_effort/`)

| Script | Purpose |
|--------|---------|
| `deep_m_effort.py` | Consolidated Deep-M logic: effort zones, adaptive MA |
| `deep_m_effort_nq.py` | NQ-specific matplotlib recreation of DeepCharts indicator |
| `deep_m_effort_scid.py` | Deep-M with native SCID reader (Plotly output) |
| `deep_m_effort_scid_optimized.py` | Optimized SCID reader with numpy memmap |

---

## Creating the Training Dataset

The training NPZ is generated in two steps:

### Step 1: Generate full 58-feature NPZ

```bash
# Option A: wyckoff_features.py (has CLI flags)
python wyckoff_effort/pipeline/wyckoff_features.py \
  --scid wyckoff_effort/datasets/NQZ25-CME.scid \
  --bar-size 40.0 \
  --reversal 40.0 \
  --output-dir wyckoff_effort/pipeline_output/ \
  --no-importance

# Option B: feature_extraction.py (reads defaults from config.py)
python -m wyckoff_effort.pipeline.feature_extraction
```

Both produce `wyckoff_nq_40pt.npz` containing `close_ary`, `tech_ary` (n_bars × 58 features),
`feature_names`, and `dates_ary`. A matching Parquet file with the raw OHLCV bars is also saved.

### Step 2: Feature pruning at training time

The NPZ ships with all 58 features. Feature selection happens **downstream** at RL training
time — `wyckoff_rl/feature_config.py` defines `SELECTED_INDICES` (33 of 58) based on
correlation/redundancy analysis. The training environment (`WyckoffTradingVecEnv`) slices
`tech_ary[:, SELECTED_INDICES]` so the agent only sees the curated 33-feature subset.
This keeps the NPZ as a single source of truth while allowing feature selection to be
iterated independently without regenerating data.

### Other CLI

```bash
# Meta-labeling (Phase 4b)
python run_40pt_meta_labeling.py
# Expects: pipeline_output/wyckoff_nq_40pt.npz + wyckoff_nq_40pt_bars.parquet

# Validate NPZ before training
python -m wyckoff_rl.tools.validate_training_data_npz \
  --npz wyckoff_effort/pipeline_output/wyckoff_nq_40pt.npz

# Smoke test
python -m wyckoff_effort.pipeline.test_smoke
```

---

## Output Files

| File | Format | Content |
|------|--------|---------|
| `pipeline_output/wyckoff_nq_40pt.npz` | NumPy zip | close_ary (7635,1), tech_ary (7635,58), feature_names |
| `pipeline_output/wyckoff_nq_40pt_bars.parquet` | Parquet | Canonical OHLCV range bars |
| `pipeline_output/meta_label_model.pkl` | Pickle | Trained RF/XGBoost classifier |
| `pipeline_output/signal_events.parquet` | Parquet | Wyckoff events + labels + features |

---

## Feature Summary

58 raw features organized as:
- **Bar microstructure** (9): body_ratio, upper_wick_ratio, delta_ratio, vol_vs_ma20, ...
- **Weis Wave** (14): wave_progress, wave_displacement_norm, wave_vol_cumulative_norm, ...
- **Wyckoff events** (5): spring_score, upthrust_score, sc_score, bc_score, absorption_score
- **Range/context** (3): pct_in_range, range_width_norm, bars_in_range
- **Returns/momentum/CVD/misc**: remaining features

After feature selection (see `wyckoff_rl/feature_config.py`): **33 features** used for RL training.
