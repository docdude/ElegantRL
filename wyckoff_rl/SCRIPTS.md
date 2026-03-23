# Wyckoff RL — Scripts & CLI Reference

## Directory Structure

```
wyckoff_rl/
├── run_train.py               # Main training entry point
├── config.py                  # Global config (paths, hyperparams, env params)
├── feature_config.py          # 58 → 33 feature selection
├── function_train_test.py     # Train/test helpers, agent factory, data slicing
├── wyckoff_agent.py           # PPO agent with CNN temporal encoder
├── wyckoff_wave_agent.py      # PPO agent with WaveNet temporal encoder
├── wyckoff_actor_critic.py    # CNN actor/critic networks
├── wyckoff_wave_actor_critic.py # WaveNet actor/critic networks
├── eval_all_checkpoints.py    # Evaluate all checkpoints per split
├── normalize_npz.py           # Pre-normalize NPZ to [-1,1]
├── tools/                     # Replay, veto discovery, analysis, validation
├── live/                      # Live/paper trading system
├── data_pipeline/             # Feature extraction & selection (Phase 1-2)
└── cpcv_pipeline/             # Combinatorial purged cross-validation
```

---

## Training

### `run_train.py` — Main Training Script

Runs Adaptive CPCV training across splits with GPU-vectorized environments.

```bash
# WaveNet weighted (current best config)
python -m wyckoff_rl.run_train \
  --model wyckoff_wave_ppo --continuous \
  --trade-reward-weight 0.5 --loss-weight 2.0 \
  --splits 0-9

# CNN weighted
python -m wyckoff_rl.run_train \
  --model wyckoff_ppo --continuous \
  --trade-reward-weight 0.5 --loss-weight 2.0 \
  --splits 5-9

# CNN no-weights (original)
python -m wyckoff_rl.run_train \
  --model wyckoff_ppo --continuous \
  --trade-reward-weight 0.0 --loss-weight 1.0 \
  --splits 0-9

# Single split, dry run
python -m wyckoff_rl.run_train --model wyckoff_wave_ppo --continuous --split 3 --dry-run
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | ppo | `ppo` (MLP), `wyckoff_ppo` (CNN), `wyckoff_wave_ppo` (WaveNet), `sac`, `td3` |
| `--reward` | pnl | `pnl`, `log_ret`, `sharpe`, `sortino` |
| `--continuous` | False | Continuous [-1,+1] sizing vs binary {-1,0,+1} |
| `--loss-weight` | 2.0 | Asymmetric advantage penalty (1.0=symmetric PPO, 2.0=2x loss penalty) |
| `--trade-reward-weight` | 0.5 | Trade-close PnL bonus (0.0=bar-only, 0.5=adds concentrated bonus) |
| `--npz` | config default | Path to Wyckoff NPZ file |
| `--net-dims` | "128,64" | MLP hidden dims (post-encoder) |
| `--lr` | 1e-4 | Learning rate |
| `--batch-size` | 512 | Batch size |
| `--break-step` | 2,000,000 | Total training steps |
| `--n-groups` | 5 | CPCV groups |
| `--k-test` | 2 | CPCV test groups → 10 splits |
| `--embargo` | 100 | CPCV embargo bars |
| `--split` | None | Run only this split (0-based) |
| `--splits` | None | Range or list: `"0-4"` or `"0,2,5"` |
| `--gpu` | 0 | GPU device ID |
| `--dry-run` | False | Print splits without training |
| `--continue` | False | Continue training from last checkpoint |

---

## Tools (`tools/`)

### `run_replay.py` — OOS Replay

Precomputes range bars + features once, then replays each checkpoint on SCID data.

```bash
# Basic replay
python -m wyckoff_rl.tools.run_replay \
  -c wyckoff_rl/live/checkpoints/best_models/wavenet_weighted_split3.pt \
  -l my_label --continuous --export-logs

# With veto rules
python -m wyckoff_rl.tools.run_replay \
  -c wyckoff_rl/live/checkpoints/best_models/wavenet_weighted_split3.pt \
  -l wnet_veto --continuous \
  --veto "large_wave_score > 0.950, range_width_norm < 0.176"

# With veto preset
python -m wyckoff_rl.tools.run_replay \
  -c path/to/actor.pt -l test --continuous \
  --veto-preset wavenet_pair

# Compare with vs without veto
python -m wyckoff_rl.tools.run_replay \
  -c path/to/actor.pt -l compare --continuous \
  --veto "large_wave_score > 0.950" --compare-veto

# Multiple checkpoints
python -m wyckoff_rl.tools.run_replay \
  -c split3.pt split4.pt split5.pt \
  --continuous --export-logs
```

| Flag | Description |
|------|-------------|
| `-c / --checkpoints` | Actor .pt file(s) |
| `-l / --label` | Run label (used for log directory name) |
| `--continuous` | Continuous sizing mode |
| `--veto` | Inline veto rules: `"feature > thresh, feature2 < thresh2"` |
| `--veto-preset` | Named preset: `cnn_split3`, `cnn_split4`, `wavenet_single`, `wavenet_pair` |
| `--compare-veto` | Run both with and without veto side-by-side |
| `--scid` | SCID file path |
| `--start-date / --end-date` | Replay date range |
| `--export-logs` | Export trade CSV to `live_logs/replay/<label>/` |

### `discover_veto.py` — Veto Rule Discovery

Sweeps feature thresholds to find entry veto rules that improve OOS performance.

```bash
python -m wyckoff_rl.tools.discover_veto \
  -c path/to/actor.pt --continuous

# With critic value gate
python -m wyckoff_rl.tools.discover_veto \
  -c path/to/actor.pt --critics path/to/cri.pth --continuous
```

### `analyze_trades.py` — Trade Analysis

```bash
# Entry feature profiles (Z-scores)
python -m wyckoff_rl.tools.analyze_trades profile -c actor.pt --continuous

# Win/loss comparison
python -m wyckoff_rl.tools.analyze_trades losses -c actor.pt --continuous

# Deep-dive on specific trade
python -m wyckoff_rl.tools.analyze_trades deep-dive -c actor.pt --continuous --trade 42
```

### `validate_training_data_npz.py` — NPZ Validation

```bash
python -m wyckoff_rl.tools.validate_training_data_npz --npz path/to/wyckoff_nq_40pt.npz
```

Checks: NaN/Inf, constant columns, outliers, scale mismatches, high correlation.

### Other Tools

| Script | Purpose |
|--------|---------|
| `validate_replay.py` | Regression test: compare replay results vs known-good trade logs |
| `test_range_bars.py` | Test range bar construction + Deep-M effort visualization |
| `test_real_nq.py` | Unit test: SCID parse → Wyckoff analysis event counts |
| `test_volume.py` | Volume sanity check on SCID data |

---

## Live Trading (`live/`)

### `trader.py` — Main Entry Point

```bash
# Paper trading via IB TWS (market hours)
python -m wyckoff_rl.live.trader live \
  --checkpoint wyckoff_rl/live/checkpoints/best_models/wavenet_weighted_split3.pt \
  --continuous --expiry 20260618 \
  --veto-preset wavenet_pair

# SCID replay (offline, any time)
python -m wyckoff_rl.live.trader replay \
  --checkpoint path/to/actor.pt --continuous \
  --scid /opt/SierraChart/Data/NQH26-CME.scid \
  --start-date 2026-03-18

# Tail SC data file + IB execution
python -m wyckoff_rl.live.trader tail \
  --checkpoint path/to/actor.pt --continuous \
  --scid /opt/SierraChart/Data/NQM26-CME.scid \
  --expiry 20260618
```

| Mode | Data Source | Execution | Use Case |
|------|------------|-----------|----------|
| `replay` | SCID file | SimExecutor | Offline backtesting |
| `live` | IB real-time ticks | IB orders | Live/paper trading |
| `tail` | SC .scid file (live) | IB orders | SC data feed + IB execution |

### Live System Architecture

```
DataAdapter (SCID / IB / SCIDTail)
    │
    ▼
RangeBarBuilder  ──→  40pt range bars from tick stream
    │
    ▼
LiveFeatureEngine  ──→  36-feature vector per bar (200-bar rolling buffer)
    │
    ▼
InferenceEngine  ──→  Actor forward pass → target position [-1, +1]
    │
    ▼
VetoFilter (optional)  ──→  Block entries when veto rules fire
    │
    ▼
OrderAdapter (Sim / IB)  ──→  Market orders → position changes
```

### Live Module Files

| File | Purpose |
|------|---------|
| `trader.py` | Main orchestrator + CLI + VETO_PRESETS |
| `inference.py` | Load actor checkpoint, maintain sliding window, get_action() |
| `precompute.py` | Precomputed replay (SCID → features once, then run N checkpoints) |
| `live_features.py` | Real-time Wyckoff feature engine (36 of 58 features) |
| `range_bar_builder.py` | Tick stream → 40pt range bars |
| `adapters.py` | Data adapters (SCID replay, IB live, SCID tail) + order adapters |
| `order_manager.py` | Position management + risk controls |
| `ib_connector.py` | IB TWS/Gateway connection via ib_insync |
| `.env` | Live config: checkpoint path, IB connection, contract expiry |

### Veto Presets (in `trader.py`)

| Preset | Rules | Model |
|--------|-------|-------|
| `wavenet_single` | `large_wave_score > 0.950` | WaveNet weighted |
| `wavenet_pair` | `large_wave_score > 0.950, range_width_norm < 0.176` | WaveNet weighted |
| `cnn_split3` | `cvd_slope_fast > 0.026, wave_delta_ratio < -0.053` | CNN studio2 |
| `cnn_split4` | `wave_progress > 0.250, upper_wick_ratio > 0.590` | CNN studio2 |

---

## Models

| `--model` flag | Agent Class | Encoder | Architecture |
|---------------|-------------|---------|--------------|
| `ppo` | AgentPPO | None (MLP only) | Vanilla PPO |
| `wyckoff_ppo` | AgentPPO_Wyckoff | 1D CNN TemporalEncoder | Conv1d → GELU → adaptive pool → MLP |
| `wyckoff_wave_ppo` | AgentPPO_WaveNet | WaveNet encoder | Dilated causal conv + gated activations + skip connections |
| `sac` | AgentSAC | None | Soft Actor-Critic |
| `td3` | AgentTD3 | None | Twin Delayed DDPG |

### State Layout

```
state = [position, pnl_norm, cash_norm | window(30 × 33 features)]
         ───────── 3 dims ──────────   ──── 990 dims ────────────
         Total: 993 dims (flattened)
```

---

## Checkpoints

### Best Models (`live/checkpoints/best_models/`)

| File | Model | OOS Result |
|------|-------|------------|
| `wavenet_weighted_split3.pt` | WaveNet + weights (s3) | +$42K raw, +$57K w/ pair veto |
| `cnn_split3.pt` | CNN studio2 continuous (s3) | +$32K raw, +$46K w/ pair veto |
| `cnn_split4.pt` | CNN studio2 continuous (s4) | — |

### Other Checkpoint Directories

| Directory | Description |
|-----------|-------------|
| `continuous/` | CNN continuous, Lightning studio 1 (splits 0, 3, 5) |
| `studio2_binary/` | CNN binary, studio 2 (splits 4, 5, 7) |
| `studio2_continuous/` | CNN continuous, studio 2 (splits 2, 3, 4, 8) |
| `wavenet_ppo/` | Initial WaveNet (Kaiming defaults, no weights) |
| `wavenet_ppo_noweights/` | WaveNet Xavier+bias init (no weights, splits 0, 3) |
| `wavenet_ppo_weighted/` | WaveNet Xavier+bias + weighted rewards (split 3) |

---

## Evaluation

### `eval_all_checkpoints.py`

```bash
python -m wyckoff_rl.eval_all_checkpoints \
  --results-dir /path/to/cpcv_run --split 3 --gpu 0
```

### `normalize_npz.py`

```bash
python -m wyckoff_rl.normalize_npz --input raw.npz --output normalized.npz
```

---

## Config Reference (`config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_GROUPS` | 5 | CPCV groups |
| `K_TEST_GROUPS` | 2 | Test groups → 10 splits, 4 paths |
| `EMBARGO_BARS` | 100 | Purge leakage buffer |
| `WINDOW_SIZE` | 30 | Sliding window bars |
| `N_SELECTED_FEATURES` | 33 | Features per bar (from 58) |
| `num_envs` | 256 | Parallel GPU environments |
| `episode_len` | 1024 | Sub-episode length |
| `batch_size` | 512 | Training batch size |
| `break_step` | 2,000,000 | Total training steps |
| `loss_weight` | 2.0 | Asymmetric advantage (2x loss penalty) |
| `trade_reward_weight` | 0.5 | Trade-close bonus weight |
| `reward_scale` | 256 | Reward normalization |
| `cost_per_trade` | 0.5 | Points per side (commission + slippage) |
