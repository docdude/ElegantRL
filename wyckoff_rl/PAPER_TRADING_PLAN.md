# Wyckoff RL — Paper Trading & Live Deployment Plan

## Overview

Once CPCV validation confirms the strategy generalizes, deploy the trained
actor for live NQ futures paper trading. Two viable architectures exist;
the pure-Python path is preferred for simplicity.

---

## Architecture: Pure Python (Recommended)

```
IB TWS API (ib_insync)
  → reqTickByTickData('AllLast') for NQ
  → RangeBarBuilder: accumulate ticks → emit 40pt range bars (OHLCV + delta)
  → WyckoffFeatureEngine: same code as wyckoff_analyzer.py (58 features)
  → Sliding window: last 30 bars × 36 selected features
  → Prepend agent state: [position, unrealized_pnl_norm, cash_norm]
  → actor(state_tensor) → action ∈ [-1, 1]
  → Map to target position: >0.33 → long, <-0.33 → short, else flat
  → IB order execution (MKT order for NQ futures)
  → Log equity, positions, trades
```

### Key advantage
The training NPZ was built from SCID data → `wyckoff_analyzer.py`. Using the
same Python feature code for live bars gives **exact feature parity** — zero
train/live skew.

---

## Architecture: SierraChart Bridge (Alternative)

```
SierraChart (C++ ACSIL)                Python (RL Agent)
├─ Live NQ tick stream                 ├─ Load actor.pth
├─ Range 40 bars (native)              ├─ Read shared state (mmap/socket/file)
├─ WyckoffWeis_NQ.cpp                  ├─ Forward through network
│  (all Wyckoff features computed)     ├─ Write target position
├─ DeepMEffort_NQ.cpp                  └─ Python polls on bar completion
│  (effort/absorption zones)
├─ Export feature vector → shared mem
├─ Read target position ← shared mem
└─ Execute via IB (SC native broker)
```

### When to use this path
- Want visual overlay (waves, zones, arrows) on SC chart while agent trades
- SC already connected to IB with live data feed
- Feature computation stays in C++ (lower latency, though irrelevant for range bars)

### Communication options
| Method              | Latency | Notes                                    |
|---------------------|---------|------------------------------------------|
| Shared memory (mmap)| ~μs     | SC DLL writes, Python reads via `mmap`   |
| Local socket        | ~ms     | SC DLL opens socket, Python connects     |
| File-based          | ~100ms  | Simplest; fine for range bars            |

---

## Components to Build

### 1. `RangeBarBuilder` (Python)
- Consumes IB tick stream (price, volume, bid_size, ask_size)
- Accumulates into current bar: tracks open, high, low, close, volume,
  ask_volume (uptick), bid_volume (downtick), num_trades
- When `high - low >= 40.0` (range size): emit completed bar, start new bar
- Handle session boundaries (RTH: 09:30-16:00 ET for NQ)
- Must match the SCID → range bar conversion logic exactly

### 2. `WyckoffFeatureEngine` (Python)
- Wraps the same functions from `wyckoff_analyzer.py`:
  - Delta & CVD
  - Weis Wave segmentation (zigzag reversal)
  - Effort vs Result ratio
  - Absorption detection
  - Volume/time strength tiers (normalized emphasized)
  - Phase classification
- Operates incrementally: on each new bar, update rolling state
- Output: 58-feature vector (or 36 selected features matching training)

### 3. `WyckoffPaperTrader` (main loop)
- Analogous to FinRL's `AlpacaPaperTrading` class
- Init: load actor.pth, connect to IB, subscribe to NQ ticks
- Main loop (event-driven, triggered by bar completion):
  1. `RangeBarBuilder` emits completed bar
  2. `WyckoffFeatureEngine` computes features
  3. Build state: `[pos, unrealized_pnl_norm, cash_norm, window_flat]`
  4. `actor(state)` → action → target position {-1, 0, +1}
  5. If position change needed: submit IB order
  6. Log trade, equity snapshot
- Risk controls: max position size, daily loss limit, session-only trading

### 4. `IBBrokerInterface`
- Wraps `ib_insync` for NQ futures:
  - `connect()`: connect to TWS/IB Gateway
  - `subscribe_ticks(contract)`: real-time tick stream
  - `get_position()`: current NQ position
  - `place_order(target_pos)`: compute delta, submit MKT order
  - `get_account_value()`: current equity
- Contract: `Future('NQ', 'CME', '202506')` (or continuous front month)

---

## Broker: Interactive Brokers

- **Paper account**: Free, no minimum deposit, 15-min delayed data
- **Live account**: Funded, real-time CME data ~$10-15/month add-on
- **API**: TWS API via `ib_insync` (synchronous Python wrapper)
- **Futures support**: Native NQ/ES/etc.
- Paper account sufficient for full pipeline validation

---

## Feature Parity Checklist

Before going live, verify features match between training and live:

- [ ] Range bar construction: same 40pt threshold, same OHLCV aggregation
- [ ] Delta calculation: askVol - bidVol (uptick/downtick rule)
- [ ] CVD: cumulative delta, reset at session boundaries
- [ ] Weis Wave: same reversal threshold and mode (% vs points)
- [ ] Effort/Result: same lookback period
- [ ] Absorption: same volume and ER thresholds
- [ ] Volume/time normalization: same period, same tier boundaries
- [ ] Feature selection: same 36 of 58 features, same column ordering
- [ ] Sliding window: same size (30), same zero-padding at start
- [ ] Agent state normalization: same tanh scaling of PnL and cash

**Test**: Run live feature engine on historical SCID data, compare output
to training NPZ feature-by-feature. Max acceptable deviation: 1e-5.

---

## Overfitting Mitigations (from CPCV analysis)

Before deploying, address overfitting observed in training:

1. **Adaptive break_step**: Scale training steps with fold size
   - Empirical sweet spot: 300K-800K steps for ~4000-bar folds
   - Rule of thumb: `break_step = min(800_000, n_train_bars * 200)`
2. **Checkpoint selection**: Use best OOS checkpoint, not final model
   - Split 0: best at 1.9M steps (large fold), Split 1: best at 311K (small fold)
3. **Ensemble**: Average actions from top-K checkpoints across CPCV paths
4. **Stronger entropy**: Increase `lambda_entropy` from -0.005 to -0.01
5. **Weight decay**: Add `weight_decay=1e-4` to Adam optimizer
6. **LR schedule**: Cosine decay from 1e-4 to 1e-5

---

## Timeline

1. **Now**: Complete CPCV evaluation (10 MLP splits + 10 CNN splits)
2. **After CPCV**: Aggregate results, confirm strategy generalizes
3. **If positive**: Build RangeBarBuilder + WyckoffFeatureEngine
4. **Validate**: Compare live features vs NPZ on historical data
5. **Paper trade**: IB paper account, monitor for 2-4 weeks
6. **If stable**: Fund IB account, add real-time data, go live with 1 contract
