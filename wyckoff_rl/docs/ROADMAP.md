# Wyckoff RL Roadmap

## Tier 1: Do Now (highest leverage)

### 4. Analyze bad trades first — **#1 PRIORITY**
- [ ] Cluster losing trades by feature regime (trending vs. ranging, volatility regime, time-of-day)
- [ ] Check if losses concentrate in specific market conditions the training data underrepresents
- [ ] Look for systematic patterns: overtrading in low-volume? chopped in consolidation zones?
- Tooling: `analyze_trade_features.py` already exists
- Directly informs items 3, 5, and 6

### 1. Paper trade on live IB data (parallel with #4)
- [ ] Start with split 4 continuous model (already wired: `ib_connector.py`, `order_manager.py`)
- [ ] Run both winning models simultaneously, log everything, don't change anything yet
- [ ] Collect 2 months of paper data: latency, slippage, real-time feature validation
- Zero-risk, validates range bar builder on streaming ticks vs. SCID replay

---

## Tier 2: Do Next (after bad trade analysis)

### 3. Stabilize the model

**a) Reward shaping improvements:**
- [ ] Add regime penalty — penalize trading during low-quality setups (quality score 0-100 from SC study logic)
- [ ] Add time-in-trade penalty to discourage overtrading/churning
- [ ] Consider asymmetric penalties — losses hurt 2x vs equivalent gains

**b) Split selection / ensembling:**
- [ ] Ensemble top 3-4 splits with majority voting or averaged position sizing
- [ ] Use PBO (probability of backtest overfitting) to quantify which splits generalize vs. lucky

**c) Observation normalization:**
- [ ] Monitor for distributional drift during paper trading (z-scored features vs. training stats)

### 6. Re-weight rare features
- [ ] Keep all 58 features but apply class-balanced weighting (don't prune, re-weight)
- [ ] Add binary indicator features (spring_detected, upthrust_detected) as always-available channels
- [ ] Consider temporal attention (Transformer head) for sparse, irregularly-spaced Wyckoff events

---

## Tier 3: Higher Risk / Longer Horizon

### 2. Train on 2026 SCID, test on 2025
- [ ] Rolling 6-month window, walk-forward validate (better than single train/test across years)
- Caveat: market microstructure changed between 2025-2026 (different vol regime, participants)

### 5. Synthetic data augmentation (GAN/GBM/Diffusion)
- **Try simpler augmentation first before synthetic data:**
  - [ ] Temporal jitter: randomly shift range bar boundaries ±1-2 ticks
  - [ ] Noise injection: small Gaussian noise to features (not prices) for regularization
  - [ ] Replay with different starting capital for different position-sizing trajectories
  - [ ] Bootstrap aggregation: train on random 80% subsets, ensemble models
- Diffusion models most promising if simple augmentation insufficient
- GAN risk: mode collapse → unrealistic paths → poison RL agent
- WaveNet-Lambert-GAN-2 results strong (MMD=0.0076, DiscAcc=0.55) — viable candidate

---

## Additional Recommendations

### A. Architecture: Position-aware attention
- [x] wyckoff_wave_ppo implemented (WaveNet dilated causal convolutions)
- [ ] Consider 1D CNN → LSTM/GRU hybrid or lightweight Transformer encoder (2-4 heads, 2 layers)

### B. Pipeline: Market regime classifier
- [ ] Train simple model to classify regime: trending-up, trending-down, ranging, volatile
- [ ] Use as additional input feature AND/OR gate (only trade when confident)

### C. Risk management layer (non-learned, outside RL agent)
- [ ] Max daily loss limit (circuit breaker)
- [ ] No trading in first/last 15 minutes
- [ ] Minimum feature quality threshold before executing
- [ ] Position timeout (force close after N bars if flat P&L)

---

## Suggested Execution Order
1. **Now**: Start paper trading + analyze bad trades (parallel)
2. **Week 2-3**: Implement reward shaping fixes based on bad trade analysis
3. **Week 3-4**: Add rare feature re-weighting + try ensemble of top splits
4. **Month 2**: Train 2026→2025 cross-validation, add regime classifier
5. **Month 2-3**: Experiment with data augmentation (noise injection, temporal jitter)
6. **Month 3+**: Evaluate synthetic data (diffusion) if simpler augmentation isn't sufficient

**Key insight**: Preprocessing and feature engine are strong (agent detects absorption, front-runs SC signals). Biggest gains come from *reducing bad trades* (risk management, reward shaping, regime filtering) rather than making the model more complex.
