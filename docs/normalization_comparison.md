# Normalization Comparison: This Project vs All Upstream ElegantRL StockTrading Envs

## Environments Inventory

There are **5 different StockTradingEnv implementations** across the ElegantRL repo, plus this project's custom env:

| # | Location | Class Name | Type | Data Format | Lineage |
|---|----------|-----------|------|-------------|---------|
| 1 | **This project** | `CryptoTradingEnv` | Single, numpy | Pre-split arrays | Paper authors' custom env |
| 2 | `elegantrl/envs/StockTradingEnv.py` | `StockTradingEnv` | Single, numpy | Pre-split arrays | **YonV1943 (canonical)** |
| 3 | `elegantrl/envs/StockTradingEnv.py` | `StockTradingVecEnv` | Vec, torch (GPU) | Pre-split arrays | **YonV1943 (canonical, newest)** |
| 4 | `helloworld/StockTradingVmapEnv.py` | `StockTradingVmapEnv` | Vec, torch (GPU) | Pre-split arrays | YonV1943 (older version of #3) |
| 5 | `examples/demo_FinRL_ElegantRL_China_A_shares.py` | `StockTradingEnv` | Single, numpy | Pre-split arrays | YonV1943 (duplicate of #2) |
| 6 | `helloworld/helloworld_REINFORCE_single_file.py` | `StockTradingEnv(gym.Env)` | Single, numpy | DataFrame | **FinRL lineage (completely different)** |

---

## State Normalization Code in `get_state()`

### 1. This Project (`environment_Alpaca.py`)
```python
def get_state(self):
    state = np.hstack((self.cash * self.norm_cash, self.stocks * self.norm_stocks))
    for i in range(self.lookback):
        tech_i = self.tech_array[self.time - i]
        normalized_tech_i = tech_i * self.norm_tech
        state = np.hstack((state, normalized_tech_i)).astype(np.float32)
    return state
```
- `norm_cash = 2⁻¹²`, `norm_stocks = 2⁻⁸`, `norm_tech = 2⁻¹⁵`, `norm_reward = 2⁻¹⁰`, `norm_action = 10000`
- State dim = 81: `[1 cash, 10 shares, 7×10 tech indicators]`
- All linear scaling, no bounding functions

### 2. Canonical Single Env (`elegantrl/envs/StockTradingEnv.py` → `StockTradingEnv`)
```python
def get_state(self) -> ARY:
    state = np.hstack((np.tanh(np.array(self.amount * 2 ** -16)),
                       self.shares * 2 ** -9,
                       self.close_ary[self.day] * 2 ** -7,
                       self.tech_ary[self.day] * 2 ** -6,))
    return state
```
- `reward_scale = 2⁻¹²`
- State = `[tanh(cash), shares, close_prices, tech]`
- Cash bounded via `tanh()`, rest linear

### 3. Canonical Vec Env (`elegantrl/envs/StockTradingEnv.py` → `StockTradingVecEnv`)
```python
def get_state(self):
    return self.vmap_get_state((self.amount * 2 ** -18).tanh(),
                               (self.shares * 2 ** -10).tanh(),
                               self.close_price[self.day] * 2 ** -7,
                               self.tech_factor[self.day] * 2 ** -6)
```
- `reward_scale = 2⁻¹²`
- Both cash AND shares bounded via `tanh()`
- Most aggressively normalized of all versions

### 4. Older VmapEnv (`helloworld/StockTradingVmapEnv.py` → `StockTradingVmapEnv`)
```python
def get_state(self):
    return self.vmap_get_state(self.amount * 2 ** 16,       # ⚠️ BUG: positive exponent!
                               self.shares * 2 ** -9,
                               self.close_price[self.day] * 2 ** -7,
                               self.tech_factor[self.day] * 2 ** -6)
```
- No `reward_scale` attribute — reward scaled inline: `(total_asset - self.total_asset) * 2 ** -6`
- **⚠️ Likely bug**: `amount * 2 ** 16` (positive!) makes cash = 1M × 65536 = 65.5 billion. Should probably be `2 ** -16` like the single env.
- Older version that was superseded by `StockTradingVecEnv`

### 5. Demo China A-shares (`examples/demo_FinRL_ElegantRL_China_A_shares.py`)
```python
def get_state(self) -> ARY:
    state = np.hstack((np.tanh(np.array(self.amount * 2 ** -16)),
                       self.shares * 2 ** -9,
                       self.close_ary[self.day] * 2 ** -7,
                       self.tech_ary[self.day] * 2 ** -6,))
    return state
```
- Identical to #2 (canonical single env). `reward_scale = 2⁻¹²`.

### 6. FinRL-style REINFORCE Env (`helloworld/helloworld_REINFORCE_single_file.py`)
```python
def _initiate_state(self):
    # For multiple stock
    state = (
        [self.initial_amount]                          # raw cash (1,000,000)
        + self.data.close.values.tolist()               # raw close prices
        + self.num_stock_shares                         # raw share counts
        + sum((self.data[tech].values.tolist()
               for tech in self.tech_indicator_list), [])  # raw tech values
    )
    return state

def _update_state(self):
    state = (
        [self.state[0]]                                # raw cash
        + self.data.close.values.tolist()               # raw close prices
        + list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])  # raw shares
        + sum((self.data[tech].values.tolist()
               for tech in self.tech_indicator_list), [])  # raw tech values
    )
    return state
```
- **NO state normalization at all** — raw values go directly into state
- `reward_scaling = 1e-4` (≈ `2⁻¹³.³`)
- `hmax = 100` (action scaling)
- `initial_amount = 1,000,000`
- Completely different architecture: DataFrame-based, uses `_initiate_state()`/`_update_state()` instead of `get_state()`
- From FinRL/FinRL-Meta lineage, not YonV1943's array-based design
- State = `[cash(raw), close_prices(raw), shares(raw), tech(raw)]` — all unnormalized

---

## Scaling Factor Comparison (all 6 environments)

| Parameter | This Project | Single Env (#2) | Vec Env (#3) | VmapEnv (#4) | REINFORCE (#6) |
|-----------|-------------|-----------------|-------------|-------------|----------------|
| **Cash** | `× 2⁻¹²` (linear) | `tanh(× 2⁻¹⁶)` | `tanh(× 2⁻¹⁸)` | `× 2¹⁶` ⚠️ | **raw** (no norm) |
| **Shares** | `× 2⁻⁸` (linear) | `× 2⁻⁹` (linear) | `tanh(× 2⁻¹⁰)` | `× 2⁻⁹` | **raw** (no norm) |
| **Close Prices** | N/A (in tech) | `× 2⁻⁷` | `× 2⁻⁷` | `× 2⁻⁷` | **raw** (no norm) |
| **Tech** | `× 2⁻¹⁵` | `× 2⁻⁶` | `× 2⁻⁶` | `× 2⁻⁶` | **raw** (no norm) |
| **Reward** | `(Δbot−Δeqw) × 2⁻¹⁰` | `Δasset × 2⁻¹²` | `Δasset × 2⁻¹²` | `Δasset × 2⁻⁶` | `Δasset × 1e-4` |
| **Action** | `10000` (dynamic) | `100` (fixed) | `100` (fixed) | `100` (fixed) | `100` (fixed) |
| **Fees** | 0.3% | 0.1% | 0.1% | 0.1% buy/sell | 0.1% buy/sell |

---

## Effective State Magnitudes (initial_amount=1M, typical crypto/stock prices)

| Component | Example | This Project | Single (#2) | Vec (#3) | VmapEnv (#4) | REINFORCE (#6) |
|-----------|---------|-------------|------------|---------|-------------|----------------|
| **Cash** | 1M | `≈ 244` | `≈ 1.0` | `≈ 1.0` | `≈ 6.55e10` ⚠️ | `1,000,000` |
| **Shares** | 50 | `≈ 0.20` | `≈ 0.098` | `≈ 0.049` | `≈ 0.098` | `50` |
| **Close** | 30k | N/A | `≈ 234` | `≈ 234` | `≈ 234` | `30,000` |
| **Tech (RSI)** | 50 | `≈ 0.0015` | `≈ 0.78` | `≈ 0.78` | `≈ 0.78` | `50` |
| **Tech (OBV)** | 1e8 | `≈ 3052` | `≈ 1.56e6` | `≈ 1.56e6` | `≈ 1.56e6` | `1e8` |

### Key Observations
1. **Cash is unbounded at ~244** in this project vs bounded to ~1.0 upstream (via `tanh`)
2. **Tech indicators are ~512× smaller** in this project (`2⁻¹⁵` vs `2⁻⁶`), potentially making them near-invisible to the network
3. **State magnitude range is extreme** in this project — cash≈244 vs tech(RSI)≈0.0015, a ratio of ~163,000:1
4. **VmapEnv has a likely bug** — `amount * 2**16` makes cash astronomically large instead of small
5. **REINFORCE env uses NO normalization** — raw values (1M cash, 30k prices) go directly into state. Only reward is scaled by `1e-4`. This is a completely different design philosophy (relies on the network learning to handle scale).
6. The canonical vec env (#3) is the most aggressively bounded, using `tanh()` on both cash and shares

---

## Reward Formulation

| Aspect | This Project | YonV1943 Envs (#2/#3/#4) | REINFORCE Env (#6) |
|--------|-------------|--------------------------|-------------------|
| **Formula** | `(Δ_portfolio − Δ_equal_weight) × norm_reward` | `(total_asset − prev_total_asset) × reward_scale` | `(end_total_asset − begin_total_asset) × reward_scaling` |
| **Type** | Differential (vs benchmark) | Absolute change | Absolute change |
| **Terminal bonus** | `gamma_return` (cumulative discounted) | `mean(rewards) / (1 − γ)` | None |
| **Scale** | `2⁻¹⁰ = 0.000977` | Single/Vec: `2⁻¹²`, Vmap: `2⁻⁶` | `1e-4 ≈ 2⁻¹³·³` |

---

## Agent-Level Normalization

| Feature | This Project (ElegantRL v0.3.6) | ElegantRL Current Main Branch |
|---------|--------------------------------|-------------------------------|
| **ActorPPO `state_norm()`** | ❌ Not present | ✅ `(state − state_avg) / (state_std + 1e-4)` with learnable params |
| **`state_avg` / `state_std`** | ❌ Not present | ✅ `nn.Parameter` (requires_grad=False), default 0/1 (identity) |
| **`reward_scale` in AgentBase** | Exists but = 1.0 (no-op, never set by DRLAgent wrapper) | Same — default `2⁰ = 1.0` |
| **`replay_buffer` reward scaling** | `reward * self.reward_scale` in `update_buffer()` — no-op at 1.0 | Same mechanism |

---

## State Structure Comparison

### This Project
```
State dim = 81
[cash(1), stocks(10), tech_t0(70)] 
                       ↑ 7 indicators × 10 tickers, lookback=1
```
7 tech indicators: `close, volume, rsi, dx, ultosc, obv, ht_dcphase`  
Close prices are embedded within tech indicators, not separate.

### Upstream ElegantRL
```
State dim = shares_num + close_prices + tech_features + 1(amount)
[cash(1), shares(N), close_prices(N), tech(M)]
```
Close prices are a **separate** state component from tech indicators.

---

## Action Normalization

| Aspect | This Project | ElegantRL (both) |
|--------|-------------|-----------------|
| **Mechanism** | Dynamic per-ticker based on `floor(log₁₀(price))` | Fixed `max_stock = 100` |
| **Scale** | `norm_action = 10000` multiplied by per-ticker magnitude | `action_int = (action * max_stock).astype(int)` |
| **Dead zone** | ❌ None | ✅ `action[abs(action) < 0.1] = 0` |
| **Fees** | 0.3% (buy & sell) | 0.1% (buy & sell) |

---

## Why REINFORCE Env (#6) Works Without State Normalization

The REINFORCE env passes completely raw state values (cash=1M, prices=30k, etc.) yet still trains. It has **two hidden normalization mechanisms** that compensate:

### 1. TransformerActor with Built-in LayerNorm

The `ActorREINFORCE` wraps a `TransformerActor`, not a plain MLP:

```python
class TransformerActor(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=64, nhead=4, num_layers=2):
        self.state_embedding = nn.Linear(state_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, action_dim)
```

`nn.TransformerEncoderLayer` includes **LayerNorm by default** — it normalizes activations internally after the attention and feedforward sublayers. Even though raw values like cash=1M go in, the `state_embedding` linear layer projects them into a 64-dim space, and then LayerNorm standardizes the activations to zero-mean/unit-variance before each sublayer. The network self-normalizes.

This is fundamentally different from the **plain MLPs** used in the YonV1943 envs and this project (`Linear → ReLU → Linear → ReLU → Linear`), which have **no normalization layers** — making env-level scaling essential.

### 2. Return Standardization in `update_net()`

```python
def update_net(self, buffer):
    reward_sums = self.compute_reward_sums(rewards, self.gamma)
    reward_sums = (reward_sums - reward_sums.mean()) / (reward_sums.std() + 1e-5)  # ← standardized!
```

The discounted returns are standardized to zero-mean/unit-variance before computing the REINFORCE policy gradient. This is the classic variance reduction baseline trick. Combined with `reward_scaling = 1e-4` in the env, gradient magnitudes stay reasonable.

### Comparison: Why Plain MLP Envs Need Manual Normalization

| | REINFORCE Env (#6) | YonV1943 Envs (#2/#3) / This Project |
|---|---|---|
| **Network** | TransformerActor (built-in LayerNorm) | Plain MLP (no normalization layers) |
| **Return processing** | Standardized: `(R − μ) / σ` | Raw discounted returns |
| **State normalization** | Not needed (LayerNorm handles scale) | **Must be manual** in env |
| **Config** | Demo: small net `[64,32]`, `break_step=1e5` | Production: large nets, long training |

**Bottom line**: The REINFORCE env doesn't "get away with" skipping normalization — it delegates normalization to the network architecture (LayerNorm inside Transformer) instead of doing it manually in the env. If that `TransformerActor` were replaced with a plain MLP, it would likely struggle badly with raw 1M cash values alongside 0.01 tech indicator values.

---

## Summary of Potential Issues

1. **Tech indicator magnitudes near zero** (`2⁻¹⁵` scaling) — neural network may effectively ignore technical indicators
2. **Cash magnitude too large** (~244 unbounded vs ~1.0 bounded upstream) — may dominate gradient signals
3. **No `tanh()` bounding** on cash or shares — state values can grow unbounded during training
4. **Large state magnitude spread** — 163,000:1 ratio between largest and smallest state components
5. **Norm values are hardcoded single-value** (not searched by Optuna) — `trial.suggest_categorical("norm_cash", [2 ** -12])` provides only one option

---

## FinRL's Stable-Baselines3 Normalization Approach

FinRL does **NOT** use SB3's `VecNormalize` wrapper. Their `DRLAgent` class (`finrl/agents/stablebaselines3/models.py`) wraps environments only with `DummyVecEnv`:
```python
from stable_baselines3.common.vec_env import DummyVecEnv  # only import, never VecNormalize

class DRLAgent:
    def __init__(self, env):
        self.env = env  # passed directly, no VecNormalize wrapping
```

### Two Completely Different Envs per DRL Library

FinRL maintains **separate environment implementations** for SB3 vs ElegantRL:

| Aspect | FinRL + SB3 (`env_stocktrading.py`) | FinRL + ElegantRL (`env_stocktrading_np.py`) |
|--------|-------------------------------------|----------------------------------------------|
| **Observation** | **RAW, unnormalized** — `[cash, prices, shares, tech]` in original scale | Manually scaled: `cash × 2⁻¹²`, `price × 2⁻⁶`, `stocks × 2⁻⁶`, `tech × 2⁻⁷` |
| **Cash in state** | `1,000,000` (literal) | `1e6 × 2⁻¹² ≈ 244` |
| **Price in state** | `~150` (literal) | `150 × 2⁻⁶ ≈ 2.34` |
| **Reward formula** | `(end_asset − begin_asset) × reward_scaling` | `(total − prev) × 2⁻¹¹` |
| **reward_scaling** | `1e-4` (typical, user-supplied) | `2⁻¹¹ ≈ 4.88e-4` |
| **obs_space** | `Box(low=-np.inf, high=np.inf)` | `Box(low=-3000, high=3000)` |
| **VecNormalize** | ❌ Never used | ❌ N/A (ElegantRL, not SB3) |
| **State construction** | `_initiate_state()` / `_update_state()` (DataFrame) | `get_state()` (pre-split arrays) |

### FinRL SB3 State Code (`env_stocktrading.py`)
```python
def _initiate_state(self):
    state = (
        [self.initial_amount]                    # raw cash: 1,000,000
        + self.data.close.values.tolist()         # raw prices: ~100-300
        + self.num_stock_shares                   # raw shares: 0-100s
        + sum((self.data[tech].values.tolist()    # raw tech: varies wildly
               for tech in self.tech_indicator_list), [])
    )
    return state
```
Note: this is the same design as REINFORCE env (#6) in this repo — identical FinRL/FinRL-Meta lineage.

### FinRL ElegantRL State Code (`env_stocktrading_np.py`)
```python
def get_state(self, price):
    amount = np.array(self.amount * (2**-12), dtype=np.float32)
    scale = np.array(2**-6, dtype=np.float32)
    return np.hstack((amount, self.turbulence_ary[self.day],
                      self.turbulence_bool[self.day],
                      price * scale, self.stocks * scale,
                      self.stocks_cool_down, self.tech_ary[self.day]))
```

### FinRL Crypto Env (`env_multiple_crypto.py`)
```python
def get_state(self):
    state = np.hstack((self.cash * 2**-18, self.stocks * 2**-3))
    for i in range(self.lookback):
        normalized_tech_i = self.tech_array[self.time - i] * 2**-15
        state = np.hstack((state, normalized_tech_i)).astype(np.float32)
    return state
```
- Much more aggressive cash scaling (`2⁻¹⁸` vs `2⁻¹²`), less stock scaling (`2⁻³` vs `2⁻⁶`/`2⁻⁹`)
- Tech scaling `2⁻¹⁵` matches "This Project" — both are crypto-focused

### Key Insight

**FinRL relies entirely on environment-internal normalization**, never on SB3's runtime `VecNormalize`.
The SB3 env passes completely raw observations to the agent and relies on `reward_scaling` alone.
The ElegantRL env hardcodes power-of-2 scaling in `get_state()`, identical to our upstream ElegantRL.

This confirms the correct approach: **hardcode normalization in the environment, do not use VecNormalize**.
Our experimental results corroborate this — VecNormalize caused degradation for all agents tested
(TD3: catastrophic divergence via replay buffer staleness, A2C: 2.8× worse objC, 12× more volatile objA).

### Why SB3 Envs Use Raw Unnormalized Observations

FinRL's SB3 environments (`env_stocktrading.py`, `cashpenalty`, `stoploss`) pass raw observations — cash at 1,000,000, prices at ~150, tech indicators spanning orders of magnitude — directly into `MlpPolicy`. This works *adequately* (not optimally) for several reasons:

**1. SB3's adaptive optimizers absorb scale differences**

SB3's PPO/A2C/TD3/SAC all use Adam optimizer by default, which maintains per-parameter adaptive learning rates. Adam's second-moment estimate ($v_t$) effectively normalizes gradients per-weight, so weights connected to large-magnitude inputs (cash=1M) get automatically smaller learning rates than weights for small-magnitude inputs (RSI=50). This doesn't fully solve the problem — the initial random weights still produce wildly different activation magnitudes — but it prevents total divergence.

**2. FinRL's SB3 envs are demos, not production**

The SB3 envs in FinRL are intentionally simple reference implementations. Key tells:
- `observation_space = Box(-inf, inf)` — no bounded obs space, no care for stability
- Default `hmax=100` (stock) / `hmax=10` (cashpenalty) — small action scales
- `reward_scaling = 1e-4` is the **only** normalization knob, and it's user-supplied
- No turbulence indicator, no cooldown mechanism, no sophisticated risk control

The ElegantRL envs in the same repo (`env_stocktrading_np.py`) are the production-grade versions — they have `obs_space = Box(-3000, 3000)`, manual `2^-N` scaling, turbulence gating, cooldown arrays. The SB3 envs exist so users can quickly get SB3 agents running without understanding ElegantRL's array-based env protocol.

**3. SB3 users typically *should* use VecNormalize — FinRL just doesn't**

The SB3 documentation explicitly recommends `VecNormalize` for continuous control tasks. FinRL's choice to skip it is a simplification, not a best practice. In the general SB3 ecosystem:
- `VecNormalize(norm_obs=True, norm_reward=True)` is standard for MuJoCo/robotics
- It maintains running mean/std and normalizes observations online
- It works well for **on-policy** algorithms (PPO, A2C) where all data is fresh

The reason VecNormalize fails for **off-policy** algorithms like TD3/SAC (as our experiments showed) is replay buffer staleness: observations stored under old normalization statistics become invalid as the running statistics update, creating distribution shift. This is a known issue in the SB3 community.

**4. Raw observations + `MlpPolicy` = it trains, but suboptimally**

A plain `MlpPolicy` MLP (`Linear(obs_dim, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, act_dim)`) *can* learn with raw observations, but:
- The first layer's weights will span ~5 orders of magnitude to compensate for input scale differences
- Weight initialization (Kaiming/Xavier) assumes roughly unit-variance inputs — violated badly
- Gradient flow through the first layer is unbalanced — cash-connected weights dominate
- Training requires more episodes to converge compared to pre-normalized inputs
- The `reward_scaling=1e-4` prevents reward magnitudes from overwhelming the value function

In practice, FinRL's SB3 demos produce mediocre but non-catastrophic results on stock trading (where price ranges are ~$50-$500 and cash is ~$1M, a ~2000:1 ratio). For crypto (where BTC=$30k and OBV=$10⁸, a ~3000:1 ratio within tech indicators alone), raw observations would likely cause much worse optimization landscape issues, which is why the crypto envs (`env_multiple_crypto.py`, `env_btc_ccxt.py`) always use manual scaling even though they target ElegantRL.

**Bottom line**: SB3 envs use raw values because they're simple demos relying on Adam's per-parameter adaptivity as an implicit weak normalizer. The ElegantRL envs invest in manual `2^-N` scaling because YonV1943's plain MLP actors (no LayerNorm, no state_norm in v0.3.6) genuinely need it. Neither approach uses VecNormalize — FinRL's pattern is consistently "normalize in the env or don't normalize at all."

---

## Complete FinRL Environment Normalization Reference

All environments from [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL), organized by DRL library target and asset class.

### Master Scaling Factor Table

#### Stock Trading — ElegantRL-targeted (manual `2^-N` scaling in `get_state()`)

| Parameter | `env_stocktrading_np.py` | `env_nas100_wrds.py` | `env_stock_papertrading.py` | `paper_trading/alpaca.py` |
|-----------|--------------------------|----------------------|-----------------------------|---------------------------|
| **Class** | `StockTradingEnv` | `StockEnvNAS100` | `AlpacaPaperTrading` | `PaperTradingAlpaca` |
| **Cash** | `× 2⁻¹²` | `max(amt,1e4) × 2⁻¹²` | `× 2⁻¹²` | `× 2⁻¹²` |
| **Close Prices** | `× 2⁻⁶` | `× 2⁻⁶` | `× 2⁻⁶` | `× 2⁻⁶` |
| **Shares** | `× 2⁻⁶` | `× 2⁻⁶` | `× 2⁻⁶` | `× 2⁻⁶` |
| **Tech (init)** | `× 2⁻⁷` | `× 2⁻⁷` | `× 2⁻⁷` | `× 2⁻⁷` |
| **Turbulence** | `sigmoid_sign × 2⁻⁵` | `sigmoid_sign × 2⁻⁵` | `sigmoid_sign × 2⁻⁵` | `sigmoid_sign × 2⁻⁵` |
| **Reward** | `Δasset × 2⁻¹¹` | `Δasset × 2⁻¹¹` | N/A (live trading) | N/A (live trading) |
| **Action** | `× max_stock (100)` | `× max_stock (100)` | `× max_stock (100)` | `× max_stock (100)` |
| **obs_space** | `Box(-3000, 3000)` | `Box(-3000, 3000)` | N/A | N/A |
| **NaN/Inf guard** | ❌ | ❌ | ❌ | ✅ `state[isnan]=0, state[isinf]=0` |
| **Data format** | Pre-split arrays | Pre-split arrays | Live Alpaca API | Live Alpaca API |

#### Cryptocurrency — ElegantRL-targeted (manual `2^-N` scaling)

| Parameter | `env_multiple_crypto.py` | `env_btc_ccxt.py` |
|-----------|--------------------------|-------------------|
| **Class** | `CryptoEnv` | `BitcoinEnv` |
| **Cash** | `× 2⁻¹⁸` | `× 2⁻¹⁸` |
| **Close Prices** | N/A (embedded in tech) | `× 2⁻¹⁵` |
| **Shares** | `× 2⁻³` | `× 2⁻⁴` |
| **Tech** | `× 2⁻¹⁵` (uniform) | **per-indicator** (see below) |
| **Turbulence** | N/A | N/A |
| **Reward** | `Δasset × 2⁻¹⁶` | `Δasset × 2⁻¹⁶` |
| **Action** | Dynamic normalizer¹ | Raw (single continuous) |
| **Data format** | Pre-split arrays | Pre-split arrays (CCXT) |

¹ `CryptoEnv._generate_action_normalizer()`: `norm = 1 / 10^(len(str(price₀)) − 7)` per-ticker, adapts to price magnitude at t=0.

**`BitcoinEnv` per-indicator tech scaling** (7 indicators from CCXT):
| Index | Indicator | Scale |
|-------|-----------|-------|
| 0 | MACD | `× 2⁻¹` |
| 1 | boll_ub | `× 2⁻¹⁵` |
| 2 | boll_lb | `× 2⁻¹⁵` |
| 3 | rsi_30 | `× 2⁻⁶` |
| 4 | dx_30 | `× 2⁻⁶` |
| 5 | close_30_sma | `× 2⁻¹⁵` |
| 6 | close_60_sma | `× 2⁻¹⁵` |

#### Stock Trading — SB3-targeted (raw, unnormalized)

| Parameter | `env_stocktrading.py` | `env_stocktrading_cashpenalty.py` | `env_stocktrading_stoploss.py` |
|-----------|----------------------|----------------------------------|-------------------------------|
| **Class** | `StockTradingEnv` | `StockTradingEnvCashpenalty` | `StockTradingEnvStopLoss` |
| **Cash** | **raw** (1,000,000) | **raw** (1,000,000) | **raw** (1,000,000) |
| **Close Prices** | **raw** (~100-300) | **raw** (~100-300) | **raw** (~100-300) |
| **Shares** | **raw** (0-100s) | **raw** (0-100s) | **raw** (0-100s) |
| **Tech** | **raw** (varies wildly) | **raw** (varies wildly) | **raw** (varies wildly) |
| **Turbulence** | **raw** (optional) | **raw** (optional) | **raw** (optional) |
| **Reward** | `Δasset × 1e-4` | `((assets−penalty)/init−1)/step` | `((assets−penalties+bonus)/init−1)/step` |
| **Action** | `× hmax (100)` | `× hmax (10)` then `÷ close` | `× hmax (10)` then `÷ close` |
| **obs_space** | `Box(-inf, inf)` | `Box(-inf, inf)` | `Box(-inf, inf)` |
| **Fees** | 0.1% buy/sell | 0.3% buy/sell | 0.3% buy/sell |
| **State method** | `_initiate_state()` | `[cash, shares, daily_info]` | `[cash, shares, daily_info]` |

#### Portfolio — SB3-targeted (different paradigm)

| Parameter | `env_portfolio.py` | `env_portfolio_optimization.py` |
|-----------|-------------------|--------------------------------|
| **Class** | `StockPortfolioEnv` | `PortfolioOptimizationEnv` |
| **State shape** | 2D: `(N+tech_count, N)` | 3D: `(features, tics, time_window)` |
| **State content** | Covariance matrix + tech indicators (raw) | Feature time-series (configurable norm) |
| **Normalization** | **None** (raw covariance + raw tech) | Configurable: `by_previous_time` (default), `by_first_time_window_value`, `by_COLUMN`, or custom fn |
| **Reward** | `new_portfolio_value` (raw!) | `ln(return) × reward_scaling` (default scaling=1) |
| **reward_scaling** | ~~`self.reward_scaling`~~ (commented out!) | Configurable (default 1) |
| **Action** | `softmax(actions)` → weights ∈ [0,1], Σ=1 | `softmax(actions)` → weights ∈ [0,1], Σ=1 |
| **Action space** | `Box(low=0, high=1, shape=(stock_dim,))` | `Box(low=0, high=1, shape=(stock_dim+1,))` — includes cash weight |
| **Commission** | None (assumed zero) | `trf` or `wvm` model (configurable %) |

### Cross-Environment Scaling Comparison (unified view)

This table compares every environment's effective cash normalization and reward scaling side-by-side:

| Environment | Library | Cash Scale | Price Scale | Share Scale | Tech Scale | Reward Scale |
|-------------|---------|------------|-------------|-------------|------------|--------------|
| **This Project** | ElegantRL | `2⁻¹²` | N/A (in tech) | `2⁻⁸` | `2⁻¹⁵` | `2⁻¹⁰` |
| Single Env (#2) | ElegantRL | `tanh(2⁻¹⁶)` | `2⁻⁷` | `2⁻⁹` | `2⁻⁶` | `2⁻¹²` |
| Vec Env (#3) | ElegantRL | `tanh(2⁻¹⁸)` | `2⁻⁷` | `tanh(2⁻¹⁰)` | `2⁻⁶` | `2⁻¹²` |
| FinRL `stocktrading_np` | ElegantRL | `2⁻¹²` | `2⁻⁶` | `2⁻⁶` | `2⁻⁷` | `2⁻¹¹` |
| FinRL `nas100_wrds` | ElegantRL | `clamp→2⁻¹²` | `2⁻⁶` | `2⁻⁶` | `2⁻⁷` | `2⁻¹¹` |
| FinRL `papertrading` (×2) | ElegantRL | `2⁻¹²` | `2⁻⁶` | `2⁻⁶` | `2⁻⁷` | N/A (live) |
| FinRL `multiple_crypto` | ElegantRL | `2⁻¹⁸` | N/A (in tech) | `2⁻³` | `2⁻¹⁵` | `2⁻¹⁶` |
| FinRL `btc_ccxt` | ElegantRL | `2⁻¹⁸` | `2⁻¹⁵` | `2⁻⁴` | per-indicator | `2⁻¹⁶` |
| FinRL `stocktrading` (SB3) | SB3 | **raw** | **raw** | **raw** | **raw** | `1e-4` |
| FinRL `cashpenalty` | SB3 | **raw** | **raw** | **raw** | **raw** | custom² |
| FinRL `stoploss` | SB3 | **raw** | **raw** | **raw** | **raw** | custom³ |
| FinRL `portfolio` | SB3 | N/A | **raw** (cov) | N/A | **raw** | raw value! |
| FinRL `portfolio_opt` | SB3 | N/A | df-norm | N/A | df-norm | `ln(r)×s` |
| REINFORCE (#6) | SB3-style | **raw** | **raw** | **raw** | **raw** | `1e-4` |

² `cashpenalty` reward: `((total_assets − max(0, assets×0.1 − cash)) / initial − 1) / steps`  
³ `stoploss` reward: `((total_assets − cash_penalty − stop_loss_penalty − low_profit_penalty + profit_bonus) / initial − 1) / steps`

### Key Patterns

1. **ElegantRL envs always use manual `2^-N` scaling** — stock envs use `2⁻⁶` to `2⁻¹²`, crypto envs use more aggressive `2⁻¹⁵` to `2⁻¹⁸`
2. **SB3 envs pass completely raw observations** — rely on `reward_scaling` alone (or custom reward formulas)
3. **Crypto envs scale cash ~64× more aggressively** than stock envs (`2⁻¹⁸` vs `2⁻¹²`) because initial_cash=1M but crypto prices are ~30,000× higher per unit
4. **`BitcoinEnv` is the only env with per-indicator tech scaling** — all others use uniform scaling
5. **Portfolio envs are a different paradigm** — state is covariance matrix or feature time-series, actions are softmax-normalized weights, no share-level trading
6. **`env_portfolio.py` has reward_scaling commented out** — `self.reward = new_portfolio_value` passes raw portfolio value (~1M) as reward, likely a bug
7. **No FinRL environment ever uses `VecNormalize`** — all use `DummyVecEnv` wrapping only

---

## VecNormalize Experimental Results Summary

| Agent | VecNormalize Config | objC (mean last 50) | objA (std last 50) | Verdict |
|-------|-------------------|--------------------|--------------------|---------|
| TD3 | None | 0.97 (stable) | stable | ✅ Best |
| TD3 | norm_obs=True | 2,108 (exploding) | diverged | ❌ Catastrophic |
| A2C | None | 1.61 | 0.80 | ✅ Best |
| A2C | norm_obs=True | 4.50 (2.8× worse) | 9.65 (12× worse) | ❌ Degraded |
| A2C | norm_obs+norm_reward | diverged | 1,830 | ❌ Catastrophic |

Root cause for TD3: replay buffer stores observations normalized with **old** running statistics,
creating distribution shift as VecNormalize statistics update during training.

Root cause for A2C: triple normalization chain — `get_state()` scales → `VecNormalize` rescales → `state_norm()` rescales again.
