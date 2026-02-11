# DRLEnsembleAgent Implementation Plan for ElegantRL

**Goal:** Recreate FinRL's `DRLEnsembleAgent` using ElegantRL's GPU-parallel VecEnv infrastructure, implementing the methodology from "Revisiting Ensemble Methods for Stock Trading" (arXiv 2501.10709v1).

**References:**
- Paper: [Revisiting Ensemble Methods for Stock Trading](https://arxiv.org/abs/2501.10709) (ICAIF 2024)
- FinRL Contest Paper: [arXiv 2504.02281v4](https://arxiv.org/abs/2504.02281)
- FinRL Source: `finrl.agents.stablebaselines3.models.DRLEnsembleAgent`
- ElegantRL Source: `elegantrl.train.run.train_agent_single_process`

---

## 📊 Architecture Overview

### FinRL DRLEnsembleAgent (Reference)
```
run_ensemble_strategy()
├── Rolling Window Loop (rebalance_window + validation_window)
│   ├── _train_window() for each model (A2C, PPO, DDPG, SAC, TD3)
│   │   ├── get_model() → SB3 model
│   │   ├── train_model() → model.learn(total_timesteps)
│   │   └── get_validation_sharpe() → CSV-based Sharpe calculation
│   ├── Model Selection: max(sharpes) → "winner-takes-all"
│   └── DRL_prediction() → Trade using best model
└── Output: df_summary with Sharpe ratios per window
```

### Target ElegantRL DRLEnsembleAgent
```
run_ensemble_strategy()
├── Rolling Window Loop (rebalance_window + validation_window)
│   ├── train_agent() for each model (PPO, SAC, DDPG)
│   │   ├── create_env_from_dataframe() → StockTradingVecEnv (GPU)
│   │   ├── explore_env() + update_net() loop
│   │   └── get_validation_sharpe() → VecEnv-based Sharpe
│   ├── Ensemble Weights: softmax(sharpes) → weighted averaging (paper method)
│   └── DRL_prediction() → Weighted ensemble action
└── Output: df_summary + full account values
```

---

## 🚀 Staged Implementation Plan

### Stage 1: Basic Ensemble (MVP)
**Goal:** Get rolling-window training + validation + ensemble trading working with ElegantRL agents.

#### 1.1 Environment Adapter
Create `DOW30VecEnv` that accepts DataFrame and produces ElegantRL-compatible VecEnv:

```python
def create_env_from_dataframe(
    df: pd.DataFrame,
    config: TradingConfig,
    beg_idx: int = 0,
    end_idx: Optional[int] = None,
    save_path: str = "datasets/temp_data"
) -> StockTradingVecEnv:
    """
    Convert FinRL DataFrame to ElegantRL VecEnv.
    
    1. Pivot DataFrame to get close_ary [time, stocks]
    2. Extract tech indicators to tech_ary [time, stocks * num_indicators]
    3. Override load_data_from_disk() to return our arrays
    4. Create StockTradingVecEnv with correct env_args
    """
```

**Key Mapping:**
| FinRL | ElegantRL |
|-------|-----------|
| `df.pivot('close')` | `close_ary` [T, N] |
| `df.pivot(tech_indicators)` | `tech_ary` [T, N×I] |
| `DummyVecEnv([lambda: env])` | `StockTradingVecEnv(num_envs=K)` |

#### 1.2 Agent Factory
```python
def create_agent(agent_type: str, env: StockTradingVecEnv, config: TradingConfig) -> AgentBase:
    """
    Create ElegantRL agent matching FinRL's API.
    
    Args:
        agent_type: 'PPO', 'SAC', 'DDPG' (paper Section 5.2.1)
        
    Returns:
        Initialized agent with correct net_dims, learning_rate, etc.
    """
```

**Paper Hyperparameters (Table 3):**
| Parameter | Value |
|-----------|-------|
| Hidden Layers | 2 × (64, 32) |
| Learning Rate | 3e-4 |
| Batch Size | 64 |
| PPO n_steps | 2048 |
| SAC buffer_size | 300,000 |

#### 1.3 Training Loop
```python
def train_model(
    agent: AgentBase,
    env: StockTradingVecEnv,
    total_timesteps: int,
    model_name: str,
    iter_num: int
) -> AgentBase:
    """
    Train using ElegantRL's explore_env() + update_net() pattern.
    
    Key differences from FinRL's model.learn():
    - Uses VecEnv with num_envs parallel environments
    - PPO: on-policy with horizon_len trajectory collection
    - SAC/DDPG: off-policy with ReplayBuffer
    """
    while total_step < total_timesteps:
        buffer_items = agent.explore_env(env, horizon_len)
        total_step += horizon_len * env.num_envs
        
        if agent.if_off_policy:
            buffer.update(buffer_items)
            if buffer.cur_size < buffer_init_size:
                continue
        else:
            buffer[:] = buffer_items
        
        agent.update_net(buffer)
```

#### 1.4 Validation Sharpe
```python
def get_validation_sharpe(agent, val_env, model_name: str, iter_num: int) -> float:
    """
    Calculate Sharpe ratio on validation environment.
    
    FinRL Pattern: Saves to CSV, reads back, calculates from daily_return column
    ElegantRL Pattern: Collect rewards from VecEnv, calculate directly
    
    Formula: Sharpe = (mean(daily_return) / std(daily_return)) * sqrt(252)
    """
    daily_rewards = []
    state, _ = val_env.reset()
    
    for t in range(val_env.max_step):
        action = agent.act(state)
        state, reward, done, _, _ = val_env.step(action)
        daily_rewards.append(reward.mean().item())
        if done.all():
            break
    
    rewards_array = np.array(daily_rewards)
    if rewards_array.std() > 1e-8:
        sharpe = rewards_array.mean() / rewards_array.std() * np.sqrt(252)
    else:
        sharpe = 0.0
    return sharpe
```

#### 1.5 Ensemble Trading
```python
def DRL_prediction(
    agents: Dict[str, AgentBase],
    weights: Dict[str, float],
    trade_env: StockTradingVecEnv,
    ensemble_method: str = "sharpe_weighted"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trade using ensemble of agents.
    
    FinRL: Winner-takes-all (uses best Sharpe model only)
    Paper: Weighted action averaging
    
    Paper Formula (Section 4):
        w_i = softmax(sharpe_i) = exp(sharpe_i) / Σ exp(sharpe_j)
        action = Σ w_i * action_i
    """
    for t in range(trade_env.max_step):
        ensemble_action = torch.zeros(...)
        for model_name, agent in agents.items():
            action = agent.act(state)
            ensemble_action += weights[model_name] * action
        
        state, reward, done, _, _ = trade_env.step(ensemble_action)
```

#### 1.6 Rolling Window Orchestration
```python
def run_ensemble_strategy(
    df: pd.DataFrame,
    config: TradingConfig,
    model_names: List[str] = ['ppo', 'sac', 'ddpg'],
    rolling_window_length: int = 5,
    validation_window_length: int = 5
) -> pd.DataFrame:
    """
    Full rolling window ensemble strategy.
    
    Paper Settings (Section 5.2.2):
        - Train window: 30 days
        - Validation window: 5 days  
        - Trade window: 5 days
    """
    train_starts, train_ends, trade_starts, trade_ends = \
        calc_train_trade_starts_ends_if_rolling(train_dates, trade_dates, rolling_window_length)
    
    for i in range(len(train_starts)):
        # 1. Create train/val/trade environments
        train_env = create_env_from_dataframe(train_data, config)
        val_env = create_env_from_dataframe(val_data, config)
        trade_env = create_env_from_dataframe(trade_data, config)
        
        # 2. Train all models
        for model_name in model_names:
            agent = create_agent(model_name, train_env, config)
            agent = train_model(agent, train_env, timesteps, model_name, i)
            sharpe = get_validation_sharpe(agent, val_env, model_name, i)
            trained_agents[model_name] = agent
            sharpes[model_name] = sharpe
        
        # 3. Calculate ensemble weights (softmax)
        weights = softmax([sharpes[n] for n in model_names])
        
        # 4. Trade
        df_account, df_actions = DRL_prediction(trained_agents, weights, trade_env)
```

#### Stage 1 Success Criteria
- [ ] Individual agents achieve Sharpe > 0 on validation
- [ ] Ensemble produces different actions than individual agents
- [ ] Rolling window loop completes without errors
- [ ] Final cumulative return is positive

---

### Stage 2: Adding Perturbation
**Goal:** Implement price perturbation for training diversity (Paper Section 5.1).

#### 2.1 Price Perturbation
```python
def create_env_from_dataframe(
    df: pd.DataFrame,
    config: TradingConfig,
    apply_perturbation: bool = False,
    perturbation_range: float = 0.01  # ±1%
) -> StockTradingVecEnv:
    """
    Paper Quote (Section 5.1):
    "For each stock, a random percentage change ranging from -1% to 1%
     is generated and applied to its prices, which shifts the price scale
     while preserving the original price trends."
    """
    if apply_perturbation:
        # Per-stock perturbation factor
        perturbation = np.random.uniform(
            1 - perturbation_range,
            1 + perturbation_range,
            size=(1, num_stocks)
        )
        close_ary = close_ary * perturbation
```

#### 2.2 Integration Points
- Apply perturbation during **training only** (not validation/trading)
- Each agent can receive different perturbations for diversity
- Store perturbation factors for reproducibility

#### Stage 2 Success Criteria
- [ ] Different agents trained on perturbed data have different policies
- [ ] Action diversity increases compared to Stage 1
- [ ] Ensemble Sharpe ratio improves or maintains

---

### Stage 3: Adding KL Divergence
**Goal:** Enforce policy diversity via KL divergence penalty in loss function (Paper Section 5.1).

#### 3.1 KL Divergence Loss Modification
```python
def compute_kl_divergence_penalty(
    current_agent: AgentBase,
    other_agents: List[AgentBase],
    states: torch.Tensor,
    kl_lambda: float = 0.1
) -> torch.Tensor:
    """
    Paper Formula (Section 5.1):
    L_new(θ_A) = L_original(θ_A) - λ * Σ_{A≠B} KL(π_θB || π_θA)
    
    The negative sign encourages DIVERGENCE from other policies.
    
    For Gaussian policies (PPO, SAC, DDPG):
    KL(π_B || π_A) = log(σ_A/σ_B) + (σ_B² + (μ_B - μ_A)²) / (2σ_A²) - 0.5
    """
```

#### 3.2 Agent Modifications

**PPO with KL Penalty:**
```python
class AgentPPO_KL(AgentPPO):
    def __init__(self, ..., other_agents: List[AgentBase] = None, kl_lambda: float = 0.1):
        super().__init__(...)
        self.other_agents = other_agents or []
        self.kl_lambda = kl_lambda
    
    def update_net(self, buffer):
        # Original PPO loss
        obj_surrogate, obj_entropy = self.compute_ppo_loss(...)
        
        # Add KL divergence penalty
        if self.other_agents:
            kl_penalty = self.compute_kl_divergence_penalty(states)
            obj_critic = obj_surrogate + obj_entropy - self.kl_lambda * kl_penalty
```

**SAC with KL Penalty:**
```python
class AgentSAC_KL(AgentSAC):
    def update_net(self, buffer):
        # Original SAC actor loss
        obj_actor = (log_prob * alpha - q_value).mean()
        
        # Add KL divergence penalty
        if self.other_agents:
            kl_penalty = self.compute_kl_divergence_penalty(states)
            obj_actor = obj_actor - self.kl_lambda * kl_penalty
```

#### 3.3 Training Coordination
```python
def train_all_agents_with_kl(
    agents: Dict[str, AgentBase],
    envs: Dict[str, StockTradingVecEnv],
    config: TradingConfig
):
    """
    Train agents with KL divergence penalty.
    
    Challenge: Need to train iteratively so each agent knows other policies.
    
    Approach 1 (Sequential): Train one agent, then next with reference to previous
    Approach 2 (Alternating): Train all agents in parallel, update KL every K steps
    """
```

#### Stage 3 Success Criteria
- [ ] KL divergence between agent policies increases during training
- [ ] Agent actions become more diverse (lower correlation)
- [ ] Ensemble performance improves due to complementary strategies

---

### Stage 4: Adding Stock Subsets
**Goal:** Train each agent on different stock subsets for strategy diversity (Paper Section 5.1).

#### 4.1 Stock Subset Selection
```python
def select_stock_subset(
    all_tickers: List[str],
    subset_ratio: float = 0.7,  # Use 70% of stocks
    seed: int = None
) -> List[str]:
    """
    Paper Quote (Section 5.1):
    "Agents are also trained on different stocks from the test set.
     It enables agents to learn various strategies for a broader range
     of stocks rather than reacting to a limited number of stocks."
    """
    np.random.seed(seed)
    n_select = int(len(all_tickers) * subset_ratio)
    return list(np.random.choice(all_tickers, n_select, replace=False))
```

#### 4.2 Integration with Environment Creation
```python
def create_env_from_dataframe(
    df: pd.DataFrame,
    config: TradingConfig,
    stock_subset: Optional[List[str]] = None
) -> StockTradingVecEnv:
    """
    Filter DataFrame to stock subset before creating environment.
    """
    if stock_subset is not None:
        df = df[df['tic'].isin(stock_subset)]
```

#### 4.3 Action Dimension Handling
```python
# Challenge: Different agents have different action dimensions
# Solution: Pad actions to full dimension during ensemble

def pad_action_to_full(
    action: torch.Tensor,  # [num_envs, subset_action_dim]
    subset_indices: List[int],
    full_action_dim: int
) -> torch.Tensor:
    """
    Pad agent action (trained on subset) to full action dimension.
    Non-selected stocks get action=0 (hold).
    """
    full_action = torch.zeros(action.shape[0], full_action_dim, device=action.device)
    full_action[:, subset_indices] = action
    return full_action
```

#### 4.4 Modified Ensemble Trading
```python
def DRL_prediction_with_subsets(
    agents: Dict[str, AgentBase],
    weights: Dict[str, float],
    stock_subsets: Dict[str, List[str]],  # {model_name: [tickers]}
    trade_env: StockTradingVecEnv,
    all_tickers: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trade using ensemble where each agent may operate on different stocks.
    """
    for t in range(trade_env.max_step):
        ensemble_action = torch.zeros(...)
        
        for model_name, agent in agents.items():
            # Get subset action
            subset_state = extract_subset_state(state, stock_subsets[model_name])
            action = agent.act(subset_state)
            
            # Pad to full dimension
            subset_indices = [all_tickers.index(t) for t in stock_subsets[model_name]]
            full_action = pad_action_to_full(action, subset_indices, trade_env.action_dim)
            
            ensemble_action += weights[model_name] * full_action
```

#### Stage 4 Success Criteria
- [ ] Each agent trained on unique stock subset
- [ ] Agents develop specialized strategies for their stocks
- [ ] Ensemble covers all stocks despite individual agent limitations
- [ ] Overall Sharpe ratio comparable to or better than full-stock training

---

## 📋 Implementation Checklist

### Stage 1: Basic Ensemble ☐
- [ ] `DOW30VecEnv` adapter class
- [ ] `create_env_from_dataframe()` function
- [ ] `create_agent()` factory function
- [ ] `train_model()` with ElegantRL loop
- [ ] `get_validation_sharpe()` from VecEnv
- [ ] `DRL_prediction()` with softmax weights
- [ ] `run_ensemble_strategy()` rolling window loop
- [ ] Unit tests for each component
- [ ] Integration test with small dataset

### Stage 2: Perturbation ☐
- [ ] `apply_perturbation` parameter in env creation
- [ ] Per-stock random perturbation (±1%)
- [ ] Logging of perturbation factors
- [ ] Comparison test: with vs without perturbation

### Stage 3: KL Divergence ☐
- [ ] `compute_kl_divergence_penalty()` function
- [ ] `AgentPPO_KL` subclass
- [ ] `AgentSAC_KL` subclass
- [ ] `AgentDDPG_KL` subclass
- [ ] Training coordination logic
- [ ] KL divergence logging/monitoring

### Stage 4: Stock Subsets ☐
- [ ] `select_stock_subset()` function
- [ ] Environment creation with subset filtering
- [ ] `pad_action_to_full()` function
- [ ] State extraction for subsets
- [ ] Modified ensemble trading
- [ ] Index mapping validation

---

## 🎯 Performance Targets

From Paper (Table 4, DOW30 Stock Dataset):

| Metric | Paper Result | Target |
|--------|-------------|--------|
| Cumulative Return | 62.60% | > 50% |
| Sharpe Ratio | 1.48 | > 1.2 |
| Max Drawdown | -10.18% | < -15% |
| Annual Return | 62.06% | > 50% |
| Annual Volatility | 18.79% | < 25% |

Individual Agent Benchmarks (Table 5):
| Agent | Sharpe (Validation) |
|-------|---------------------|
| PPO | ~1.55 |
| SAC | ~1.27 |
| DDPG | ~1.51 |

---

## 📁 File Structure

```
elegantrl/
├── envs/
│   └── DOW30VecEnv.py              # Stage 1: DataFrame adapter
├── agents/
│   ├── AgentPPO_KL.py              # Stage 3: KL-augmented PPO
│   ├── AgentSAC_KL.py              # Stage 3: KL-augmented SAC
│   └── AgentDDPG_KL.py             # Stage 3: KL-augmented DDPG
└── train/
    └── ensemble.py                  # DRLEnsembleAgent class

examples/
└── demo_DRLEnsembleAgent.py        # Full working example

unit_tests/
└── ensemble/
    ├── test_env_adapter.py
    ├── test_validation_sharpe.py
    ├── test_ensemble_weights.py
    └── test_rolling_window.py
```

---

## 🔧 Configuration Template

```python
@dataclass
class EnsembleConfig:
    """Configuration for DRLEnsembleAgent."""
    
    # Data
    tickers: List[str] = field(default_factory=lambda: DOW_30_TICKER)
    train_start: str = "2021-01-01"
    train_end: str = "2022-06-01"
    val_start: str = "2022-06-01"
    test_end: str = "2024-01-01"
    
    # Environment
    initial_amount: float = 1_000_000
    max_stock: int = 100
    cost_pct: float = 0.001
    num_envs: int = 96  # Reduced for memory
    
    # Rolling window
    rebalance_window: int = 30  # days
    validation_window: int = 5  # days
    trading_window: int = 5     # days
    
    # Network
    net_dims: Tuple[int, int] = (64, 32)
    learning_rate: float = 3e-4
    batch_size: int = 64
    
    # Agents
    agent_types: List[str] = field(default_factory=lambda: ['ppo', 'sac', 'ddpg'])
    timesteps_dict: Dict[str, int] = field(default_factory=lambda: {
        'ppo': 100_000,
        'sac': 50_000,
        'ddpg': 50_000,
    })
    
    # Stage 2: Perturbation
    use_perturbation: bool = True
    perturbation_range: float = 0.01  # ±1%
    
    # Stage 3: KL Divergence
    use_kl_divergence: bool = True
    kl_lambda: float = 0.1
    
    # Stage 4: Stock Subsets
    use_stock_subsets: bool = True
    stock_subset_ratio: float = 0.7
    
    # Ensemble
    ensemble_method: str = 'sharpe_weighted'  # or 'winner_takes_all'
```

---

## ⚠️ Known Issues & Solutions

### Issue 1: VecEnv Auto-Reset
**Problem:** ElegantRL VecEnv auto-resets on `done=True`, corrupting final portfolio value.
**Solution:** Read `env.cumulative_returns` before reset, or capture `total_asset` at step T-1.

### Issue 2: Action Dead Zone
**Problem:** ElegantRL applies dead zone: actions in (-0.1, 0.1) → 0.
**Solution:** Account for this when interpreting action statistics; may need larger action magnitudes.

### Issue 3: Memory with Multiple Agents
**Problem:** Training 3 agents × 2048 envs exceeds GPU memory.
**Solution:** Reduce `num_envs` to 96-256 for off-policy, clean up between agents with `torch.cuda.empty_cache()`.

### Issue 4: Sharpe = 0 for SAC
**Problem:** SAC often returns Sharpe = 0 on short validation windows.
**Solution:** Increase training timesteps, use VecNormalize, ensure proper exploration.

---

## 📚 References

1. **Revisiting Ensemble Methods for Stock Trading** (arXiv 2501.10709v1)
   - Core methodology: rolling windows, Sharpe weighting, diversity methods

2. **FinRL-Contest: Market Synthesis** (arXiv 2504.02281v4)
   - Benchmark: 227K sps with 2048 envs for PPO on A100

3. **ElegantRL StockTradingVecEnv**
   - Source: `elegantrl/envs/StockTradingEnv.py`
   - Features: vmap parallelization, tanh normalization, dead zone

4. **FinRL DRLEnsembleAgent**
   - Source: `finrl/agents/stablebaselines3/models.py`
   - Pattern: rolling window, winner-takes-all selection
