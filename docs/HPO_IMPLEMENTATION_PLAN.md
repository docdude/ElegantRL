# Hypersweeper HPO Implementation Plan for ElegantRL Stock Trading

## Overview

Integrate AutoML's Hypersweeper framework with our GPU-parallel VecEnv stock trading agents to systematically optimize hyperparameters using walk-forward validation.

**Hardware Setup:**
- **HPO Development/Testing:** RTX 2080 (on-policy agents first - PPO)
- **Production Training:** H200 (after HPO finds best configs)

---

## Phase 1: Prepare ElegantRL for HPO

### 1.1 Create HPO-Ready Training Script

Copy and modify `demo_FinRL_Alpaca_VecEnv.py` → `hpo_alpaca_vecenv.py`

**Key Modifications:**

```python
# hpo_alpaca_vecenv.py

import hydra
from omegaconf import DictConfig
import numpy as np

@hydra.main(config_path="configs", config_name="alpaca_ppo", version_base="1.1")
def train_and_evaluate(cfg: DictConfig) -> float:
    """
    Hydra-compatible training function for Hypersweeper.
    
    Returns:
        float: Negative Sharpe ratio (minimize for HPO)
    """
    # === SEED MANAGEMENT ===
    seed = cfg.seed
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    
    # === WALK-FORWARD FOLD (if enabled) ===
    fold_id = getattr(cfg, 'fold_id', 0)
    train_data, val_data = get_walk_forward_split(fold_id, cfg.n_folds)
    
    # === BUILD CONFIG FROM HYDRA ===
    args = build_elegantrl_config(cfg)
    
    # === TRAIN ===
    trained_agent = train_agent(args, train_data)
    
    # === EVALUATE ON VALIDATION SET ===
    val_sharpe = evaluate_on_validation(trained_agent, val_data)
    
    # Return negative Sharpe (Hypersweeper minimizes by default)
    return -val_sharpe

if __name__ == "__main__":
    train_and_evaluate()
```

### 1.2 Implement Walk-Forward Data Splits

```python
# elegantrl/train/walk_forward.py

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

@dataclass
class WalkForwardConfig:
    """Walk-forward validation configuration."""
    n_folds: int = 5
    train_ratio: float = 0.7  # 70% train, 30% val within each fold
    expanding: bool = False   # False = rolling window, True = expanding
    gap_days: int = 0         # Gap between train and val (avoid lookahead)

def get_walk_forward_splits(
    total_days: int,
    config: WalkForwardConfig
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Generate walk-forward train/val splits.
    
    Returns:
        List of ((train_start, train_end), (val_start, val_end))
    """
    splits = []
    fold_size = total_days // config.n_folds
    
    for fold in range(config.n_folds):
        if config.expanding:
            # Expanding window: train always starts at 0
            train_start = 0
            train_end = (fold + 1) * fold_size - int(fold_size * (1 - config.train_ratio))
        else:
            # Rolling window
            train_start = fold * fold_size
            train_end = train_start + int(fold_size * config.train_ratio)
        
        val_start = train_end + config.gap_days
        val_end = min((fold + 1) * fold_size, total_days)
        
        if val_end > val_start:
            splits.append(((train_start, train_end), (val_start, val_end)))
    
    return splits
```

### 1.3 Seed Management Utility

```python
# elegantrl/train/seed_utils.py

import random
import numpy as np
import torch as th

def set_all_seeds(seed: int, cuda_deterministic: bool = False):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    
    if cuda_deterministic:
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

def get_seed_for_trial(base_seed: int, trial_id: int) -> int:
    """Generate unique seed for each HPO trial."""
    return (base_seed + trial_id * 1000) % (2**32 - 1)
```

---

## Phase 2: Install & Configure Hypersweeper

### 2.1 Installation

```bash
# In your ElegantRL venv
cd /mnt/ssd_backup/ElegantRL
source .venv/bin/activate

# Install Hypersweeper with SMAC and DEHB
pip install hydra-core omegaconf

# Clone Hypersweeper (not yet on PyPI as stable)
git clone https://github.com/automl/hypersweeper.git
cd hypersweeper
pip install -e ".[smac,dehb]"

# Verify installation
python -c "from hydra_plugins.hyper_smac import HyperSMACConfig; print('SMAC OK')"
python -c "from hydra_plugins.hyper_dehb import HyperDEHBConfig; print('DEHB OK')"
```

### 2.2 Create Directory Structure

```
ElegantRL/
├── examples/
│   ├── hpo_alpaca_vecenv.py          # Main HPO training script
│   └── configs/
│       ├── alpaca_ppo.yaml           # Base config for PPO
│       ├── alpaca_sac.yaml           # Base config for SAC  
│       ├── alpaca_ddpg.yaml          # Base config for DDPG
│       ├── search_space/
│       │   ├── ppo_space.yaml        # PPO hyperparameter space
│       │   ├── sac_space.yaml        # SAC hyperparameter space
│       │   └── ddpg_space.yaml       # DDPG hyperparameter space
│       └── sweeper/
│           ├── smac_bo.yaml          # SMAC Bayesian Optimization
│           ├── dehb_mf.yaml          # DEHB Multi-fidelity
│           └── pb2_pbt.yaml          # PB2 Population-based
```

### 2.3 Example Config Files

**`configs/alpaca_ppo.yaml`** (Base config):
```yaml
defaults:
  - search_space: ppo_space
  - sweeper: smac_bo
  - _self_

# Target function settings
agent: ppo
gpu_id: 0  # RTX 2080
num_envs: 512  # Reduced for 2080's 8GB VRAM

# Walk-forward settings
n_folds: 3
fold_id: 0  # Overridden by HPO for cross-validation

# Training settings (defaults, overridden by HPO)
seed: 42
break_step: 500000  # Reduced for HPO speed

# Fixed settings (not tuned)
state_dim: 283
action_dim: 28
max_step: 602

hydra:
  sweeper:
    n_trials: 50
  run:
    dir: outputs/${agent}/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: outputs/${agent}_sweep/${now:%Y-%m-%d_%H-%M-%S}
```

**`configs/search_space/ppo_space.yaml`**:
```yaml
search_space:
  hyperparameters:
    learning_rate:
      type: float
      lower: 1e-5
      upper: 1e-3
      log: true
      default: 3e-4
    
    gamma:
      type: float
      lower: 0.95
      upper: 0.999
      default: 0.985
    
    net_dims_0:  # First layer width
      type: int
      lower: 64
      upper: 512
      log: true
      default: 256
    
    net_dims_1:  # Second layer width
      type: int
      lower: 64
      upper: 256
      log: true
      default: 128
    
    batch_size:
      type: int
      lower: 128
      upper: 1024
      log: true
      default: 512
    
    horizon_len:
      type: int
      lower: 256
      upper: 2048
      log: true
      default: 512
    
    ratio_clip:
      type: float
      lower: 0.1
      upper: 0.4
      default: 0.25
    
    lambda_gae_adv:
      type: float
      lower: 0.9
      upper: 0.99
      default: 0.95
    
    lambda_entropy:
      type: float
      lower: 0.0001
      upper: 0.1
      log: true
      default: 0.01
    
    repeat_times:
      type: int
      lower: 4
      upper: 32
      default: 16
```

**`configs/sweeper/smac_bo.yaml`**:
```yaml
hydra:
  sweeper:
    _target_: hydra_plugins.hypersweeper.hypersweeper.Hypersweeper
    opt_constructor: hydra_plugins.hyper_smac.hyper_smac.make_smac
    n_trials: 50
    
    sweeper_kwargs:
      smac_facade: smac.facade.HyperparameterOptimizationFacade
      scenario:
        n_trials: 50
        deterministic: false
        n_workers: 1  # Sequential for GPU
      
      # Warmstart from known good config
      initial_design:
        n_configs: 5  # Random configs before BO kicks in
```

---

## Phase 3: Start with On-Policy (PPO) on RTX 2080

### 3.1 Why PPO First?

| Reason | Benefit |
|--------|---------|
| **No replay buffer** | Lower VRAM usage on 2080 (8GB) |
| **Faster iterations** | On-policy = no buffer warmup |
| **Simpler HP space** | No buffer_size, soft_update_tau |
| **Your best results** | PPO reached 189% avgR |

### 3.2 Initial HPO Run

```bash
# Activate environment
cd /mnt/ssd_backup/ElegantRL
source .venv/bin/activate

# Run PPO HPO with SMAC (50 trials)
CUDA_VISIBLE_DEVICES=1 python examples/hpo_alpaca_vecenv.py \
    --config-name=alpaca_ppo \
    -m \
    hydra.sweeper.n_trials=50 \
    gpu_id=1 \
    num_envs=512

# Monitor progress
tail -f outputs/ppo_sweep/*/runhistory.csv
```

### 3.3 Expected Timeline

| Stage | Trials | Time/Trial | Total |
|-------|--------|------------|-------|
| Random init | 5 | ~10 min | 50 min |
| BO exploration | 25 | ~10 min | 4 hrs |
| BO exploitation | 20 | ~10 min | 3 hrs |
| **Total** | 50 | | **~8 hrs** |

---

## Phase 4: Optimizer Selection by Agent Type

### 4.1 Recommended Optimizers

| Agent | Optimizer | Why |
|-------|-----------|-----|
| **PPO** | SMAC (BO) | Moderate HP space (~10 HPs), BO efficient |
| **SAC** | DEHB | Larger space, multi-fidelity saves compute |
| **DDPG** | SMAC (BO) | Small space (~8 HPs), BO sufficient |
| **ModSAC** | PB2 | Complex agent, benefits from population |

### 4.2 Optimizer Comparison

| Optimizer | Best For | Parallelization | Multi-Fidelity |
|-----------|----------|-----------------|----------------|
| **SMAC** | Small-medium spaces, few trials | Limited | Via Hyperband |
| **DEHB** | Large spaces, many trials | Excellent | Native |
| **PB2** | RL agents, online adaptation | Native (population) | Native (budgets) |
| **HEBO** | Noisy objectives, mixed types | Good | No |

### 4.3 Sweeper Configs by Agent

**`configs/sweeper/dehb_mf.yaml`** (for SAC):
```yaml
hydra:
  sweeper:
    _target_: hydra_plugins.hypersweeper.hypersweeper.Hypersweeper
    opt_constructor: hydra_plugins.hyper_dehb.hyper_dehb.make_dehb
    n_trials: 100
    budget_variable: break_step  # Multi-fidelity on training steps
    
    sweeper_kwargs:
      min_fidelity: 100000    # 100K steps (cheap eval)
      max_fidelity: 1000000   # 1M steps (full eval)
      eta: 3                   # Halving factor
```

**`configs/sweeper/pb2_pbt.yaml`** (for ModSAC):
```yaml
hydra:
  sweeper:
    _target_: hydra_plugins.hypersweeper.hypersweeper.Hypersweeper
    opt_constructor: hydra_plugins.hyper_pbt.hyper_pbt.make_pb2
    n_trials: 20  # Population size
    
    sweeper_kwargs:
      population_size: 20
      perturbation_interval: 50000  # Steps between adaptations
      quantile_fraction: 0.25       # Bottom 25% replaced
```

---

## Phase 5: Post-HPO Analysis

### 5.1 Parameter Importance (LPI)

After SMAC run completes:

```bash
# Run LPI analysis
python examples/hpo_alpaca_vecenv.py \
    --config-name=alpaca_ppo_lpi \
    -m \
    hydra.sweeper.sweeper_kwargs.data_path=outputs/ppo_sweep/latest/runhistory.csv
```

### 5.2 DeepCAVE Visualization

```python
# In notebook or script
from deepcave import Recorder, Objective
from deepcave.runs.converters.deepcave import DeepCAVERun

# Load HPO results
run = DeepCAVERun.from_path("outputs/ppo_sweep/latest")

# HP Importance plot
from deepcave.plugins.hyperparameter.importances import Importances
imp = Importances()
imp.generate_outputs(run)
```

### 5.3 Walk-Forward Aggregation

```python
# After running HPO with n_folds > 1
def aggregate_walk_forward_results(runhistory_path: str):
    """Average Sharpe across walk-forward folds."""
    df = pd.read_csv(runhistory_path)
    
    # Group by config (excluding fold_id and seed)
    config_cols = [c for c in df.columns if c not in ['fold_id', 'seed', 'performance']]
    
    results = df.groupby(config_cols).agg({
        'performance': ['mean', 'std', 'count']
    }).reset_index()
    
    results.columns = config_cols + ['mean_sharpe', 'std_sharpe', 'n_folds']
    return results.sort_values('mean_sharpe')  # Best (most negative) first
```

---

## Implementation Checklist

### Phase 1: Preparation (Day 1-2)
- [ ] Create `hpo_alpaca_vecenv.py` from demo script
- [ ] Implement `walk_forward.py` module
- [ ] Implement `seed_utils.py` module
- [ ] Create config directory structure
- [ ] Write base YAML configs

### Phase 2: Setup (Day 2)
- [ ] Install Hypersweeper in venv
- [ ] Verify SMAC/DEHB imports work
- [ ] Test Hydra config loading
- [ ] Run single trial manually to verify

### Phase 3: PPO HPO (Day 3-4)
- [ ] Run initial 10-trial test on 2080
- [ ] Verify runhistory.csv logging
- [ ] Run full 50-trial SMAC sweep
- [ ] Analyze results with LPI

### Phase 4: Expand to Other Agents (Day 5+)
- [ ] Create SAC search space
- [ ] Run DEHB sweep for SAC on H200
- [ ] Create DDPG search space
- [ ] Compare optimizers

### Phase 5: Validation (Day 6+)
- [ ] Run walk-forward on top 3 configs
- [ ] Test on held-out test set
- [ ] Compare to baseline (current best)

---

## Quick Reference Commands

```bash
# PPO with SMAC (50 trials)
CUDA_VISIBLE_DEVICES=1 python examples/hpo_alpaca_vecenv.py -m --config-name=alpaca_ppo

# SAC with DEHB (100 trials, multi-fidelity)
CUDA_VISIBLE_DEVICES=0 python examples/hpo_alpaca_vecenv.py -m --config-name=alpaca_sac

# LPI Analysis after HPO
python examples/hpo_alpaca_vecenv.py -m --config-name=alpaca_ppo_lpi

# Single trial test (no sweep)
python examples/hpo_alpaca_vecenv.py --config-name=alpaca_ppo
```

---

## Expected Outcomes

| Metric | Current Best | After HPO (Expected) |
|--------|--------------|---------------------|
| PPO Sharpe | 2.64 | 2.8 - 3.0 |
| DDPG Sharpe | 2.88 | 3.0 - 3.2 |
| ModSAC Sharpe | 2.64 | 2.8 - 3.0 |
| Generalization | Unknown | Validated on 3+ folds |

**Key Success Criteria:**
1. Find configs that outperform manual tuning
2. Identify which HPs matter most (fANOVA/LPI)
3. Validate robustness across time periods
4. Reduce future tuning effort with prior knowledge
