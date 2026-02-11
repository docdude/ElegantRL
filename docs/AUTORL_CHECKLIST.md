# AutoRL Reproducibility Checklist

**Based on:** [how-to-autorl checklist](https://github.com/facebookresearch/how-to-autorl/blob/main/checklist.pdf)

## 1. Train/Test Separation & Validation Methodology

| Item | Status | Notes |
|------|--------|-------|
| Separate training and test settings available? | ✅ | Walk-forward CV with time-based splits |
| Training setting used only for training? | ✅ | `beg_idx=train_start, end_idx=train_end` |
| Training setting used only for tuning? | ✅ | HPO uses training data only |
| Final results reported on test setting? | ✅ | Validation split (val_start:val_end) |

### Validation Strategy Options

**Current:** `n_folds: 1` = Simple 80/20 holdout (fast but higher variance)

**Recommended for production HPO:** `n_folds: 3-5` = Walk-forward CV

| Method | Pros | Cons | Config |
|--------|------|------|--------|
| **Holdout (n_folds=1)** | Fast, cheap | Overfits to single split | `n_folds: 1` |
| **Walk-Forward CV (n_folds=3)** | Robust, tests multiple regimes | 3x slower | `n_folds: 3` |
| **Expanding Window** | Uses all early data | Later folds dominate | Custom |
| **Combinatorial Purged CV** | Most robust for finance | Complex, very slow | Not implemented |

### Why Not Simple 80/20?

Per [Schneider et al. 2024](https://arxiv.org/abs/2405.15393):
- Fixed holdout splits risk **overfitting to the validation set** during HPO
- Reshuffling splits per trial improves generalization (but N/A for time series)
- For time series: **walk-forward with multiple folds** is the correct analog

Per [Bischl et al. 2023 (WIREs)](https://doi.org/10.1002/widm.1484):
- 5-fold CV recommended for robust hyperparameter selection
- Holdout variance is high, especially with limited data

### Recommended Settings

```yaml
# For quick testing
n_folds: 1
break_step: 50000

# For proper HPO (production)
n_folds: 3
break_step: 200000

# For thorough validation (final tuning)
n_folds: 5
break_step: 500000
gap_days: 5  # Purging gap to avoid lookahead
```

### Walk-Forward Implementation

```python
# Current get_walk_forward_splits() with n_folds=3:
# Fold 0: Train [0-200], Val [200-238] (2016-2017)
# Fold 1: Train [238-438], Val [438-476] (2018-2019)  
# Fold 2: Train [476-676], Val [676-714] (2020-2021)
# Each fold tests different market regime!
```

## 2. HPO Method

| Item | Value |
|------|-------|
| Package used | `hypersweeper` 0.2.3 |
| Optimization method | SMAC (Bayesian Optimization), DEHB (Multi-fidelity) |
| Seeds implementation | Multi-seed averaging via `sweeper_kwargs.seeds: [0, 1, 2]` |

## 3. Configuration Space

### PPO (alpaca_ppo.yaml)
| Hyperparameter | Type | Range/Choices | Default |
|----------------|------|---------------|---------|
| learning_rate | uniform_float (log) | [1e-5, 1e-3] | 3e-4 |
| gamma | categorical | [0.9, 0.95, 0.98, 0.99, 0.995, 0.999] | 0.99 |
| net_arch | categorical | [small, medium, large] | medium |
| batch_size | categorical | [128, 256, 512, 1024] | 512 |
| horizon_len | categorical | [256, 512, 1024, 2048] | 512 |
| ratio_clip | categorical | [0.1, 0.2, 0.25, 0.3, 0.4] | 0.25 |
| lambda_gae_adv | categorical | [0.8, 0.9, 0.92, 0.95, 0.98, 0.99] | 0.95 |
| lambda_entropy | uniform_float (log) | [1e-8, 0.1] | 0.01 |
| repeat_times | categorical | [4, 8, 10, 16, 20, 32] | 16 |
| max_grad_norm | categorical | [0.3, 0.5, 0.7, 1.0, 2.0, 5.0] | 1.0 |
| vf_coef | uniform_float | [0.1, 1.0] | 0.5 |

### SAC (alpaca_sac.yaml)
| Hyperparameter | Type | Range/Choices | Default |
|----------------|------|---------------|---------|
| learning_rate | uniform_float (log) | [1e-5, 1e-3] | 1e-4 |
| gamma | categorical | [0.9, 0.95, 0.98, 0.99, 0.995, 0.999] | 0.99 |
| net_arch | categorical | [small, medium, big] | medium |
| batch_size | categorical | [64, 128, 256, 512, 1024] | 256 |
| buffer_size | categorical | [10K, 50K, 100K, 500K, 1M] | 100K |
| soft_update_tau | categorical | [0.001, 0.005, 0.01, 0.02, 0.05] | 0.005 |
| horizon_len | categorical | [64, 128, 256, 512] | 128 |
| repeat_times | categorical | [1, 2, 4, 8] | 2 |
| lambda_entropy | uniform_float (log) | [0.01, 0.5] | 0.1 |

### DDPG (alpaca_ddpg.yaml)
| Hyperparameter | Type | Range/Choices | Default |
|----------------|------|---------------|---------|
| learning_rate | uniform_float (log) | [1e-5, 1e-3] | 1e-4 |
| gamma | categorical | [0.9, 0.95, 0.98, 0.99, 0.995, 0.999] | 0.99 |
| net_arch | categorical | [small, medium, big] | medium |
| batch_size | categorical | [64, 128, 256, 512, 1024] | 256 |
| buffer_size | categorical | [10K, 50K, 100K, 500K, 1M] | 100K |
| soft_update_tau | categorical | [0.001, 0.005, 0.01, 0.02, 0.05, 0.08] | 0.005 |
| horizon_len | categorical | [64, 128, 256, 512] | 128 |
| repeat_times | categorical | [1, 2, 4, 8] | 2 |
| explore_noise_std | uniform_float | [0.05, 0.5] | 0.1 |

## 4. Budget Consistency

| Item | Status | Notes |
|------|--------|-------|
| Same search space for shared HPs across algorithms? | ✅ | gamma, learning_rate, net_arch consistent |
| Same tuning budget for all methods? | ✅ | 50 trials PPO/DDPG, 100 trials SAC (larger space) |
| Comparable hardware for all runs? | ✅ | RTX 2080 for all HPO, H200 for final training |

## 5. Seeds

| Setting | Seeds | Purpose |
|---------|-------|---------|
| **Tuning Seeds** | [0, 1, 2] | Multi-seed averaging during HPO |
| **Test Seeds** | [5, 6, 7, 8, 9] | Final evaluation (separate!) |

### Implementation
```yaml
# In config file
hydra:
  sweeper:
    sweeper_kwargs:
      seeds: [0, 1, 2]  # Tuning seeds
```

## 6. Cost Metric

| Metric | Formula | Notes |
|--------|---------|-------|
| **Primary** | Negative Sharpe Ratio | `return -sharpe` (minimization) |
| **Secondary** | Cumulative Return (%) | Logged but not optimized |

## 7. Final Reporting Template

After HPO, report:

```
# Algorithm: PPO
# Environment: Alpaca Stock Trading (28 stocks, DJIA)
# Tuning Seeds: [0, 1, 2]
# Test Seeds: [5, 6, 7, 8, 9]
# HPO Trials: 50
# Final Configuration:
  - learning_rate: <optimized_value>
  - gamma: <optimized_value>
  - net_arch: <optimized_value>
  - batch_size: <optimized_value>
  - ...

# Test Results (mean ± std across test seeds):
  - Sharpe Ratio: X.XX ± Y.YY
  - Cumulative Return: X.XX% ± Y.YY%
  - Max Drawdown: X.XX% ± Y.YY%
```

## 8. Commands

### Run HPO (Tuning)
```bash
# PPO with SMAC
CUDA_VISIBLE_DEVICES=1 python examples/hpo_alpaca_vecenv.py -m --config-name=alpaca_ppo

# SAC with DEHB (multi-fidelity)
CUDA_VISIBLE_DEVICES=1 python examples/hpo_alpaca_vecenv.py -m --config-name=alpaca_sac

# DDPG with SMAC
CUDA_VISIBLE_DEVICES=1 python examples/hpo_alpaca_vecenv.py -m --config-name=alpaca_ddpg
```

### Run Test (Final Evaluation)
```bash
# After HPO completes, get incumbent from outputs/ppo_sweep/.../incumbent.csv
# Then evaluate on test seeds [5,6,7,8,9]
python examples/hpo_alpaca_vecenv.py --config-name=alpaca_ppo \
    seed=5 \
    learning_rate=<from_incumbent> \
    gamma=<from_incumbent> \
    ...
```

### Run Post-Hoc Analysis (Built into Hypersweeper)
```bash
# LPI - Local Parameter Importance (which HPs matter around incumbent?)
python examples/hpo_alpaca_vecenv.py -m --config-name=alpaca_ppo_lpi \
    hydra.sweeper.sweeper_kwargs.data_path=outputs/ppo_sweep/<timestamp>/runhistory.csv

# Ablation Paths (which HP changes helped most from default→incumbent?)
python examples/hpo_alpaca_vecenv.py -m --config-name=alpaca_ppo_ablation \
    hydra.sweeper.sweeper_kwargs.data_path=outputs/ppo_sweep/<timestamp>/runhistory.csv

# Output files:
#   LPI:      outputs/ppo_lpi/<timestamp>/lpi_scores.csv, lpi_plot.png
#   Ablation: outputs/ppo_ablation/<timestamp>/ablation_path.csv, ablation_path.png
```

## 9. File Structure

```
examples/
├── hpo_alpaca_vecenv.py          # Main HPO script (Hydra target)
└── configs/
    ├── alpaca_ppo.yaml           # PPO base config
    ├── alpaca_sac.yaml           # SAC base config
    ├── alpaca_ddpg.yaml          # DDPG base config
    └── search_space/
        ├── ppo_space.yaml        # PPO search space
        ├── sac_space.yaml        # SAC search space
        └── ddpg_space.yaml       # DDPG search space
```

## 10. References

- [Hypersweeper](https://github.com/automl/hypersweeper) - HPO framework
- [how-to-autorl](https://github.com/facebookresearch/how-to-autorl) - Best practices
- [SMAC3](https://github.com/automl/SMAC3) - Bayesian optimization
- [DEHB](https://github.com/automl/DEHB) - Multi-fidelity optimization
---

## Appendix: Validation Methodology Deep Dive

### Literature Summary

**Key Papers:**
1. **Bischl et al. 2023** - "Hyperparameter Optimization: Foundations, Algorithms, Best Practices"
   - [DOI: 10.1002/widm.1484](https://doi.org/10.1002/widm.1484)
   - Recommends 5-fold CV over holdout for robustness
   - Discusses budget allocation, multi-fidelity, and early stopping

2. **Schneider et al. 2024** - "Reshuffling Resampling Splits Can Improve Generalization"
   - [arXiv:2405.15393](https://arxiv.org/abs/2405.15393)
   - **Key finding:** Reshuffled holdout ≈ 5-fold CV performance
   - Reduces overfitting to validation set during HPO
   - Time series caveat: Can't reshuffle; use multiple temporal folds

3. **López de Prado 2018** - "Advances in Financial Machine Learning"
   - Combinatorial Purged Cross-Validation (CPCV)
   - Embargo periods to prevent information leakage
   - Addresses non-IID nature of financial data

### Validation Methods for Time Series RL

| Method | Description | Complexity | Implemented |
|--------|-------------|------------|-------------|
| **Holdout** | Fixed 80/20 split | O(1) | ✅ Default |
| **Walk-Forward CV** | Rolling train/val windows | O(k) | ✅ `n_folds > 1` |
| **Expanding Window** | Growing training set | O(k) | ❌ Easy to add |
| **Purged K-Fold** | K-fold with embargo | O(k) | ⚠️ via `gap_days` |
| **CPCV** | Combinatorial purged | O(n choose k) | 🔗 External repo |

### When to Use What

```
Quick Testing:      n_folds=1, break_step=50K   (~5 min/trial)
Standard HPO:       n_folds=3, break_step=200K  (~15 min/trial)  
Thorough Tuning:    n_folds=5, break_step=500K  (~40 min/trial)
Publication-Ready:  n_folds=5, break_step=1M, gap_days=5
```

### Future Enhancements (Not Implemented)

1. **Expanding Window Validation**
   ```python
   # All data up to fold_end for training, next window for validation
   train_start = 0  # Always start from beginning
   train_end = fold * fold_size + int(fold_size * train_ratio)
   ```

2. **Combinatorial Purged CV (CPCV)**
   - Generates all combinations of train/test splits
   - Purges overlapping samples + embargo period
   - Very robust but computationally expensive
   - **Available in separate repo** (long processing times)
   - See: `mlfinlab.cross_validation.CombinatorialPurgedKFold`

3. **Anchored Walk-Forward (FinRL default)**
   - Train: All data up to validation window
   - Val: Next fixed-length window
   - Common in financial backtesting

### Trade-offs

| Concern | Holdout | Walk-Forward 3-Fold | Walk-Forward 5-Fold |
|---------|---------|---------------------|---------------------|
| Compute Cost | 1x | 3x | 5x |
| Variance | High | Medium | Low |
| Regime Robustness | Poor | Good | Best |
| Early Stopping Risk | High | Medium | Low |
| Recommended For | Debugging | HPO | Final tuning |