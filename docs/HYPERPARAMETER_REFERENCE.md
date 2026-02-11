# ElegantRL Hyperparameter Reference

## Overview

This document describes the hyperparameters available in ElegantRL for different agent types, with specific notes for financial trading applications.

---

## Common Parameters (All Agents)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `net_dims` | varies | Hidden layer dimensions for actor/critic networks. Larger = more capacity but slower training. **Off-policy: [256, 128], On-policy: [128, 64]** in our setup. |
| `gamma` | 0.99 | Discount factor for future rewards. Higher (0.99) = long-term focus, Lower (0.9) = short-term. Finance often uses 0.99 for multi-day strategies. |
| `learning_rate` | 1e-4 | Step size for gradient updates. Too high = unstable, too low = slow convergence. 1e-4 to 3e-4 typical for finance. |
| `break_step` | 1e6 | Total environment steps before stopping. More steps = longer training but potentially better policy. |
| `reward_scale` | 1.0 | **Multiplier for rewards before training.** Critical for SAC to balance profit signal vs entropy. Try 2.0-4.0 if SAC is stuck in "entropy trap". |
| `state_value_tau` | 0.0 | Running mean/std normalization of states (0 = disabled). Set to 0 when using VecNormalize. |

---

## On-Policy Agents (PPO, A2C)

On-policy agents collect fresh experience each iteration and discard old data. They're generally more stable but less sample-efficient.

### PPO (Proximal Policy Optimization)

**Best for:** Stable training, good default choice, handles continuous actions well.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon_len` | 2048 | Steps collected per iteration before update. Larger = more stable gradients but slower updates. |
| `repeat_times` | 16 | Number of SGD passes over collected data. Higher = more learning per sample but risk of overfitting to batch. PPO paper uses 3-10, we use 16 for faster convergence. |
| `lambda_entropy` | 0.01 | **Entropy coefficient.** Encourages exploration by rewarding action randomness. Too high = random actions, too low = premature convergence. 0.01 is good default. |
| `lambda_gae_adv` | 0.95 | GAE lambda for advantage estimation. Higher = lower bias but higher variance. 0.95-0.98 typical. |
| `ratio_clip` | 0.25 | PPO clipping parameter. Limits policy update size. 0.1-0.3 typical, 0.2 is standard. |
| `num_envs` | 2048 | Parallel environments. More = faster data collection, better gradient estimates. Limited by GPU memory. |

**PPO-Specific Notes for Finance:**
- PPO's clipping prevents catastrophic policy changes (important when trading real money)
- `lambda_entropy=0.01` provides exploration without excessive randomness
- High `num_envs` (2048) gives diverse market scenarios per update

### A2C (Advantage Actor-Critic)

**Best for:** Simpler than PPO, faster wall-clock time, good for quick experiments.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon_len` | 2048 | Same as PPO |
| `repeat_times` | 8 | Typically lower than PPO (no clipping protection) |
| `lambda_entropy` | 0.01 | Same as PPO |
| `lambda_gae_adv` | 0.95 | Same as PPO |
| `num_envs` | 2048 | Same as PPO |

**A2C vs PPO:**
- A2C has no clipping → larger updates → potentially faster but less stable
- A2C typically uses lower `repeat_times` to avoid overfitting
- In our tests, A2C + VecNormalize achieved 16.90% return (best Sharpe 2.56)

---

## Off-Policy Agents (SAC, TD3, DDPG)

Off-policy agents store experience in a replay buffer and can reuse old data. More sample-efficient but can suffer from value overestimation.

### SAC (Soft Actor-Critic)

**Best for:** Sample efficiency, automatic entropy tuning, handles exploration well.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 256 | Samples per gradient update. Larger = more stable but slower. 256-512 typical. |
| `buffer_size` | 1e5 | Replay buffer capacity. Larger = more diverse experience but more memory. 1e5-1e6 typical. |
| `soft_update_tau` | 5e-3 | Target network update rate. Lower = more stable but slower learning. 0.005 typical. |
| `num_envs` | 96 | Fewer than on-policy (buffer provides diversity). 64-128 typical. |
| `alpha_log` (internal) | -1.0 | Log of entropy coefficient (auto-tuned). Initial value affects early exploration. |
| `target_entropy` (internal) | log(action_dim) | Target entropy for auto-tuning. **For 28 stocks = 3.33.** Lower = less randomness. |

**SAC-Specific Issues for Finance:**

⚠️ **The Entropy Trap:**
```
objA = Q(s,a) - alpha * log_prob
     = profit_signal - entropy_bonus
```

If `objA` and `etc` (entropy) keep rising (56 → 153), the agent is being rewarded for staying random rather than learning a strategy. Solutions:

1. **Increase `reward_scale`** (2.0-4.0): Amplifies profit signal to compete with entropy
2. **Lower `target_entropy`**: Reduce to `0.5 * log(action_dim)` or even negative
3. **Use reward normalization**: VecNormalize with `norm_reward=True`

**Diagnostic Signs:**
| Metric | Healthy | Entropy Trap |
|--------|---------|--------------|
| avgR | Increasing | Flat (162-170) |
| objA/etc | Stable or decreasing | Rising unbounded |
| expR | Close to avgR | Much lower than avgR |

### TD3 (Twin Delayed DDPG)

**Best for:** Deterministic policies, addresses DDPG's overestimation.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 256 | Same as SAC |
| `buffer_size` | 1e5 | Same as SAC |
| `soft_update_tau` | 5e-3 | Same as SAC |
| `policy_noise` | 0.2 | Noise added to target actions (smoothing). Higher = more robust but slower. |
| `noise_clip` | 0.5 | Clips target policy noise. Prevents extreme actions. |
| `policy_delay` | 2 | Update actor every N critic updates. Delays help stability. |
| `explore_noise` | 0.1 | Gaussian noise for exploration during training. |

**TD3 vs SAC:**
- TD3 is deterministic (no entropy) → can converge to local optima
- TD3 uses explicit noise injection vs SAC's stochastic policy
- TD3 may be more stable but less exploratory

### DDPG (Deep Deterministic Policy Gradient)

**Best for:** Simple off-policy baseline, fast training.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 256 | Same as above |
| `buffer_size` | 1e5 | Same as above |
| `soft_update_tau` | 5e-3 | Same as above |
| `explore_noise` | 0.1 | Noise for exploration |

**DDPG Issues:**
- Prone to Q-value overestimation (TD3 and SAC fix this)
- Sensitive to hyperparameters
- Generally superseded by TD3/SAC

---

## VecNormalize Settings

| Parameter | On-Policy (PPO/A2C) | Off-Policy (SAC/TD3/DDPG) |
|-----------|---------------------|---------------------------|
| `norm_obs` | ✅ True | ✅ True |
| `norm_reward` | ❌ False | ✅ True |
| `clip_obs` | 10.0 | 10.0 |
| `clip_reward` | - | 10.0 |

**Why different reward normalization?**
- **On-policy:** GAE internally normalizes advantages; external reward normalization can interfere
- **Off-policy:** No internal normalization; reward normalization helps stabilize Q-values and prevents entropy trap

---

## Recommended Settings by Agent

### PPO (Finance)
```python
net_dims = [128, 64]  # Smaller for on-policy (2048 envs provide diversity)
gamma = 0.99
learning_rate = 1e-4
horizon_len = 2048
repeat_times = 16
lambda_entropy = 0.01
lambda_gae_adv = 0.95
ratio_clip = 0.25
num_envs = 2048
# VecNormalize: norm_obs=True, norm_reward=False
```

### A2C (Finance)
```python
net_dims = [128, 64]  # Smaller for on-policy
gamma = 0.99
learning_rate = 1e-4
horizon_len = 2048
repeat_times = 8
lambda_entropy = 0.01
lambda_gae_adv = 0.95
num_envs = 2048
# VecNormalize: norm_obs=True, norm_reward=False
```

### SAC (Finance - Anti-Entropy-Trap)
```python
net_dims = [256, 128]  # Larger for off-policy (fewer envs, more capacity needed)
gamma = 0.99
learning_rate = 3e-4
batch_size = 256
buffer_size = 100000
soft_update_tau = 0.005
num_envs = 96
reward_scale = 2.0  # Increase if entropy trap persists
# VecNormalize: norm_obs=True, norm_reward=True
# Consider: ModSAC for better finance performance
```

### TD3 (Finance)
```python
net_dims = [256, 128]  # Larger for off-policy
gamma = 0.99
learning_rate = 3e-4
batch_size = 256
buffer_size = 100000
soft_update_tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
explore_noise = 0.1
num_envs = 96
# VecNormalize: norm_obs=True, norm_reward=True
```

---

## Experiment Results Summary

| Model | Normalization | Best Return | Sharpe | vs DJIA |
|-------|--------------|-------------|--------|---------|
| **PPO baseline** | none | 18.29% | 2.14 | +4.29% |
| **A2C + norm** | obs only | 16.90% | 2.56 | +2.90% |
| PPO + norm | obs only | TBD | TBD | TBD |
| SAC baseline | none | 13.22% | 1.69 | -0.78% |
| SAC + norm | obs + reward | TBD | TBD | TBD |
| A2C baseline | none | 10.28% | 1.76 | -3.72% |

**Key Findings:**
1. On-policy agents (PPO, A2C): `--best` flag correctly selects optimal checkpoint
2. Off-policy agents (SAC): Training avgR doesn't predict test performance
3. VecNormalize significantly improves A2C (+6.6% return improvement)
4. PPO baseline surprisingly strong without normalization

---

## Next Steps: ModSAC

For financial applications, consider **ModSAC** (Modified SAC) which includes:
- Reliability update mechanism
- Better handling of sparse rewards
- Reduced entropy trap susceptibility

See: `examples/demo_DDPG_TD3_SAC.py` for ModSAC implementation.
