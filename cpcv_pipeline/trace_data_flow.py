#!/usr/bin/env python3
"""
Trace the exact data flow through the env for non-contiguous CPCV training data.

Answers the question: "What does the RL agent see when fed non-contiguous
segments via save_sliced_data → NPZ → StockTradingVecEnv?"

We pick Split 2 (one of the worst gaps: ~158 trading days missing) and:
1. Show the original indices (two non-contiguous segments)
2. Show what save_sliced_data produces (concatenated array)
3. Show what the env sees at the gap boundary (obs, reward, prices)
4. Compare with/without VecNormalize
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.dirname(__file__) or '.')

import numpy as np
import torch as th
import itertools as itt

ALPACA_NPZ_PATH = 'datasets/alpaca_stock_data.numpy.npz'

# ─── Inline CPCV split generation (avoid heavy import chain) ────────────────
def _generate_cpcv_splits(total_days, n_groups=5, n_test_groups=2, embargo_days=7):
    """Generate CPCV splits directly without importing the full pipeline."""
    indices = np.arange(total_days)
    fold_bounds = [(fold[0], fold[-1] + 1) for fold in np.array_split(indices, n_groups)]
    selected = list(itt.combinations(fold_bounds, n_test_groups))
    selected.reverse()
    
    splits = []
    for fold_bound_list in selected:
        # Test indices
        test_indices = np.empty(0, dtype=int)
        for fold_start, fold_end in fold_bound_list:
            test_indices = np.union1d(test_indices, indices[fold_start:fold_end]).astype(int)
        
        # Train indices (complement minus embargo)
        train_indices = np.setdiff1d(indices, test_indices)
        
        # Apply embargo around test boundaries
        for fold_start, fold_end in fold_bound_list:
            embargo_start = max(0, fold_start - embargo_days)
            embargo_end = min(total_days, fold_end + embargo_days)
            embargo_zone = np.arange(embargo_start, embargo_end)
            train_indices = np.setdiff1d(train_indices, embargo_zone)
        
        train_indices = np.setdiff1d(train_indices, test_indices)
        splits.append((train_indices, test_indices))
    
    return splits

# ─── 1. Load full dataset and generate splits ────────────────────────────────
print("="*80)
print("STEP 1: Load full dataset")
print("="*80)

full_data = np.load(os.path.join('..', ALPACA_NPZ_PATH))
close_ary = full_data['close_ary']
tech_ary = full_data['tech_ary']
print(f"Full dataset: {close_ary.shape[0]} days, {close_ary.shape[1]} stocks")
print(f"Date range: rows 0..{close_ary.shape[0]-1}")

splits_raw = _generate_cpcv_splits(close_ary.shape[0])
splits = [{'train_indices': t, 'test_indices': v} for t, v in splits_raw]
print(f"\nGenerated {len(splits)} CPCV splits")

# ─── 2. Analyze Split 2 (non-contiguous) ─────────────────────────────────────
print("\n" + "="*80)
print("STEP 2: Analyze Split 2 train indices")
print("="*80)

split = splits[1]  # Split 2 (0-indexed)
train_idx = np.sort(split['train_indices'])
test_idx = np.sort(split['test_indices'])

print(f"Split 2 train indices: {len(train_idx)} days")
print(f"Split 2 test indices:  {len(test_idx)} days")

# Find gaps in train indices
gaps = np.where(np.diff(train_idx) > 1)[0]
print(f"\nNumber of gaps in train data: {len(gaps)}")

for g_i, g in enumerate(gaps):
    seg_end = train_idx[g]
    seg_start = train_idx[g + 1]
    gap_days = seg_start - seg_end - 1
    print(f"\n  Gap {g_i+1}:")
    print(f"    Segment 1 ends at original day {seg_end} (position {g} in sliced array)")
    print(f"    Segment 2 starts at original day {seg_start} (position {g+1} in sliced array)")
    print(f"    Gap size: {gap_days} trading days missing")
    
    # Show price jump at gap for first few stocks
    print(f"\n    Price data at gap boundary (first 5 stocks):")
    print(f"    {'Stock':>6} | {'Day '+str(seg_end):>12} | {'Day '+str(seg_start):>12} | {'% Change':>10} | {'Typical daily %':>16}")
    print(f"    {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*16}")
    
    for s in range(min(5, close_ary.shape[1])):
        price_before = close_ary[seg_end, s]
        price_after = close_ary[seg_start, s]
        pct_gap = (price_after - price_before) / price_before * 100
        
        # Typical daily change for this stock (using whole dataset)
        daily_returns = np.diff(close_ary[:, s]) / close_ary[:-1, s]
        typical_daily = np.std(daily_returns) * 100
        
        print(f"    {s:>6} | {price_before:>12.2f} | {price_after:>12.2f} | {pct_gap:>+9.1f}% | {typical_daily:>15.2f}%")

# ─── 3. What save_sliced_data produces ───────────────────────────────────────
print("\n" + "="*80)
print("STEP 3: What save_sliced_data does (concatenate non-contiguous segments)")
print("="*80)

sliced_close = close_ary[train_idx]
sliced_tech = tech_ary[train_idx]

print(f"Sliced array shape: {sliced_close.shape}")
print(f"Original train indices span: [{train_idx[0]}..{train_idx[g]}] + [{train_idx[g+1]}..{train_idx[-1]}]")
print(f"Sliced array is contiguous: rows 0..{len(train_idx)-1}")

gap_pos = gaps[0]  # Position in sliced array where gap occurs
print(f"\nIn the sliced array, the gap is at position {gap_pos} → {gap_pos+1}")
print(f"  sliced_close[{gap_pos}] = day {train_idx[gap_pos]} prices")
print(f"  sliced_close[{gap_pos+1}] = day {train_idx[gap_pos+1]} prices (jumped {train_idx[gap_pos+1] - train_idx[gap_pos]} original days)")

# ─── 4. What the env sees ────────────────────────────────────────────────────
print("\n" + "="*80)
print("STEP 4: What StockTradingVecEnv sees (step-by-step at gap)")
print("="*80)

from elegantrl.envs.StockTradingEnv import StockTradingVecEnv

# Save temp NPZ
tmp_npz = '/tmp/trace_split2_train.npz'
np.savez_compressed(tmp_npz, close_ary=sliced_close, tech_ary=sliced_tech)

env = StockTradingVecEnv(
    npz_path=tmp_npz,
    num_envs=1,
    gpu_id=-1,  # CPU
    beg_idx=0,
    end_idx=len(train_idx),
)

# Disable random reset for reproducibility
env.if_random_reset = False
state, _ = env.reset()

print(f"Env max_step: {env.max_step} (walks through {env.max_step+1} rows)")
print(f"State dim: {env.state_dim}")
print(f"Gap position in env: day {gap_pos} → {gap_pos+1}")

# Run through to gap boundary
print(f"\nStepping through env to gap boundary...")
print(f"\n{'Env Day':>8} | {'Orig Day':>9} | {'Price[0]':>10} | {'Price[0]*2^-7':>14} | {'Reward[0]':>10} | {'Note':>20}")
print(f"{'-'*8}-+-{'-'*9}-+-{'-'*10}-+-{'-'*14}-+-{'-'*10}-+-{'-'*20}")

# Show a few steps before gap, at gap, and after gap
sample_days = list(range(max(0, gap_pos - 3), min(env.max_step + 1, gap_pos + 5)))

# Use random actions every step (buy/hold/sell) to simulate a real agent
th.manual_seed(42)

for step in range(min(env.max_step, gap_pos + 5)):
    act = th.rand(1, env.action_dim) * 2 - 1  # uniform [-1, 1] each step
    state, reward, done, truncate, info = env.step(act)
    
    if step + 1 in sample_days:  # step+1 because day increments at start of step
        orig_day = train_idx[step + 1] if step + 1 < len(train_idx) else "?"
        price0 = env.close_price[env.day, 0].item()
        scaled_price = price0 * 2**-7
        rew = reward[0].item()
        
        note = ""
        if step + 1 == gap_pos:
            note = "← LAST BEFORE GAP"
        elif step + 1 == gap_pos + 1:
            note = "← FIRST AFTER GAP"
        
        print(f"{env.day:>8} | {orig_day:>9} | {price0:>10.2f} | {scaled_price:>14.6f} | {rew:>+10.6f} | {note:>20}")

# ─── 5. Reward spike analysis ────────────────────────────────────────────────
print("\n" + "="*80)
print("STEP 5: Reward distribution - normal vs gap step")
print("="*80)

# Re-run full episode to collect all rewards
env2 = StockTradingVecEnv(
    npz_path=tmp_npz,
    num_envs=1,
    gpu_id=-1,
    beg_idx=0,
    end_idx=len(train_idx),
)
env2.if_random_reset = False
env2.reset()

all_rewards = []
th.manual_seed(42)  # same seed for reproducibility
for step in range(env2.max_step):
    act = th.rand(1, env2.action_dim) * 2 - 1  # random buy/hold/sell
    _, reward, done, _, _ = env2.step(act)
    if not done:
        all_rewards.append(reward[0].item())

all_rewards = np.array(all_rewards)
gap_reward = all_rewards[gap_pos] if gap_pos < len(all_rewards) else np.nan

normal_rewards = np.concatenate([all_rewards[:gap_pos-5], all_rewards[gap_pos+5:]]) if gap_pos + 5 < len(all_rewards) else all_rewards[:gap_pos-5]

print(f"Total steps: {len(all_rewards)}")
print(f"Normal rewards (excl. ±5 days of gap):")
print(f"  mean   = {np.mean(normal_rewards):+.8f}")
print(f"  std    = {np.std(normal_rewards):.8f}")
print(f"  |max|  = {np.max(np.abs(normal_rewards)):.8f}")

print(f"\nGap step reward (day {gap_pos}→{gap_pos+1}):")
print(f"  reward = {gap_reward:+.8f}")
print(f"  ratio  = {abs(gap_reward) / np.std(normal_rewards):.1f}x std")

# ─── 6. VecNormalize impact ──────────────────────────────────────────────────
print("\n" + "="*80)
print("STEP 6: VecNormalize running stats pollution")
print("="*80)

try:
    from elegantrl.envs.vec_normalize import VecNormalize, RunningMeanStd
    
    # Simulate what happens to ret_rms with gap vs without
    # Simulate accumulation with Welford's algorithm
    
    device = th.device('cpu')
    
    # With gap
    rms_with_gap = RunningMeanStd(shape=(), device=device)
    for r in all_rewards:
        rms_with_gap.update(th.tensor([[r]], dtype=th.float32))
    
    # Without gap (replace gap reward with typical)
    rewards_no_gap = all_rewards.copy()
    rewards_no_gap[gap_pos] = np.mean(normal_rewards)
    
    rms_no_gap = RunningMeanStd(shape=(), device=device)
    for r in rewards_no_gap:
        rms_no_gap.update(th.tensor([[r]], dtype=th.float32))
    
    print(f"ret_rms WITH gap artifact:")
    print(f"  mean = {rms_with_gap.mean.item():.8f}")
    print(f"  var  = {rms_with_gap.var.item():.8f}")
    print(f"  std  = {th.sqrt(rms_with_gap.var).item():.8f}")
    
    print(f"\nret_rms WITHOUT gap artifact:")
    print(f"  mean = {rms_no_gap.mean.item():.8f}")  
    print(f"  var  = {rms_no_gap.var.item():.8f}")
    print(f"  std  = {th.sqrt(rms_no_gap.var).item():.8f}")
    
    var_inflation = (rms_with_gap.var / rms_no_gap.var).item()
    print(f"\n  Variance inflation factor: {var_inflation:.2f}x")
    print(f"  This means normal rewards get divided by {np.sqrt(var_inflation):.2f}x larger std")
    print(f"  → Normal reward signal is squished by {1/np.sqrt(var_inflation)*100:.1f}% of its true scale")

except ImportError:
    print("  [VecNormalize not available]")

# ─── 7. Observation space analysis at gap ─────────────────────────────────────
print("\n" + "="*80)
print("STEP 7: Observation vector at gap boundary")  
print("="*80)

env3 = StockTradingVecEnv(
    npz_path=tmp_npz,
    num_envs=1,
    gpu_id=-1,
    beg_idx=0,
    end_idx=len(train_idx),
)
env3.if_random_reset = False
env3.reset()

th.manual_seed(42)  # same seed for reproducibility
states_at_gap = {}

for step in range(min(env3.max_step, gap_pos + 3)):
    act = th.rand(1, env3.action_dim) * 2 - 1  # random buy/hold/sell
    state, _, done, _, _ = env3.step(act)
    if step + 1 in [gap_pos-1, gap_pos, gap_pos+1, gap_pos+2]:
        states_at_gap[step+1] = state[0].cpu().numpy().copy()

# State layout: [tanh(amount*2^-18), tanh(shares*2^-10) * num_shares, close*2^-7 * num_shares, tech*2^-6 * num_tech]
num_shares = env3.num_shares
print(f"State layout: [amount(1), shares({num_shares}), close_scaled({num_shares}), tech_scaled(?)]")

print(f"\nScaled close prices in observation (first 5 stocks):")
print(f"{'Day':>6} | {'Orig':>6} | {'close[0]*2^-7':>14} | {'close[1]*2^-7':>14} | {'close[2]*2^-7':>14} | {'close[3]*2^-7':>14} | {'close[4]*2^-7':>14}")
start_idx = 1 + num_shares  # after amount + shares
for day, obs in sorted(states_at_gap.items()):
    orig = train_idx[day] if day < len(train_idx) else "?"
    close_vals = obs[start_idx:start_idx+min(5, num_shares)]
    vals_str = " | ".join(f"{v:>14.6f}" for v in close_vals)
    label = " ← GAP" if day == gap_pos + 1 else ""
    print(f"{day:>6} | {orig:>6} | {vals_str}{label}")

# ─── 8. Compare with FinRL_Crypto approach ────────────────────────────────────
print("\n" + "="*80)
print("STEP 8: Comparison with FinRL_Crypto and RiskLabAI")
print("="*80)

print("""
FinRL_Crypto CPCV approach (from GitHub source):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. function_train_test.py:train_agent():
     price_array_train = price_array[train_indices, :]
     tech_array_train  = tech_array[train_indices, :]
   → SAME concatenation as our save_sliced_data()

2. Environment (CryptoEnvAlpaca/CryptoEnvCCXT):
   - Takes price_array directly in config dict
   - Walks linearly: self.time += 1 each step
   - get_state: state = [cash * norm_cash, stocks * norm_stocks, tech * norm_tech]
   - reward = (delta_bot - delta_eqw) * norm_reward
   → SAME linear walking, SAME gap problem

3. NO VecNormalize wrapper used:
   - Uses env_params with manual scaling: norm_cash=2^-12, norm_stocks=2^-8, 
     norm_tech=2^-15, norm_reward=2^-10
   → Gap still creates a reward spike, but it's a FIXED multiplier, not adaptive

4. Critical difference: FinRL_Crypto reward = (delta_bot - delta_eqw)
   where delta_eqw uses equal-weight portfolio that ALSO jumps at the gap.
   If bot holds stocks, both delta_bot and delta_eqw jump similarly,
   partially canceling the gap effect in the reward!

RiskLabAI CPCV approach (from GitHub source):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Uses supervised ML models (not RL):
   - estimator.fit(X_train, y_train, sample_weight=weights_train)
   - estimator.predict(X_test)
   
2. Train/test with non-contiguous indices:
   - X_train = single_data.iloc[train_indices]  
   - Each sample is INDEPENDENT (no sequential time dependency)
   - No "walking through prices" → NO gap artifact
   
3. Their CPCV implementation handles non-contiguous test sets explicitly:
   - CombinatorialPurged._single_split() passes continous_test_times=False
   - Purging is done properly against non-contiguous boundaries

Key insight: The gap problem is UNIQUE to sequential RL environments.
Supervised ML doesn't have this issue because each sample is independent.
""")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("="*80)
print("SUMMARY: Data flow trace findings")
print("="*80)

reward_std = np.std(normal_rewards) if len(normal_rewards) > 0 else 1e-10
gap_ratio = abs(gap_reward) / reward_std if reward_std > 1e-12 else float('nan')

try:
    var_infl = var_inflation
except NameError:
    var_infl = float('nan')

print(f"""
DATA FLOW:
  1. CPCV generates train_indices = [0..{train_idx[gaps[0]]}] + [{train_idx[gaps[0]+1]}..{train_idx[-1]}]
  2. save_sliced_data() sorts & concatenates → contiguous array of {len(train_idx)} rows
  3. StockTradingVecEnv loads NPZ → self.close_price tensor ({len(train_idx)} x {close_ary.shape[1]})
  4. env.reset() sets self.day = 0
  5. env.step() increments self.day += 1 linearly
  6. At env day {gap_pos}→{gap_pos+1}, prices jump from original day {train_idx[gaps[0]]}→{train_idx[gaps[0]+1]}

THE ENV'S PERSPECTIVE:
  - The env has NO knowledge of gaps. It sees a smooth array of prices.
  - At the gap boundary, it just sees an unusually large price move.
  - The reward = (total_asset_new - total_asset_old) * 2^-12
  - If the agent holds stocks, this one-step return is ~{gap_ratio:.0f}x normal std.

IMPACT ON LEARNING (without VecNormalize):
  - The gap reward is just ONE step out of ~{len(train_idx)} per episode
  - With fixed 2^-12 scaling, the gap step has a larger-than-usual reward
  - But the PPO clip ratio limits how much any single step can change the policy
  - The agent may learn "sometimes prices jump a lot" which isn't great but isn't catastrophic
  - This is IDENTICAL to FinRL_Crypto's behavior (they have the same issue)

IMPACT ON LEARNING (with VecNormalize):
  - ret_rms variance inflated by ~{var_infl:.1f}x
  - ALL OTHER normal rewards get squished by this inflated std
  - The gap step gets a "normal-looking" normalized reward
  - But the ~{len(train_idx)-1} normal steps get WEAKER signal
  - This happens EVERY episode, compounding the damage
  - Then during EVAL, these polluted stats normalize the test observations/rewards
  - → VecNormalize makes the gap problem MUCH worse than without it

RECOMMENDATION:
  - The gap in the concatenated data is EXPECTED for CPCV + RL
  - FinRL_Crypto has the SAME issue and accepts it
  - The env itself handles it reasonably (fixed scaling, one bad step per episode)
  - VecNormalize amplifies the problem via adaptive statistics → DISABLE IT
  - The env already has hardcoded scaling: amount*2^-18, shares*2^-10, close*2^-7, 
    tech*2^-6, reward*2^-12
""")

# Cleanup
os.remove(tmp_npz)
print("[Done. Temp file cleaned up.]")
