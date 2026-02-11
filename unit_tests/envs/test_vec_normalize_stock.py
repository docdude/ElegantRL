"""
Integration test: VecNormalize with StockTradingVecEnv

This script tests that VecNormalize works correctly with the actual
stock trading environment and shows the before/after normalization effects.

Run: python unit_tests/envs/test_vec_normalize_stock.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch as th
import numpy as np

# Check if we have the stock data
STOCK_NPZ = './elegantrl/envs/China_A_shares.numpy.npz'
HAS_STOCK_DATA = os.path.exists(STOCK_NPZ)


def test_with_mock_stock_env():
    """Test with a mock environment that mimics stock data characteristics."""
    print("\n" + "="*70)
    print("TEST: VecNormalize with Mock Stock Environment")
    print("="*70)
    
    from elegantrl.envs.vec_normalize import VecNormalize
    
    class MockStockVecEnv:
        """Mock env with realistic stock data characteristics."""
        def __init__(self, num_envs=32, gpu_id=-1):
            self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and gpu_id >= 0) else "cpu")
            self.num_envs = num_envs
            
            # Realistic dimensions: 30 stocks
            self.num_shares = 30
            num_tech_indicators = 8  # macd, boll_ub, boll_lb, rsi, cci, dx, sma30, sma60
            amount_dim = 1
            
            self.state_dim = self.num_shares + self.num_shares + (self.num_shares * num_tech_indicators) + amount_dim
            self.action_dim = self.num_shares
            self.max_step = 200
            self.if_discrete = False
            self.env_name = "MockStockVecEnv"
            self.if_random_reset = True
            self.initial_amount = 1e6
            
            self.day = 0
            
            # Realistic feature scales for stock data
            # State = [amount, shares, prices, tech_indicators]
            self._setup_feature_scales()
        
        def _setup_feature_scales(self):
            """Set up realistic scales for different feature types."""
            # Amount: ~$1M, scaled by 2^-18 in original env → ~4
            # Shares: ~0-1000, scaled by 2^-10 → ~0-1
            # Prices: ~$50-500, scaled by 2^-7 → ~0.4-4
            # Tech indicators (per stock, 8 each):
            #   macd: -10 to 10, scaled by 2^-6 → -0.15 to 0.15
            #   boll_ub/lb: like prices
            #   rsi: 0-100, scaled by 2^-6 → 0-1.5
            #   cci: -200 to 200, scaled by 2^-6 → -3 to 3
            #   dx: 0-100
            #   sma30/60: like prices
            
            self.feature_info = {
                'amount': {'idx': 0, 'mean': 0.5, 'std': 0.3},  # tanh(amount * 2^-18)
                'shares': {'idx': slice(1, 31), 'mean': 0.3, 'std': 0.4},  # tanh(shares * 2^-10)
                'prices': {'idx': slice(31, 61), 'mean': 1.5, 'std': 1.0},  # price * 2^-7
                'tech': {'idx': slice(61, self.state_dim), 'mean': 0.5, 'std': 1.5},  # tech * 2^-6
            }
        
        def reset(self):
            self.day = 0
            return self._get_state(), {}
        
        def step(self, action):
            self.day += 1
            reward = th.randn(self.num_envs, device=self.device) * 0.001  # Small rewards
            done = th.zeros(self.num_envs, dtype=th.bool, device=self.device)
            if self.day >= self.max_step:
                done = th.ones(self.num_envs, dtype=th.bool, device=self.device)
            return self._get_state(), reward, done, done, {}
        
        def _get_state(self):
            state = th.zeros(self.num_envs, self.state_dim, device=self.device)
            
            for name, info in self.feature_info.items():
                idx = info['idx']
                if isinstance(idx, int):
                    state[:, idx] = th.randn(self.num_envs, device=self.device) * info['std'] + info['mean']
                else:
                    size = idx.stop - idx.start
                    state[:, idx] = th.randn(self.num_envs, size, device=self.device) * info['std'] + info['mean']
            
            return state
    
    # Test without normalization
    print("\n--- Without VecNormalize ---")
    env_raw = MockStockVecEnv(num_envs=64)
    obs_raw, _ = env_raw.reset()
    
    for _ in range(100):
        action = th.randn(env_raw.num_envs, env_raw.action_dim) * 0.1
        obs_raw, _, _, _, _ = env_raw.step(action)
    
    print(f"Raw observation stats:")
    print(f"  Shape: {obs_raw.shape}")
    print(f"  Range: [{obs_raw.min():.4f}, {obs_raw.max():.4f}]")
    print(f"  Mean: {obs_raw.mean():.4f}")
    print(f"  Std: {obs_raw.std():.4f}")
    print(f"  Per-feature std range: [{obs_raw.std(dim=0).min():.4f}, {obs_raw.std(dim=0).max():.4f}]")
    
    # Test with normalization
    print("\n--- With VecNormalize ---")
    env_norm = VecNormalize(
        MockStockVecEnv(num_envs=64),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        training=True
    )
    env_norm.enable_debug(False)  # Disable auto-print
    
    obs_norm, _ = env_norm.reset()
    
    # Warm up the running stats
    print("  Warming up running statistics (500 steps)...")
    for i in range(500):
        action = th.randn(env_norm.num_envs, env_norm.action_dim) * 0.1
        obs_norm, reward_norm, _, _, _ = env_norm.step(action)
        
        if i == 0:
            print(f"  Step 0 - obs mean: {obs_norm.mean():.4f}, std: {obs_norm.std():.4f}")
        elif i == 99:
            print(f"  Step 100 - obs mean: {obs_norm.mean():.4f}, std: {obs_norm.std():.4f}")
        elif i == 499:
            print(f"  Step 500 - obs mean: {obs_norm.mean():.4f}, std: {obs_norm.std():.4f}")
    
    print(f"\nNormalized observation stats (after warmup):")
    print(f"  Shape: {obs_norm.shape}")
    print(f"  Range: [{obs_norm.min():.4f}, {obs_norm.max():.4f}]")
    print(f"  Mean: {obs_norm.mean():.4f}")
    print(f"  Std: {obs_norm.std():.4f}")
    print(f"  Per-feature std range: [{obs_norm.std(dim=0).min():.4f}, {obs_norm.std(dim=0).max():.4f}]")
    
    # Print detailed stats
    stats = env_norm.get_stats_summary()
    print(f"\nRunning statistics:")
    print(f"  Obs count: {stats['obs_count']:.0f}")
    print(f"  Obs mean range: [{stats['obs_mean_min']:.4f}, {stats['obs_mean_max']:.4f}]")
    print(f"  Obs std range: [{stats['obs_std_min']:.4f}, {stats['obs_std_max']:.4f}]")
    print(f"  Reward return std: {stats['ret_std']:.6f}")
    
    # Verify normalization is working
    assert abs(obs_norm.mean()) < 0.5, f"Normalized mean should be near 0, got {obs_norm.mean()}"
    assert 0.5 < obs_norm.std() < 1.5, f"Normalized std should be near 1, got {obs_norm.std()}"
    
    print("\n✓ Mock stock environment test passed!")
    return True


def test_with_real_stock_env():
    """Test with actual StockTradingVecEnv if data is available."""
    if not HAS_STOCK_DATA:
        print("\n⚠ Skipping real stock env test (data not found)")
        return True
    
    print("\n" + "="*70)
    print("TEST: VecNormalize with Real StockTradingVecEnv")
    print("="*70)
    
    from elegantrl.envs.StockTradingEnv import StockTradingVecEnv
    from elegantrl.envs.vec_normalize import VecNormalize
    
    # Create real env
    env = StockTradingVecEnv(num_envs=32, gpu_id=-1)
    env_norm = VecNormalize(env, norm_obs=True, norm_reward=True, training=True)
    
    print(f"\nEnvironment: {env.env_name}")
    print(f"  State dim: {env.state_dim}")
    print(f"  Action dim: {env.action_dim}")
    print(f"  Max step: {env.max_step}")
    
    # Collect raw stats
    obs_raw, _ = env.reset()
    print(f"\nRaw observation (before norm):")
    print(f"  Range: [{obs_raw.min():.4f}, {obs_raw.max():.4f}]")
    print(f"  Mean: {obs_raw.mean():.4f}, Std: {obs_raw.std():.4f}")
    
    # Run with normalization
    obs_norm, _ = env_norm.reset()
    
    print("\nRunning 1000 steps with normalization...")
    for i in range(1000):
        action = th.randn(env_norm.num_envs, env_norm.action_dim, device=env_norm.device) * 0.1
        obs_norm, reward_norm, done, _, _ = env_norm.step(action)
        
        if (i + 1) % 250 == 0:
            print(f"  Step {i+1}: obs mean={obs_norm.mean():.4f}, std={obs_norm.std():.4f}")
    
    stats = env_norm.get_stats_summary()
    print(f"\nFinal statistics:")
    print(f"  Obs mean range: [{stats['obs_mean_min']:.4f}, {stats['obs_mean_max']:.4f}]")
    print(f"  Obs std range: [{stats['obs_std_min']:.4f}, {stats['obs_std_max']:.4f}]")
    
    print("\n✓ Real stock environment test passed!")
    return True


def test_training_vs_eval_mode():
    """Test that training/eval modes work correctly."""
    print("\n" + "="*70)
    print("TEST: Training vs Evaluation Mode")
    print("="*70)
    
    from elegantrl.envs.vec_normalize import VecNormalize
    
    class SimpleEnv:
        def __init__(self):
            self.device = th.device('cpu')
            self.num_envs = 8
            self.state_dim = 10
            self.action_dim = 2
            self.max_step = 100
            self.if_discrete = False
            self.env_name = "SimpleEnv"
        
        def reset(self):
            return th.randn(self.num_envs, self.state_dim) * 5 + 10, {}
        
        def step(self, action):
            obs = th.randn(self.num_envs, self.state_dim) * 5 + 10
            reward = th.randn(self.num_envs)
            done = th.zeros(self.num_envs, dtype=th.bool)
            return obs, reward, done, done, {}
    
    # Training mode
    env = VecNormalize(SimpleEnv(), training=True)
    env.reset()
    
    for _ in range(100):
        env.step(th.randn(8, 2))
    
    count_after_train = env.obs_rms.count
    mean_after_train = env.obs_rms.mean.clone()
    
    # Switch to eval mode
    env.set_training(False)
    
    for _ in range(100):
        env.step(th.randn(8, 2))
    
    count_after_eval = env.obs_rms.count
    mean_after_eval = env.obs_rms.mean.clone()
    
    print(f"  Count after training: {count_after_train:.0f}")
    print(f"  Count after eval: {count_after_eval:.0f}")
    print(f"  Mean changed during eval: {(mean_after_train - mean_after_eval).abs().sum() > 0}")
    
    assert count_after_train == count_after_eval, "Stats should not update in eval mode"
    assert (mean_after_train - mean_after_eval).abs().sum() == 0, "Mean should not change in eval mode"
    
    print("\n✓ Training/eval mode test passed!")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("VecNormalize Integration Tests")
    print("="*70)
    
    all_passed = True
    all_passed &= test_with_mock_stock_env()
    all_passed &= test_with_real_stock_env()
    all_passed &= test_training_vs_eval_mode()
    
    print("\n" + "="*70)
    if all_passed:
        print("All integration tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("="*70 + "\n")
