import os
import numpy as np
import pandas as pd
import torch as th
import matplotlib.pyplot as plt

from elegantrl import Config
from elegantrl import train_agent
from elegantrl.train.config import build_env
from elegantrl.agents import AgentPPO, AgentA2C
from elegantrl.agents import AgentSAC, AgentTD3, AgentDDPG
from elegantrl.envs.vec_normalize import VecNormalize

# FinRL backtest imports
try:
    from finrl.plot import backtest_stats, get_baseline, plot_return, backtest_plot
    FINRL_AVAILABLE = True
except ImportError:
    FINRL_AVAILABLE = False
    print("Warning: finrl.plot not available. Install finrl for full backtest stats.")


# =============================================================================
# AGENT REGISTRY
# =============================================================================

AGENT_REGISTRY = {
    # On-policy agents (can handle larger num_envs)
    'ppo': AgentPPO,
    'a2c': AgentA2C,
    # Off-policy agents (need reduced params for GPU memory)
    'sac': AgentSAC,
    'td3': AgentTD3,
    'ddpg': AgentDDPG,
}

ON_POLICY_AGENTS = {'ppo', 'a2c'}
OFF_POLICY_AGENTS = {'sac', 'td3', 'ddpg'}


def valid_agent_vecenv_with_backtest(env_class, env_args: dict, net_dims: list, agent_class, 
                                      actor_path: str, gpu_id: int = 0, 
                                      initial_amount: float = 1e6,
                                      baseline_ticker: str = '000001.SS',
                                      trade_start_date: str = '2020-07-01',
                                      trade_end_date: str = '2024-06-01',
                                      save_path: str = None,
                                      agent_name: str = 'PPO'):
    """
    Validate agent on VecEnv with FinRL backtest stats.
    
    Uses env_id=0 to get a representative trajectory for backtest stats.
    Also reports aggregate statistics across all VecEnv parallel runs.
    """
    env = build_env(env_class, env_args, gpu_id=gpu_id)
    
    # Disable random reset for fair backtest comparison
    env.if_random_reset = False
    
    print(f"\n{'='*60}")
    print(f"BACKTEST VALIDATION")
    print(f"{'='*60}")
    print(f"| Loading actor from: {actor_path}")
    
    # Load actor
    actor = th.load(actor_path, map_location=f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu', weights_only=False)
    actor.eval()
    
    device = env.device
    max_step = env.max_step
    num_envs = env.num_envs
    
    # Track account values for env_id=0 (representative trajectory)
    account_values = [initial_amount]
    
    # Run episode
    state, info_dict = env.reset()
    
    for t in range(max_step):
        action = actor(state.to(device))
        state, reward, terminal, truncate, info_dict = env.step(action)
        
        # Track total_asset for env 0
        # Note: VecEnv auto-resets on terminal, so we must handle the final step specially
        if hasattr(env, 'total_asset'):
            if t < max_step - 1:
                # Normal step: read total_asset directly
                account_value = env.total_asset[0].cpu().item()
                account_values.append(account_value)
            else:
                # Final step: env has reset, use cumulative_returns to compute final value
                # cumulative_returns = (final_asset / initial_amount) * 100
                if hasattr(env, 'cumulative_returns') and env.cumulative_returns is not None:
                    cr = env.cumulative_returns
                    if isinstance(cr, list):
                        final_value = initial_amount * cr[0] / 100
                    elif hasattr(cr, 'cpu'):
                        final_value = initial_amount * cr[0].cpu().item() / 100
                    else:
                        final_value = initial_amount * float(cr) / 100
                    account_values.append(final_value)
                else:
                    account_values.append(account_values[-1])  # fallback
    
    # Get final cumulative returns across all envs
    if hasattr(env, 'cumulative_returns'):
        returns = env.cumulative_returns
        if hasattr(returns, 'cpu'):
            returns = returns.cpu().numpy()
        elif isinstance(returns, list):
            returns = np.array(returns)
        else:
            returns = np.array([returns])
    else:
        returns = np.array([0.0])
    
    print(f"\n{'='*60}")
    print(f"VECENV RESULTS (across {num_envs} parallel envs)")
    print(f"{'='*60}")
    print(f"| Avg Cumulative Return: {returns.mean():.2f}%")
    print(f"| Std Cumulative Return: {returns.std():.2f}%")
    print(f"| Min Return: {returns.min():.2f}%")
    print(f"| Max Return: {returns.max():.2f}%")
    
    # Create DataFrame for FinRL backtest (single trajectory from env 0)
    # Generate trading dates
    dates = pd.bdate_range(start=trade_start_date, periods=len(account_values))
    df_account = pd.DataFrame({
        'date': dates.astype(str),
        'account_value': account_values
    })
    
    print(f"\n{'='*60}")
    print(f"BACKTEST STATISTICS (Env 0 Representative)")
    print(f"{'='*60}")
    print(f"| Trading Days: {len(account_values)}")
    print(f"| Initial Amount: ${initial_amount:,.2f}")
    print(f"| Final Amount: ${account_values[-1]:,.2f}")
    print(f"| Total Return: {(account_values[-1]/initial_amount - 1)*100:.2f}%")
    
    # Use FinRL backtest_stats
    if FINRL_AVAILABLE and len(account_values) > 2:
        print(f"\n{'='*60}")
        print(f"FINRL BACKTEST STATS")
        print(f"{'='*60}")
        try:
            perf_stats = backtest_stats(account_value=df_account)
            print(perf_stats)
        except Exception as e:
            print(f"| Warning: FinRL backtest_stats failed: {e}")
        
        # Get baseline comparison
        if baseline_ticker:
            print(f"\n{'='*60}")
            print(f"BASELINE COMPARISON: {baseline_ticker}")
            print(f"{'='*60}")
            try:
                baseline_df = get_baseline(
                    ticker=baseline_ticker,
                    start=trade_start_date,
                    end=trade_end_date
                )
                if baseline_df is not None and len(baseline_df) > 0:
                    # Normalize baseline to initial_amount
                    baseline_df['account_value'] = baseline_df['close'] / baseline_df['close'].iloc[0] * initial_amount
                    baseline_return = (baseline_df['account_value'].iloc[-1] / initial_amount - 1) * 100
                    print(f"| {baseline_ticker} Return: {baseline_return:.2f}%")
                    print(f"| Strategy vs Baseline: {(account_values[-1]/initial_amount - 1)*100 - baseline_return:+.2f}%")
                    
                    # Baseline stats
                    baseline_for_stats = pd.DataFrame({
                        'date': baseline_df['date'],
                        'account_value': baseline_df['account_value']
                    })
                    print(f"\n| {baseline_ticker} Stats:")
                    try:
                        baseline_stats = backtest_stats(account_value=baseline_for_stats)
                        print(baseline_stats)
                    except Exception as e:
                        print(f"| Warning: Baseline stats failed: {e}")
            except Exception as e:
                print(f"| Warning: Could not fetch baseline {baseline_ticker}: {e}")
    
    # Save results
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Save account values
        df_account.to_csv(f"{save_path}/account_values.csv", index=False)
        print(f"\n| Saved account values to: {save_path}/account_values.csv")
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Account value over time with baseline comparison
        ax1 = axes[0, 0]
        ax1.plot(range(len(account_values)), account_values, label=f'{agent_name.upper()} Strategy ({(account_values[-1]/initial_amount - 1)*100:.1f}%)', linewidth=2, color='blue')
        
        # Add baseline if available
        if baseline_ticker:
            try:
                baseline_df = get_baseline(
                    ticker=baseline_ticker,
                    start=df_account.loc[0, "date"],
                    end=df_account.loc[len(df_account) - 1, "date"]
                )
                if baseline_df is not None and len(baseline_df) > 0:
                    baseline_values = baseline_df['close'] / baseline_df['close'].iloc[0] * initial_amount
                    baseline_return = (baseline_values.iloc[-1] / initial_amount - 1) * 100
                    # Align baseline to strategy timeline
                    baseline_x = np.linspace(0, len(account_values)-1, len(baseline_values))
                    ax1.plot(baseline_x, baseline_values.values, label=f'{baseline_ticker} ({baseline_return:.1f}%)', linewidth=2, color='orange', linestyle='--')
            except Exception as e:
                print(f"| Warning: Could not plot baseline in 4-panel: {e}")
        
        ax1.axhline(initial_amount, color='gray', linestyle=':', alpha=0.5, label='Initial $1M')
        ax1.set_title(f'Portfolio Value: Strategy vs {baseline_ticker} Baseline')
        ax1.set_xlabel('Trading Day')
        ax1.set_ylabel('Account Value ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add annotation for alpha
        strategy_return = (account_values[-1]/initial_amount - 1)*100
        if 'baseline_return' in dir():
            alpha = strategy_return - baseline_return
            ax1.annotate(f'Alpha: {alpha:+.2f}%', xy=(0.98, 0.02), xycoords='axes fraction',
                        ha='right', va='bottom', fontsize=11, fontweight='bold',
                        color='green' if alpha > 0 else 'red')
        
        # Plot 2: Return distribution across envs
        ax2 = axes[0, 1]
        ax2.hist(returns, bins=50, edgecolor='black', alpha=0.7, color='green')
        ax2.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
        ax2.set_title(f'Return Distribution ({num_envs} envs)')
        ax2.set_xlabel('Cumulative Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Daily returns
        ax3 = axes[1, 0]
        account_arr = np.array(account_values)
        daily_returns = np.diff(account_arr) / account_arr[:-1] * 100
        ax3.bar(range(len(daily_returns)), daily_returns, alpha=0.7, color='purple')
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('Daily Returns')
        ax3.set_xlabel('Trading Day')
        ax3.set_ylabel('Daily Return (%)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Drawdown
        ax4 = axes[1, 1]
        peak = np.maximum.accumulate(account_arr)
        drawdown = (account_arr - peak) / peak * 100
        ax4.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color='red')
        ax4.set_title('Drawdown')
        ax4.set_xlabel('Trading Day')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"{save_path}/backtest_results.png"
        plt.savefig(plot_path, dpi=150)
        print(f"| Saved backtest plot to: {plot_path}")
        plt.close()
    
        # Create baseline comparison plot (saves to file, works in VS Code)
        print(f"\n==============Compare to {baseline_ticker}===========")
        try:
            baseline_df = get_baseline(
                ticker=baseline_ticker,
                start=df_account.loc[0, "date"],
                end=df_account.loc[len(df_account) - 1, "date"]
            )
            if baseline_df is not None and len(baseline_df) > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Normalize both to percentage returns from initial
                strategy_returns = (np.array(account_values) / initial_amount - 1) * 100
                baseline_returns = (baseline_df['close'] / baseline_df['close'].iloc[0] - 1) * 100
                
                # Plot
                ax.plot(range(len(strategy_returns)), strategy_returns, 
                       label=f'{agent_name.upper()} Strategy ({strategy_returns[-1]:.1f}%)', linewidth=2, color='blue')
                ax.plot(range(len(baseline_returns)), baseline_returns.values, 
                       label=f'{baseline_ticker} ({baseline_returns.iloc[-1]:.1f}%)', linewidth=2, color='orange')
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
                
                # Fill between to show alpha
                min_len = min(len(strategy_returns), len(baseline_returns))
                ax.fill_between(range(min_len), 
                               strategy_returns[:min_len], 
                               baseline_returns.values[:min_len],
                               alpha=0.2, color='green', 
                               where=strategy_returns[:min_len] > baseline_returns.values[:min_len],
                               label='Outperformance')
                ax.fill_between(range(min_len), 
                               strategy_returns[:min_len], 
                               baseline_returns.values[:min_len],
                               alpha=0.2, color='red', 
                               where=strategy_returns[:min_len] < baseline_returns.values[:min_len],
                               label='Underperformance')
                
                ax.set_title(f'{agent_name.upper()} Strategy vs {baseline_ticker} Baseline')
                ax.set_xlabel('Trading Day')
                ax.set_ylabel('Cumulative Return (%)')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                comparison_path = f"{save_path}/baseline_comparison.png"
                plt.savefig(comparison_path, dpi=150)
                print(f"| Saved baseline comparison to: {comparison_path}")
                plt.close()
        except Exception as e:
            print(f"| Warning: Could not create baseline comparison plot: {e}")

    return df_account, returns


def valid_agent_vecenv(env_class, env_args: dict, net_dims: list, agent_class, actor_path: str, gpu_id: int = 0):
    """Validate agent on VecEnv with GPU support - uses cumulative_returns from env."""
    env = build_env(env_class, env_args, gpu_id=gpu_id)
    
    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']
    
    print(f"| Validating with VecEnv on GPU:{gpu_id}")
    print(f"| Loading actor from: {actor_path}")
    
    # ElegantRL saves full actor module, not state_dict
    actor = th.load(actor_path, map_location=f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu', weights_only=False)
    actor.eval()
    
    device = env.device
    max_step = env.max_step
    
    # Run one episode through VecEnv
    state, info_dict = env.reset()
    for t in range(max_step):
        action = actor(state.to(device))
        state, reward, terminal, truncate, info_dict = env.step(action)
    
    # Get cumulative returns from the VecEnv (tracked internally)
    if hasattr(env, 'cumulative_returns'):
        returns = env.cumulative_returns
        # Handle both tensor and list formats
        if hasattr(returns, 'cpu'):
            returns = returns.cpu().numpy()
        else:
            returns = np.array(returns)
        avg_return = returns.mean()
        std_return = returns.std()
        print(f"| VecEnv Validation Results:")
        print(f"|   Num Envs: {env.num_envs}")
        print(f"|   Avg Cumulative Return: {avg_return:.4f}")
        print(f"|   Std Cumulative Return: {std_return:.4f}")
        print(f"|   Min Return: {returns.min():.4f}")
        print(f"|   Max Return: {returns.max():.4f}")
        return avg_return, std_return
    else:
        print("| WARNING: env.cumulative_returns not available")
        return None, None


def draw_learning_curve_using_recorder(cwd: str):
    recorder = np.load(f"{cwd}/recorder.npy")

    import matplotlib as mpl
    mpl.use('Agg')  # write  before `import matplotlib.pyplot as plt`. `plt.savefig()` without a running X server
    import matplotlib.pyplot as plt
    x_axis = recorder[:, 0]
    y_axis = recorder[:, 2]
    plt.plot(x_axis, y_axis)
    plt.xlabel('#samples (Steps)')
    plt.ylabel('#Rewards (Score)')
    plt.grid()

    file_path = f"{cwd}/LearningCurve.jpg"
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()
    plt.savefig(file_path)
    print(f"| Save learning curve in {file_path}")


'''run'''


def run(gpu_id: int = 0, eval_only: bool = False, checkpoint: str = None,
        agent_name: str = 'ppo', use_vec_normalize: bool = False, run_suffix: str = None):
    
    # Resolve agent class
    agent_name_lower = agent_name.lower()
    if agent_name_lower not in AGENT_REGISTRY:
        print(f"❌ Unknown agent: {agent_name}")
        print(f"   Available agents: {list(AGENT_REGISTRY.keys())}")
        return
    
    agent_class = AGENT_REGISTRY[agent_name_lower]
    is_off_policy = agent_name_lower in OFF_POLICY_AGENTS
    
    from elegantrl.envs.StockTradingEnv import StockTradingVecEnv
    
    # Data split indices
    id0 = 0
    id1 = int(1113 * 0.8)  # ~890 training days
    id2 = 1113              # ~223 test days

    gamma = 0.99
    env_class = StockTradingVecEnv
    max_step = id2 - id1 - 1  # ~222 days
    
    # Configure based on agent type
    if is_off_policy:
        # Off-policy: SAC, TD3, DDPG - reduced params to fit GPU memory
        num_envs = 256
        batch_size = 256
        horizon_len = 256
        buffer_size = int(1e5)  # 100K
        repeat_times = 1
        learning_rate = 1e-4
        net_dims = [256, 128]
        state_value_tau = 0
        soft_update_tau = 5e-3
        break_step = int(5e5)
        print(f"   ⚡ Off-policy mode: num_envs={num_envs}, buffer_size={buffer_size:,}")
    else:
        # On-policy: PPO, A2C
        num_envs = 2 ** 11  # 2048
        batch_size = None
        horizon_len = max_step
        buffer_size = None
        repeat_times = 16
        learning_rate = 1e-4  # Reduced from 2e-4 for stability
        net_dims = [128, 64]
        state_value_tau = 0.1  # Will be overridden to 0 if using VecNormalize
        lambda_entropy = 0.01  # Entropy coefficient for exploration
        break_step = int(1e6)
        print(f"   🚀 On-policy mode: num_envs={num_envs}, lr={learning_rate}, entropy={lambda_entropy}")
    
    # If using VecNormalize, disable agent's internal state normalization to avoid double-normalization
    if use_vec_normalize and state_value_tau > 0:
        print(f"   ⚠️  Disabling state_value_tau (was {state_value_tau}) - VecNormalize handles normalization")
        state_value_tau = 0
    
    # VecNormalize settings - different for on-policy vs off-policy agents
    # On-policy (PPO/A2C): Normalize both obs and rewards - data is consumed once and discarded,
    #   ElegantRL's env already scales rewards (reward_scale=2^-12), so VecNormalize norm_reward
    #   on top of that creates double-normalization for on-policy agents, causing GAE noise amplification.
    # Off-policy (SAC/TD3/DDPG): norm_reward=True provides variance reduction for Q-learning
    #   which is self-consistent regardless of reward scale.
    norm_reward = is_off_policy  # Off-policy: extra normalization helps Q-learning; On-policy: env scaling sufficient
    
    vec_normalize_kwargs = {
        'norm_obs': True,
        'norm_reward': norm_reward,
        'clip_obs': 10.0,
        'clip_reward': 10.0 if norm_reward else None,
        'gamma': gamma,
        'training': True,
    }
    
    if use_vec_normalize:
        if is_off_policy:
            print(f"   📊 VecNormalize: norm_obs=True, norm_reward=False (off-policy: stale replay buffer)")
        else:
            print(f"   📊 VecNormalize: norm_obs=True, norm_reward=True (on-policy: data consumed once)")
    
    env_args = {
        'env_name': 'StockTradingVecEnv-v2',
        'num_envs': num_envs,
        'max_step': max_step,
        'state_dim': 151,
        'action_dim': 15,
        'if_discrete': False,
        'gamma': gamma,
        'beg_idx': id0,
        'end_idx': id1,
        # VecNormalize: running observation/reward normalization
        'use_vec_normalize': use_vec_normalize,
        'vec_normalize_kwargs': vec_normalize_kwargs,
    }

    args = Config(agent_class, env_class, env_args)

    args.gpu_id = gpu_id
    args.random_seed = (args.random_seed or 0) + gpu_id + 1943

    # Custom checkpoint directory with optional suffix
    base_cwd = f'./StockTradingVecEnv-v2_{agent_class.__name__[5:]}_{args.random_seed}'
    if run_suffix:
        args.cwd = f'{base_cwd}_{run_suffix}'
        print(f"   📁 Checkpoint dir: {args.cwd}")
    
    # SAFETY: Don't auto-remove existing checkpoints
    args.if_remove = False

    args.break_step = break_step
    args.net_dims = net_dims
    args.gamma = gamma
    args.horizon_len = horizon_len
    args.repeat_times = repeat_times
    args.learning_rate = learning_rate
    args.state_value_tau = state_value_tau
    args.reward_scale = 2 ** 0

    # Off-policy specific overrides
    if is_off_policy:
        args.batch_size = batch_size
        args.buffer_size = buffer_size
        args.soft_update_tau = soft_update_tau
    else:
        # On-policy specific settings (PPO/A2C)
        args.lambda_entropy = lambda_entropy

    args.eval_times = 2 ** 14
    args.eval_per_step = int(2e4)
    args.num_workers = 2
    args.eval_env_class = StockTradingVecEnv
    args.eval_env_args = {
        'env_name': 'StockTradingVecEnv-v2',
        'num_envs': num_envs,
        'max_step': max_step,
        'state_dim': 151,
        'action_dim': 15,
        'if_discrete': False,
        'beg_idx': id1,
        'end_idx': id2,
        # VecNormalize for eval env - uses same settings but won't update stats
        'use_vec_normalize': use_vec_normalize,
        'vec_normalize_kwargs': {**vec_normalize_kwargs, 'training': False},
    }

    # Train or skip
    if not eval_only:
        train_agent(args, if_single_process=False)
    else:
        print("| Skipping training (eval-only mode)")
    
    # Find checkpoint
    actor_path = None
    if checkpoint:
        if checkpoint.endswith('.pt'):
            actor_path = checkpoint
        elif os.path.isdir(checkpoint):
            pt_files = sorted([s for s in os.listdir(checkpoint)
                               if s.endswith('.pt') and s.startswith('actor')])
            if pt_files:
                actor_path = f"{checkpoint}/{pt_files[-1]}"
    
    # Initialize save_path
    save_path = None
    
    if not eval_only:
        if input("| Press 'y' to load actor.pt and validate:") != 'y':
            return
        if not actor_path:
            pt_files = sorted([s for s in os.listdir(args.cwd)
                               if s.endswith('.pt') and s.startswith('actor')])
            if not pt_files:
                print(f"❌ No actor .pt checkpoint files found in {args.cwd}")
                print(f"   Files present: {os.listdir(args.cwd)}")
                return
            actor_path = f"{args.cwd}/{pt_files[-1]}"
        save_path = args.cwd
    else:
        if not actor_path:
            # Try to find latest checkpoint directory
            import glob
            pattern = f"./StockTradingVecEnv-v2_{agent_class.__name__[5:]}_*"
            dirs = sorted(glob.glob(pattern))
            if dirs:
                pt_files = sorted([s for s in os.listdir(dirs[-1])
                                   if s.endswith('.pt') and s.startswith('actor')])
                if pt_files:
                    actor_path = f"{dirs[-1]}/{pt_files[-1]}"
                    save_path = dirs[-1]
        
        if not actor_path:
            print(f"❌ No checkpoint found! Provide --checkpoint path")
            print(f"   Example: --checkpoint ./StockTradingVecEnv-v2_{agent_class.__name__[5:]}_1943/actor.pt")
            return
    
    # Determine save path from checkpoint if not set
    if checkpoint and not save_path:
        if checkpoint.endswith('.pt'):
            save_path = os.path.dirname(checkpoint)
        else:
            save_path = checkpoint
    
    print(f"| Loading: {actor_path}")
    
    # Validate on HELD-OUT TEST DATA ONLY (not training data)
    val_env_args = env_args.copy()
    val_env_args['beg_idx'], val_env_args['end_idx'] = (id1, id2)  # Test data only!
    val_env_args['max_step'] = id2 - id1 - 1
    
    # Calculate approximate test dates
    test_start_date = '2024-01-01'
    test_end_date = '2024-12-01'
    
    print(f"| Test data range: idx {id1} to {id2} ({id2-id1} trading days)")
    print(f"| NOTE: Evaluating on HELD-OUT test data only (not seen during training)")
    
    # Run comprehensive backtest with FinRL stats
    valid_agent_vecenv_with_backtest(
        env_class, val_env_args, args.net_dims, agent_class, actor_path, 
        gpu_id=gpu_id, 
        initial_amount=1e6,
        baseline_ticker='000001.SS',  # Shanghai Composite Index
        trade_start_date=test_start_date,
        trade_end_date=test_end_date,
        save_path=save_path,
        agent_name=agent_name
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ElegantRL China A-shares VecEnv Demo')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--agent', type=str, default='ppo', 
                        choices=list(AGENT_REGISTRY.keys()),
                        help='Agent to use: ppo, a2c (on-policy) | sac, td3, ddpg (off-policy)')
    parser.add_argument('--eval', action='store_true', dest='eval_only',
                        help='Skip training, run validation only')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint dir or .pt file (for --eval mode)')
    parser.add_argument('--normalize', action='store_true', dest='use_vec_normalize',
                        help='Enable VecNormalize for obs/reward normalization')
    parser.add_argument('--suffix', type=str, default=None,
                        help='Suffix to append to checkpoint directory (e.g., v2, norm)')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"ElegantRL VecEnv {'Validation' if args.eval_only else 'Training'}")
    print(f"{'='*60}")
    print(f"Agent: {args.agent.upper()}")
    print(f"GPU: {args.gpu}")
    print(f"VecNormalize: {'Enabled' if args.use_vec_normalize else 'Disabled'}")
    if args.eval_only:
        print(f"Mode: Eval-only")
        if args.checkpoint:
            print(f"Checkpoint: {args.checkpoint}")
    else:
        print(f"Available agents: {list(AGENT_REGISTRY.keys())}")
    
    run(gpu_id=args.gpu, eval_only=args.eval_only, checkpoint=args.checkpoint,
        agent_name=args.agent, use_vec_normalize=args.use_vec_normalize, 
        run_suffix=args.suffix)
