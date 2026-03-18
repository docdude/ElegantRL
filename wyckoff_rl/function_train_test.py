"""
Wyckoff RL — Train / Test functions.

Mirrors cpcv_pipeline/function_train_test.py for the single-instrument
WyckoffTradingEnv. Key differences:
  - Single asset (action_dim=1, no portfolio rebalancing)
  - Range bars (not daily), so "days" → "bars"
  - Data is pre-sliced from close_ary/tech_ary → temp NPZ
  - GPU-vectorized WyckoffTradingVecEnv with num_envs parallel episodes
  - Multiprocessing training with GPU memory auto-scaling
"""

import os
import sys
import json
import numpy as np
import torch as th
from datetime import datetime
from typing import Optional, Tuple

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from wyckoff_rl.config import (
    WYCKOFF_NPZ_PATH, RESULTS_DIR,
    DEFAULT_ERL_PARAMS, DEFAULT_ENV_PARAMS,
    RANDOM_SEED, GPU_ID,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & slicing
# ─────────────────────────────────────────────────────────────────────────────

def load_wyckoff_data(npz_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load close_ary and tech_ary from Wyckoff NPZ."""
    path = npz_path or WYCKOFF_NPZ_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Wyckoff NPZ not found: {path}")
    data = np.load(path, allow_pickle=True)
    return data['close_ary'], data['tech_ary']


def save_sliced_data(
    indices: np.ndarray,
    output_path: str,
    close_ary: np.ndarray,
    tech_ary: np.ndarray,
) -> int:
    """Save pre-sliced arrays to a temporary NPZ."""
    indices = np.sort(indices)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        close_ary=close_ary[indices],
        tech_ary=tech_ary[indices],
    )
    return len(indices)


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────────────────────

def get_agent_class(model_name: str):
    model_name = model_name.lower()
    if model_name == "ppo":
        from elegantrl.agents.AgentPPO import AgentPPO
        return AgentPPO
    elif model_name == "sac":
        from elegantrl.agents.AgentSAC import AgentSAC
        return AgentSAC
    elif model_name == "td3":
        from elegantrl.agents.AgentTD3 import AgentTD3
        return AgentTD3
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Training a single CPCV split
# ─────────────────────────────────────────────────────────────────────────────

def train_split(
    split_idx: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    close_ary: np.ndarray,
    tech_ary: np.ndarray,
    model_name: str = "ppo",
    erl_params: dict = None,
    env_params: dict = None,
    cwd_base: str = None,
    gpu_id: int = GPU_ID,
    random_seed: int = RANDOM_SEED,
    continue_train: bool = False,
) -> dict:
    """
    Train a DRL agent on one CPCV split.

    1. Pre-slices arrays → temp NPZ files
    2. Creates WyckoffTradingVecEnv pointing at train NPZ
    3. Trains with ElegantRL's multiprocessing trainer (GPU-vectorized)
    4. Evaluates on test NPZ
    5. Returns metrics
    """
    from elegantrl.train.config import Config
    from elegantrl.train.run import train_agent
    from elegantrl.envs.WyckoffTradingEnv import WyckoffTradingVecEnv

    erl_params = {**DEFAULT_ERL_PARAMS, **(erl_params or {})}
    env_params = {**DEFAULT_ENV_PARAMS, **(env_params or {})}

    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)
    n_train = len(train_indices)
    n_test = len(test_indices)

    print(f"\n{'='*60}")
    print(f"SPLIT {split_idx}: Train {n_train} bars, Test {n_test} bars")

    # Leakage check
    overlap = np.intersect1d(train_indices, test_indices)
    if len(overlap) > 0:
        raise ValueError(f"LEAKAGE in split {split_idx}: {len(overlap)} overlapping")
    print(f"  ✓ No leakage")
    print(f"{'='*60}")

    # ── 1. Save sliced data ──────────────────────────────────────────────
    split_data_dir = os.path.join(RESULTS_DIR, "split_data")
    os.makedirs(split_data_dir, exist_ok=True)

    train_npz = os.path.join(split_data_dir, f"split{split_idx}_train.npz")
    test_npz = os.path.join(split_data_dir, f"split{split_idx}_test.npz")

    save_sliced_data(train_indices, train_npz, close_ary, tech_ary)
    save_sliced_data(test_indices, test_npz, close_ary, tech_ary)
    print(f"  Saved: train={n_train} bars, test={n_test} bars")

    # Compute dims
    n_features = tech_ary.shape[1]
    state_dim = 3 + n_features  # position + unrealized_pnl + cash + tech
    action_dim = 1
    train_max_step = n_train - 1
    test_max_step = n_test - 1

    num_envs = env_params.get('num_envs', 256)
    episode_len = env_params.get('episode_len', 4096)

    # ── 2. Build Config ──────────────────────────────────────────────────
    agent_class = get_agent_class(model_name)
    is_off_policy = model_name.lower() in ("sac", "td3")

    # Auto-scale num_envs and num_workers based on GPU memory
    _cpu_count = os.cpu_count() or 8
    _num_workers = min(_cpu_count // 4, 4) if not is_off_policy else min(_cpu_count // 2, 6)

    if not is_off_policy:
        try:
            gpu_free, gpu_total = th.cuda.mem_get_info(gpu_id)
        except Exception:
            gpu_free = 7 * 1024**3
            gpu_total = 8 * 1024**3
        _reserve = 512 * 1024**2  # 512 MiB for fragmentation/allocator headroom
        usable = max(gpu_free - _reserve, gpu_total - 2 * 1024**3)
        gpu_used = gpu_total - gpu_free
        # Fixed overhead: CUDA contexts for each training process
        # Measured: workers ~445 MiB, learner ~205 MiB, evaluator ~164 MiB
        fixed_overhead = _num_workers * 445 * 1024**2 + 205 * 1024**2 + 164 * 1024**2
        buffer_budget = (usable - fixed_overhead) * 0.75  # 25% safety margin
        # 5 tensors per step per env: states(f32), actions(f32), rewards(f32),
        # + terminals(bool), truncates(bool)
        bytes_per_step = (state_dim + action_dim + 3) * 4 + 2
        # Workers + learner both hold num_workers copies; evaluator holds 1 copy
        # PPO buffer is horizon_len (=episode_len), not full train_max_step
        eff_horizon = episode_len if episode_len < train_max_step else train_max_step
        total_steps_per_env = 2 * _num_workers * eff_horizon + test_max_step
        max_envs = max(1, int(buffer_budget // (total_steps_per_env * bytes_per_step)))
        # Round down to nearest 64 for GPU efficiency
        if max_envs >= 64:
            max_envs = (max_envs // 64) * 64
        if max_envs < num_envs:
            proj_gb = (fixed_overhead + max_envs * total_steps_per_env * bytes_per_step) / 1024**3
            print(f"  GPU scaling: num_envs {num_envs}→{max_envs} "
                  f"(GPU={gpu_total / 1024**3:.1f} GiB, free={gpu_free / 1024**3:.1f} GiB, "
                  f"used={gpu_used / 1024**2:.0f} MiB, horizon={eff_horizon}, "
                  f"workers={_num_workers}, proj={proj_gb:.1f} GiB)")
            num_envs = max_envs

    seed = random_seed + split_idx
    if cwd_base is None:
        cwd_base = os.path.join(RESULTS_DIR, f"Wyckoff_{model_name.upper()}_{random_seed}")
    cwd = os.path.join(cwd_base, f"split_{split_idx}")

    env_args = {
        'env_name': 'WyckoffTradingVecEnv-v1',
        'num_envs': num_envs,
        'max_step': train_max_step,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'if_discrete': False,
        'beg_idx': 0,
        'end_idx': n_train,
        'npz_path': train_npz,
        'initial_amount': env_params['initial_amount'],
        'cost_per_trade': env_params['cost_per_trade'],
        'gamma': erl_params.get('gamma', 0.99),
        'reward_mode': env_params.get('reward_mode', 'pnl'),
        'reward_scale': env_params.get('reward_scale', 1.0),
        'gpu_id': gpu_id,
        'episode_len': episode_len,
    }

    # Cap eval walkthrough for training speed — full eval done post-training
    eval_max_step = min(test_max_step, env_params.get('eval_max_step', 20_000))

    eval_env_args = env_args.copy()
    eval_env_args.update({
        'max_step': eval_max_step,
        'end_idx': min(n_test, eval_max_step + 1),
        'npz_path': test_npz,
        'episode_len': None,  # eval walks test data, no sub-episodes
    })

    args = Config(agent_class, WyckoffTradingVecEnv, env_args)
    args.eval_env_class = WyckoffTradingVecEnv
    args.eval_env_args = eval_env_args

    # Agent hyperparams
    args.learning_rate = erl_params['learning_rate']
    args.gamma = erl_params['gamma']
    args.clip_grad_norm = erl_params['clip_grad_norm']
    args.net_dims = erl_params['net_dims']
    args.batch_size = erl_params['batch_size']
    args.break_step = erl_params['break_step']
    args.eval_per_step = erl_params['eval_per_step']
    args.eval_times = erl_params['eval_times']

    if not is_off_policy:
        # PPO — use episode_len for fast iterations (sub-episodes w/ auto-reset)
        args.horizon_len = episode_len
        args.repeat_times = erl_params['repeat_times']
        args.lambda_entropy = erl_params['lambda_entropy']
        args.ratio_clip = erl_params['ratio_clip']
        args.lambda_gae_adv = erl_params['lambda_gae_adv']
        args.if_use_v_trace = erl_params.get('if_use_v_trace', True)
    else:
        # SAC / TD3
        args.horizon_len = train_max_step // 4
        args.repeat_times = erl_params.get('repeat_times', 2.0)
        args.buffer_size = erl_params.get('buffer_size', int(1e5))
        args.soft_update_tau = erl_params.get('soft_update_tau', 5e-3)

    # GPU-vectorized env, multiprocessing training
    args.gpu_id = gpu_id
    args.random_seed = seed
    args.cwd = cwd
    args.continue_train = continue_train
    args.if_remove = not continue_train
    args.if_keep_save = True
    args.if_over_write = False
    args.num_workers = _num_workers
    args.state_value_tau = 0  # no internal norm

    # ── 3. Save metadata ─────────────────────────────────────────────────
    meta = {
        'split_idx': split_idx,
        'n_train': n_train,
        'n_test': n_test,
        'train_npz': train_npz,
        'test_npz': test_npz,
        'model_name': model_name,
        'seed': seed,
        'num_envs': num_envs,
        'num_workers': _num_workers,
        'gpu_id': gpu_id,
        'env_params': env_params,
        'erl_params': {k: v for k, v in erl_params.items()
                       if not isinstance(v, (np.ndarray, th.Tensor))},
        'timestamp': datetime.now().isoformat(),
    }
    os.makedirs(cwd, exist_ok=True)
    with open(os.path.join(cwd, 'split_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    # Also save to parent (survives if_remove)
    os.makedirs(cwd_base, exist_ok=True)
    with open(os.path.join(cwd_base, f'split_{split_idx}_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"  CWD:        {cwd}")
    print(f"  Agent:      {model_name.upper()}")
    print(f"  state_dim:  {state_dim}")
    print(f"  num_envs:   {num_envs}")
    print(f"  num_workers: {_num_workers}")
    print(f"  Reward:     {env_params.get('reward_mode', 'pnl')}")
    print(f"  break_step: {args.break_step:,}")

    # ── 4. Train ─────────────────────────────────────────────────────────
    train_agent(args, if_single_process=False)

    # ── 5. Collect results ───────────────────────────────────────────────
    result = {
        'split_idx': split_idx,
        'cwd': cwd,
        'train_bars': n_train,
        'test_bars': n_test,
        'model_name': model_name,
        'seed': seed,
    }

    recorder_path = os.path.join(cwd, 'recorder.npy')
    if os.path.exists(recorder_path):
        recorder = np.load(recorder_path)
        if len(recorder) > 0:
            best_idx = np.argmax(recorder[:, 1])
            result['best_step'] = int(recorder[best_idx, 0])
            result['best_cumR'] = float(recorder[best_idx, 1])
            result['final_cumR'] = float(recorder[-1, 1])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    actor_path: str,
    npz_path: str,
    env_params: dict = None,
    state_dim: int = None,
    action_dim: int = 1,
    net_dims: list = None,
    gpu_id: int = 0,
) -> dict:
    """
    Deterministic evaluation of a saved actor on a test set.

    Returns metrics: total_return, sharpe, sortino, max_drawdown, n_trades.
    """
    from elegantrl.envs.WyckoffTradingEnv import WyckoffTradingEnv
    from elegantrl.agents.AgentPPO import ActorPPO

    env_params = {**DEFAULT_ENV_PARAMS, **(env_params or {})}
    net_dims = net_dims or [128, 64]

    data = np.load(npz_path, allow_pickle=True)
    n_bars = data['close_ary'].shape[0]
    n_features = data['tech_ary'].shape[1]
    if state_dim is None:
        state_dim = 3 + n_features

    device = th.device(f"cuda:{gpu_id}" if th.cuda.is_available() else "cpu")

    # Load actor
    actor = ActorPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(device)
    actor.load_state_dict(th.load(actor_path, map_location=device))
    actor.eval()

    # Build env
    env = WyckoffTradingEnv(
        npz_path=npz_path,
        beg_idx=0,
        end_idx=n_bars,
        **env_params,
    )

    state, _ = env.reset()
    portfolio_values = [env.initial_amount]
    positions = [0]
    n_trades = 0

    with th.no_grad():
        for t in range(env.max_step):
            state_t = th.tensor(state, dtype=th.float32, device=device).unsqueeze(0)
            action = actor(state_t).squeeze(0).cpu().numpy()
            old_pos = env.position
            state, reward, terminal, truncate, _ = env.step(action)
            if env.position != old_pos:
                n_trades += 1
            portfolio_values.append(env.total_asset)
            positions.append(env.position)

    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Metrics
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe = 0.0
    down = returns[returns < 0]
    sortino = returns.mean() / down.std() * np.sqrt(252) if len(down) > 0 and down.std() > 0 else 0.0
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    max_dd = drawdowns.min()

    return {
        'total_return': float(total_return),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_drawdown': float(max_dd),
        'n_trades': n_trades,
        'n_bars': n_bars,
        'positions': positions,
        'portfolio_values': portfolio_values.tolist(),
    }
