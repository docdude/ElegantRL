"""
Train and test functions for CPCV / WF / KCV pipeline.

Key design: The env always loads full data from disk and slices with beg_idx:end_idx.
To support disjoint CPCV index sets without modifying the env:
  1. Pre-slice arrays: price_array[train_indices]
  2. Save sliced arrays to a temporary .npz
  3. Point the env at the temp .npz, use beg_idx=0, end_idx=len(sliced)

This avoids ALL env modifications and matches FinRL_Crypto's approach
(price_array[train_indices, :]) while working with ElegantRL's existing arch.
"""

import os
import sys
import json
import shutil
import numpy as np
import torch as th
from datetime import datetime
from typing import Optional, Tuple

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cpcv_pipeline.config import (
    ALPACA_NPZ_PATH, RESULTS_DIR, DATA_CACHE_DIR,
    DEFAULT_ERL_PARAMS, DEFAULT_ENV_PARAMS,
    USE_VEC_NORMALIZE, VEC_NORMALIZE_KWARGS,
    RANDOM_SEED, GPU_ID,
)
from cpcv_pipeline.function_CPCV import format_segments


# ─────────────────────────────────────────────────────────────────────────────
# Data slicing for disjoint index sets
# ─────────────────────────────────────────────────────────────────────────────

def load_full_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the full (unsliced) price and tech arrays from disk."""
    if not os.path.exists(ALPACA_NPZ_PATH):
        raise FileNotFoundError(
            f"Data not found at {ALPACA_NPZ_PATH}. "
            f"Run download_and_preprocess() first."
        )
    ary = np.load(ALPACA_NPZ_PATH, allow_pickle=True)
    return ary['close_ary'], ary['tech_ary']


def load_dates_from_npz(npz_path: str = None) -> Optional[np.ndarray]:
    """Load dates_ary from an NPZ file, or None if not present.

    Parameters
    ----------
    npz_path : str, optional
        Path to NPZ.  Defaults to ALPACA_NPZ_PATH.

    Returns
    -------
    dates_ary : np.ndarray or None
        Array of date strings (YYYY-MM-DD), index-aligned with close_ary rows.
    """
    path = npz_path or ALPACA_NPZ_PATH
    if not os.path.exists(path):
        return None
    ary = np.load(path, allow_pickle=True)
    if 'dates_ary' in ary:
        return ary['dates_ary']
    return None


def save_sliced_data(
    indices: np.ndarray,
    output_path: str,
    close_ary: Optional[np.ndarray] = None,
    tech_ary: Optional[np.ndarray] = None,
    dates_ary: Optional[np.ndarray] = None,
) -> int:
    """
    Save pre-sliced arrays to a temporary .npz file.

    Parameters
    ----------
    indices : np.ndarray
        Row indices to select from the full arrays.
    output_path : str
        Path where to save the sliced .npz file.
    close_ary, tech_ary : optional
        If provided, use these instead of loading from disk.
    dates_ary : optional
        If provided, slice and include dates in the output NPZ.

    Returns
    -------
    n_rows : int
        Number of rows in the sliced arrays.
    """
    if close_ary is None or tech_ary is None:
        close_ary, tech_ary = load_full_data()

    indices = np.sort(indices)
    sliced_close = close_ary[indices]
    sliced_tech = tech_ary[indices]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {'close_ary': sliced_close, 'tech_ary': sliced_tech}
    if dates_ary is not None:
        data['dates_ary'] = dates_ary[indices]
    np.savez_compressed(output_path, **data)
    return len(indices)


def prepare_sliced_npz(
    indices: np.ndarray,
    close_ary: np.ndarray,
    tech_ary: np.ndarray,
    label: str,
) -> Tuple[str, int]:
    """
    Save pre-sliced arrays and return (npz_path, n_rows).

    This is the simpler alternative to SlicedAlpacaEnv context manager.
    Used when we want to save the file and point a new env class at it.
    """
    indices = np.sort(indices)
    npz_path = os.path.join(DATA_CACHE_DIR, f"cpcv_sliced_{label}.npz")
    n_rows = save_sliced_data(indices, npz_path, close_ary, tech_ary)
    return npz_path, n_rows


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Env setup
# ─────────────────────────────────────────────────────────────────────────────
# With npz_path now a first-class constructor kwarg on StockTradingVecEnv,
# no subclass overrides are needed.  The path is passed via env_args and
# survives forkserver pickle/unpickle because kwargs_filter includes it.

from elegantrl.envs.StockTradingEnv import StockTradingVecEnv


# ─────────────────────────────────────────────────────────────────────────────
# Train and test (like FinRL_Crypto's function_train_test.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_agent_class(model_name: str):
    """Return the agent class for a given model name."""
    model_name = model_name.lower()
    if model_name == "ppo":
        from elegantrl.agents.AgentPPO import AgentPPO
        return AgentPPO
    elif model_name == "a2c":
        from elegantrl.agents.AgentA2C import AgentA2C
        return AgentA2C
    elif model_name == "sac":
        from elegantrl.agents.AgentSAC import AgentSAC
        return AgentSAC
    elif model_name == "td3":
        from elegantrl.agents.AgentTD3 import AgentTD3
        return AgentTD3
    elif model_name == "ddpg":
        from elegantrl.agents.AgentDDPG import AgentDDPG
        return AgentDDPG
    else:
        raise ValueError(f"Unknown model name: {model_name}")


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
    use_vec_normalize: bool = USE_VEC_NORMALIZE,
    vec_normalize_kwargs: dict = None,
    continue_train: bool = False,
    cwd_suffix: str = None,
    num_workers: int = None,
) -> dict:
    """
    Train a DRL agent on one CPCV split and evaluate on the test set.

    This is the core function called for each split. It:
    1. Pre-slices arrays for train/test using fancy indexing
    2. Saves sliced arrays to separate .npz files
    3. Creates dynamic env classes pointing at each .npz
    4. Trains the agent using ElegantRL's train_agent()
    5. Returns metrics (Sharpe, return, etc.)

    Parameters
    ----------
    split_idx : int
        Index of this CPCV split (0-based).
    train_indices : np.ndarray
        Training row indices (from CombPurgedKFoldCV.split()).
    test_indices : np.ndarray
        Test row indices.
    close_ary : np.ndarray
        Full price array (all days).
    tech_ary : np.ndarray
        Full technical indicator array (all days).
    model_name : str
        DRL algorithm name (ppo, a2c, sac, etc.).
    erl_params : dict
        Agent hyperparameters (overrides DEFAULT_ERL_PARAMS).
    env_params : dict
        Environment parameters (overrides DEFAULT_ENV_PARAMS).
    cwd_base : str
        Base directory for saving checkpoints.
    gpu_id : int
        GPU device ID.
    random_seed : int
        Random seed.
    use_vec_normalize : bool
        Whether to wrap env with VecNormalize.
    vec_normalize_kwargs : dict
        VecNormalize parameters.
    continue_train : bool
        If True, resume from existing checkpoints in cwd (loads act, cri,
        optimizers, VecNormalize stats, and recorder). Skips if_remove.
    cwd_suffix : str, optional
        Subdirectory name within cwd_base.  Defaults to ``f"split_{split_idx}"``.
        Override for B-CPCV to use e.g. ``f"bag_{bag_idx}"``.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - split_idx, train_days, test_days
        - cwd (checkpoint directory)
        - final_return, sharpe, etc. (from evaluation)
    """
    from elegantrl.train.config import Config
    from elegantrl.train.run import train_agent

    if erl_params is None:
        erl_params = DEFAULT_ERL_PARAMS.copy()
    if env_params is None:
        env_params = DEFAULT_ENV_PARAMS.copy()
    if vec_normalize_kwargs is None:
        vec_normalize_kwargs = VEC_NORMALIZE_KWARGS.copy()

    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)
    n_train = len(train_indices)
    n_test = len(test_indices)

    print(f"\n{'='*60}")
    print(f"SPLIT {split_idx + 1}: Train {n_train} days, Test {n_test} days")
    print(f"  Train indices: {format_segments(train_indices)}")
    print(f"  Test indices:  {format_segments(test_indices)}")
    overlap = np.intersect1d(train_indices, test_indices)
    if len(overlap) > 0:
        raise ValueError(
            f"LEAKAGE DETECTED in split {split_idx}: "
            f"{len(overlap)} overlapping indices!"
        )
    print(f"  ✓ No leakage (0 overlapping indices)")
    print(f"{'='*60}")

    # ── 1. Save sliced data ──────────────────────────────────────────────
    split_data_dir = os.path.join(DATA_CACHE_DIR, "cpcv_splits")
    os.makedirs(split_data_dir, exist_ok=True)

    train_npz = os.path.join(split_data_dir, f"split{split_idx}_train.npz")
    test_npz = os.path.join(split_data_dir, f"split{split_idx}_test.npz")

    # Load dates from the full NPZ (if available) so sliced NPZs carry them
    dates_ary = load_dates_from_npz()

    save_sliced_data(train_indices, train_npz, close_ary, tech_ary, dates_ary=dates_ary)
    save_sliced_data(test_indices, test_npz, close_ary, tech_ary, dates_ary=dates_ary)

    print(f"  Saved train data: {train_npz} ({n_train} rows)")
    print(f"  Saved test data:  {test_npz}  ({n_test} rows)")

    # Compute dimensions from sliced data
    state_dim = close_ary.shape[1] + close_ary.shape[1] + tech_ary.shape[1] + 1
    action_dim = close_ary.shape[1]
    train_max_step = n_train - 1
    test_max_step = n_test - 1

    num_envs = env_params.get('num_envs', 2048)

    # ── 3. Build Config ──────────────────────────────────────────────────
    agent_class = get_agent_class(model_name)
    is_off_policy = model_name.lower() in ("sac", "td3", "ddpg")

    # Scale num_envs to fit GPU memory — same pattern as demo_FinRL_Alpaca_VecEnv_CV.py
    # Memory layout (multiprocessing, all on same GPU):
    #   Workers: num_workers × (horizon × num_envs × 5_tensors × float32)
    #   Learner: horizon × (num_workers × num_envs) × 5_tensors  (concatenated)
    #   Evaluator: test_horizon × num_envs × 5_tensors
    # Total variable ≈ (2 × num_workers × train_horizon + test_horizon) × num_envs × bps
    # Fixed overhead measured on RTX 2080 / CUDA 12.9:
    #   Worker CUDA context: ~445 MiB each  (includes model copy + PyTorch alloc)
    #   Learner context: ~205 MiB  (includes model + optimizer)
    #   Evaluator context: ~164 MiB
    _cpu_count = os.cpu_count() or 8
    _num_workers = min(_cpu_count // 4, 4) if not is_off_policy else min(_cpu_count // 2, 6)
    if num_workers is not None:
        _num_workers = num_workers
    if not is_off_policy:
        try:
            gpu_free, gpu_total = th.cuda.mem_get_info(gpu_id)
        except Exception:
            gpu_free = 7 * 1024**3
            gpu_total = 8 * 1024**3
        # Use actual free memory (accounts for other processes like Docker
        # containers, display servers, etc.) with a reserve for PyTorch
        # fragmentation and allocator overhead.
        # Previously used (total - 1 GiB) which caused OOM when external
        # processes (e.g. gunicorn ~848 MiB) exceeded the 1 GiB reserve.
        _reserve = 512 * 1024**2  # 512 MiB for fragmentation/allocator headroom
        usable = max(gpu_free - _reserve, gpu_total - 2 * 1024**3)
        gpu_used = gpu_total - gpu_free
        # Fixed overhead: CUDA contexts for each training process
        # Measured: workers ~445 MiB, learner ~205 MiB, evaluator ~164 MiB
        fixed_overhead = _num_workers * 445 * 1024**2 + 205 * 1024**2 + 164 * 1024**2
        buffer_budget = usable - fixed_overhead
        # 5 tensors per step per env: states, actions, rewards, logprobs, undones
        bytes_per_step = (state_dim + action_dim + 3) * 4
        # Workers + learner both hold num_workers copies; evaluator holds 1 copy
        total_steps_per_env = 2 * _num_workers * train_max_step + test_max_step
        max_envs = max(1, int(buffer_budget // (total_steps_per_env * bytes_per_step)))
        # Round down to nearest 256 for memory alignment
        max_envs = max(256, (max_envs // 256) * 256)
        if max_envs < num_envs:
            proj_gb = (fixed_overhead + max_envs * total_steps_per_env * bytes_per_step) / 1024**3
            print(f"  GPU scaling: num_envs {num_envs}→{max_envs} "
                  f"(GPU={gpu_total / 1024**3:.1f} GiB, free={gpu_free / 1024**3:.1f} GiB, "
                  f"used={gpu_used / 1024**2:.0f} MiB, horizon={train_max_step}, "
                  f"workers={_num_workers}, proj={proj_gb:.1f} GiB)")
            num_envs = max_envs

    # Set up cwd
    seed = random_seed + split_idx
    if cwd_base is None:
        cwd_base = os.path.join(
            RESULTS_DIR,
            f"CPCV_{model_name.upper()}_{random_seed}"
        )
    cwd = os.path.join(cwd_base, cwd_suffix or f"split_{split_idx}")

    env_args = {
        'env_name': 'AlpacaStockVecEnv-v1',
        'num_envs': num_envs,
        'max_step': train_max_step,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'if_discrete': False,
        'gamma': erl_params.get('gamma', 0.995),
        'beg_idx': 0,           # ALWAYS 0 — data is pre-sliced
        'end_idx': n_train,     # ALWAYS full length of sliced array
        'npz_path': train_npz,  # forkserver-safe: path serialized with env_args
        'initial_amount': env_params.get('initial_amount', 1e6),
        'max_stock': env_params.get('max_stock', 1e2),
        'cost_pct': env_params.get('cost_pct', 1e-3),
        'use_vec_normalize': use_vec_normalize,
        'vec_normalize_kwargs': vec_normalize_kwargs,
    }

    eval_env_args = env_args.copy()
    eval_env_args.update({
        'max_step': test_max_step,
        'beg_idx': 0,
        'end_idx': n_test,
        'npz_path': test_npz,  # forkserver-safe: different NPZ for eval
        # eval uses same VecNormalize but in eval mode (frozen stats)
        # run.py handles syncing stats from train env → eval env
    })

    args = Config(agent_class, StockTradingVecEnv, env_args)
    args.eval_env_class = StockTradingVecEnv
    args.eval_env_args = eval_env_args

    # Agent hyperparams — shared
    args.learning_rate = erl_params.get('learning_rate', 1e-4)
    args.gamma = erl_params.get('gamma', 0.99)
    args.clip_grad_norm = erl_params.get('clip_grad_norm', 3.0)

    if not is_off_policy:
        # On-policy (PPO / A2C)
        args.net_dims = erl_params.get('net_dims', [128, 64])
        args.batch_size = erl_params.get('batch_size', 128)
        args.break_step = erl_params.get('break_step', 1_000_000)
        args.repeat_times = erl_params.get('repeat_times', 16.0)
        args.horizon_len = train_max_step
        args.lambda_entropy = erl_params.get('lambda_entropy', 0.01)
        args.ratio_clip = erl_params.get('ratio_clip', 0.25)
        args.lambda_gae_adv = erl_params.get('lambda_gae_adv', 0.95)
        args.if_use_v_trace = erl_params.get('if_use_v_trace', True)
    else:
        # Off-policy (SAC / TD3 / DDPG)
        args.net_dims = erl_params.get('net_dims', [256, 128])
        args.batch_size = erl_params.get('batch_size', 256)
        args.break_step = erl_params.get('break_step', 500_000)
        args.repeat_times = erl_params.get('repeat_times', 2.0)
        args.horizon_len = train_max_step // 4
        args.buffer_size = erl_params.get('buffer_size', int(1e5))
        args.soft_update_tau = erl_params.get('soft_update_tau', 5e-3)

    # Disable agent's internal state normalization when VecNormalize handles it
    if use_vec_normalize:
        args.state_value_tau = 0

    # Training settings
    args.gpu_id = gpu_id
    args.random_seed = seed
    args.cwd = cwd
    args.continue_train = continue_train
    args.if_remove = not continue_train  # preserve checkpoints when resuming
    args.if_keep_save = True
    args.if_over_write = False
    args.eval_per_step = erl_params.get('eval_per_step', 5_000)
    args.eval_times = erl_params.get('eval_times', 32)
    args.num_workers = _num_workers

    # ── 4. Save split metadata ────────────────────────────────────────
    # Save to BOTH cwd (for post-training tools) and parent dir (immune to if_remove).
    # if_remove=True deletes cwd at the start of train_agent(), so the cwd copy
    # is lost until re-saved after training. The parent copy survives for tools
    # like eval_all_checkpoints that run mid-training.
    meta = {
        'split_idx': split_idx,
        'train_indices': train_indices.tolist(),
        'test_indices': test_indices.tolist(),
        'n_train': n_train,
        'n_test': n_test,
        'train_npz': train_npz,
        'test_npz': test_npz,
        'model_name': model_name,
        'seed': seed,
        'gpu_id': gpu_id,
        'use_vec_normalize': use_vec_normalize,
        'erl_params': {k: v for k, v in erl_params.items()
                      if not isinstance(v, (np.ndarray, th.Tensor))},
        'env_params': env_params,
        'timestamp': datetime.now().isoformat(),
    }
    os.makedirs(cwd, exist_ok=True)
    with open(os.path.join(cwd, 'split_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    # Also save to parent (survives if_remove)
    os.makedirs(cwd_base, exist_ok=True)
    with open(os.path.join(cwd_base, f'split_{split_idx}_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n  Starting training...")
    print(f"  CWD:        {cwd}")
    print(f"  Agent:      {model_name.upper()}")
    print(f"  Seed:       {seed}")
    print(f"  num_envs:   {num_envs}")
    print(f"  break_step: {args.break_step:,}")

    # ── 5. Train ─────────────────────────────────────────────────────────
    train_agent(args, if_single_process=False)

    # ── 6. Re-save metadata (if_remove may have deleted pre-training copy)
    os.makedirs(cwd, exist_ok=True)
    with open(os.path.join(cwd, 'split_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    # ── 7. Collect results ───────────────────────────────────────────────
    result = {
        'split_idx': split_idx,
        'cwd': cwd,
        'train_days': n_train,
        'test_days': n_test,
        'model_name': model_name,
        'seed': seed,
    }

    # Try to load best checkpoint result
    recorder_path = os.path.join(cwd, 'recorder.npy')
    if os.path.exists(recorder_path):
        recorder = np.load(recorder_path)
        if len(recorder) > 0:
            best_idx = np.argmax(recorder[:, 1])
            result['best_step'] = int(recorder[best_idx, 0])
            result['best_avgR'] = float(recorder[best_idx, 1])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Test / evaluate a trained agent on OOS data
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent_on_indices(
    checkpoint_path: str,
    indices: np.ndarray,
    close_ary: np.ndarray,
    tech_ary: np.ndarray,
    model_name: str = "ppo",
    net_dims: list = None,
    gpu_id: int = 0,
    vec_normalize_path: str = None,
    num_envs: int = 1,
) -> dict:
    """
    Evaluate a trained agent on a specific set of indices using VecEnv.

    Uses the same VecEnv class as training (not single-env) so that
    observation shapes and VecNormalize stats are consistent.

    Parameters
    ----------
    checkpoint_path : str
        Path to actor .pt / .pth file.
    indices : np.ndarray
        Row indices to evaluate on.
    close_ary, tech_ary : np.ndarray
        Full arrays.
    model_name : str
        Agent class name.
    net_dims : list
        Network dimensions.
    gpu_id : int
        GPU device.
    vec_normalize_path : str or None
        Path to vec_normalize.pt saved during training.
        If provided, wraps the eval env with VecNormalize in eval mode.
    num_envs : int
        Number of parallel environments for evaluation.

    Returns
    -------
    result : dict
        final_return, sharpe, max_drawdown, daily_returns, account_values
    """
    from elegantrl.envs.vec_normalize import VecNormalize

    if net_dims is None:
        net_dims = [128, 64]

    indices = np.sort(indices)
    n_rows = len(indices)

    # Save sliced test data
    test_npz = os.path.join(DATA_CACHE_DIR, "cpcv_splits", "_eval_temp.npz")
    save_sliced_data(indices, test_npz, close_ary, tech_ary)

    state_dim = close_ary.shape[1] * 2 + tech_ary.shape[1] + 1
    action_dim = close_ary.shape[1]
    gamma = DEFAULT_ERL_PARAMS.get('gamma', 0.995)

    env = StockTradingVecEnv(
        npz_path=test_npz,
        initial_amount=DEFAULT_ENV_PARAMS.get('initial_amount', 1e6),
        max_stock=DEFAULT_ENV_PARAMS.get('max_stock', 1e2),
        cost_pct=DEFAULT_ENV_PARAMS.get('cost_pct', 1e-3),
        gamma=gamma,
        beg_idx=0,
        end_idx=n_rows,
        num_envs=num_envs,
        gpu_id=gpu_id,
    )
    env.if_random_reset = False

    # Wrap with VecNormalize if stats file exists (eval mode = frozen stats)
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        env = VecNormalize(env, training=False)
        env.load(vec_normalize_path, verbose=True)
        print(f"    Loaded VecNormalize stats from {vec_normalize_path}")

    # Load actor
    agent_class = get_agent_class(model_name)
    device = th.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")

    try:
        # Try loading full actor object first
        actor = th.load(checkpoint_path, map_location=device, weights_only=False)
        actor.eval()
    except Exception:
        # Fall back to state_dict
        agent = agent_class(net_dims, state_dim, action_dim, gpu_id=gpu_id)
        agent.act.load_state_dict(
            th.load(checkpoint_path, map_location=device, weights_only=False)
        )
        actor = agent.act
        actor.eval()

    # Run episode — VecEnv returns tensors, use env_id=0 for metrics
    state, _ = env.reset()
    max_step = env.max_step
    account_values = [DEFAULT_ENV_PARAMS.get('initial_amount', 1e6)]

    for t in range(max_step):
        with th.no_grad():
            action = actor(state.to(device) if hasattr(state, 'to') else state)
        state, reward, terminal, truncate, info = env.step(action)

        if hasattr(env, 'total_asset'):
            account_values.append(env.total_asset[0].cpu().item())

    account_values = np.array(account_values)
    daily_returns = np.diff(account_values) / (account_values[:-1] + 1e-9)

    final_return = (account_values[-1] / account_values[0]) - 1.0
    sharpe = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-9)) * np.sqrt(252)
    max_drawdown = np.min(
        account_values / np.maximum.accumulate(account_values)
    ) - 1.0

    # Clean up temp file
    if os.path.exists(test_npz):
        os.remove(test_npz)

    return {
        'final_return': float(final_return),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'account_values': account_values.tolist(),
        'daily_returns': daily_returns.tolist(),
    }
