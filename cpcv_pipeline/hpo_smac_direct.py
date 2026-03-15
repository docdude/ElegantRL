#!/usr/bin/env python3
"""
Direct SMAC3 HPO with Instance-Based Cross-Validation.

Each CPCV/ACPCV fold maps to a SMAC *instance*, enabling:
  - Per-split persistence in runhistory.json (crash recovery)
  - Early termination of unpromising configs (after 3-4 splits, not all 10)
  - Resume from last completed instance on restart (overwrite=False)
  - Estimated 2.5-3× speedup vs full-evaluation approach

Architecture:
    SMAC's Intensifier decides which (config, instance/split) to evaluate next.
    Bad configs are killed early after poor performance on initial splits.
    Each split evaluation is independently recorded in runhistory.json.

    Replaces: Hydra + Hypersweeper (atomic 10-split evaluation, no crash recovery)
    Keeps:    train_split(), evaluate_agent_on_indices(), composite objective

Usage:
    # New run (overwrite any existing SMAC state)
    python -m cpcv_pipeline.hpo_smac_direct --overwrite

    # Resume crashed run (automatic — picks up from runhistory.json)
    python -m cpcv_pipeline.hpo_smac_direct

    # Custom config
    python -m cpcv_pipeline.hpo_smac_direct --config configs/cpcv_ppo_acpcv.yaml

References:
    - SMAC3 intensify_crossvalidation.py example (instances = CV folds)
    - López de Prado (2018): CPCV — "Advances in Financial Machine Learning"
    - Bailey et al. (2014): PBO — "The Probability of Backtest Overfitting"
"""

import argparse
import os
import sys
import random
import json
import re as _re
import time as _time
import yaml

import numpy as np
import torch as th

from pathlib import Path
from typing import Dict

from ConfigSpace import ConfigurationSpace, Float, Categorical, Configuration
from smac import HyperparameterOptimizationFacade, Scenario
from smac.intensifier.intensifier import Intensifier

# ── Project imports ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from cpcv_pipeline.function_CPCV import (
    CombPurgedKFoldCV,
    AdaptiveCombPurgedKFoldCV,
    BaggedCombPurgedKFoldCV,
    verify_no_leakage,
    compute_external_feature,
)
from cpcv_pipeline.function_train_test import (
    load_full_data,
    train_split,
    evaluate_agent_on_indices,
)
from cpcv_pipeline.hpo_cpcv import (
    set_all_seeds,
    get_cv_splits,
    compute_equal_weight_benchmark,
    ALPHA_WEIGHT,
)

# ── Constants ──
ON_POLICY_AGENTS = {'ppo', 'a2c'}
OFF_POLICY_AGENTS = {'sac', 'modsac', 'td3', 'ddpg'}
NET_ARCH_MAP = {
    "small": [64, 64],
    "medium": [256, 128],
    "large": [512, 256],
    "big": [400, 300],
}


# =============================================================================
# CONFIGSPACE BUILDER
# =============================================================================

def build_configspace(space_path: str, defaults: dict) -> ConfigurationSpace:
    """
    Build SMAC ConfigSpace from Hypersweeper-format search space YAML.

    Default values come from the base config dict (not from the YAML's
    ``${interpolation}`` references, which require Hydra to resolve).
    """
    with open(space_path) as f:
        raw = yaml.safe_load(f)

    cs = ConfigurationSpace(seed=0)
    hps = []

    for name, spec in raw['hyperparameters'].items():
        default = defaults.get(name)

        if spec['type'] == 'uniform_float':
            hp = Float(
                name,
                (float(spec['lower']), float(spec['upper'])),
                log=spec.get('log', False),
                default=float(default) if default is not None else None,
            )
            hps.append(hp)

        elif spec['type'] == 'categorical':
            choices = list(spec['choices'])
            # Convert booleans → strings for ConfigSpace compatibility
            has_bool = any(isinstance(c, bool) for c in choices)
            if has_bool:
                choices = [str(c).lower() for c in choices]
                if default is not None:
                    default = str(default).lower()
            hp = Categorical(name, choices, default=default)
            hps.append(hp)

    cs.add(hps)
    return cs


# =============================================================================
# TARGET FUNCTION FACTORY
# =============================================================================

def make_target_function(
    cfg: dict,
    splits: list,
    hpo_close: np.ndarray,
    hpo_tech: np.ndarray,
    output_dir: str,
):
    """
    Build the SMAC target function (closure).

    Returns a callable with signature:
        target_fn(config, instance, seed) -> float  (cost; SMAC minimizes)

    Each call trains ONE split, evaluates ALL checkpoints, and returns
    the negative composite objective (SR + 0.1 × alpha).
    """
    agent_name = cfg['agent'].lower()
    gpu_id = cfg.get('gpu_id', 0)
    training_seed = cfg.get('seed', 1943)
    is_off_policy = agent_name in OFF_POLICY_AGENTS
    use_vec_normalize = cfg.get('use_vec_normalize', False)
    norm_obs = cfg.get('norm_obs', True)
    norm_reward = cfg.get('norm_reward', False)
    num_workers_cfg = cfg.get('num_workers', None)
    break_step = cfg.get('break_step', int(5e5))
    num_envs = cfg.get('num_envs', 896)

    # PBO persistence directory
    pbo_dir = os.path.join(output_dir, "pbo_returns")
    os.makedirs(pbo_dir, exist_ok=True)

    # ── Config → trial_id mapping (persisted for resume) ──
    config_map_path = os.path.join(output_dir, "config_trial_map.json")

    if os.path.exists(config_map_path):
        with open(config_map_path) as f:
            config_trial_map: Dict[str, int] = json.load(f)
    else:
        config_trial_map = {}

    # Resume: find highest existing trial_id
    existing_ids = []
    if os.path.isdir(pbo_dir):
        for fname in os.listdir(pbo_dir):
            if fname.startswith("trial_") and fname.endswith("_summary.json"):
                try:
                    existing_ids.append(int(fname.split("_")[1]))
                except (ValueError, IndexError):
                    pass
    # Also include IDs from the config map
    if config_trial_map:
        existing_ids.extend(config_trial_map.values())
    next_trial_id = [max(existing_ids, default=-1) + 1]  # mutable for closure

    def target_fn(config: Configuration, instance: str, seed: int = 0) -> float:
        split_idx = int(instance)
        train_indices, val_indices = splits[split_idx]
        n_val = len(val_indices)
        if n_val < 3:
            print(f"  Split {split_idx}: skipping (val too short: {n_val})")
            return 10.0  # high cost = bad config

        # ── Assign trial_id (consistent across resume) ──
        config_key = json.dumps(dict(config), sort_keys=True, default=str)
        if config_key not in config_trial_map:
            config_trial_map[config_key] = next_trial_id[0]
            next_trial_id[0] += 1
            # Persist map for crash recovery
            with open(config_map_path, 'w') as f:
                json.dump(config_trial_map, f, indent=2)
        trial_id = config_trial_map[config_key]

        # ── Build hyperparams from SMAC config ──
        net_arch = config.get("net_arch", "small")
        net_dims = NET_ARCH_MAP.get(net_arch, [64, 64])
        learning_rate = float(config.get("learning_rate", 1e-4))
        gamma = float(config.get("gamma", 0.99))
        batch_size = int(config.get("batch_size", 512))
        repeat_times = int(config.get("repeat_times", 16))
        clip_grad_norm = float(config.get("clip_grad_norm", 3.0))

        erl_params = {
            'net_dims': net_dims,
            'learning_rate': learning_rate,
            'gamma': gamma,
            'batch_size': batch_size,
            'repeat_times': repeat_times,
            'clip_grad_norm': clip_grad_norm,
            'break_step': break_step,
        }

        if not is_off_policy:
            erl_params['ratio_clip'] = float(config.get('ratio_clip', 0.25))
            erl_params['lambda_gae_adv'] = float(
                config.get('lambda_gae_adv', 0.95)
            )
            erl_params['lambda_entropy'] = float(
                config.get('lambda_entropy', 0.01)
            )
            v_trace = config.get('if_use_v_trace', 'true')
            erl_params['if_use_v_trace'] = (
                v_trace == 'true' if isinstance(v_trace, str)
                else bool(v_trace)
            )
        else:
            erl_params['buffer_size'] = int(cfg.get('buffer_size', int(1e5)))
            erl_params['soft_update_tau'] = float(
                cfg.get('soft_update_tau', 5e-3)
            )

        env_params = {
            'num_envs': num_envs,
            'initial_amount': 1e6,
            'max_stock': 100,
            'cost_pct': 1e-3,
        }

        vec_normalize_kwargs = {
            'norm_obs': norm_obs,
            'norm_reward': norm_reward,
            'clip_obs': 10.0,
            'clip_reward': None,
            'gamma': gamma,
            'training': True,
        }

        # Per-trial, per-split working directory
        trial_cwd = os.path.join(output_dir, f"trial_{trial_id:04d}")

        print(f"\n{'='*60}")
        print(f"Trial {trial_id} | Split {split_idx}/{len(splits)} | "
              f"{agent_name.upper()} | net={net_dims} lr={learning_rate:.1e}")
        print(f"{'='*60}")

        # ── Train ──
        set_all_seeds(training_seed)
        t0 = _time.time()

        result = train_split(
            split_idx=split_idx,
            train_indices=train_indices,
            test_indices=val_indices,
            close_ary=hpo_close,
            tech_ary=hpo_tech,
            model_name=agent_name,
            erl_params=erl_params,
            env_params=env_params,
            cwd_base=trial_cwd,
            gpu_id=gpu_id,
            random_seed=training_seed,
            use_vec_normalize=use_vec_normalize,
            vec_normalize_kwargs=vec_normalize_kwargs,
            continue_train=False,
            num_workers=num_workers_cfg,
        )
        train_time = _time.time() - t0

        # ── Evaluate ALL checkpoints → pick best by composite ──
        split_cwd = result.get('cwd', '')
        sharpe_agent = 0.0
        agent_return = 0.0

        bench = compute_equal_weight_benchmark(hpo_close, val_indices)
        sharpe_bench = bench['sharpe']
        bench_return = bench['total_return']

        if split_cwd and os.path.isdir(split_cwd):
            _ckpt_pattern = _re.compile(
                r'^actor__(\d+)(?:_(\d+\.\d+))?\.pt$'
            )
            all_ckpts = []
            for f in os.listdir(split_cwd):
                m = _ckpt_pattern.match(f)
                if m:
                    all_ckpts.append({
                        'path': os.path.join(split_cwd, f),
                        'filename': f,
                        'step': int(m.group(1)),
                    })
            act_path = os.path.join(split_cwd, 'act.pth')
            if os.path.exists(act_path):
                max_step = max((c['step'] for c in all_ckpts), default=0)
                all_ckpts.append({
                    'path': act_path,
                    'filename': 'act.pth',
                    'step': max_step + 1,
                })

            vec_norm_path = (
                os.path.join(split_cwd, 'vec_normalize.pt')
                if use_vec_normalize else None
            )

            best_composite = -float('inf')
            best_eval = None
            best_ckpt_name = None
            n_ckpts = len(all_ckpts)
            print(f"  Evaluating {n_ckpts} checkpoints on val fold...")

            for ckpt in all_ckpts:
                try:
                    eval_result = evaluate_agent_on_indices(
                        checkpoint_path=ckpt['path'],
                        indices=val_indices,
                        close_ary=hpo_close,
                        tech_ary=hpo_tech,
                        model_name=agent_name,
                        net_dims=net_dims,
                        gpu_id=gpu_id,
                        vec_normalize_path=vec_norm_path,
                        num_envs=1,
                    )
                    sr = eval_result['sharpe']
                    ret_pct = eval_result['final_return'] * 100
                    _alpha = ret_pct / 100.0 - bench_return
                    _comp = sr + ALPHA_WEIGHT * _alpha
                    if _comp > best_composite:
                        best_composite = _comp
                        best_eval = eval_result
                        best_ckpt_name = ckpt['filename']
                except Exception as e:
                    print(f"    {ckpt['filename']}: eval failed ({e})")
                    continue

            if best_eval is not None:
                sharpe_agent = best_eval['sharpe']
                agent_return = best_eval['final_return'] * 100

                # Save PBO daily returns
                daily_rets = np.array(best_eval['daily_returns'])
                npy_path = os.path.join(
                    pbo_dir, f"trial_{trial_id}_split_{split_idx}.npy"
                )
                np.save(npy_path, daily_rets)
                print(f"  Best: {best_ckpt_name} "
                      f"(SR={sharpe_agent:.4f}, ret={agent_return:.1f}%)")
            else:
                print(f"  Split {split_idx}: all evals failed, using SR=0")

        # ── Composite objective ──
        alpha = agent_return / 100.0 - bench_return
        composite = sharpe_agent + ALPHA_WEIGHT * alpha

        print(f"  Split {split_idx}: composite={composite:.4f}, "
              f"SR={sharpe_agent:.4f}, alpha={alpha:+.4f}, "
              f"bench_SR={sharpe_bench:.4f} | {train_time:.0f}s")

        # Free GPU memory between splits
        th.cuda.empty_cache()

        # SMAC minimizes cost → return negative composite
        return -composite

    return target_fn


# =============================================================================
# RESULTS EXTRACTION
# =============================================================================

def extract_results(smac, incumbent, n_splits: int) -> dict:
    """Extract per-split results for the incumbent from SMAC's runhistory."""
    rh = smac.runhistory

    # Iterate runhistory to find incumbent's per-instance costs
    per_split_composite = {}
    try:
        for trial_key in rh:
            trial_value = rh[trial_key]
            config = rh.get_config(trial_key.config_id)
            if config == incumbent:
                instance = trial_key.instance
                per_split_composite[instance] = float(-trial_value.cost)
    except Exception as e:
        print(f"Warning: could not extract per-split results: {e}")

    if per_split_composite:
        mean_composite = float(np.mean(list(per_split_composite.values())))
    else:
        # Fallback: try average_cost API
        try:
            mean_composite = float(-rh.average_cost(incumbent))
        except Exception:
            mean_composite = 0.0

    return {
        'incumbent': {k: str(v) for k, v in dict(incumbent).items()},
        'mean_composite': mean_composite,
        'per_split_composite': per_split_composite,
        'n_splits_evaluated': len(per_split_composite),
        'n_splits_total': n_splits,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Direct SMAC3 HPO with Instance-Based Cross-Validation"
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/cpcv_ppo_acpcv.yaml",
        help="Config YAML path (relative to cpcv_pipeline/)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="Override number of configs to evaluate (total budget = n_trials × n_splits)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing SMAC output (default: resume)",
    )
    parser.add_argument(
        "--space", type=str, default=None,
        help="Search space YAML path (default: auto-detect from config)",
    )
    args = parser.parse_args()

    # ── Load config ──
    config_path = _SCRIPT_DIR / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Extract Hydra/SMAC settings before removing
    hydra_cfg = cfg.pop('hydra', {})
    defaults = cfg.pop('defaults', [])

    smac_scenario = (
        hydra_cfg.get('sweeper', {})
        .get('sweeper_kwargs', {})
        .get('optimizer_kwargs', {})
        .get('scenario', {})
    )
    smac_seed = smac_scenario.get('seed', 1493)
    n_trials_cfg = hydra_cfg.get('sweeper', {}).get('n_trials', 50)

    # Determine search space path
    if args.space:
        space_path = _SCRIPT_DIR / args.space
    else:
        space_name = "ppo_space"
        for d in (defaults or []):
            if isinstance(d, dict) and 'search_space' in d:
                space_name = d['search_space']
                break
        space_path = config_path.parent / "search_space" / f"{space_name}.yaml"

    # ── Setup ──
    agent_name = cfg['agent'].lower()
    seed = cfg.get('seed', 1943)
    cv_method = cfg.get('cv_method', 'acpcv')

    set_all_seeds(seed)

    # ── Load data ──
    print("Loading data...")
    close_ary, tech_ary = load_full_data()
    num_days_total = close_ary.shape[0]

    hpo_pool_end = int(num_days_total * (1.0 - cfg.get('test_ratio', 0.2)))
    hpo_close = close_ary[:hpo_pool_end]
    hpo_tech = tech_ary[:hpo_pool_end]

    print(f"\n{'='*60}")
    print(f"SMAC Direct HPO | {agent_name.upper()} | {cv_method.upper()}")
    print(f"{'='*60}")
    print(f"  Total days: {num_days_total}, HPO pool: {hpo_pool_end}")
    print(f"  Seed: {seed}")

    # ── Generate CV splits ──
    external_feature = None
    if cv_method == 'acpcv':
        external_feature = compute_external_feature(
            hpo_close,
            feature_name=cfg.get('acpcv_feature', 'drawdown'),
            window=cfg.get('acpcv_window', 63),
            tech_ary=hpo_tech,
        )

    splits = get_cv_splits(
        total_days=hpo_pool_end,
        cv_method=cv_method,
        n_folds=cfg.get('n_folds', 3),
        n_groups=cfg.get('n_groups', 5),
        n_test_groups=cfg.get('n_test_groups', 2),
        gap_days=cfg.get('gap_days', 7),
        embargo_days=cfg.get('embargo_days', 7),
        external_feature=external_feature,
        n_subsplits=cfg.get('n_subsplits', 3),
        lower_quantile=cfg.get('lower_quantile', 0.25),
        upper_quantile=cfg.get('upper_quantile', 0.75),
    )
    n_splits = len(splits)

    # ── Build ConfigSpace ──
    cs = build_configspace(str(space_path), cfg)
    print(f"\nConfigSpace: {len(cs)} hyperparameters")
    print(cs)

    # ── Output directory ──
    output_dir = os.path.join(
        str(_SCRIPT_DIR / "train_results"),
        f"hpo_smac_{cv_method}_{agent_name}_seed{seed}",
    )
    smac_output_dir = os.path.join(output_dir, "smac_output")
    os.makedirs(output_dir, exist_ok=True)

    # ── SMAC Scenario ──
    # Each CV split = one instance. SMAC evaluates configs across instances
    # with intensification (early termination of bad configs).
    instances = [str(i) for i in range(n_splits)]
    instance_features = {str(i): [float(i)] for i in range(n_splits)}

    # Budget: n_trials configs × n_splits instances = total evaluations.
    # With early termination, actual evaluations << budget → more exploration.
    n_trials_per_config = n_splits
    n_configs = args.n_trials or n_trials_cfg
    total_budget = n_configs * n_trials_per_config

    scenario = Scenario(
        configspace=cs,
        instances=instances,
        instance_features=instance_features,
        deterministic=True,
        n_trials=total_budget,
        seed=smac_seed,
        output_directory=smac_output_dir,
        n_workers=1,
    )

    # ── Intensifier: evaluate up to n_splits instances per config ──
    intensifier = Intensifier(
        scenario=scenario,
        max_config_calls=n_splits,
        max_incumbents=10,
    )

    # ── Target function ──
    target_fn = make_target_function(
        cfg=cfg,
        splits=splits,
        hpo_close=hpo_close,
        hpo_tech=hpo_tech,
        output_dir=output_dir,
    )

    # ── Initial design: cap at reasonable size ──
    # Default is 10*D+1 = 101 for 10 hyperparameters, which would exhaust
    # the budget on random search alone. Cap at max(5, n_configs // 5).
    n_initial = max(5, n_configs // 5)
    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario=scenario,
        n_configs=n_initial,
    )

    # ── Run SMAC ──
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=target_fn,
        intensifier=intensifier,
        initial_design=initial_design,
        overwrite=args.overwrite,
    )

    print(f"\n{'='*60}")
    print(f"Starting SMAC optimization")
    print(f"{'='*60}")
    print(f"  Instances:  {n_splits} {cv_method.upper()} splits")
    print(f"  Budget:     {total_budget} evaluations "
          f"(≤{n_configs} configs × {n_splits} splits)")
    print(f"  Resume:     {not args.overwrite}")
    print(f"  Output:     {smac_output_dir}")
    print(f"  SMAC seed:  {smac_seed}")

    incumbent = smac.optimize()

    # ── Results ──
    results = extract_results(smac, incumbent, n_splits)
    results['cv_method'] = cv_method
    results['seed'] = seed
    results['smac_seed'] = smac_seed

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Best config: {results['incumbent']}")
    print(f"  Mean composite: {results['mean_composite']:.4f}")
    print(f"  Splits evaluated: {results['n_splits_evaluated']}/{n_splits}")
    for inst, comp in sorted(results['per_split_composite'].items(),
                              key=lambda x: int(x[0])):
        print(f"    Split {inst}: {comp:.4f}")

    # Save summary
    results_path = os.path.join(output_dir, "smac_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
