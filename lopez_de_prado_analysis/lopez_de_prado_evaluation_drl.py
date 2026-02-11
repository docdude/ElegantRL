"""
López de Prado Evaluation Framework for Deep Reinforcement Learning

Implementation of advanced backtesting and evaluation methods from:
- "Advances in Financial Machine Learning" (2018) - López de Prado
- "Machine Learning for Asset Managers" (2020) - López de Prado
- "Deep RL for Cryptocurrency Trading" (2023) - Gort et al. (FinRL_Crypto)

Key methods adapted for DRL:
1. Combinatorial Purged Cross-Validation (CPCV) with proper pred/eval times
2. Probability of Backtest Overfitting (PBO) for checkpoint selection
3. Anchored (expanding-window) Walk-Forward Analysis
4. Proper purging and embargo for time-series (FinRL_Crypto methodology)

DRL-Specific Adaptations:
- Episodes span multiple days (not single-step predictions)
- Evaluation time = when reward is known (episode end / position close)
- Checkpoint comparison via PBO to detect overfitting
- Daily returns-based Sharpe calculation
"""

import os
import sys
import warnings
from typing import List, Tuple, Dict
from pathlib import Path
import itertools as itt

import numpy as np
import pandas as pd

# Add pypbo to path for López de Prado metrics (canonical implementation)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pypbo'))
from pypbo.pbo import pbo as pbo_func  # noqa: E402
from pypbo import perf  # noqa: E402

# Optional: torch for DRL model loading
try:
    import torch as th
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. DRL checkpoint loading disabled.")

warnings.filterwarnings('ignore')


# =============================================================================
# PROPER PURGE/EMBARGO FUNCTIONS (from FinRL_Crypto / López de Prado)
# =============================================================================

def purge_proper(indices: np.ndarray, train_indices: np.ndarray, 
                 test_fold_start: int, test_fold_end: int,
                 pred_times: pd.Series, eval_times: pd.Series) -> np.ndarray:
    """
    Proper purging from López de Prado / FinRL_Crypto.
    
    Removes train samples whose eval_time is AFTER the pred_time
    of the first test sample. This prevents data leakage where
    the training set "sees" information from the test period.
    
    Args:
        indices: All available indices
        train_indices: Current training indices
        test_fold_start: Start index of test fold
        test_fold_end: End index of test fold  
        pred_times: When predictions are made (pd.Series with datetime)
        eval_times: When outcomes are known (pd.Series with datetime)
        
    Returns:
        Purged training indices
    """
    time_test_fold_start = pred_times.iloc[test_fold_start]
    
    # Train indices BEFORE test fold: only keep if eval_time < test start
    train_indices_1 = np.intersect1d(
        train_indices, 
        indices[eval_times < time_test_fold_start]
    )
    
    # Train indices AFTER test fold: keep all
    train_indices_2 = np.intersect1d(
        train_indices, 
        indices[test_fold_end:]
    )
    
    return np.concatenate((train_indices_1, train_indices_2))


def embargo_proper(indices: np.ndarray, train_indices: np.ndarray,
                   test_indices: np.ndarray, test_fold_end: int,
                   pred_times: pd.Series, eval_times: pd.Series,
                   embargo_td: pd.Timedelta) -> np.ndarray:
    """
    Proper embargo from López de Prado / FinRL_Crypto.
    
    After purging, also remove train samples whose pred_time is within
    embargo_td of the test set's last eval_time. This handles
    temporal autocorrelation in financial time series.
    
    Args:
        indices: All available indices
        train_indices: Current training indices (after purging)
        test_indices: Test set indices
        test_fold_end: End index of test fold
        pred_times: When predictions are made
        eval_times: When outcomes are known
        embargo_td: Embargo time delta (e.g., pd.Timedelta(days=5))
        
    Returns:
        Training indices with embargo applied
    """
    # Get the last evaluation time in the test set
    test_indices_in_fold = test_indices[test_indices <= test_fold_end]
    if len(test_indices_in_fold) == 0:
        return train_indices
        
    last_test_eval_time = eval_times.iloc[test_indices_in_fold].max()
    
    # Find first training index allowed after embargo
    min_train_index = len(
        pred_times[pred_times <= last_test_eval_time + embargo_td]
    )
    
    if min_train_index < len(indices):
        allowed_indices = np.concatenate((
            indices[:test_fold_end], 
            indices[min_train_index:]
        ))
        train_indices = np.intersect1d(train_indices, allowed_indices)
    
    return train_indices

class DRLEvaluator:
    """
    Advanced evaluation using López de Prado's methods adapted for DRL trading.
    
    Key differences from classification ML:
    - Episodes span multiple days (prediction → evaluation gap is larger)
    - Evaluation uses portfolio returns, not class predictions
    - Multiple checkpoints are compared via PBO
    - Sharpe ratio is the primary metric (not AUC/accuracy)
    """
    
    def __init__(self, embargo_td: pd.Timedelta = pd.Timedelta(days=5), 
                 n_splits: int = 5):
        """
        Initialize DRL evaluator.
        
        Args:
            embargo_td: Time delta for embargo period (default 5 days)
                       For daily trading, 5 days handles weekly autocorrelation
            n_splits: Number of CV splits for CPCV
        """
        self.embargo_td = embargo_td
        self.n_splits = n_splits
        self.results_ = {}
    
    # =========================================================================
    # DRL-SPECIFIC TIME HANDLING
    # =========================================================================
    
    def get_drl_times(self, dates: pd.DatetimeIndex, 
                      episode_length: int = 1) -> Tuple[pd.Series, pd.Series]:
        """
        Generate pred_times and eval_times for DRL trading.
        
        For DRL:
        - pred_time: When the agent makes a decision (daily)
        - eval_time: When the outcome is known
        
        For simple daily trading: eval_time = pred_time + 1 day (next day's close)
        For multi-day episodes: eval_time = pred_time + episode_length
        
        Args:
            dates: DatetimeIndex of trading dates
            episode_length: Average episode length in days (default 1 for daily)
            
        Returns:
            pred_times: Series with prediction timestamps
            eval_times: Series with evaluation timestamps
        """
        n = len(dates)
        pred_times = pd.Series(dates, index=dates)
        
        # Evaluation time = prediction time + episode length
        eval_times = pd.Series(
            [dates[min(i + episode_length, n-1)] for i in range(n)],
            index=dates
        )
        
        return pred_times, eval_times
    
    def get_drl_times_from_episodes(self, episode_starts: np.ndarray,
                                    episode_ends: np.ndarray,
                                    dates: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
        """
        Generate pred_times and eval_times from actual episode boundaries.
        
        Use this when you have episode start/end information from training.
        More accurate than fixed episode_length assumption.
        
        Args:
            episode_starts: Array of episode start indices
            episode_ends: Array of episode end indices
            dates: DatetimeIndex of trading dates
            
        Returns:
            pred_times, eval_times Series
        """
        n = len(dates)
        pred_times = pd.Series(dates, index=dates)
        
        # Build eval_times from episode boundaries
        eval_times_list = []
        episode_idx = 0
        
        for i in range(n):
            # Find which episode this index belongs to
            while episode_idx < len(episode_starts) - 1 and i >= episode_starts[episode_idx + 1]:
                episode_idx += 1
            
            # Eval time is the episode end
            if episode_idx < len(episode_ends):
                eval_idx = min(episode_ends[episode_idx], n - 1)
            else:
                eval_idx = min(i + 1, n - 1)
            
            eval_times_list.append(dates[eval_idx])
        
        eval_times = pd.Series(eval_times_list, index=dates)
        
        return pred_times, eval_times
    
    # =========================================================================
    # COMBINATORIAL PURGED CROSS-VALIDATION (CPCV) FOR DRL
    # =========================================================================
    
    def cpcv_split(self, n_samples: int, dates: pd.DatetimeIndex,
                   n_test_splits: int = 2,
                   episode_length: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate CPCV splits with proper purging and embargo for DRL.
        
        This implements the FinRL_Crypto methodology which is faithful to
        López de Prado's original CPCV from "Advances in Financial ML".
        
        Args:
            n_samples: Number of samples
            dates: DatetimeIndex for the data
            n_test_splits: Number of folds in test set per combination (default 2)
            episode_length: DRL episode length for eval_times
            
        Yields:
            (train_indices, test_indices) tuples for each combination
        """
        indices = np.arange(n_samples)
        
        # Get pred/eval times for DRL
        pred_times, eval_times = self.get_drl_times(dates, episode_length)
        
        # Create fold boundaries (equal-sized folds)
        fold_bounds = [(fold[0], fold[-1] + 1) 
                       for fold in np.array_split(indices, self.n_splits)]
        
        # Generate all combinations of test folds
        test_combinations = list(itt.combinations(fold_bounds, n_test_splits))
        
        # Reverse so first combination has test at end (more realistic)
        test_combinations.reverse()
        
        for fold_bound_list in test_combinations:
            # Compute test indices
            test_indices = np.empty(0, dtype=int)
            test_fold_bounds = []
            
            for fold_start, fold_end in fold_bound_list:
                # Merge contiguous folds
                if not test_fold_bounds or fold_start != test_fold_bounds[-1][-1]:
                    test_fold_bounds.append((fold_start, fold_end))
                elif fold_start == test_fold_bounds[-1][-1]:
                    test_fold_bounds[-1] = (test_fold_bounds[-1][0], fold_end)
                
                test_indices = np.union1d(
                    test_indices, 
                    indices[fold_start:fold_end]
                ).astype(int)
            
            # Compute train indices (complement of test)
            train_indices = np.setdiff1d(indices, test_indices)
            
            # Apply purging and embargo for each test fold
            for test_fold_start, test_fold_end in test_fold_bounds:
                # Purge: remove train samples whose eval overlaps test pred
                train_indices = purge_proper(
                    indices, train_indices, 
                    test_fold_start, test_fold_end,
                    pred_times, eval_times
                )
                
                # Embargo: remove train samples too close to test
                train_indices = embargo_proper(
                    indices, train_indices, test_indices, test_fold_end,
                    pred_times, eval_times, self.embargo_td
                )
            
            yield train_indices, test_indices
    
    def walk_forward_splits(self, n_samples: int, dates: pd.DatetimeIndex,
                            val_ratio: float = 0.2,
                            n_folds: int = 3,
                            gap_days: int = 5) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Anchored (expanding-window) walk-forward splits for DRL evaluation.
        
        The training window always starts at day 0 and grows with each fold,
        matching the approach in demo_FinRL_Alpaca_VecEnv_CV.py.  This is the
        standard methodology for time-series because it uses all available
        historical data.
        
        Example with n_folds=3, 900 days, val_ratio=0.2:
            Fold 0: Train [0:240]  -> Val [240:300]    (240 train days)
            Fold 1: Train [0:480]  -> Val [480:600]    (480 train days)
            Fold 2: Train [0:720]  -> Val [720:900]    (720 train days)
        
        Args:
            n_samples: Total number of trading days
            dates: DatetimeIndex
            val_ratio: Fraction of each fold period used for validation
            n_folds: Number of walk-forward folds
            gap_days: Gap between train and val (embargo/purging)
            
        Returns:
            List of ((train_start, train_end), (val_start, val_end))
        """
        if n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {n_folds}")
        
        splits = []
        fold_size = n_samples // n_folds
        
        for fold in range(n_folds):
            val_end = (fold + 1) * fold_size
            val_days = int(fold_size * val_ratio)
            val_start = val_end - val_days
            train_end = val_start - gap_days
            train_start = 0  # Always anchored at start
            
            if train_end > train_start and val_end > val_start:
                splits.append(((train_start, train_end), (val_start, val_end)))
        
        return splits
    
    # =========================================================================
    # DRL CHECKPOINT EVALUATION
    # =========================================================================
    
    def evaluate_drl_checkpoint(self, actor, env_class, env_args: dict,
                                device: str = 'cuda:0',
                                initial_amount: float = 1e6) -> Dict:
        """
        Evaluate a single DRL checkpoint and return portfolio metrics.
        
        Args:
            actor: Trained actor network (torch model)
            env_class: Environment class
            env_args: Environment arguments
            device: Device for inference
            initial_amount: Initial portfolio value
            
        Returns:
            Dictionary with:
            - daily_returns: Array of daily percentage returns
            - account_values: Array of portfolio values
            - sharpe_ratio: Annualized Sharpe ratio
            - cumulative_return: Total return percentage
            - max_drawdown: Maximum drawdown percentage
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for DRL checkpoint evaluation")
        
        # Create environment
        env = env_class(**env_args)
        
        # Disable random reset for reproducible evaluation
        if hasattr(env, 'if_random_reset'):
            env.if_random_reset = False
        
        # Run episode and track account values
        state, _ = env.reset()
        account_values = [initial_amount]
        
        with th.no_grad():
            for t in range(env.max_step):
                if hasattr(state, 'to'):
                    state_tensor = state.to(device)
                else:
                    state_tensor = th.tensor(state, dtype=th.float32, device=device)
                
                action = actor(state_tensor)
                state, reward, terminal, truncate, info = env.step(action)
                
                # Track actual portfolio value (env 0 if vectorized)
                if hasattr(env, 'total_asset'):
                    if hasattr(env.total_asset, '__getitem__'):
                        current_value = float(env.total_asset[0].cpu().item() 
                                             if hasattr(env.total_asset[0], 'cpu') 
                                             else env.total_asset[0])
                    else:
                        current_value = float(env.total_asset)
                    account_values.append(current_value)
        
        # Handle final value from cumulative_returns if available
        if hasattr(env, 'cumulative_returns') and env.cumulative_returns is not None:
            cr = env.cumulative_returns
            if hasattr(cr, '__getitem__'):
                final_cr = float(cr[0].cpu().item() if hasattr(cr[0], 'cpu') else cr[0])
            else:
                final_cr = float(cr)
            account_values[-1] = initial_amount * final_cr / 100
        
        account_values = np.array(account_values)
        
        # Compute daily returns (percentage)
        daily_returns = np.diff(account_values) / account_values[:-1]
        
        # Annualized Sharpe ratio
        if daily_returns.std() > 1e-8:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Cumulative return
        cumulative_return = (account_values[-1] / initial_amount - 1) * 100
        
        # Maximum drawdown
        peak = np.maximum.accumulate(account_values)
        drawdown = (peak - account_values) / peak
        max_drawdown = drawdown.max() * 100
        
        return {
            'daily_returns': daily_returns,
            'account_values': account_values,
            'sharpe_ratio': sharpe,
            'cumulative_return': cumulative_return,
            'max_drawdown': max_drawdown,
            'n_days': len(daily_returns)
        }
    
    def evaluate_checkpoints_directory(self, checkpoint_dir: str, 
                                       env_class, env_args: dict,
                                       device: str = 'cuda:0',
                                       pattern: str = '*.pt') -> Dict[str, Dict]:
        """
        Evaluate all checkpoints in a directory.
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
            env_class: Environment class
            env_args: Environment arguments
            device: Device for inference
            pattern: Glob pattern for checkpoint files
            
        Returns:
            Dictionary mapping checkpoint_name -> evaluation results
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for checkpoint evaluation")
        
        checkpoint_dir = Path(checkpoint_dir)
        pt_files = sorted(checkpoint_dir.glob(pattern))
        
        if not pt_files:
            raise ValueError(f"No checkpoints found in {checkpoint_dir} matching {pattern}")
        
        print(f"🔍 Evaluating {len(pt_files)} checkpoints from {checkpoint_dir}")
        
        results = {}
        for i, pt_file in enumerate(pt_files):
            print(f"   [{i+1}/{len(pt_files)}] {pt_file.name}...", end=' ')
            
            try:
                actor = th.load(pt_file, map_location=device, weights_only=False)
                actor.eval()
                
                eval_result = self.evaluate_drl_checkpoint(
                    actor, env_class, env_args, device
                )
                results[pt_file.name] = eval_result
                
                print(f"Sharpe: {eval_result['sharpe_ratio']:.3f}, "
                      f"Return: {eval_result['cumulative_return']:.2f}%")
            except Exception as e:
                print(f"FAILED: {e}")
                results[pt_file.name] = {'error': str(e)}
        
        return results
    
    # =========================================================================
    # PROBABILITY OF BACKTEST OVERFITTING (PBO) FOR DRL
    # =========================================================================
    
    def compute_drl_pbo(self, checkpoint_results: Dict[str, Dict],
                        n_splits: int = 16) -> Dict:
        """
        Compute PBO across multiple DRL checkpoints.
        
        This treats each checkpoint as a "strategy" for PBO computation.
        Uses pypbo library (canonical López de Prado implementation).
        
        Args:
            checkpoint_results: Dict from evaluate_checkpoints_directory()
                Maps checkpoint_name -> {'daily_returns': array, ...}
            n_splits: Number of CSCV splits (must be even)
            
        Returns:
            PBO results dictionary
        """
        print("🔍 Computing DRL PBO across checkpoints...")
        
        # Extract daily returns for each checkpoint (as strategies)
        valid_checkpoints = {
            k: v for k, v in checkpoint_results.items() 
            if 'daily_returns' in v and 'error' not in v
        }
        
        if len(valid_checkpoints) < 2:
            print("⚠️  Need at least 2 valid checkpoints for PBO")
            return {
                'pbo': None,
                'interpretation': 'N/A - Need multiple checkpoints'
            }
        
        # Find minimum length (align all returns)
        min_len = min(len(v['daily_returns']) for v in valid_checkpoints.values())
        
        # Build returns matrix: (observations x strategies)
        checkpoint_names = list(valid_checkpoints.keys())
        strategy_returns = np.column_stack([
            valid_checkpoints[name]['daily_returns'][:min_len]
            for name in checkpoint_names
        ])
        
        print(f"   Checkpoints: {len(checkpoint_names)}")
        print(f"   Trading days: {min_len}")
        print(f"   Matrix shape: {strategy_returns.shape}")
        
        # Use pypbo's pbo function
        return self.probability_backtest_overfitting(
            strategy_returns, n_splits=n_splits
        )
    
    def compute_pbo_from_hyperparameters(self, hp_results: List[Dict],
                                         n_splits: int = 16) -> Dict:
        """
        Compute PBO across hyperparameter optimization trials.
        
        Each HP trial is treated as a "strategy" - if the best HP set
        has high PBO, the optimization is overfit.
        
        Args:
            hp_results: List of dicts, each with 'daily_returns' array
            n_splits: Number of CSCV splits
            
        Returns:
            PBO results
        """
        print("🔍 Computing PBO across hyperparameter trials...")
        
        if len(hp_results) < 2:
            print("⚠️  Need at least 2 HP trials for PBO")
            return {'pbo': None, 'interpretation': 'N/A'}
        
        # Find minimum length
        min_len = min(len(r['daily_returns']) for r in hp_results 
                      if 'daily_returns' in r)
        
        # Build returns matrix
        strategy_returns = np.column_stack([
            r['daily_returns'][:min_len] for r in hp_results
            if 'daily_returns' in r
        ])
        
        print(f"   HP trials: {strategy_returns.shape[1]}")
        print(f"   Trading days: {min_len}")
        
        return self.probability_backtest_overfitting(
            strategy_returns, n_splits=n_splits
        )
    
    def probability_backtest_overfitting(self, strategy_returns: np.ndarray,
                                         n_splits: int = 16) -> Dict:
        """
        Calculate PBO using pypbo library (López de Prado implementation).
        
        PBO measures probability that IS performance ranking is false.
        Uses Combinatorial Symmetric Cross-Validation (CSCV) method.
        
        Args:
            strategy_returns: Matrix (T_observations x N_strategies)
            n_splits: Number of CSCV splits (must be even)
            
        Returns:
            PBO statistics including probability, logits, and diagnostics
        """
        print("🔍 Calculating PBO (using pypbo library)...")
        
        if strategy_returns.ndim != 2:
            raise ValueError(f"strategy_returns must be 2D, got {strategy_returns.shape}")
        
        n_observations, n_strategies = strategy_returns.shape
        
        if n_strategies < 2:
            return {'pbo': None, 'interpretation': 'Need ≥2 strategies'}
        
        print(f"   Shape: ({n_observations} obs, {n_strategies} strategies)")
        
        # Define metric (Sharpe ratio)
        def metric_func(returns):
            return perf.sharpe_iid(returns, bench=0, factor=1, log=False)
        
        try:
            pbo_result = pbo_func(
                M=strategy_returns,
                S=n_splits,
                metric_func=metric_func,
                threshold=0,
                n_jobs=-1,
                verbose=False,
                plot=False
            )
            
            results = {
                'pbo': pbo_result.pbo,
                'prob_oos_loss': pbo_result.prob_oos_loss,
                'lambda_values': pbo_result.logits,
                'mean_logit': np.mean(pbo_result.logits),
                'std_logit': np.std(pbo_result.logits),
                'n_strategies': n_strategies,
                'n_splits': len(pbo_result.logits),
                'performance_degradation': {
                    'slope': pbo_result.linear_model.slope,
                    'r_squared': pbo_result.linear_model.rvalue ** 2,
                    'p_value': pbo_result.linear_model.pvalue
                },
                'interpretation': self._interpret_pbo(pbo_result.pbo)
            }
            
            print(f"✅ PBO Results:")
            print(f"   Strategies: {results['n_strategies']}")
            print(f"   CSCV splits: {results['n_splits']}")
            print(f"   PBO: {results['pbo']:.3f}")
            print(f"   Prob OOS Loss: {results['prob_oos_loss']:.3f}")
            print(f"   Mean λ: {results['mean_logit']:.3f}")
            print(f"   Degradation: slope={results['performance_degradation']['slope']:.3f}")
            print(f"   {results['interpretation']}")
            
            return results
            
        except Exception as e:
            print(f"⚠️  PBO failed: {e}")
            return {'pbo': None, 'interpretation': f'Error: {e}'}
    
    def _interpret_pbo(self, pbo: float) -> str:
        """Interpret PBO value"""
        if pbo < 0.3:
            return "✅ Low overfitting risk - Results likely robust"
        elif pbo < 0.5:
            return "⚠️ Moderate overfitting risk - Exercise caution"
        elif pbo < 0.7:
            return "🔴 High overfitting risk - Results questionable"
        else:
            return "🚨 Very high overfitting risk - Likely spurious"
    
    # =========================================================================
    # DEFLATED SHARPE RATIO (DSR) FOR DRL
    # =========================================================================
    
    def compute_deflated_sharpe(self, sharpe_ratio: float, n_trials: int,
                                returns: np.ndarray) -> Dict:
        """
        Compute Deflated Sharpe Ratio (DSR) from pypbo.
        
        DSR adjusts Sharpe ratio for multiple testing bias.
        Given N trials, what's the probability the observed Sharpe
        could have been achieved by chance?
        
        Args:
            sharpe_ratio: Observed Sharpe ratio
            n_trials: Number of trials/strategies tested
            returns: Array of daily returns
            
        Returns:
            DSR results including probability
        """
        print("🔍 Computing Deflated Sharpe Ratio (DSR)...")
        
        # Import pbo module (DSR is in pbo, not perf)
        import pypbo as pbo_module
        
        # Compute return statistics
        T = len(returns)
        skew = float(pd.Series(returns).skew())
        kurt = float(pd.Series(returns).kurtosis() + 3)  # Excess to regular kurtosis
        
        # Estimate std of sharpe ratios across trials
        # For DRL: we approximate using the variance of the observed sharpe
        # across different configurations/checkpoints
        sharpe_std = sharpe_ratio / np.sqrt(2 * n_trials) if n_trials > 1 else 0.1
        
        # Use pypbo's DSR
        dsr = pbo_module.dsr(
            test_sharpe=sharpe_ratio,
            sharpe_std=sharpe_std,
            N=n_trials,
            T=T,
            skew=skew,
            kurtosis=kurt
        )
        
        results = {
            'dsr': dsr,
            'original_sharpe': sharpe_ratio,
            'n_trials': n_trials,
            'n_observations': len(returns),
            'interpretation': self._interpret_dsr(dsr)
        }
        
        print(f"✅ DSR Results:")
        print(f"   Original Sharpe: {sharpe_ratio:.3f}")
        print(f"   Trials tested: {n_trials}")
        print(f"   DSR (prob not luck): {dsr:.3f}")
        print(f"   {results['interpretation']}")
        
        return results
    
    def _interpret_dsr(self, dsr: float) -> str:
        """Interpret DSR value"""
        if dsr > 0.95:
            return "✅ High confidence - Unlikely due to luck"
        elif dsr > 0.75:
            return "⚠️ Moderate confidence - Some multiple testing bias"
        elif dsr > 0.5:
            return "🔴 Low confidence - Likely inflated by selection"
        else:
            return "🚨 Very low confidence - Probably luck"
    
    # =========================================================================
    # DRL WALK-FORWARD ANALYSIS
    # =========================================================================
    
    def drl_walk_forward_analysis(self, checkpoint_dir: str,
                                  env_class, env_args_template: dict,
                                  dates: pd.DatetimeIndex,
                                  n_folds: int = 3,
                                  val_ratio: float = 0.2,
                                  gap_days: int = 5,
                                  device: str = 'cuda:0') -> Dict:
        """
        DRL Anchored Walk-Forward Analysis.
        
        For each fold:
        1. Train on expanding historical window (anchored at day 0)
        2. Gap period (embargo)
        3. Evaluate checkpoint on validation period
        
        This assumes you have pre-trained checkpoints for each fold.
        The env_args_template should have 'start_date' and 'end_date' that
        will be modified for each fold.
        
        Args:
            checkpoint_dir: Directory with fold checkpoints (fold_0.pt, fold_1.pt, etc.)
            env_class: Environment class
            env_args_template: Template env args (dates will be modified)
            dates: Full date range
            n_folds: Number of WFA folds
            val_ratio: Fraction of each fold period used for validation
            gap_days: Embargo gap in days
            device: Inference device
            
        Returns:
            WFA results with per-fold and aggregate metrics
        """
        print("🔍 DRL Anchored Walk-Forward Analysis...")
        
        # Generate splits
        splits = self.walk_forward_splits(
            len(dates), dates, val_ratio, n_folds, gap_days
        )
        
        checkpoint_dir = Path(checkpoint_dir)
        results = {
            'folds': [],
            'sharpe_ratios': [],
            'cumulative_returns': [],
            'max_drawdowns': []
        }
        
        for fold, ((train_start, train_end), (val_start, val_end)) in enumerate(splits):
            print(f"\n   Fold {fold + 1}/{n_folds}")
            print(f"   Train: {dates[train_start].date()} - {dates[train_end-1].date()}")
            print(f"   Val:   {dates[val_start].date()} - {dates[val_end-1].date()}")
            
            # Look for checkpoint
            checkpoint_patterns = [
                f'fold_{fold}.pt',
                f'fold{fold}.pt', 
                f'actor_fold_{fold}.pt',
                f'actor_{fold}.pt'
            ]
            
            checkpoint_path = None
            for pattern in checkpoint_patterns:
                candidate = checkpoint_dir / pattern
                if candidate.exists():
                    checkpoint_path = candidate
                    break
            
            if checkpoint_path is None:
                print(f"   ⚠️ No checkpoint found for fold {fold}")
                continue
            
            print(f"   Loading: {checkpoint_path.name}")
            
            # Create validation env args
            val_env_args = env_args_template.copy()
            val_env_args['start_date'] = dates[val_start].strftime('%Y-%m-%d')
            val_env_args['end_date'] = dates[val_end-1].strftime('%Y-%m-%d')
            
            try:
                actor = th.load(checkpoint_path, map_location=device, weights_only=False)
                actor.eval()
                
                eval_result = self.evaluate_drl_checkpoint(
                    actor, env_class, val_env_args, device
                )
                
                results['folds'].append({
                    'fold': fold,
                    'train_dates': (dates[train_start], dates[train_end-1]),
                    'val_dates': (dates[val_start], dates[val_end-1]),
                    'sharpe': eval_result['sharpe_ratio'],
                    'return': eval_result['cumulative_return'],
                    'mdd': eval_result['max_drawdown'],
                    'daily_returns': eval_result['daily_returns']
                })
                
                results['sharpe_ratios'].append(eval_result['sharpe_ratio'])
                results['cumulative_returns'].append(eval_result['cumulative_return'])
                results['max_drawdowns'].append(eval_result['max_drawdown'])
                
                print(f"   Sharpe: {eval_result['sharpe_ratio']:.3f}, "
                      f"Return: {eval_result['cumulative_return']:.2f}%, "
                      f"MDD: {eval_result['max_drawdown']:.2f}%")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Aggregate statistics
        if results['sharpe_ratios']:
            results['mean_sharpe'] = np.mean(results['sharpe_ratios'])
            results['std_sharpe'] = np.std(results['sharpe_ratios'])
            results['mean_return'] = np.mean(results['cumulative_returns'])
            results['std_return'] = np.std(results['cumulative_returns'])
            results['mean_mdd'] = np.mean(results['max_drawdowns'])
            
            print(f"\n✅ WFA Summary:")
            print(f"   Sharpe: {results['mean_sharpe']:.3f} ± {results['std_sharpe']:.3f}")
            print(f"   Return: {results['mean_return']:.2f}% ± {results['std_return']:.2f}%")
            print(f"   MDD: {results['mean_mdd']:.2f}%")
        
        return results
    
    # =========================================================================
    # COMPREHENSIVE DRL EVALUATION
    # =========================================================================
    
    def comprehensive_drl_evaluation(self, checkpoint_dir: str,
                                     env_class, env_args: dict,
                                     device: str = 'cuda:0',
                                     n_splits: int = 16) -> Dict:
        """
        Run comprehensive López de Prado evaluation for DRL.
        
        This performs:
        1. Evaluate all checkpoints in directory
        2. Compute PBO across checkpoints
        3. Compute DSR for best checkpoint
        4. Generate summary report
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            env_class: Environment class
            env_args: Environment arguments
            device: Inference device
            n_splits: CSCV splits for PBO
            
        Returns:
            Comprehensive evaluation results
        """
        print("=" * 80)
        print("🚀 LÓPEZ DE PRADO DRL COMPREHENSIVE EVALUATION")
        print("=" * 80)
        
        all_results = {}
        
        # 1. Evaluate all checkpoints
        print("\n1️⃣ CHECKPOINT EVALUATION")
        print("-" * 40)
        checkpoint_results = self.evaluate_checkpoints_directory(
            checkpoint_dir, env_class, env_args, device
        )
        all_results['checkpoints'] = checkpoint_results
        
        # Find best checkpoint by Sharpe
        valid_results = {k: v for k, v in checkpoint_results.items() 
                        if 'sharpe_ratio' in v}
        
        if valid_results:
            best_name = max(valid_results.keys(), 
                           key=lambda k: valid_results[k]['sharpe_ratio'])
            best_result = valid_results[best_name]
            all_results['best_checkpoint'] = {
                'name': best_name,
                **best_result
            }
            print(f"\n   🏆 Best: {best_name}")
            print(f"      Sharpe: {best_result['sharpe_ratio']:.3f}")
            print(f"      Return: {best_result['cumulative_return']:.2f}%")
        
        # 2. PBO across checkpoints
        print("\n2️⃣ PROBABILITY OF BACKTEST OVERFITTING (PBO)")
        print("-" * 40)
        
        if len(valid_results) >= 2:
            pbo_results = self.compute_drl_pbo(checkpoint_results, n_splits)
            all_results['pbo'] = pbo_results
        else:
            print("   ⚠️ Need ≥2 valid checkpoints for PBO")
            all_results['pbo'] = {'pbo': None}
        
        # 3. Deflated Sharpe Ratio for best checkpoint
        print("\n3️⃣ DEFLATED SHARPE RATIO (DSR)")
        print("-" * 40)
        
        if valid_results and 'best_checkpoint' in all_results:
            best = all_results['best_checkpoint']
            dsr_results = self.compute_deflated_sharpe(
                sharpe_ratio=best['sharpe_ratio'],
                n_trials=len(valid_results),
                returns=best['daily_returns']
            )
            all_results['dsr'] = dsr_results
        else:
            print("   ⚠️ No valid checkpoints for DSR")
            all_results['dsr'] = {'dsr': None}
        
        # 4. Summary
        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)
        
        all_results['summary'] = self._generate_drl_summary(all_results)
        print(all_results['summary'])
        
        return all_results
    
    def _generate_drl_summary(self, results: Dict) -> str:
        """Generate summary report for DRL evaluation"""
        report = []
        
        # Best checkpoint
        if 'best_checkpoint' in results:
            best = results['best_checkpoint']
            report.append(f"🏆 Best Checkpoint: {best['name']}")
            report.append(f"   Sharpe: {best['sharpe_ratio']:.3f}")
            report.append(f"   Return: {best['cumulative_return']:.2f}%")
            report.append(f"   MDD: {best['max_drawdown']:.2f}%")
        
        # PBO
        if 'pbo' in results and results['pbo'].get('pbo') is not None:
            pbo = results['pbo']
            report.append(f"\n📊 Overfitting Risk:")
            report.append(f"   PBO: {pbo['pbo']:.3f}")
            report.append(f"   {pbo['interpretation']}")
        
        # DSR
        if 'dsr' in results and results['dsr'].get('dsr') is not None:
            dsr = results['dsr']
            report.append(f"\n📊 Statistical Significance:")
            report.append(f"   DSR: {dsr['dsr']:.3f}")
            report.append(f"   {dsr['interpretation']}")
        
        # Overall assessment
        report.append(f"\n🎯 OVERALL ASSESSMENT:")
        
        pbo_val = results.get('pbo', {}).get('pbo')
        dsr_val = results.get('dsr', {}).get('dsr')
        sharpe = results.get('best_checkpoint', {}).get('sharpe_ratio', 0)
        
        if pbo_val is not None and dsr_val is not None:
            if pbo_val < 0.5 and dsr_val > 0.75 and sharpe > 0.5:
                report.append("   ✅ ROBUST - Low overfitting, statistically significant")
            elif pbo_val < 0.7 and dsr_val > 0.5:
                report.append("   ⚠️ MODERATE - Some overfitting risk")
            else:
                report.append("   🔴 CONCERNING - High overfitting or low significance")
        
        return "\n".join(report)


# =============================================================================
# DEMO FUNCTION
# =============================================================================

def demo_drl_evaluation():
    """
    Demo function showing how to use the DRL evaluator.
    
    This demonstrates:
    1. PBO computation with simulated strategy returns
    2. DSR computation
    3. How to integrate with actual checkpoints
    """
    print("🚀 López de Prado DRL Evaluation Demo")
    print("=" * 60)
    
    # Create simulated strategy returns (as if from different checkpoints)
    np.random.seed(42)
    n_strategies = 10  # 10 different checkpoints
    n_days = 252  # 1 year of trading
    
    print("\n📊 Simulating checkpoint returns...")
    
    # Simulate returns with different Sharpe ratios
    strategy_returns_list = []
    for i in range(n_strategies):
        # Varying quality: some checkpoints better than others
        mean_return = np.random.uniform(-0.0005, 0.002)
        volatility = np.random.uniform(0.01, 0.03)
        returns = np.random.normal(mean_return, volatility, n_days)
        strategy_returns_list.append(returns)
        
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        print(f"   Checkpoint {i+1}: Sharpe = {sharpe:.2f}")
    
    # Convert to matrix: (observations x strategies)
    strategy_returns = np.column_stack(strategy_returns_list)
    print(f"\n   Matrix shape: {strategy_returns.shape}")
    print(f"   (observations={strategy_returns.shape[0]}, strategies={strategy_returns.shape[1]})")
    
    # Create evaluator
    evaluator = DRLEvaluator(embargo_td=pd.Timedelta(days=5), n_splits=5)
    
    # 1. Compute PBO
    print("\n" + "=" * 60)
    print("1️⃣ PROBABILITY OF BACKTEST OVERFITTING (PBO)")
    print("=" * 60)
    
    pbo_results = evaluator.probability_backtest_overfitting(
        strategy_returns, n_splits=8
    )
    
    # 2. Compute DSR for best strategy
    print("\n" + "=" * 60)
    print("2️⃣ DEFLATED SHARPE RATIO (DSR)")
    print("=" * 60)
    
    # Find best strategy by Sharpe
    sharpe_ratios = [
        r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0 
        for r in strategy_returns.T
    ]
    best_idx = np.argmax(sharpe_ratios)
    best_sharpe = sharpe_ratios[best_idx]
    best_returns = strategy_returns[:, best_idx]
    
    print(f"\n   Best checkpoint: #{best_idx + 1}")
    print(f"   Sharpe ratio: {best_sharpe:.3f}")
    
    dsr_results = evaluator.compute_deflated_sharpe(
        sharpe_ratio=best_sharpe,
        n_trials=n_strategies,
        returns=best_returns
    )
    
    # 3. Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    print(f"\n🏆 Best Checkpoint: #{best_idx + 1}")
    print(f"   Sharpe: {best_sharpe:.3f}")
    
    if pbo_results.get('pbo') is not None:
        print(f"\n📊 Overfitting Risk:")
        print(f"   PBO: {pbo_results['pbo']:.3f}")
        print(f"   {pbo_results['interpretation']}")
    
    if dsr_results.get('dsr') is not None:
        print(f"\n📊 Statistical Significance:")
        print(f"   DSR: {dsr_results['dsr']:.3f}")
        print(f"   {dsr_results['interpretation']}")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    pbo = pbo_results.get('pbo')
    dsr = dsr_results.get('dsr')
    
    if pbo is not None and dsr is not None:
        if pbo < 0.5 and dsr > 0.75 and best_sharpe > 0.5:
            print("   ✅ ROBUST - Low overfitting risk, statistically significant")
        elif pbo < 0.7 and dsr > 0.5:
            print("   ⚠️ MODERATE - Some overfitting risk, moderate significance")
        else:
            print("   🔴 CONCERNING - High overfitting or low significance")
    
    return {
        'pbo': pbo_results,
        'dsr': dsr_results,
        'best_sharpe': best_sharpe,
        'sharpe_ratios': sharpe_ratios
    }


def demo_cpcv_splits():
    """Demo CPCV split generation with proper purging/embargo"""
    print("🔍 CPCV Split Demo")
    print("=" * 60)
    
    # Create dates
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    n_samples = len(dates)
    
    evaluator = DRLEvaluator(embargo_td=pd.Timedelta(days=5), n_splits=5)
    
    print(f"\nGenerating CPCV splits for {n_samples} samples...")
    print(f"Number of folds: {evaluator.n_splits}")
    print(f"Embargo: {evaluator.embargo_td}")
    
    for i, (train_idx, test_idx) in enumerate(evaluator.cpcv_split(n_samples, dates)):
        print(f"\n   Combination {i+1}:")
        print(f"      Train: {len(train_idx)} samples, "
              f"{dates[train_idx[0]].date()} - {dates[train_idx[-1]].date()}")
        print(f"      Test:  {len(test_idx)} samples, "
              f"{dates[test_idx[0]].date()} - {dates[test_idx[-1]].date()}")
        
        if i >= 4:  # Just show first 5
            remaining = sum(1 for _ in evaluator.cpcv_split(n_samples, dates)) - i - 1
            print(f"\n   ... and {remaining} more combinations")
            break


if __name__ == "__main__":
    # Run PBO/DSR demo
    results = demo_drl_evaluation()
    
    print("\n\n")
    
    # Run CPCV demo
    demo_cpcv_splits()
