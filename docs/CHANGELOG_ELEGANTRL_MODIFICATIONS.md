# ElegantRL Modifications Changelog

All changes made to the [AI4Finance-Foundation/ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL) codebase, based on commit `95a1a6bb` (master).

---

## Summary

| Category | Files Modified | Files Added | Lines Changed |
|---|---|---|---|
| Core Library (elegantrl/) | 7 | 1 | ~257 |
| Examples/Scripts | 1 | 6 + 8 configs | ~4,200 new |
| Evaluation (López de Prado) | 0 | 6 | ~3,500 new |
| Tests | 0 | 1 | ~285 new |
| **Total** | **8** | **22+** | **~8,200** |

---

## 1. Core Library Changes (`elegantrl/`)

### 1.1 `elegantrl/envs/vec_normalize.py` (NEW — 491 lines)

VecNormalize observation/reward normalization wrapper for GPU-based VecEnvs.

- **RunningMeanStd**: Welford's online algorithm for running mean/variance on GPU tensors
- **VecNormalize**: Wraps any VecEnv to normalize observations and optionally rewards
  - `norm_obs`: Normalize observations using running statistics (default: True)
  - `norm_reward`: Normalize rewards using return-based statistics (default: True)
  - `training`: Toggle stats updates vs frozen inference mode
  - `save()`/`load()`: Persist normalization statistics to disk
  - `__setattr__` forwarding: Ensures env-specific attributes (e.g. `if_random_reset`) are written to the inner env, not the wrapper — **critical bug fix** that prevented deterministic evaluation
  - `__getattr__` forwarding: Proxies attribute reads (`total_asset`, `cumulative_returns`, etc.) to inner env

### 1.2 `elegantrl/envs/StockTradingEnv.py` (MODIFIED — +76 lines)

Stock cooling-down period and minimum holding constraints for both scalar and VecEnv.

- **`stock_cd_steps`**: Simulates T+1 settlement — after buying, must wait N steps before selling
- **`min_stock_rate`**: Prevents selling below a minimum portfolio allocation threshold (reduces overtrading)
- **StockTradingEnv (scalar)**:
  - Added `stock_cd` counter array, decremented each step
  - Buy sets `stock_cd[i] = stock_cd_steps`; sell blocked if `stock_cd[i] > 0`
  - Sell blocked if total share value < `min_stock_rate * initial_amount`
- **StockTradingVecEnv (GPU)**:
  - Same logic vectorized with `th.where` masks
  - `reset()` now returns `(state, {})` tuple (was just `state`) for Gymnasium compatibility
  - `step()` now returns 5-tuple `(state, reward, done, truncate, {})` (was 4-tuple)
  - **Removed `action = th.ones_like(action)` debug line** that was overriding all agent actions with buy-all

### 1.3 `elegantrl/envs/__init__.py` (MODIFIED — +9 lines)

Added exports: `VecNormalize`, `RunningMeanStd`, `StockTradingEnv`, `StockTradingVecEnv`.

### 1.4 `elegantrl/agents/AgentPPO.py` (MODIFIED — +23/-13 lines)

**AgentA2C.update_objectives()** — Fixed 2D batch sampling for VecEnv compatibility:

- Old: 1D flat indexing `states[indices]` — broke with `(sample_len, num_seqs, state_dim)` tensors
- New: 2D sampling with `ids0 = ids % sample_len`, `ids1 = ids // sample_len` — matches PPO's approach
- Changed `value.squeeze(1)` → `value.squeeze(-1)` for shape robustness

### 1.5 `elegantrl/agents/AgentBase.py` (MODIFIED — +1 line)

Added `self.explore_rate` attribute (epsilon for DQN-style exploration, default 0.0).

### 1.6 `elegantrl/train/run.py` (MODIFIED — +33 lines)

VecNormalize integration into all three training modes:

- **`train_agent_single_process()`**:
  - Loads VecNormalize stats on `continue_train`
  - Saves stats every eval cycle (periodic) and at training end (final)
- **`Worker` class (multiprocessing/multi-GPU)**:
  - Worker-0 loads VecNormalize stats on `continue_train`
  - Worker-0 saves stats every 10 collection cycles and at shutdown

### 1.7 `elegantrl/train/config.py` (MODIFIED — +9 lines)

`build_env()` now optionally wraps with VecNormalize when `env_args['use_vec_normalize'] = True`.
Accepts `vec_normalize_kwargs` dict for configuration.

### 1.8 `elegantrl/train/evaluator.py` (MODIFIED — +62/-50 lines)

**`draw_learning_curve()`** visual overhaul:

- Clearer color scheme: blue (eval return), orange (explore reward), green (actor obj), crimson (critic loss)
- Added `fill_between` for ±1 std band on episode return
- Combined legends for twin-axis panels
- Grid alpha, line widths, and titles improved

### 1.9 `elegantrl/__init__.py` (MODIFIED — +3 lines)

Added exports: `valid_agent`, `Evaluator`.

---

## 2. New Training & Evaluation Scripts (`examples/`)

### 2.1 `examples/demo_FinRL_Alpaca_VecEnv.py` (NEW — 966 lines)

Full-featured Alpaca 28-stock trading demo with VecEnv:

- Downloads and preprocesses Alpaca market data (28 DOW stocks, 2020-2024)
- Supports PPO, A2C, SAC, TD3, DDPG, ModSAC agents
- VecNormalize integration with `--normalize` flag
- `--eval` mode for checkpoint evaluation with FinRL backtest stats
- `--best` auto-selects highest-avgR checkpoint
- `--continue` for resuming training with saved normalization stats
- DJIA baseline comparison with plots

### 2.2 `examples/demo_FinRL_Alpaca_VecEnv_CV.py` (NEW — 1,446 lines)

Cross-validation training with three CV methodologies:

- **Holdout**: Simple train/test split
- **Anchored Walk-Forward**: Expanding training window, always anchored at day 0
- **CPCV**: Combinatorial Purged Cross-Validation (López de Prado 2018) with embargo
- Unified `get_cv_splits()` dispatcher
- `compute_equal_weight_sharpe()` benchmark for excess Sharpe calculation
- Multi-fold training loop with per-fold and aggregate statistics
- CLI: `--cv-method`, `--n-groups`, `--n-test-groups`, `--embargo-pct`

### 2.3 `examples/hpo_alpaca_vecenv.py` (NEW — 799 lines)

Hyperparameter optimization with Hydra + Hypersweeper:

- Same three CV generators as the CV demo script
- Hydra-configured HPO with per-agent YAML configs
- Excess Sharpe (over equal-weight benchmark) as objective
- Multi-fold training with aggregated metrics
- `@hydra.main` entry point for sweep orchestration

### 2.4 `examples/eval_all_checkpoints.py` (NEW — 309 lines)

Batch checkpoint evaluation and comparison:

- Evaluates all `actor__*.pt` files in a directory
- VecNormalize-aware (loads `vec_normalize.pt` if present)
- DJIA baseline comparison with alpha, Sharpe, max drawdown
- Consistency metric: % of days beating baseline
- 4-panel comparison plot + CSV export
- Recommendations: best return, best Sharpe, most consistent

### 2.5 `examples/demo_FinRL_ElegantRL_China_A_shares_vec_env.py` (NEW — 647 lines)

China A-shares trading demo with GPU VecEnv (similar structure to Alpaca demo).

### 2.6 `examples/demo_A2C_PPO.py` (MODIFIED — +41/-13 lines)

- Added SAC and DDPG agent support (`AgentClassList = [PPO, A2C, SAC, DDPG]`)
- Off-policy memory guards (`buffer_size`, `batch_size`, `horizon_len` overrides)
- Added ENV_ID 6 (`stock_trading`) and 7 (`stock_trading_vec`) routes
- Increased `break_step` to 1e6 for VecEnv training

### 2.7 `examples/configs/` (NEW — 8 YAML files)

Hydra configuration files for HPO per agent type:
`alpaca_ppo.yaml`, `alpaca_a2c.yaml`, `alpaca_sac.yaml`, `alpaca_td3.yaml`, `alpaca_ddpg.yaml`, `alpaca_modsac.yaml`, `alpaca_ppo_lpi.yaml`, `alpaca_ppo_ablation.yaml`

Each configures: `cv_method`, `n_folds`, `n_groups`, `n_test_groups`, `embargo_pct`, `test_ratio`, `gap_days`, and agent-specific hyperparameters.

---

## 3. López de Prado Evaluation Framework (`lopez_de_prado_analysis/`)

### 3.1 `lopez_de_prado_evaluation_drl.py` (NEW — 1,160 lines)

Advanced DRL evaluation using methods from *Advances in Financial ML* (2018):

- **`DRLEvaluator`** class:
  - `cpcv_split()`: CPCV with proper purging and embargo using `pred_times`/`eval_times`
  - `walk_forward_splits()`: Anchored (expanding-window) walk-forward
  - `evaluate_drl_checkpoint()`: Single checkpoint → portfolio metrics
  - `compute_drl_pbo()`: PBO across multiple checkpoints (uses `pypbo` library)
  - `compute_deflated_sharpe()`: DSR for selection bias correction
  - `drl_walk_forward_analysis()`: Full WFA with pre-trained fold checkpoints
  - `comprehensive_drl_evaluation()`: All-in-one pipeline (eval + PBO + DSR + report)
- `purge_proper()` / `embargo_proper()`: FinRL_Crypto methodology for time-series leakage prevention
- Demo functions for simulated and CPCV split visualization

### 3.2 `multi_agent_evaluation.py` (NEW — 574 lines)

Multi-agent/HPO comparison with PBO and DSR:

- Evaluates multiple agent directories or HPO trial results
- `compute_pbo_analysis()`: PBO with zero-variance filtering
- `compute_dsr_analysis()`: DSR for best agent selection bias
- CSV import for Hypersweeper results
- JSON export, summary tables, CLI with `--dirs`/`--pattern`/`--csv`

### 3.3 `test_drl_evaluation.py` (NEW — 266 lines)

Test harness for evaluating PPO checkpoints against López de Prado metrics.

### 3.4 Supporting Files

- `lopez_de_prado_evaluation.py`: General (non-DRL) evaluation framework
- `test_pbo.py`: PBO unit tests
- `pypbo/`: Bundled [pypbo](https://github.com/chrischoy/pypbo) library (canonical PBO/DSR implementation)
- `PYPBO_INTEGRATION.md`: Integration documentation

---

## 4. Tests

### 4.1 `unit_tests/envs/test_vec_normalize_stock.py` (NEW — 285 lines)

VecNormalize integration tests:

- Running statistics correctness (mean, variance, batch updates)
- Observation normalization (clipping, shape preservation)
- Save/load round-trip (statistics persistence)
- Training vs inference mode toggle
- Integration with StockTradingVecEnv

---

## 5. Key Bug Fixes

| Bug | Impact | Fix |
|---|---|---|
| `action = th.ones_like(action)` in StockTradingVecEnv.step() | Agent actions ignored — all envs bought everything every step | Removed debug line, use `action.clone()` |
| StockTradingVecEnv.reset() returned `state` not `(state, {})` | Incompatible with Gymnasium API | Return tuple |
| StockTradingVecEnv.step() returned 4-tuple | Missing `truncate` for Gymnasium 0.26+ | Return 5-tuple |
| AgentA2C 1D batch indexing | Crashed with VecEnv `(T, N, dim)` buffers | 2D sampling matching PPO |
| VecNormalize `__setattr__` not forwarding to inner env | `if_random_reset = False` didn't reach inner env → random starting positions at eval → inflated returns (85% → actual 20%) | Added `__setattr__` with forwarding set |
| Checkpoint auto-selection picked `vec_normalize.pt` over actor files | Eval tried to load normalization dict as actor model → crash | Filter `.pt` files to `s.startswith('actor')` |

---

## 6. Documentation Files (NEW)

- `docs/AUTORL_CHECKLIST.md`: AutoRL implementation checklist
- `docs/HPO_IMPLEMENTATION_PLAN.md`: Hypersweeper HPO integration plan
- `docs/HYPERPARAMETER_REFERENCE.md`: Agent hyperparameter reference
- `docs/DRLEnsembleAgent_Implementation_Plan.md`: Ensemble agent plan
