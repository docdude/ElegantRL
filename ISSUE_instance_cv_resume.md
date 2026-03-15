## Feature Request: Instance-based CV support and resume-safe state management

* HyperSweeper version: 0.0.1 (main branch, commit 6918ecb)
* Python version: 3.12
* Operating System: Ubuntu 22.04

### Description

When using HyperSMAC with SMAC's instance-based cross-validation (where each CV fold is a SMAC instance), two issues arise:

1. **Instance field is not passed through the ask/tell pipeline.** `HyperSMACAdapter.ask()` does not forward `smac_info.instance` to the `Info` dataclass, and `tell()` does not pass `info.instance` back to `TrialInfo`. The `Info` dataclass in `utils.py` also lacks an `instance` field. Similarly, `run_configs()` in `HypersweeperSweeper` never injects `instance=` into the Hydra overrides. This means SMAC's intensifier cannot track per-instance evaluations, breaking instance-based intensification (e.g., running CV folds as separate SMAC instances for early termination of poor configs).

2. **On process restart, `trials_run` and `job_idx` reset to 0.** SMAC correctly resumes from its persisted `runhistory.json` and `intensifier.json` via `SMBO._initialize_state()`, but `HypersweeperSweeper.__init__()` always sets `self.trials_run = 0` and `self.job_idx = 0`. This means the sweeper gets a fresh `n_trials` budget on every restart, re-running evaluations that SMAC has already completed. Hydra's `job.num` also restarts from 0, potentially overwriting output directories.

Additionally, `make_smac()` fails when the scenario dict contains OmegaConf `DictConfig`/`ListConfig` objects (e.g., `instances: ["0", "1", ...]`), because SMAC's internal JSON serialization doesn't handle OmegaConf types.

### Steps/Code to Reproduce

**Instance support:**

```yaml
# configs/mlp_smac_cv.yaml
hydra:
  sweeper:
    n_trials: 50
    sweeper_kwargs:
      max_parallelization: 0
      optimizer_kwargs:
        intensifier:
          _target_: smac.intensifier.intensifier.Intensifier
          _partial_: true
          max_config_calls: 5  # 5 CV folds
        scenario:
          instances: ["0", "1", "2", "3", "4"]
          instance_features:
            "0": [0]
            "1": [1]
            "2": [2]
            "3": [3]
            "4": [4]
```

```python
# Target function expects instance as a fold index
@hydra.main(config_path="configs", config_name="mlp_smac_cv")
def mlp_from_cfg(cfg):
    instance = cfg.get("instance", None)  # <-- never set by hypersweeper
    fold_idx = int(instance)
    # ... evaluate only this fold
```

**Resume:**

```bash
# Run 1: completes 50 evaluations
python -m examples.mlp --config-name=mlp_smac_cv -m

# Run 2: restart — should continue from eval 51, but starts from 0
python -m examples.mlp --config-name=mlp_smac_cv -m
```

### Expected Results

1. The target function receives the `instance` value from SMAC's intensifier, enabling per-fold CV evaluation and early termination of bad configs.
2. On restart, `trials_run` and `job_idx` initialize from SMAC's persisted runhistory count, so the sweeper continues from where it left off.

### Actual Results

1. `instance` is always `None` in the target function. SMAC's intensifier cannot distribute configs across CV folds.
2. On restart, the sweeper re-runs the full `n_trials` budget from scratch (even though SMAC internally skips duplicate evaluations, the sweeper's termination counter is wrong).
3. `make_smac()` crashes with `TypeError: Object of type DictConfig is not JSON serializable` when `instances` or `instance_features` are OmegaConf objects.

### Proposed Fix

I have a working implementation in my fork: https://github.com/docdude/hypersweeper/commit/548bfc8

**Changes:**

| File | Change |
|------|--------|
| `utils.py` | Add `instance` field to `Info` dataclass |
| `hyper_smac.py` | Forward `instance` in `ask()` and `tell()` |
| `hyper_smac.py` | Add `get_n_completed_trials()` to `HyperSMACAdapter` |
| `hyper_smac.py` | Convert OmegaConf → plain dict before passing to `Scenario()` |
| `hyper_smac.py` | Add `instance=None` param to `dummy_func` |
| `hypersweeper_sweeper.py` | Add `_restore_state_from_optimizer()` — restores `trials_run`/`job_idx` from optimizer state on init |
| `hypersweeper_sweeper.py` | Forward `instance` in `run_configs()` overrides |
| `examples/` | Add `mlp_smac_cv.yaml` example config for instance-based CV |
| `examples/mlp.py` | Add per-fold evaluation when `instance` is set |

The resume mechanism is generic: any optimizer adapter can implement `get_n_completed_trials()` and the sweeper will use it. Happy to open a PR if there's interest.

### Additional Info

- Did you try upgrading to the most current version? Yes (main branch at 6918ecb)
- Are you using a supported operating system (version)? Yes (Ubuntu 22.04)
- How did you install this package? GitHub (editable install from source)
