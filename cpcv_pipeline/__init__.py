"""
CPCV Pipeline — Leak-free cross-validation for DRL trading agents.

Modules:
    function_CPCV:      CombPurgedKFoldCV splits and backtest path generation
    function_train_test: Pre-sliced training/evaluation with dynamic env factory
    config:             Centralized settings
    optimize_cpcv:      CPCV training script
    optimize_wf:        Walk-forward training script
    optimize_kcv:       K-fold CV training script
    test_leakage:       Verification tests (run before GPU training)
"""
