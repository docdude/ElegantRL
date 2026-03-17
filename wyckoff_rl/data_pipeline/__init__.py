"""
Wyckoff → ElegantRL preprocessing pipeline.

Converts raw SCID tick data through Wyckoff analysis into NPZ arrays
compatible with ElegantRL's StockTradingVecEnv / CPCV pipeline.

Modules:
    feature_extraction  — Phase 1: SCID → analyze_wyckoff() → NPZ
    feature_selection   — Phase 2: RiskLabAI denoising + PCA + MDI/MDA
    signal_extraction   — Phase 4B: Wyckoff events → triple barrier labels
    meta_labeling       — Phase 4B: Train meta-label classifier with CPCV
    bet_sizing          — Phase 4B: RiskLabAI bet sizing integration
    evaluation          — PBO + PSR backtest evaluation
"""
