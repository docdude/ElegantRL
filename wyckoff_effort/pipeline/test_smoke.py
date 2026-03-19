#!/usr/bin/env python3
"""
Smoke test for all preprocessing scripts.
Run: python -m wyckoff_effort.pipeline.test_smoke
"""
import logging
import traceback
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = []


def report(name, passed, detail=""):
    tag = PASS if passed else FAIL
    results.append((name, passed))
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))


# ── Synthetic data ───────────────────────────────────────────────────────────
np.random.seed(42)
N_BARS = 500
N_FEAT = 22  # matches len(WYCKOFF_FEATURE_COLUMNS)

from wyckoff_effort.pipeline.config import WYCKOFF_FEATURE_COLUMNS

tech_ary = np.random.randn(N_BARS, N_FEAT).astype(np.float32)
labels = np.random.randint(0, 2, N_BARS)

# Synthetic analyzed DataFrame for signal_extraction
prices = 20000 + np.cumsum(np.random.randn(N_BARS) * 10)
df_analyzed = pd.DataFrame({
    "close": prices,
    "high": prices + np.abs(np.random.randn(N_BARS) * 5),
    "low": prices - np.abs(np.random.randn(N_BARS) * 5),
    "volume": np.random.randint(100, 10000, N_BARS),
    "delta": np.random.randn(N_BARS) * 500,
    "cvd": np.cumsum(np.random.randn(N_BARS) * 100),
    "Spring": 0,
    "Upthrust": 0,
    "SellingClimax": 0,
    "BuyingClimax": 0,
    "VolumeStrength": np.random.randint(0, 5, N_BARS),
})
# Inject some events
for col in ["Spring", "Upthrust", "SellingClimax", "BuyingClimax"]:
    idx = np.random.choice(N_BARS, size=8, replace=False)
    df_analyzed.loc[idx, col] = 1

# Add all WYCKOFF_FEATURE_COLUMNS
for c in WYCKOFF_FEATURE_COLUMNS:
    if c not in df_analyzed.columns:
        df_analyzed[c] = np.random.randn(N_BARS)

# ═════════════════════════════════════════════════════════════════════════════
print("\n=== 1. signal_extraction (no RiskLabAI) ===")
# ═════════════════════════════════════════════════════════════════════════════

signals_df = None
labeled_df = None
sample_weights = None

try:
    from wyckoff_effort.pipeline.signal_extraction import extract_wyckoff_signals
    signals_df = extract_wyckoff_signals(df_analyzed)
    report("extract_wyckoff_signals", len(signals_df) > 0, f"{len(signals_df)} signals")
except Exception as e:
    report("extract_wyckoff_signals", False, str(e))
    traceback.print_exc()

try:
    from wyckoff_effort.pipeline.signal_extraction import apply_triple_barrier
    labeled_df = apply_triple_barrier(df_analyzed, signals_df)
    n_tp = (labeled_df["label"] == 1).sum()
    report("apply_triple_barrier", len(labeled_df) > 0,
           f"{len(labeled_df)} labeled, {n_tp} TP")
except Exception as e:
    report("apply_triple_barrier", False, str(e))
    traceback.print_exc()

try:
    from wyckoff_effort.pipeline.signal_extraction import compute_sample_weights
    sample_weights = compute_sample_weights(labeled_df, N_BARS)
    report("compute_sample_weights", sample_weights is not None,
           f"shape={sample_weights.shape}")
except Exception as e:
    report("compute_sample_weights", False, str(e))
    traceback.print_exc()

# ═════════════════════════════════════════════════════════════════════════════
print("\n=== 2. feature_selection (RiskLabAI: denoise, PCA, MDI, MDA) ===")
# ═════════════════════════════════════════════════════════════════════════════

try:
    from wyckoff_effort.pipeline.feature_selection import denoise_features
    cov = denoise_features(tech_ary)
    report("denoise_features (RMT)", cov.shape == (N_FEAT, N_FEAT),
           f"shape={cov.shape}")
except Exception as e:
    report("denoise_features (RMT)", False, str(e))
    traceback.print_exc()

try:
    from wyckoff_effort.pipeline.feature_selection import pca_features
    ortho, eigen = pca_features(tech_ary)
    report("pca_features", ortho.shape[0] == N_BARS,
           f"shape={ortho.shape}")
except Exception as e:
    report("pca_features", False, str(e))
    traceback.print_exc()

try:
    from wyckoff_effort.pipeline.feature_selection import compute_feature_importance_mdi
    mdi = compute_feature_importance_mdi(tech_ary, labels)
    report("compute_feature_importance_mdi", len(mdi) > 0,
           f"top={mdi.index[0]}")
except Exception as e:
    report("compute_feature_importance_mdi", False, str(e))
    traceback.print_exc()

try:
    from wyckoff_effort.pipeline.feature_selection import compute_feature_importance_mda
    mda = compute_feature_importance_mda(tech_ary, labels)
    report("compute_feature_importance_mda", len(mda) > 0,
           f"top={mda.index[0]}")
except Exception as e:
    report("compute_feature_importance_mda", False, str(e))
    traceback.print_exc()

# ═════════════════════════════════════════════════════════════════════════════
print("\n=== 3. meta_labeling (RiskLabAI: CPCV) ===")
# ═════════════════════════════════════════════════════════════════════════════

try:
    from wyckoff_effort.pipeline.meta_labeling import train_meta_label_model
    if labeled_df is not None and len(labeled_df) >= 20:
        meta_result = train_meta_label_model(labeled_df, n_splits=3, n_test_groups=1)
        report("train_meta_label_model (CPCV)", meta_result["model"] is not None,
               f"F1={meta_result['cv_results']['f1'].mean():.3f}")
    else:
        report("train_meta_label_model (CPCV)", False, "not enough labeled signals")
except Exception as e:
    report("train_meta_label_model (CPCV)", False, str(e))
    traceback.print_exc()

# ═════════════════════════════════════════════════════════════════════════════
print("\n=== 4. bet_sizing ===")
# ═════════════════════════════════════════════════════════════════════════════

probs = np.random.rand(20)
sides = np.random.choice([-1, 1], size=20).astype(float)

try:
    from wyckoff_effort.pipeline.bet_sizing import probability_to_bet_size
    sizes_k = probability_to_bet_size(probs, sides, method="kelly")
    report("bet_sizing (kelly)", sizes_k is not None,
           f"avg_abs={np.abs(sizes_k).mean():.3f}")
except Exception as e:
    report("bet_sizing (kelly)", False, str(e))
    traceback.print_exc()

try:
    sizes_t = probability_to_bet_size(probs, sides, method="tiered")
    report("bet_sizing (tiered)", sizes_t is not None,
           f"avg_abs={np.abs(sizes_t).mean():.3f}")
except Exception as e:
    report("bet_sizing (tiered)", False, str(e))
    traceback.print_exc()

try:
    sizes_c = probability_to_bet_size(probs, sides, method="cdf")
    report("bet_sizing (cdf / RiskLabAI)", sizes_c is not None,
           f"avg_abs={np.abs(sizes_c).mean():.3f}")
except Exception as e:
    report("bet_sizing (cdf / RiskLabAI)", False, str(e))
    traceback.print_exc()

try:
    from wyckoff_effort.pipeline.bet_sizing import compute_average_active_bet_size
    fake_signals = pd.DataFrame({
        "bar_idx": np.arange(0, 40, 2),
        "exit_bar": np.arange(0, 40, 2) + 5,
    })
    avg = compute_average_active_bet_size(fake_signals, np.random.rand(20), 50)
    report("compute_average_active_bet_size (RiskLabAI)", avg is not None,
           f"shape={avg.shape}")
except Exception as e:
    report("compute_average_active_bet_size (RiskLabAI)", False, str(e))
    traceback.print_exc()

# ═════════════════════════════════════════════════════════════════════════════
print("\n=== 5. evaluation (RiskLabAI: Sharpe, PSR, PBO) ===")
# ═════════════════════════════════════════════════════════════════════════════

returns = np.random.randn(252) * 0.01

try:
    from wyckoff_effort.pipeline.evaluation import compute_sharpe
    sr = compute_sharpe(returns)
    report("compute_sharpe (RiskLabAI)", isinstance(sr, float), f"SR={sr:.4f}")
except Exception as e:
    report("compute_sharpe (RiskLabAI)", False, str(e))
    traceback.print_exc()

try:
    from wyckoff_effort.pipeline.evaluation import compute_sortino
    so = compute_sortino(returns)
    report("compute_sortino", isinstance(so, float), f"Sortino={so:.4f}")
except Exception as e:
    report("compute_sortino", False, str(e))
    traceback.print_exc()

try:
    from wyckoff_effort.pipeline.evaluation import compute_max_drawdown, compute_calmar
    mdd = compute_max_drawdown(returns)
    cal = compute_calmar(returns)
    report("compute_max_drawdown / calmar", True,
           f"MDD={mdd:.4f}, Calmar={cal:.4f}")
except Exception as e:
    report("compute_max_drawdown / calmar", False, str(e))
    traceback.print_exc()

try:
    from wyckoff_effort.pipeline.evaluation import compute_psr
    psr = compute_psr(returns, benchmark_sharpe=0.0)
    report("compute_psr (RiskLabAI)", isinstance(psr, float), f"PSR={psr:.4f}")
except Exception as e:
    report("compute_psr (RiskLabAI)", False, str(e))
    traceback.print_exc()

try:
    from wyckoff_effort.pipeline.evaluation import compute_pbo
    variants = np.random.randn(252, 4) * 0.01
    pbo, logits = compute_pbo(variants, n_partitions=4)
    report("compute_pbo (RiskLabAI)", isinstance(pbo, float), f"PBO={pbo:.4f}")
except Exception as e:
    report("compute_pbo (RiskLabAI)", False, str(e))
    traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
passed = sum(1 for _, p in results if p)
total = len(results)
print(f"Results: {passed}/{total} passed")
if passed < total:
    print("Failed:")
    for name, p in results:
        if not p:
            print(f"  - {name}")
print("=" * 60)
