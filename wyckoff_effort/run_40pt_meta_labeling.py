"""
Run meta-labeling pipeline on the CORRECT data:
  - 40pt NQ range bars (7,635 bars) — the actual RL training timeframe
  - All 58 Wyckoff features (not the original 21)

Previous run was on 4pt bars (442K bars × 21 features) — wrong timeframe.
"""

import logging
import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wyckoff_rl.data_pipeline.meta_labeling import (
    train_meta_label_model,
    save_meta_label_model,
)
from wyckoff_rl.data_pipeline.signal_extraction import (
    compute_atr,
    compute_sample_weights,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_output")


def build_40pt_dataframe():
    """Build full DataFrame from 40pt NPZ + parquet OHLCV."""
    npz_path = os.path.join(OUTPUT_DIR, "wyckoff_nq_40pt.npz")
    bars_path = os.path.join(OUTPUT_DIR, "wyckoff_nq_40pt_bars.parquet")

    data = np.load(npz_path, allow_pickle=True)
    tech = data["tech_ary"]          # (7635, 58)
    closes = data["close_ary"][:, 0] # (7635,)
    feature_names = [str(n) for n in data["feature_names"]]

    bars = pd.read_parquet(bars_path)
    assert len(bars) == len(tech), f"Shape mismatch: bars={len(bars)}, npz={len(tech)}"

    # Combine OHLCV with all 58 features
    df = bars[["open", "high", "low", "close", "volume"]].copy()
    df = df.reset_index(drop=True)  # integer index for bar positions
    for i, name in enumerate(feature_names):
        df[name] = tech[:, i]

    logger.info(f"Built 40pt DataFrame: {df.shape[0]} bars × {df.shape[1]} columns")
    return df, feature_names


def extract_signals_from_scores(df, feature_names, min_score=0.0):
    """
    Extract Wyckoff signals from continuous score features.

    Maps score > threshold to signal events.
    """
    score_to_side = {
        "spring_score": +1,       # long (Spring = buying opportunity)
        "upthrust_score": -1,     # short (Upthrust = selling opportunity)
        "sc_score": +1,           # long (SellingClimax = exhaustion of selling)
        "bc_score": -1,           # short (BuyingClimax = exhaustion of buying)
    }

    events = []
    for score_col, side in score_to_side.items():
        if score_col not in feature_names:
            logger.warning(f"Missing score column: {score_col}")
            continue

        mask = df[score_col] > min_score
        event_bars = df[mask]
        event_type = score_col.replace("_score", "").capitalize()

        for idx in event_bars.index:
            row = df.loc[idx]
            event = {
                "bar_idx": idx,
                "event_type": event_type,
                "side": side,
                "close": row["close"],
                "score": row[score_col],  # keep the score for analysis
            }
            # Capture ALL 58 feature snapshots
            for feat in feature_names:
                event[feat] = row[feat]
            events.append(event)

    signals_df = pd.DataFrame(events)
    if len(signals_df) > 0:
        signals_df = signals_df.sort_values("bar_idx").reset_index(drop=True)

    # Show distributions
    for et in score_to_side:
        name = et.replace("_score", "").capitalize()
        n = len(signals_df[signals_df["event_type"] == name]) if len(signals_df) > 0 else 0
        logger.info(f"  {name}: {n} signals")

    logger.info(f"Total signals extracted: {len(signals_df)}")
    return signals_df


def apply_triple_barrier_40pt(
    df, signals_df,
    pt_multiplier=1.5, sl_multiplier=1.0,
    vertical_bars=20, atr_period=20,
):
    """
    Triple barrier for 40pt bars.
    vertical_bars=20 (not 50) because each bar is 10× larger than 4pt bars.
    """
    atr = compute_atr(df, period=atr_period)
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    n_bars = len(df)

    labels, returns, exit_bars_list, exit_types = [], [], [], []

    for _, signal in signals_df.iterrows():
        entry_idx = int(signal["bar_idx"])
        entry_price = closes[entry_idx]
        side = signal["side"]
        barrier_width = atr.iloc[entry_idx]

        if side > 0:
            tp_price = entry_price + pt_multiplier * barrier_width
            sl_price = entry_price - sl_multiplier * barrier_width
        else:
            tp_price = entry_price - pt_multiplier * barrier_width
            sl_price = entry_price + sl_multiplier * barrier_width

        max_bar = min(entry_idx + vertical_bars, n_bars - 1)
        exit_bar = max_bar
        exit_type = "vertical"

        for j in range(entry_idx + 1, max_bar + 1):
            if side > 0:
                if highs[j] >= tp_price:
                    exit_bar = j
                    exit_type = "tp"
                    break
                if lows[j] <= sl_price:
                    exit_bar = j
                    exit_type = "sl"
                    break
            else:
                if lows[j] <= tp_price:
                    exit_bar = j
                    exit_type = "tp"
                    break
                if highs[j] >= sl_price:
                    exit_bar = j
                    exit_type = "sl"
                    break

        exit_price = closes[exit_bar]
        ret = side * np.log(exit_price / entry_price)
        label = 1 if exit_type == "tp" else 0

        labels.append(label)
        returns.append(ret)
        exit_bars_list.append(exit_bar)
        exit_types.append(exit_type)

    labeled_df = signals_df.copy()
    labeled_df["label"] = labels
    labeled_df["ret"] = returns
    labeled_df["exit_bar"] = exit_bars_list
    labeled_df["exit_type"] = exit_types

    n_tp = sum(1 for t in exit_types if t == "tp")
    n_sl = sum(1 for t in exit_types if t == "sl")
    n_vert = sum(1 for t in exit_types if t == "vertical")
    wr = sum(labels) / max(len(labels), 1) * 100
    logger.info(
        f"Triple barrier: {len(labeled_df)} events → "
        f"TP={n_tp} ({100*n_tp/len(labels):.1f}%), "
        f"SL={n_sl} ({100*n_sl/len(labels):.1f}%), "
        f"Vertical={n_vert} ({100*n_vert/len(labels):.1f}%), "
        f"Win rate={wr:.1f}%"
    )
    return labeled_df


def main():
    print("=" * 70)
    print("META-LABELING ON 40pt NQ BARS (CORRECT DATA)")
    print("=" * 70)

    # ── 1. Build DataFrame ──
    df, feature_names = build_40pt_dataframe()

    # ── 2. Extract signals ──
    # Use score > 0 to match "event detected" semantics
    signals = extract_signals_from_scores(df, feature_names, min_score=0.0)

    if len(signals) == 0:
        logger.error("No signals extracted!")
        return

    # Remove duplicate bar_idx (keep highest score if multiple events on same bar)
    before = len(signals)
    signals = signals.sort_values("score", ascending=False).drop_duplicates("bar_idx", keep="first")
    signals = signals.sort_values("bar_idx").reset_index(drop=True)
    logger.info(f"Deduplicated: {before} → {len(signals)} signals (removed {before-len(signals)} overlapping)")

    # ── 3. Triple barrier labeling ──
    print("\n--- Triple Barrier Labeling ---")
    labeled = apply_triple_barrier_40pt(
        df, signals,
        pt_multiplier=1.5,
        sl_multiplier=1.0,
        vertical_bars=20,  # ~20 × 40pt = 800pt max holding
    )

    # Quick return analysis
    print(f"\nReturn stats:")
    print(f"  Mean: {labeled['ret'].mean():.5f}")
    print(f"  Std:  {labeled['ret'].std():.5f}")
    print(f"  Sharpe (per-trade): {labeled['ret'].mean()/labeled['ret'].std():.3f}")

    by_type = labeled.groupby("event_type").agg(
        n=("label", "count"),
        win_rate=("label", "mean"),
        avg_ret=("ret", "mean"),
    )
    print(f"\nBy event type:\n{by_type.to_string()}")

    # ── 4. Sample weights ──
    weights = compute_sample_weights(labeled, n_total_bars=len(df))

    # ── 5. Meta-labeling with ALL 58 features ──
    print("\n--- Meta-Labeling (RF, CPCV, 58 features) ---")
    result = train_meta_label_model(
        labeled_df=labeled,
        feature_columns=feature_names,  # ALL 58 features
        sample_weights=weights,
        model_type="rf",
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=20,
        n_splits=5,
        n_test_groups=2,
        embargo_pct=0.01,
    )

    # ── 6. Results ──
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    cv = result["cv_results"]
    print(f"\nCV Performance ({len(cv)} folds):")
    print(f"  F1:       {cv['f1'].mean():.3f} ± {cv['f1'].std():.3f}")
    print(f"  Precision:{cv['precision'].mean():.3f} ± {cv['precision'].std():.3f}")
    print(f"  Recall:   {cv['recall'].mean():.3f} ± {cv['recall'].std():.3f}")
    print(f"  LogLoss:  {cv['log_loss'].mean():.3f} ± {cv['log_loss'].std():.3f}")

    fi = result["feature_importance"]
    print(f"\nTop 15 Features (MDI importance):")
    for i, (feat, row) in enumerate(fi.head(15).iterrows()):
        print(f"  {i+1:2d}. {feat:30s}  {row['Mean']:.4f} ± {row['Std']:.4f}")

    # OOS probability distribution
    oos = result["oos_predictions"]
    valid = oos[~np.isnan(oos)]
    print(f"\nOOS Probability Distribution:")
    print(f"  Mean: {np.mean(valid):.4f}")
    print(f"  Std:  {np.std(valid):.4f}")
    print(f"  P5:   {np.percentile(valid, 5):.4f}")
    print(f"  P25:  {np.percentile(valid, 25):.4f}")
    print(f"  P50:  {np.percentile(valid, 50):.4f}")
    print(f"  P75:  {np.percentile(valid, 75):.4f}")
    print(f"  P95:  {np.percentile(valid, 95):.4f}")

    # Compare high-prob vs low-prob trades
    if len(valid) > 100:
        labeled_with_prob = labeled.copy()
        labeled_with_prob["meta_prob"] = oos
        valid_mask = ~np.isnan(oos)
        labeled_valid = labeled_with_prob[valid_mask]

        p50 = np.median(valid)
        high = labeled_valid[labeled_valid["meta_prob"] >= p50]
        low = labeled_valid[labeled_valid["meta_prob"] < p50]

        print(f"\nMeta-Label Discrimination:")
        print(f"  High P(success) (>= {p50:.3f}): n={len(high)}, WR={high['label'].mean():.3f}, avg_ret={high['ret'].mean():.5f}")
        print(f"  Low  P(success) (<  {p50:.3f}): n={len(low)},  WR={low['label'].mean():.3f}, avg_ret={low['ret'].mean():.5f}")

        # Quartile analysis
        print(f"\n  Quartile breakdown:")
        for q_low, q_high, name in [(0, 25, "Q1"), (25, 50, "Q2"), (50, 75, "Q3"), (75, 100, "Q4")]:
            lo = np.percentile(valid, q_low)
            hi = np.percentile(valid, q_high)
            mask = (labeled_valid["meta_prob"] >= lo) & (labeled_valid["meta_prob"] <= hi)
            q_data = labeled_valid[mask]
            if len(q_data) > 0:
                print(f"    {name} [{lo:.3f}-{hi:.3f}]: n={len(q_data)}, WR={q_data['label'].mean():.3f}, avg_ret={q_data['ret'].mean():.6f}")

    # ── 7. Save ──
    save_path = os.path.join(OUTPUT_DIR, "meta_label_model_40pt.pkl")
    save_meta_label_model(result, path=save_path)

    labeled.to_parquet(os.path.join(OUTPUT_DIR, "phase4b_labeled_signals_40pt.parquet"))

    # Also run with just the 33 selected features for comparison
    print("\n\n--- COMPARISON: Meta-Labeling with 33 SELECTED features ---")
    from wyckoff_rl.feature_config import SELECTED_FEATURES
    selected = [f for f in SELECTED_FEATURES if f in feature_names]
    logger.info(f"Using {len(selected)} of {len(SELECTED_FEATURES)} selected features")

    result_33 = train_meta_label_model(
        labeled_df=labeled,
        feature_columns=selected,
        sample_weights=weights,
        model_type="rf",
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=20,
        n_splits=5,
        n_test_groups=2,
        embargo_pct=0.01,
    )

    cv33 = result_33["cv_results"]
    print(f"\nCV Performance (33 features, {len(cv33)} folds):")
    print(f"  F1:       {cv33['f1'].mean():.3f} ± {cv33['f1'].std():.3f}")
    print(f"  Precision:{cv33['precision'].mean():.3f} ± {cv33['precision'].std():.3f}")
    print(f"  Recall:   {cv33['recall'].mean():.3f} ± {cv33['recall'].std():.3f}")
    print(f"  LogLoss:  {cv33['log_loss'].mean():.3f} ± {cv33['log_loss'].std():.3f}")

    fi33 = result_33["feature_importance"]
    print(f"\nTop 15 Features (33-feature model):")
    for i, (feat, row) in enumerate(fi33.head(15).iterrows()):
        print(f"  {i+1:2d}. {feat:30s}  {row['Mean']:.4f} ± {row['Std']:.4f}")

    oos33 = result_33["oos_predictions"]
    valid33 = oos33[~np.isnan(oos33)]
    print(f"\n33-feature OOS prob std: {np.std(valid33):.4f} (was 0.025 on 4pt)")

    save_meta_label_model(result_33, path=os.path.join(OUTPUT_DIR, "meta_label_model_40pt_33feat.pkl"))
    print("\nDone! Saved models to pipeline_output/")


if __name__ == "__main__":
    main()
