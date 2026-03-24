"""
Compare range bars built from raw ticks (live path) vs 1s pre-resampled (training path).
Quantifies the divergence caused by 1s pre-aggregation in scid_parser.py.
"""
import sys, struct, os, time as time_module
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

sys.path.insert(0, '/opt/ElegantRL')

RECORD_DTYPE = np.dtype([
    ('datetime', '<u8'), ('open', '<f4'), ('high', '<f4'),
    ('low', '<f4'), ('close', '<f4'), ('num_trades', '<i4'),
    ('volume', '<i4'), ('bid_volume', '<i4'), ('ask_volume', '<i4'),
])
OLE_TO_UNIX_US = 25569 * 86400 * 1_000_000

scid_path = '/opt/SierraChart/Data/NQH26-CME.scid'
RANGE_SIZE = 40.0

print("=" * 70)
print("COMPARING: tick-by-tick (live) vs 1s-presample (training) range bars")
print("=" * 70)

# ── Parse SCID ──
with open(scid_path, 'rb') as f:
    header = f.read(56)
header_size = struct.unpack_from('<I', header, 4)[0]
record_size = struct.unpack_from('<I', header, 8)[0]
total = (os.path.getsize(scid_path) - header_size) // record_size
data = np.memmap(scid_path, dtype=RECORD_DTYPE, mode='r', offset=header_size, shape=(total,))
ole_us = data['datetime']
unix_us = ole_us.astype(np.int64) - OLE_TO_UNIX_US

start_ts = int(datetime(2026, 1, 15, tzinfo=timezone.utc).timestamp() * 1_000_000)
end_ts = int(datetime(2026, 3, 19, tzinfo=timezone.utc).timestamp() * 1_000_000)
start_idx = int(np.searchsorted(unix_us, start_ts))
end_idx = int(np.searchsorted(unix_us, end_ts))
n = end_idx - start_idx
print(f"\nSCID records in date range: {n:,}")

closes = np.array(data['close'][start_idx:end_idx], dtype=np.float64)
volumes = np.array(data['volume'][start_idx:end_idx], dtype=np.float64)
ask_vols = np.array(data['ask_volume'][start_idx:end_idx], dtype=np.float64)
bid_vols = np.array(data['bid_volume'][start_idx:end_idx], dtype=np.float64)
deltas_arr = (ask_vols - bid_vols).astype(np.int64)
num_trades_arr = np.array(data['num_trades'][start_idx:end_idx], dtype=np.int64)
timestamps_us = unix_us[start_idx:end_idx].astype(np.float64)
timestamps_s = timestamps_us / 1_000_000

# ══════════════════════════════════════════════════════════════════════
# PATH 1: Tick-by-tick (live / precompute path)
# ══════════════════════════════════════════════════════════════════════
print("\n[PATH 1] Building range bars from raw ticks (live path)...")
t0 = time_module.time()
from wyckoff_rl.live.range_bar_builder import RangeBarBuilder
builder = RangeBarBuilder(range_size=RANGE_SIZE)
last_price = 0.0
for i in range(n):
    price = closes[i]
    vol = volumes[i]
    if vol <= 0 or price <= 0:
        continue
    ask_v = ask_vols[i]
    is_uptick = ask_v > (vol - ask_v) if vol > 0 else (price >= last_price)
    last_price = price
    builder.on_tick(price, vol, is_uptick, timestamps_s[i])
tick_bars = builder.completed_bars
t1 = time_module.time()
print(f"  Tick bars: {len(tick_bars)} ({t1-t0:.1f}s)")

# ══════════════════════════════════════════════════════════════════════
# PATH 2: 1s pre-resample (training path from scid_parser.py)
# ══════════════════════════════════════════════════════════════════════
print("\n[PATH 2] Building range bars from 1s-presampled data (training path)...")
t0 = time_module.time()
tick_df = pd.DataFrame({
    'close': closes,
    'volume': volumes.astype(np.int64),
    'bid_volume': bid_vols.astype(np.int64),
    'ask_volume': ask_vols.astype(np.int64),
    'delta': deltas_arr,
    'num_trades': num_trades_arr,
}, index=pd.DatetimeIndex(pd.to_datetime(timestamps_us, unit='us', utc=True), name='datetime'))
del data  # release memmap

sec = tick_df.resample('1s').agg({
    'close': ['first', 'max', 'min', 'last'],
    'volume': 'sum', 'bid_volume': 'sum', 'ask_volume': 'sum',
    'delta': 'sum', 'num_trades': 'sum',
})
sec.columns = ['open', 'high', 'low', 'close', 'volume', 'bid_volume', 'ask_volume', 'delta', 'num_trades']
sec = sec.dropna(subset=['close'])
sec = sec[sec['volume'] > 0]
print(f"  1s bars: {len(sec):,}")

high = sec['high'].values.astype(np.float64)
low = sec['low'].values.astype(np.float64)
close_s = sec['close'].values.astype(np.float64)
n_sec = len(sec)

# Pure-Python boundary detection (matches _range_bar_boundaries_python in scid_parser.py)
bar_ends = []; bar_opens = []; bar_highs = []; bar_lows = []; bar_closes = []
bar_open = close_s[0]; bar_high = close_s[0]; bar_low = close_s[0]
for i in range(n_sec):
    if high[i] > bar_high: bar_high = high[i]
    if low[i] < bar_low: bar_low = low[i]
    if bar_high - bar_low >= RANGE_SIZE:
        if high[i] >= bar_open + RANGE_SIZE:
            bc = bar_open + RANGE_SIZE
        elif low[i] <= bar_open - RANGE_SIZE:
            bc = bar_open - RANGE_SIZE
        else:
            bc = close_s[i]
        bar_ends.append(i); bar_opens.append(bar_open)
        bar_highs.append(bar_high); bar_lows.append(bar_low); bar_closes.append(bc)
        bar_open = bc; bar_high = bc; bar_low = bc
t1 = time_module.time()
print(f"  1s-presample bars: {len(bar_ends)} ({t1-t0:.1f}s)")

# ══════════════════════════════════════════════════════════════════════
# COMPARISON
# ══════════════════════════════════════════════════════════════════════
n_tick = len(tick_bars)
n_1s = len(bar_ends)
n_min = min(n_tick, n_1s)

print(f"\n{'='*70}")
print(f"BAR COUNTS: {n_tick} (tick) vs {n_1s} (1s) — diff = {n_tick - n_1s}")
print(f"{'='*70}")

# Aligned comparison
mismatches_open = 0
mismatches_close = 0
mismatches_any = 0
max_open_diff = 0.0
max_close_diff = 0.0
first_diverge = None
diverge_details = []

for i in range(n_min):
    tb = tick_bars[i]
    o_diff = abs(tb.open - bar_opens[i])
    c_diff = abs(tb.close - bar_closes[i])
    h_diff = abs(tb.high - bar_highs[i])
    l_diff = abs(tb.low - bar_lows[i])

    if o_diff > 0.01: mismatches_open += 1
    if c_diff > 0.01: mismatches_close += 1
    if max(o_diff, c_diff, h_diff, l_diff) > 0.01:
        mismatches_any += 1
        if o_diff > max_open_diff: max_open_diff = o_diff
        if c_diff > max_close_diff: max_close_diff = c_diff
        if len(diverge_details) < 10:
            diverge_details.append((i, tb, bar_opens[i], bar_highs[i], bar_lows[i], bar_closes[i]))

print(f"\nBar-by-bar alignment (first {n_min} bars):")
print(f"  Open mismatches:  {mismatches_open:,} / {n_min:,} ({100*mismatches_open/n_min:.2f}%)")
print(f"  Close mismatches: {mismatches_close:,} / {n_min:,} ({100*mismatches_close/n_min:.2f}%)")
print(f"  Any OHLC mismatch: {mismatches_any:,} / {n_min:,} ({100*mismatches_any/n_min:.2f}%)")
print(f"  Max open diff:  {max_open_diff:.4f}")
print(f"  Max close diff: {max_close_diff:.4f}")

if diverge_details:
    print(f"\nFirst {len(diverge_details)} divergent bars:")
    for idx, tb, o1s, h1s, l1s, c1s in diverge_details:
        print(f"  Bar {idx:5d}: tick=[{tb.open:.2f} {tb.high:.2f} {tb.low:.2f} {tb.close:.2f}]"
              f"  1s=[{o1s:.2f} {h1s:.2f} {l1s:.2f} {c1s:.2f}]"
              f"  Δopen={tb.open-o1s:+.2f} Δclose={tb.close-c1s:+.2f}")
else:
    print("\n*** ALL bars match perfectly! ***")

# Spot-check drift at various points
print(f"\nDrift check at specific bars:")
for check in [10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, n_min-1]:
    if check >= n_min:
        break
    tb = tick_bars[check]
    o_diff = tb.open - bar_opens[check]
    c_diff = tb.close - bar_closes[check]
    print(f"  Bar {check:5d}: Δopen={o_diff:+.2f}  Δclose={c_diff:+.2f}"
          f"  tick_close={tb.close:.2f}  1s_close={bar_closes[check]:.2f}")

# ══════════════════════════════════════════════════════════════════════
# FEATURE DIVERGENCE (if bars differ)
# ══════════════════════════════════════════════════════════════════════
if mismatches_any > 0:
    print(f"\n{'='*70}")
    print("FEATURE DIVERGENCE ANALYSIS")
    print(f"{'='*70}")

    from wyckoff_effort.pipeline.wyckoff_features import build_all_features

    # Build features from tick bars
    tick_df_bars = pd.DataFrame([{
        'open': b.open, 'high': b.high, 'low': b.low, 'close': b.close,
        'volume': b.volume, 'delta': b.delta,
        'duration_seconds': b.duration_seconds,
        'num_trades': b.num_trades, 'cvd': b.cvd,
        'ask_volume': b.ask_volume, 'bid_volume': b.bid_volume,
    } for b in tick_bars])

    # Build features from 1s bars
    # Need volume etc from cumsum - reconstruct
    vol_arr = sec['volume'].values.astype(np.int64)
    bid_vol = sec['bid_volume'].values.astype(np.int64)
    ask_vol = sec['ask_volume'].values.astype(np.int64)
    delta_1s = sec['delta'].values.astype(np.int64)
    nt_arr = sec['num_trades'].values.astype(np.int64)
    ts_arr = sec.index.values

    cum_vol = np.concatenate([[0], np.cumsum(vol_arr)])
    cum_bid = np.concatenate([[0], np.cumsum(bid_vol)])
    cum_ask = np.concatenate([[0], np.cumsum(ask_vol)])
    cum_delta = np.concatenate([[0], np.cumsum(delta_1s)])
    cum_nt = np.concatenate([[0], np.cumsum(nt_arr)])

    ends = np.array(bar_ends, dtype=np.int64)
    starts = np.empty(len(ends), dtype=np.int64)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1

    start_ts_ns = ts_arr[starts].astype(np.int64)
    end_ts_ns = ts_arr[ends].astype(np.int64)
    dur_ns = end_ts_ns - start_ts_ns
    dur_s = np.maximum(dur_ns / 1e9, 0.1)

    onesec_df_bars = pd.DataFrame({
        'open': bar_opens,
        'high': bar_highs,
        'low': bar_lows,
        'close': bar_closes,
        'volume': cum_vol[ends + 1] - cum_vol[starts],
        'bid_volume': cum_bid[ends + 1] - cum_bid[starts],
        'ask_volume': cum_ask[ends + 1] - cum_ask[starts],
        'delta': cum_delta[ends + 1] - cum_delta[starts],
        'num_trades': cum_nt[ends + 1] - cum_nt[starts],
        'duration_seconds': dur_s,
    })
    onesec_df_bars['cvd'] = onesec_df_bars['delta'].cumsum()

    print("Computing features for tick-bar path...")
    feat_tick, _, names_df = build_all_features(tick_df_bars, reversal_points=40.0)
    names = names_df.columns.tolist()
    print("Computing features for 1s-bar path...")
    feat_1s, _, _ = build_all_features(onesec_df_bars, reversal_points=40.0)

    n_feat_min = min(len(feat_tick), len(feat_1s))
    n_features = feat_tick.shape[1]
    print(f"\nFeature matrix: tick={feat_tick.shape}, 1s={feat_1s.shape}")
    print(f"Comparing {n_feat_min} aligned bars across {n_features} features...")

    # Skip first 30 bars (warmup)
    start = 30
    feat_tick_cmp = feat_tick[start:n_feat_min]
    feat_1s_cmp = feat_1s[start:n_feat_min]

    abs_diff = np.abs(feat_tick_cmp - feat_1s_cmp)
    mean_diff = np.nanmean(abs_diff, axis=0)
    max_diff = np.nanmax(abs_diff, axis=0)
    pct_nonzero = np.mean(abs_diff > 0.001, axis=0) * 100

    from wyckoff_rl.live.live_features import TRAINING_FEATURE_INDICES

    print(f"\n{'Idx':>3} {'Feature':<30} {'MeanDiff':>10} {'MaxDiff':>10} {'%Differ':>8}  {'USED':>4}")
    print("-" * 75)
    for j in range(n_features):
        used = "***" if j in TRAINING_FEATURE_INDICES else ""
        if max_diff[j] > 0.001:
            print(f"{j:3d} {names[j] if j < len(names) else '?':<30} {mean_diff[j]:10.6f} {max_diff[j]:10.4f} {pct_nonzero[j]:7.1f}%  {used}")

    # Summary of used features only
    used_idx = sorted(TRAINING_FEATURE_INDICES)
    used_mean = mean_diff[used_idx]
    used_max = max_diff[used_idx]
    print(f"\n{'='*70}")
    print(f"USED FEATURES ONLY ({len(used_idx)} features):")
    print(f"  Overall mean abs diff: {np.nanmean(used_mean):.6f}")
    print(f"  Overall max abs diff:  {np.nanmax(used_max):.4f}")
    print(f"  Features with any diff > 0.001: {np.sum(used_max > 0.001)}/{len(used_idx)}")
    print(f"  Features with any diff > 0.01:  {np.sum(used_max > 0.01)}/{len(used_idx)}")
    print(f"  Features with any diff > 0.1:   {np.sum(used_max > 0.1)}/{len(used_idx)}")

print(f"\n{'='*70}")
print("DONE")
