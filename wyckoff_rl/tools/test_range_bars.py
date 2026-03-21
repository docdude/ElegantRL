"""Test 40-range bar construction + Wyckoff analysis + Deep-M plot on real NQ SCID data."""
import argparse
import os
import sys
import time

# Add ElegantRL root to path so wyckoff_effort.utils is importable
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from wyckoff_effort.utils.scid_parser import SCIDReader, resample_range_bars
from wyckoff_effort.utils.wyckoff_analyzer import analyze_wyckoff
from wyckoff_effort.utils.plotter import plot_deep_m_effort

def main():
    parser = argparse.ArgumentParser(description='Wyckoff Weis Wave range-bar analysis')
    parser.add_argument('--scid', default='/opt/SierraChart/Data/NQH26-CME.scid', help='Path to SCID file')
    parser.add_argument('--date', default='2025-12-18', help='Date to analyze (YYYY-MM-DD)')
    parser.add_argument('--range-size', type=float, default=10.0, help='Range bar size in points')
    parser.add_argument('--tick-size', type=float, default=0.25, help='Instrument tick size')
    parser.add_argument('--reversal-pct', type=float, default=0.5, help='Wave reversal threshold %%')
    parser.add_argument('--zigzag', action='store_true', default=True,
                        help='Show ZigZag trendline (default: on)')
    parser.add_argument('--no-zigzag', action='store_true', help='Hide ZigZag trendline')
    parser.add_argument('--wave-compare', action='store_true', default=True,
                        help='Show wave-to-wave comparison / yellow bars (default: on)')
    parser.add_argument('--no-wave-compare', action='store_true', help='Hide wave comparison annotations')
    parser.add_argument('--wave-time', action='store_true', default=True,
                        help='Show wave time duration (default: on)')
    parser.add_argument('--no-wave-time', action='store_true', help='Hide wave time in annotations')
    parser.add_argument('--output', default='deep_m_effort_nq.html', help='Output HTML file')
    parser.add_argument('--last-n', type=int, default=0, help='Show last N bars (0 = all)')
    args = parser.parse_args()

    # Handle --no-* flags
    if args.no_zigzag:
        args.zigzag = False
    if args.no_wave_compare:
        args.wave_compare = False
    if args.no_wave_time:
        args.wave_time = False
    # Parse ticks
    t0 = time.time()
    reader = SCIDReader(args.scid)
    ticks = reader.read()
    t1 = time.time()
    print(f"Parsed {len(ticks):,} ticks in {t1-t0:.1f}s")

    # Build range bars
    t2 = time.time()
    ticks_per_range = int(args.range_size / args.tick_size)
    bars = resample_range_bars(ticks, range_size=args.range_size, tick_size=args.tick_size)
    t3 = time.time()
    print(f"Built {len(bars):,} range bars ({ticks_per_range}-range / {args.range_size}pt) in {t3-t2:.1f}s")
    print(f"Date range: {bars.index[0]} to {bars.index[-1]}")
    print(f"\nVolume stats:")
    print(bars['volume'].describe())

    # Filter to target date
    import pandas as pd
    day_start = pd.Timestamp(args.date)
    day_end = day_start + pd.Timedelta(days=1)
    day_bars = bars[(bars.index >= day_start) & (bars.index < day_end)].copy()
    print(f"\n{args.date} range bars: {len(day_bars)}")

    # Wyckoff analysis
    t4 = time.time()
    analyzed = analyze_wyckoff(day_bars, reversal_pct=args.reversal_pct)
    t5 = time.time()
    print(f"Wyckoff analysis: {t5-t4:.1f}s on {len(analyzed)} range bars")
    print(f"Springs: {analyzed.Spring.sum()}")
    print(f"Upthrusts: {analyzed.Upthrust.sum()}")
    print(f"Selling Climaxes: {analyzed.SellingClimax.sum()}")
    print(f"Buying Climaxes: {analyzed.BuyingClimax.sum()}")
    print(f"Absorption bars: {analyzed.Absorption.sum()}")

    # Wave comparison stats
    if 'LargeWave' in analyzed.columns:
        print(f"Large waves (yellow): {int(analyzed.LargeWave.sum())}")
        print(f"Large E/R (yellow): {int(analyzed.LargeER.sum())}")
        n_waves = analyzed['WaveID'].max() + 1
        print(f"Total waves: {n_waves}")

    # ── Per-wave detail table ────────────────────────────────────────
    if 'WaveEndVol' in analyzed.columns:
        import numpy as np

        def _fmt_time(s):
            if pd.isna(s) or s == 0:
                return ''
            s = float(s)
            if s >= 3600:
                return f"{s/3600:.1f}h"
            if s >= 60:
                return f"{s/60:.0f}m"
            return f"{s:.0f}s"

        # Collect per-wave rows (only wave-end bars have WaveEndVol set)
        wave_ends = analyzed.dropna(subset=['WaveEndVol']).copy()
        if len(wave_ends):
            hdr = (f"{'#':>3} {'Dir':>4} {'Start':>8} {'End':>8} "
                   f"{'Volume':>10} {'PrDisp':>8} {'E/R':>6} {'Time':>6} "
                   f"{'LgW':>4} {'LgER':>4} {'vSame':>5} {'vPrev':>5} "
                   f"{'erSm':>5} {'erPr':>5}")
            print(f"\n{'─'*len(hdr)}")
            print("Wave-by-Wave Comparison")
            print(f"{'─'*len(hdr)}")
            print(hdr)
            print(f"{'─'*len(hdr)}")

            for _, r in wave_ends.iterrows():
                wid = int(r['WaveID'])
                wdir = '↑' if r.get('WaveDir', 0) > 0 else '↓'
                # Find wave start time
                wave_mask = analyzed['WaveID'] == wid
                w_start = analyzed.index[wave_mask][0].strftime('%H:%M:%S')
                w_end = r.name.strftime('%H:%M:%S')
                vol = int(r['WaveEndVol'])
                pdisp = r.get('WavePriceDisp', float('nan'))
                pdisp_s = f"{pdisp:.1f}" if not pd.isna(pdisp) else ''
                er = r['WaveEndER']
                er_s = f"{er:.2f}" if not pd.isna(er) else ''
                wt = _fmt_time(r.get('WaveTime', 0))
                lgw = '★' if r.get('LargeWave', 0) == 1 else ''
                lger = '★' if r.get('LargeER', 0) == 1 else ''
                vs = int(r['WaveVsSame']) if not pd.isna(r.get('WaveVsSame', np.nan)) else ''
                vp = int(r['WaveVsPrev']) if not pd.isna(r.get('WaveVsPrev', np.nan)) else ''
                es = int(r['ERVsSame']) if not pd.isna(r.get('ERVsSame', np.nan)) else ''
                ep = int(r['ERVsPrev']) if not pd.isna(r.get('ERVsPrev', np.nan)) else ''
                # Highlight large waves
                mark = ' ◄◄' if lgw or lger else ''
                print(f"{wid:>3} {wdir:>4} {w_start:>8} {w_end:>8} "
                      f"{vol:>10,} {pdisp_s:>8} {er_s:>6} {wt:>6} "
                      f"{lgw:>4} {lger:>4} {vs!s:>5} {vp!s:>5} "
                      f"{es!s:>5} {ep!s:>5}{mark}")
            print(f"{'─'*len(hdr)}")
            print("Dir=wave direction  PrDisp=price displacement  E/R=effort/result")
            print("LgW=large wave vs prior 4  LgER=large E/R vs prior 4")
            print("vSame/vPrev=volume vs same-dir/prev-dir wave (+1=larger, -1=smaller)")
            print("erSm/erPr=E/R vs same-dir/prev-dir wave")

    # Strip columns the plotter shouldn't render if user disabled features
    if not args.zigzag:
        analyzed.drop(columns=['Trendline', 'TurningPoint'], inplace=True, errors='ignore')
    if not args.wave_compare:
        analyzed.drop(columns=['WaveEndVol', 'WaveEndER', 'LargeWave', 'LargeER',
                               'WaveVsSame', 'WaveVsPrev', 'ERVsSame', 'ERVsPrev'],
                      inplace=True, errors='ignore')
    if not args.wave_time:
        analyzed.drop(columns=['WaveTime'], inplace=True, errors='ignore')

    # Signal summary
    if 'Signal' in analyzed.columns:
        sig = analyzed['Signal']
        events = analyzed.loc[analyzed['SignalEvent'] != '', 'SignalEvent']
        print(f"\nSignal summary:")
        print(f"  Long bars:  {(sig == 1).sum()}")
        print(f"  Short bars: {(sig == -1).sum()}")
        print(f"  Flat bars:  {(sig == 0).sum()}")
        print(f"  Signal events ({len(events)}):")
        for ts, ev in events.items():
            print(f"    {ts}  {ev}")

    # Plot Deep-M style chart
    last_n = args.last_n if args.last_n > 0 else len(analyzed)
    print(f"\nGenerating Deep-M Effort chart for {args.date}...")
    fig = plot_deep_m_effort(analyzed, last_n=last_n,
                             title=f'Deep-M Effort (NQ) — {args.date} Full Day',
                             show=False)
    fig.write_html(args.output)
    print(f"Saved to {args.output} — open in browser")

if __name__ == '__main__':
    main()
