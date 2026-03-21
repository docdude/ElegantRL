"""Test Wyckoff analysis on real NQ SCID data."""
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from wyckoff_effort.utils.scid_parser import SCIDReader, resample_ticks
from wyckoff_effort.utils.wyckoff_analyzer import analyze_wyckoff

def main():
    t0 = time.time()
    reader = SCIDReader('data/NQZ25-CME.scid')
    ticks = reader.read()
    bars = resample_ticks(ticks, '5min')
    t1 = time.time()
    print(f'Parse+resample: {t1-t0:.1f}s -> {len(bars):,} 5-min bars')
    print(f'Date range: {bars.index[0]} to {bars.index[-1]}')
    print(f'Sample bar (index 100):')
    print(bars.iloc[100])
    print()

    subset = bars.tail(2000).copy()
    t2 = time.time()
    analyzed = analyze_wyckoff(subset)
    t3 = time.time()
    print(f'Wyckoff analysis: {t3-t2:.1f}s on {len(analyzed)} bars')
    print(f'Springs: {analyzed.Spring.sum()}')
    print(f'Upthrusts: {analyzed.Upthrust.sum()}')
    print(f'Selling Climaxes: {analyzed.SellingClimax.sum()}')
    print(f'Buying Climaxes: {analyzed.BuyingClimax.sum()}')
    print(f'Absorption bars: {analyzed.Absorption.sum()}')
    print(f'Phase distribution:')
    print(analyzed.Phase.value_counts().sort_index())

if __name__ == '__main__':
    main()
