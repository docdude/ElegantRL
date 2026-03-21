"""Check NQ SCID volume sanity."""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from wyckoff_effort.utils.scid_parser import SCIDReader, resample_ticks

reader = SCIDReader('data/NQZ25-CME.scid')
ticks = reader.read()
bars = resample_ticks(ticks, '5min')

# Check volume stats during RTH (Regular Trading Hours: 9:30-16:00 ET = 14:30-21:00 UTC)
rth = bars.between_time('14:30', '20:55')
print(f'Total 5-min bars: {len(bars):,}')
print(f'RTH bars: {len(rth):,}')
print(f'\nRTH Volume stats:')
print(rth['volume'].describe())
print(f'\nSample RTH bars (high volume):')
top = rth.nlargest(5, 'volume')
print(top[['open', 'close', 'high', 'low', 'volume', 'bid_volume', 'ask_volume', 'delta']])
print(f'\nbid_volume total: {bars.bid_volume.sum():,.0f}')
print(f'ask_volume total: {bars.ask_volume.sum():,.0f}')
print(f'volume total: {bars.volume.sum():,.0f}')
