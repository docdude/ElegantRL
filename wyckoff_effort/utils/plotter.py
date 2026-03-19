"""
Deep-M Effort Zone Plotter — Plotly visualization inspired by DeepCharts Deep-M (NQ).

Renders:
  1. Candlestick chart on 40-range bars
  2. Green/purple effort zones (bid vs ask dominance per bar)
  3. Built-in adaptive moving average with slope-based coloring
  4. Volume delta sub-chart
  5. Wyckoff event markers (Springs, Upthrusts, SC, BC)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_effort_zones(df, lookback=14):
    """
    Classify each bar's effort direction — where the path of least resistance lies.

    Logic (inspired by Deep-M):
      - Compare cumulative delta slope and volume absorption
      - Green zone: ask-side dominance  (path of least resistance = up)
      - Purple zone: bid-side dominance (path of least resistance = down)

    Returns arrays of zone colors and opacities.
    """
    n = len(df)
    delta = df['delta'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)

    # Rolling cumulative delta slope via vectorized rolling regression
    cvd = np.cumsum(delta)
    cvd_series = pd.Series(cvd)

    # Rolling slope = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    # With fixed x = 0..lookback-1, simplify using rolling sums of y
    x = np.arange(lookback, dtype=np.float64)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    denom = lookback * sum_x2 - sum_x * sum_x

    rolling_sum_y = cvd_series.rolling(window=lookback, min_periods=2).sum()
    # sum(x*y) needs rolling dot product — use index-weighted rolling sum
    # y_indexed[i] = cvd[i] * (lookback-1), cvd[i-1] * (lookback-2), ...
    # Equivalent: sum(x*y) = sum_{k=0}^{L-1} k * y[i-L+1+k]
    #           = (L-1)*y[i] + (L-2)*y[i-1] + ... + 0*y[i-L+1]
    # This equals (L-1)*rolling_sum - rolling of cumsum differences... 
    # Simpler: just use EWM slope approximation for speed
    cvd_ema_fast = cvd_series.ewm(span=lookback, adjust=False).mean()
    cvd_ema_slow = cvd_series.ewm(span=lookback * 3, adjust=False).mean()
    cvd_slope = (cvd_ema_fast - cvd_ema_slow).values

    # Rolling delta ratio: net delta / volume (how one-sided the flow is)
    avg_vol = pd.Series(volume).rolling(window=lookback, min_periods=1).mean().values
    delta_ratio = np.where(avg_vol > 0, delta / avg_vol, 0.0)

    # Effort score: combine CVD slope direction + per-bar delta ratio
    # Positive = ask dominance (bullish), Negative = bid dominance (bearish)
    scale = np.nanmean(np.abs(cvd_slope[lookback:])) + 1e-9
    cvd_norm = np.tanh(cvd_slope / scale)
    delta_norm = np.clip(delta_ratio, -1, 1)
    effort_score = 0.6 * cvd_norm + 0.4 * delta_norm

    # Classify zones
    zone_color = np.where(effort_score > 0, 'green', 'purple')
    zone_opacity = np.clip(np.abs(effort_score), 0.1, 0.8)

    return effort_score, zone_color, zone_opacity


def adaptive_ma(close, period=20):
    """Adaptive moving average — Kaufman-style efficiency ratio MA."""
    n = len(close)
    ama = np.full(n, np.nan)
    if n < period:
        return ama

    # Efficiency ratio
    direction = np.abs(close[period:] - close[:-period])
    volatility = np.zeros(n - period)
    for i in range(n - period):
        volatility[i] = np.sum(np.abs(np.diff(close[i:i + period + 1])))
    er = np.where(volatility > 0, direction / volatility, 0.0)

    # Smoothing constants
    fast_sc = 2.0 / (2 + 1)
    slow_sc = 2.0 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    ama[period - 1] = close[period - 1]
    for i in range(period, n):
        ama[i] = ama[i - 1] + sc[i - period] * (close[i] - ama[i - 1])

    return ama


def _build_zone_rectangles(effort_score, subset, min_zone_bars=3,
                           max_zone_bars=25, er_threshold=1.0):
    """
    Build DeepCharts-style effort rectangles — zones only appear where price
    is **consolidating** (high effort, low result), NOT on trending moves.

    Logic:
      1. Mark each bar as "consolidating" if its body size (|Close-Open|) is
         small relative to the rolling average body → effort is being absorbed.
         Also include bars flagged as Absorption by the Wyckoff analyzer.
      2. Group consecutive consolidating bars into clusters.
      3. For each cluster, use the effort_score sign (majority vote) to decide
         green vs purple direction.
      4. Zone height = body range of the consolidation cluster (tight).

    Trending bars (large bodies, ease of movement) get NO zone — they are
    the result, not the effort.

    Returns list of dicts: {x0, x1, y0, y1, direction, strength}
    """
    n = len(effort_score)
    zones = []
    if n == 0:
        return zones

    opn = subset['open'].values
    close = subset['close'].values
    high = subset['high'].values
    low = subset['low'].values

    # Body size per bar
    body = np.abs(close - opn)
    # Rolling median body size — bars with small bodies relative to this are consolidating
    body_series = pd.Series(body)
    rolling_body = body_series.rolling(window=14, min_periods=1).median().values

    # A bar is "consolidating" if:
    #   - body <= rolling median body (price isn't moving much despite volume), OR
    #   - it's flagged as Absorption by wyckoff_analyzer, OR
    #   - ER_Ratio is low (high effort, low result)
    is_absorption = subset['Absorption'].values if 'Absorption' in subset.columns else np.zeros(n)
    er_ratio = subset['ER_Ratio'].values if 'ER_Ratio' in subset.columns else np.ones(n)

    is_consolidating = ((body <= rolling_body) |
                        (is_absorption == 1) |
                        (er_ratio <= er_threshold))

    # Group consecutive consolidating bars into clusters
    def _body_lo(i):
        return min(opn[i], close[i])

    def _body_hi(i):
        return max(opn[i], close[i])

    in_zone = False
    start = 0

    for i in range(n):
        if is_consolidating[i] and not in_zone:
            # Start a new cluster
            start = i
            in_zone = True
        elif in_zone and (not is_consolidating[i] or (i - start) >= max_zone_bars):
            # End cluster — allow 1 non-consolidating bar gap to bridge micro-breaks
            # Check if next bar resumes consolidation (bridge gap)
            if (not is_consolidating[i] and i + 1 < n and is_consolidating[i + 1]
                    and (i - start) < max_zone_bars):
                continue  # bridge 1-bar gap

            # Flush the cluster
            end = i if not is_consolidating[i] else i + 1
            span = end - start
            if span >= min_zone_bars:
                # Direction = majority vote of effort_score in this cluster
                cluster_effort = effort_score[start:end]
                direction = 1 if np.sum(cluster_effort > 0) > np.sum(cluster_effort < 0) else -1
                avg_strength = np.mean(np.abs(cluster_effort))

                # Height = body range of the cluster with small padding
                bodies_lo = np.array([_body_lo(j) for j in range(start, end)])
                bodies_hi = np.array([_body_hi(j) for j in range(start, end)])
                zone_lo = bodies_lo.min()
                zone_hi = bodies_hi.max()
                # Small wick padding (20%)
                avg_wick_up = np.mean(high[start:end] - bodies_hi)
                avg_wick_dn = np.mean(bodies_lo - low[start:end])
                zone_lo -= avg_wick_dn * 0.2
                zone_hi += avg_wick_up * 0.2

                zones.append({
                    'x0': start, 'x1': end - 1,
                    'y0': zone_lo, 'y1': zone_hi,
                    'direction': direction,
                    'strength': avg_strength,
                })
            in_zone = False

    # Handle final cluster
    if in_zone:
        end = n
        span = end - start
        if span >= min_zone_bars:
            cluster_effort = effort_score[start:end]
            direction = 1 if np.sum(cluster_effort > 0) > np.sum(cluster_effort < 0) else -1
            avg_strength = np.mean(np.abs(cluster_effort))
            bodies_lo = np.array([_body_lo(j) for j in range(start, end)])
            bodies_hi = np.array([_body_hi(j) for j in range(start, end)])
            zone_lo = bodies_lo.min()
            zone_hi = bodies_hi.max()
            avg_wick_up = np.mean(high[start:end] - bodies_hi)
            avg_wick_dn = np.mean(bodies_lo - low[start:end])
            zone_lo -= avg_wick_dn * 0.2
            zone_hi += avg_wick_up * 0.2
            zones.append({
                'x0': start, 'x1': end - 1,
                'y0': zone_lo, 'y1': zone_hi,
                'direction': direction,
                'strength': avg_strength,
            })

    return zones


def plot_deep_m_effort(df, last_n=300, title='Deep-M Effort (NQ) — 40 Range',
                       ma_period=20, zone_lookback=14, show=True,
                       tz='US/Mountain'):
    """
    Create a Deep-M inspired Plotly chart.

    Args:
        df: DataFrame from analyze_wyckoff() with OHLCV + Delta + Wyckoff columns.
             Must have at minimum: open, high, low, close, delta, volume.
        last_n: Number of bars to display (default 300).
        title: Chart title.
        ma_period: Adaptive MA period.
        zone_lookback: Lookback for effort zone calculation.
        show: If True, opens in browser. Returns the figure either way.
        tz: Timezone for x-axis labels (default 'US/Mountain' for MST/MDT).

    Returns:
        plotly.graph_objects.Figure
    """
    subset = df.tail(last_n).copy().reset_index()

    # Need a sequential x-axis (not datetime) to avoid gaps in range bars
    x = np.arange(len(subset))

    # Compute effort zones
    effort_score, zone_color, zone_opacity = compute_effort_zones(subset, lookback=zone_lookback)

    # Adaptive MA
    close = subset['close'].values
    ma = adaptive_ma(close, period=ma_period)

    delta = subset['delta'].values

    # ─── Build figure with 2 rows: price + volume delta ───
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=[title, 'Volume Delta'],
    )

    # ─── Effort zones as DeepCharts-style rectangles ───
    # Group consecutive same-direction bars into rectangular zones spanning
    # the actual price range (High–Low) of that effort cluster.
    zones = _build_zone_rectangles(effort_score, subset, min_zone_bars=3)

    price_min = subset['low'].min()
    price_max = subset['high'].max()
    price_range = price_max - price_min
    bar_height = price_range * 1.05  # for signal shading later

    for z in zones:
        color = 'rgba(0,200,0,{a})' if z['direction'] == 1 else 'rgba(128,0,200,{a})'
        # Opacity scales with effort strength (stronger effort = more visible)
        alpha = min(0.35, max(0.10, z['strength'] * 0.5))
        border_color = 'rgba(0,200,0,0.6)' if z['direction'] == 1 else 'rgba(128,0,200,0.6)'
        fig.add_shape(
            type='rect',
            x0=z['x0'] - 0.5, x1=z['x1'] + 0.5,
            y0=z['y0'], y1=z['y1'],
            fillcolor=color.format(a=f'{alpha:.2f}'),
            line=dict(color=border_color, width=1),
            layer='below',
            row=1, col=1,
        )

    # ─── Build per-bar customdata for unified hover ───
    cvd = subset['delta'].cumsum().values if 'delta' in subset.columns else np.zeros(len(subset))
    vol = subset['volume'].values
    # Wyckoff event label per bar
    wyckoff_labels = []
    for i in range(len(subset)):
        parts = []
        for col_name in ('Spring', 'Upthrust', 'SellingClimax', 'BuyingClimax', 'Absorption'):
            if col_name in subset.columns and subset[col_name].iloc[i] == 1:
                parts.append(col_name)
        if 'Phase' in subset.columns:
            ph = subset['Phase'].iloc[i]
            phase_map = {0: '', 1: 'Accum', 2: 'Markup', 3: 'Distrib', 4: 'Markdown'}
            ph_label = phase_map.get(int(ph), '')
            if ph_label:
                parts.append(ph_label)
        wyckoff_labels.append(' | '.join(parts) if parts else '')

    # customdata: [volume, delta, cvd, effort_score, wyckoff_label, ma_value]
    customdata = np.column_stack([
        vol,
        delta,
        cvd,
        effort_score,
        wyckoff_labels,
        np.where(np.isnan(ma), '', np.round(ma, 2).astype(str)),
    ])

    # ─── Candlestick ───
    # Strength-based bar coloring: 5-tier volume strength → color intensity
    STRENGTH_COLORS = {
        (4, 1): '#1b5e20', (4, -1): '#b71c1c',   # ultra: dark green/red
        (3, 1): '#66bb6a', (3, -1): '#ef9a9a',    # high: light green/pink
        (2, 1): '#bdbdbd', (2, -1): '#bdbdbd',    # average: light gray
        (1, 1): '#9e9e9e', (1, -1): '#9e9e9e',    # low: medium gray
        (0, 1): '#616161', (0, -1): '#616161',     # lowest: dark gray
    }

    if 'VolumeStrength' in subset.columns:
        vs = subset['VolumeStrength'].values.astype(int)
        wd = subset['WaveDir'].values if 'WaveDir' in subset.columns \
            else np.where(subset['close'] >= subset['open'], 1, -1)

        bar_colors = np.array([
            STRENGTH_COLORS.get((min(int(s), 4), 1 if d >= 0 else -1), '#bdbdbd')
            for s, d in zip(vs, wd)
        ])
        is_yellow = np.zeros(len(subset), dtype=bool)
        if 'LargeER' in subset.columns:
            is_yellow = subset['LargeER'].values == 1
            bar_colors[is_yellow] = '#ffd600'

        # Build timestamps for hover
        ts_col = None
        for col_name in ('datetime', 'DateTime', 'date', 'timestamp'):
            if col_name in subset.columns:
                ts_col = col_name
                break
        if ts_col is not None:
            bar_times = pd.to_datetime(subset[ts_col]).dt.strftime('%H:%M:%S').values
        else:
            bar_times = ['' for _ in range(len(subset))]

        strength_labels = {0: 'Lowest', 1: 'Low', 2: 'Avg', 3: 'High', 4: 'Ultra'}
        hover_texts = [
            f"<b>Bar #{i}</b>  {bar_times[i]}<br>"
            f"O: {subset['open'].iloc[i]:.2f}  H: {subset['high'].iloc[i]:.2f}<br>"
            f"L: {subset['low'].iloc[i]:.2f}  C: {subset['close'].iloc[i]:.2f}<br>"
            f"Vol: {int(vol[i]):,}  Δ: {int(delta[i]):+,}<br>"
            f"CVD: {int(cvd[i]):,}  Effort: {effort_score[i]:+.2f}<br>"
            f"VolStr: {strength_labels.get(int(vs[i]), str(vs[i]))} ({vs[i]})"
            f"  MA: {customdata[i][5]}"
            + (f"<br><b style='color:#ffd600'>★ Large E/R (Yellow)</b>" if is_yellow[i] else '')
            + (f"<br><b>{wyckoff_labels[i]}</b>" if wyckoff_labels[i] else '')
            for i in range(len(subset))
        ]

        first_trace = True
        for color in dict.fromkeys(bar_colors):
            mask = bar_colors == color
            idx = np.where(mask)[0]
            fig.add_trace(go.Candlestick(
                x=x[idx],
                open=subset['open'].values[idx],
                high=subset['high'].values[idx],
                low=subset['low'].values[idx],
                close=subset['close'].values[idx],
                increasing_line_color=color,
                increasing_fillcolor=color,
                decreasing_line_color=color,
                decreasing_fillcolor=color,
                name='Price' if first_trace else '',
                showlegend=first_trace,
                hoverinfo='text',
                text=[hover_texts[i] for i in idx],
            ), row=1, col=1)
            first_trace = False
    else:
        fig.add_trace(go.Candlestick(
            x=x,
            open=subset['open'],
            high=subset['high'],
            low=subset['low'],
            close=subset['close'],
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            name='Price',
            customdata=customdata,
            hoverinfo='text',
            text=[
                f"<b>Bar #{i}</b><br>"
                f"O: {subset['open'].iloc[i]:.2f}  H: {subset['high'].iloc[i]:.2f}<br>"
                f"L: {subset['low'].iloc[i]:.2f}  C: {subset['close'].iloc[i]:.2f}<br>"
                f"Vol: {int(vol[i]):,}  Δ: {int(delta[i]):+,}<br>"
                f"CVD: {int(cvd[i]):,}  Effort: {effort_score[i]:+.2f}<br>"
                f"MA: {customdata[i][5]}"
                + (f"<br><b>{wyckoff_labels[i]}</b>" if wyckoff_labels[i] else '')
                for i in range(len(subset))
            ],
        ), row=1, col=1)

    # ─── Adaptive MA with slope-based coloring ───
    # Plot MA as two traces (above/below) instead of N individual segments
    ma_above_y = np.where(close >= ma, ma, np.nan)
    ma_below_y = np.where(close < ma, ma, np.nan)
    fig.add_trace(go.Scatter(
        x=x, y=ma_above_y, mode='lines',
        line=dict(color='#00e676', width=2),
        name='MA (bullish)', showlegend=False, hoverinfo='skip',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=ma_below_y, mode='lines',
        line=dict(color='#aa00ff', width=2),
        name='MA (bearish)', showlegend=False, hoverinfo='skip',
    ), row=1, col=1)

    # ─── ZigZag Trendline ───
    if 'Trendline' in subset.columns:
        tl = subset['Trendline'].values
        tl_x = x[~np.isnan(tl)]
        tl_y = tl[~np.isnan(tl)]
        if len(tl_x) > 1:
            fig.add_trace(go.Scatter(
                x=tl_x, y=tl_y, mode='lines',
                line=dict(color='cyan', width=1.5),
                connectgaps=True,
                name='ZigZag', showlegend=True,
                hoverinfo='skip',
            ), row=1, col=1)

    # ─── Turning Points ───
    if 'TurningPoint' in subset.columns:
        tp = subset['TurningPoint'].values
        tp_mask = ~np.isnan(tp)
        if tp_mask.any():
            fig.add_trace(go.Scatter(
                x=x[tp_mask], y=tp[tp_mask],
                mode='markers',
                marker=dict(color='red', size=5, symbol='circle'),
                name='Pivot', showlegend=True,
                hoverinfo='skip',
            ), row=1, col=1)

    # ─── Weis Wave annotations (volume, E/R, time at wave ends) ───
    if 'WaveEndVol' in subset.columns:
        wave_end_mask = subset['WaveEndVol'].notna()
        if wave_end_mask.any():
            for idx in subset.index[wave_end_mask]:
                i = int(idx)
                w_vol = subset['WaveEndVol'].iloc[i]
                w_er = subset['WaveEndER'].iloc[i]
                w_dir = subset['WaveDir'].iloc[i] if 'WaveDir' in subset.columns else 0
                w_time = subset['WaveTime'].iloc[i] if 'WaveTime' in subset.columns else 0
                is_large = (subset['LargeWave'].iloc[i] == 1 if 'LargeWave' in subset.columns else False)
                is_large_er = (subset['LargeER'].iloc[i] == 1 if 'LargeER' in subset.columns else False)

                # Format wave time
                if w_time >= 3600:
                    time_str = f"{w_time/3600:.1f}h"
                elif w_time >= 60:
                    time_str = f"{w_time/60:.0f}m"
                else:
                    time_str = f"{w_time:.0f}s"

                # Position: above for up waves, below for down waves
                y_pos = subset['high'].iloc[i] + price_range * 0.01 if w_dir == 1 \
                    else subset['low'].iloc[i] - price_range * 0.01
                txt_pos = 'top center' if w_dir == 1 else 'bottom center'
                txt_color = '#FFD700' if (is_large or is_large_er) else '#00BFFF'

                label = f"({int(w_vol):,})"
                if w_er and not np.isnan(w_er):
                    label += f" [{w_er:.1f}]"
                if w_time > 0:
                    label += f" {time_str}"

                fig.add_trace(go.Scatter(
                    x=[x[i]], y=[y_pos],
                    mode='text', text=[label],
                    textposition=txt_pos,
                    textfont=dict(size=8, color=txt_color),
                    showlegend=False, hoverinfo='skip',
                ), row=1, col=1)

    # ─── Wyckoff event markers ───
    markers = [
        ('Spring', '△', '#00e676', 'below'),
        ('Upthrust', '▽', '#aa00ff', 'above'),
        ('SellingClimax', 'SC', '#ff6d00', 'below'),
        ('BuyingClimax', 'BC', '#2979ff', 'above'),
    ]
    for col_name, symbol_text, color, position in markers:
        if col_name in subset.columns:
            mask = subset[col_name] == 1
            if mask.any():
                y_vals = subset.loc[mask, 'low' if position == 'below' else 'high']
                offset = (subset['high'].max() - subset['low'].min()) * 0.015
                y_plot = y_vals - offset if position == 'below' else y_vals + offset
                fig.add_trace(go.Scatter(
                    x=x[mask.values],
                    y=y_plot,
                    mode='markers+text',
                    marker=dict(size=10, color=color, symbol='triangle-up' if position == 'below' else 'triangle-down'),
                    text=symbol_text,
                    textposition='bottom center' if position == 'below' else 'top center',
                    textfont=dict(size=8, color=color),
                    name=col_name,
                    showlegend=True,
                ), row=1, col=1)

    # ─── Volume Delta bars ───
    delta_colors = ['#00e676' if d >= 0 else '#aa00ff' for d in delta]
    fig.add_trace(go.Bar(
        x=x,
        y=delta,
        marker_color=delta_colors,
        name='Delta',
        showlegend=False,
        hovertemplate='Δ: %{y:+,}<extra></extra>',
    ), row=2, col=1)

    # ─── Absorption highlight on delta ───
    # Tint absorption bars with a gold border instead of overlaying diamonds,
    # so delta bars remain fully readable.
    if 'Absorption' in subset.columns:
        abs_mask = subset['Absorption'] == 1
        if abs_mask.any():
            fig.add_trace(go.Bar(
                x=x[abs_mask.values],
                y=delta[abs_mask.values],
                marker=dict(
                    color='rgba(255,215,0,0.15)',
                    line=dict(width=1.5, color='rgba(255,215,0,0.7)'),
                ),
                name='Absorption',
                showlegend=True,
                hovertemplate='Absorption Δ: %{y:+,}<extra></extra>',
            ), row=2, col=1)

    # ─── Signal entry/exit markers + long/short zone shading ───
    if 'Signal' in subset.columns:
        sig = subset['Signal'].values
        sig_event = subset['SignalEvent'].values if 'SignalEvent' in subset.columns else [''] * len(subset)
        offset = (subset['high'].max() - subset['low'].min()) * 0.025

        # Long/short zone shading (semi-transparent background)
        long_mask = sig == 1
        short_mask = sig == -1
        if long_mask.any():
            fig.add_trace(go.Bar(
                x=x[long_mask], y=np.full(long_mask.sum(), bar_height),
                base=price_min - price_range * 0.025,
                marker=dict(color='rgba(0,230,118,0.07)', line_width=0),
                showlegend=False, hoverinfo='skip', name='Long Zone',
            ), row=1, col=1)
        if short_mask.any():
            fig.add_trace(go.Bar(
                x=x[short_mask], y=np.full(short_mask.sum(), bar_height),
                base=price_min - price_range * 0.025,
                marker=dict(color='rgba(255,82,82,0.07)', line_width=0),
                showlegend=False, hoverinfo='skip', name='Short Zone',
            ), row=1, col=1)

        # Entry/exit arrows at signal changes
        for i in range(1, len(sig)):
            if sig_event[i] == '':
                continue
            ev = sig_event[i]
            if sig[i] == 1:  # entered long
                fig.add_trace(go.Scatter(
                    x=[x[i]], y=[subset['low'].iloc[i] - offset],
                    mode='markers+text', text=['▲ ' + ev],
                    textposition='bottom center',
                    textfont=dict(size=9, color='#00e676'),
                    marker=dict(size=12, color='#00e676', symbol='triangle-up',
                                line=dict(width=1, color='white')),
                    name='Long Entry', showlegend=False,
                ), row=1, col=1)
            elif sig[i] == -1:  # entered short
                fig.add_trace(go.Scatter(
                    x=[x[i]], y=[subset['high'].iloc[i] + offset],
                    mode='markers+text', text=['▼ ' + ev],
                    textposition='top center',
                    textfont=dict(size=9, color='#ff5252'),
                    marker=dict(size=12, color='#ff5252', symbol='triangle-down',
                                line=dict(width=1, color='white')),
                    name='Short Entry', showlegend=False,
                ), row=1, col=1)
            elif sig[i] == 0 and sig[i - 1] != 0:  # exit to flat
                marker_color = '#ffab40'
                y_pos = subset['high'].iloc[i] + offset if sig[i - 1] == 1 else subset['low'].iloc[i] - offset
                fig.add_trace(go.Scatter(
                    x=[x[i]], y=[y_pos],
                    mode='markers+text', text=['✕ ' + ev],
                    textposition='top center' if sig[i - 1] == 1 else 'bottom center',
                    textfont=dict(size=9, color=marker_color),
                    marker=dict(size=10, color=marker_color, symbol='x',
                                line=dict(width=1, color='white')),
                    name='Exit', showlegend=False,
                ), row=1, col=1)

    # ─── Custom tick labels: show timestamps in local timezone ───
    if 'datetime' in subset.columns:
        ts = subset['datetime']
    elif subset.index.name == 'datetime':
        ts = subset.index
    else:
        ts = pd.RangeIndex(len(subset))

    # Timestamps from the SCID parser are already tz-naive in US/Mountain.
    # Only convert if they still carry an explicit UTC tzinfo.
    if hasattr(ts, 'dt'):
        ts_local = ts.dt.tz_convert(tz) if ts.dt.tz is not None else ts
    elif isinstance(ts, pd.DatetimeIndex):
        ts_local = ts.tz_convert(tz) if ts.tz is not None else ts
    else:
        ts_local = ts

    tick_step = max(1, len(x) // 20)
    tick_vals = list(range(0, len(x), tick_step))
    tick_labels = []
    for idx in tick_vals:
        val = ts_local.iloc[idx] if hasattr(ts_local, 'iloc') else ts_local[idx]
        if hasattr(val, 'strftime'):
            tick_labels.append(val.strftime('%m/%d %H:%M'))
        else:
            tick_labels.append(str(val))

    # ─── Layout ───
    fig.update_xaxes(
        tickvals=tick_vals,
        ticktext=tick_labels,
        tickangle=-45,
        row=2, col=1,
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False)

    fig.update_layout(
        template='plotly_dark',
        height=800,
        margin=dict(l=60, r=30, t=50, b=60),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='right', x=1,
            font=dict(size=10),
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.85)',
            font_size=11,
            font_family='monospace',
        ),
    )

    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Delta', row=2, col=1)

    if show:
        fig.show()

    return fig
