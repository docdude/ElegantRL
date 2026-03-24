"""
Sweep carry_cost to find the value that makes selective Wyckoff entries
uniquely optimal vs always-positioned.

Tests 5 strategies for each carry_cost value:
  1. flat       — never trade
  2. always_long — enter long bar 0, hold forever
  3. always_short — enter short bar 0, hold forever
  4. selective_hold10 — enter on signal, hold exactly vesting_bars, exit
  5. always_pos  — enter immediately, stay positioned, re-enter after signal exit
"""
import numpy as np
import sys

NPZ = "/opt/ElegantRL/wyckoff_effort/pipeline_output/wyckoff_nq_40pt.npz"

data = np.load(NPZ)
close = data["close_ary"].astype(np.float64)
tech  = data["tech_ary"].astype(np.float64)

# Drop rows with NaN close
valid = ~np.isnan(close)
close = close[valid]
tech  = tech[valid, :]
N = len(close)

# Signal columns
COL_SPRING   = 35
COL_UPTHRUST = 36
COL_WAVE_DIR = 15

EVENT_THR = 0.3

spring   = np.nan_to_num(tech[:, COL_SPRING], nan=0.0)
upthrust = np.nan_to_num(tech[:, COL_UPTHRUST], nan=0.0)

# Compute drift
diffs = np.diff(close)
drift = float(np.nanmean(diffs))
print(f"Bars={N}  drift={drift:.4f}pts  spring>0.3={np.mean(spring>EVENT_THR)*100:.1f}%  upthrust>0.3={np.mean(upthrust>EVENT_THR)*100:.1f}%")

# Detrended price changes
detrended = diffs - drift  # length N-1

def simulate(carry, bonus, vesting):
    results = {}
    
    # 1) Flat — 0 reward per bar
    results['flat'] = 0.0
    
    # 2) Always long — hold from bar 0 to end
    pnl_long = np.sum(detrended) / 2000.0  # normalized
    carry_long = (N - 1) * carry
    results['always_long'] = pnl_long - carry_long
    
    # 3) Always short
    pnl_short = -np.sum(detrended) / 2000.0
    carry_short = (N - 1) * carry
    results['always_short'] = pnl_short - carry_short
    
    # 4) Selective hold=vesting — enter on signal, hold exactly vesting bars, exit
    total_bonus = 0.0
    total_carry = 0.0
    total_pnl = 0.0
    n_trades = 0
    cooldown = 0
    
    for i in range(N - 1):
        if cooldown > 0:
            cooldown -= 1
            continue
        
        has_spring = spring[i] > EVENT_THR
        has_upthrust = upthrust[i] > EVENT_THR
        
        if has_spring or has_upthrust:
            # Determine direction
            if has_spring:
                direction = 1  # long
            else:
                direction = -1  # short
            
            # Hold for vesting bars
            hold_bars = min(vesting, N - 1 - i)
            trade_pnl = 0.0
            for j in range(hold_bars):
                trade_pnl += direction * detrended[i + j] / 2000.0
                total_carry += carry
            
            total_pnl += trade_pnl
            
            # Deferred bonus: only if held full vesting period
            if hold_bars >= vesting:
                total_bonus += bonus
            
            n_trades += 1
            cooldown = hold_bars - 1  # skip bars we're already holding
    
    results['sel_hold10'] = total_pnl + total_bonus - total_carry
    results['sel_trades'] = n_trades
    
    # 5) Always-positioned — enter immediately, never exit voluntarily
    # Mimics agent behavior: enter long, stay in, occasionally flip on signals
    total_pnl_ap = 0.0
    total_carry_ap = 0.0
    total_bonus_ap = 0.0
    n_trades_ap = 0
    pos = 0  # 0=flat, 1=long, -1=short
    bars_held = 0
    vesting_remaining = 0
    vesting_amount = 0.0
    
    for i in range(N - 1):
        has_spring = spring[i] > EVENT_THR
        has_upthrust = upthrust[i] > EVENT_THR
        
        if pos == 0:
            # Enter on ANY bar (always-positioned strategy)
            pos = 1  # default long
            bars_held = 0
            if has_spring:
                vesting_remaining = vesting
                vesting_amount = bonus
            elif has_upthrust:
                pos = -1
                vesting_remaining = vesting
                vesting_amount = bonus
            n_trades_ap += 1
        
        # Accumulate PnL
        total_pnl_ap += pos * detrended[i] / 2000.0
        total_carry_ap += carry
        bars_held += 1
        
        # Vesting countdown
        if vesting_remaining > 0:
            vesting_remaining -= 1
            if vesting_remaining == 0:
                total_bonus_ap += vesting_amount
                vesting_amount = 0.0
    
    results['always_pos'] = total_pnl_ap + total_bonus_ap - total_carry_ap
    results['always_pos_trades'] = n_trades_ap
    
    return results

print()
print(f"{'carry':>7} {'bonus':>5} {'vest':>4} | {'flat':>8} {'always_L':>8} {'always_S':>8} | {'sel_h10':>8} {'sel_N':>5} | {'always_p':>8} {'alw_N':>5} | BEST")
print("-" * 110)

for carry in [0.002, 0.005, 0.007, 0.01, 0.015, 0.02, 0.025, 0.03]:
    r = simulate(carry, bonus=0.50, vesting=10)
    
    vals = {
        'flat': float(r['flat']),
        'always_L': float(r['always_long']),
        'always_S': float(r['always_short']),
        'sel_h10': float(r['sel_hold10']),
        'always_p': float(r['always_pos']),
    }
    best = max(vals, key=vals.get)
    marker = ' <<<' if best == 'sel_h10' else ''
    
    print(f"{carry:7.3f} {0.50:5.2f} {10:4d} | {r['flat']:+8.2f} {r['always_long']:+8.2f} {r['always_short']:+8.2f} | "
          f"{r['sel_hold10']:+8.2f} {r['sel_trades']:5.0f} | {r['always_pos']:+8.2f} {r['always_pos_trades']:5.0f} | {best}{marker}")

# Now try carry=0.01 with different bonus scales
print()
print("=== BONUS SCALE SWEEP at carry=0.01 ===")
print(f"{'carry':>7} {'bonus':>5} {'vest':>4} | {'flat':>8} {'always_L':>8} {'always_S':>8} | {'sel_h10':>8} {'sel_N':>5} | {'always_p':>8} {'alw_N':>5} | BEST")
print("-" * 110)

for bonus in [0.05, 0.10, 0.20, 0.30, 0.50]:
    carry = 0.01
    r = simulate(carry, bonus=bonus, vesting=10)
    
    vals = {
        'flat': float(r['flat']),
        'always_L': float(r['always_long']),
        'always_S': float(r['always_short']),
        'sel_h10': float(r['sel_hold10']),
        'always_p': float(r['always_pos']),
    }
    best = max(vals, key=vals.get)
    marker = ' <<<' if best == 'sel_h10' else ''
    
    print(f"{carry:7.3f} {bonus:5.2f} {10:4d} | {r['flat']:+8.2f} {r['always_long']:+8.2f} {r['always_short']:+8.2f} | "
          f"{r['sel_hold10']:+8.2f} {r['sel_trades']:5.0f} | {r['always_pos']:+8.2f} {r['always_pos_trades']:5.0f} | {best}{marker}")

# Also test the interaction with different vesting periods + carry
print()
print("=== RECOMMENDED COMBOS ===")
print(f"{'carry':>7} {'bonus':>5} {'vest':>4} | {'flat':>8} {'always_L':>8} {'always_S':>8} | {'sel_h10':>8} {'sel_N':>5} | {'always_p':>8} {'alw_N':>5} | gap")
print("-" * 110)

for carry, bonus, vest in [(0.01, 0.30, 10), (0.01, 0.50, 10), (0.015, 0.50, 10), 
                             (0.02, 0.50, 10), (0.01, 0.20, 10), (0.01, 0.15, 5)]:
    r = simulate(carry, bonus=bonus, vesting=vest)
    
    vals = {
        'flat': float(r['flat']),
        'always_L': float(r['always_long']),
        'always_S': float(r['always_short']),
        'sel_h10': float(r['sel_hold10']),
        'always_p': float(r['always_pos']),
    }
    best_key = max(vals, key=vals.get)
    best_val = vals[best_key]
    second_val = sorted(vals.values(), reverse=True)[1]
    gap = best_val - second_val
    
    marker = ' <<<' if best_key == 'sel_h10' else ''
    
    print(f"{carry:7.3f} {bonus:5.2f} {vest:4d} | {r['flat']:+8.2f} {r['always_long']:+8.2f} {r['always_short']:+8.2f} | "
          f"{r['sel_hold10']:+8.2f} {r['sel_trades']:5.0f} | {r['always_pos']:+8.2f} {r['always_pos_trades']:5.0f} | {gap:+.2f}{marker}")
