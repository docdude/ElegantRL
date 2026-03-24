"""
NQ Wyckoff-Weis Trading Environment for ElegantRL
===================================================

Discrete-action RL environment for NQ futures using Wyckoff methodology
and Weis Wave analysis.  Uses precomputed features from the 58-feature
NPZ pipeline but presents them as a structured observation rather than
a raw sliding window.

Key design principles (from research: MacroHFT, EarnHFT, positional-context
intraday DRL paper):
  1. Structured observation: bar + wave + event + phase + position features
  2. Discrete semantic actions: hold / enter_long / enter_short / add / reduce / exit
  3. Shaped rewards: PnL delta  +  Wyckoff event-alignment bonuses  -  penalties
  4. Hard risk rules: position limits, invalid-action penalty
  5. Curriculum support: restrict action space / event types by stage

Two classes:
  NQWyckoffWeisEnv     — single env (Gymnasium API + ElegantRL metadata)
  NQWyckoffWeisVecEnv  — GPU-vectorized env for PPO training

Compatible with ElegantRL discrete PPO::

    from elegantrl.agents.AgentPPO import AgentDiscretePPO
    args = Config(AgentDiscretePPO, NQWyckoffWeisVecEnv, env_args)
"""

from __future__ import annotations

import os
import logging
import numpy as np
import torch as th

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# Discrete action space
ACTION_HOLD = 0
ACTION_ENTER_LONG = 1
ACTION_ENTER_SHORT = 2
ACTION_ADD = 3
ACTION_REDUCE = 4
ACTION_EXIT = 5
N_ACTIONS = 6

# ---------------------------------------------------------------------------
# Feature selection from the 58-feature tech_ary (NPZ "tech_ary" column idx)
#
# Rationale: present the agent with a "prop-trader state" — bar micro-
# structure, Weis wave context, Wyckoff event scores, phase probabilities,
# and position state.  No sliding window needed because wave/event/phase
# features already encode multi-bar temporal structure.
# ---------------------------------------------------------------------------
ENV_FEATURE_INDICES = [
    # ── Bar microstructure (10) ──
    0,   # body_ratio          — bar body/range, direction
    1,   # upper_wick_ratio    — rejection at highs
    2,   # lower_wick_ratio    — rejection at lows (spring context)
    4,   # delta_ratio         — per-bar order flow direction
    5,   # vol_vs_ma20         — relative volume (effort)
    8,   # duration_norm       — bar formation speed
    9,   # cvd_slope_fast      — short-term CVD trend
    11,  # cvd_divergence      — CVD vs price divergence
    13,  # return_5            — 5-bar momentum
    14,  # volatility_20       — realised vol regime
    # ── Weis Wave (15) ──
    15,  # wave_direction       — current wave direction ±1
    16,  # wave_progress        — % into current wave
    17,  # wave_displacement_norm — price movement of current wave
    18,  # wave_vol_cumulative_norm — volume of current wave
    19,  # wave_delta_ratio     — order flow inside current wave
    21,  # wave_vol_vs_prev     — effort comparison (current vs prior)
    23,  # wave_disp_vs_prev    — displacement comparison (shortening)
    27,  # demand_score_3wave   — 3-wave demand composite
    28,  # supply_score_3wave   — 3-wave supply composite
    29,  # wave_vol_trend_up    — up-wave volume trend
    30,  # wave_vol_trend_down  — down-wave volume trend
    34,  # large_wave_score     — climactic-volume flag
    58,  # wave_effort_result_raw — absolute wave vol/displacement
    59,  # wave_time_norm       — wave calendar duration (normalized)
    60,  # wave_velocity_norm   — wave displacement per second
    # ── Wyckoff Events (6) ──
    35,  # spring_score
    36,  # upthrust_score
    37,  # sc_score             — selling climax
    38,  # bc_score             — buying climax
    39,  # absorption_score
    41,  # stopping_action_score
    # ── Range / Phase Context (7) ──
    48,  # pct_in_range         — position inside trading range
    49,  # range_width_norm     — range size (vol proxy)
    50,  # bars_in_range        — time in current range
    53,  # phase_accum_score    — accumulation probability
    54,  # phase_markup_score   — markup probability
    55,  # phase_distrib_score  — distribution probability
    56,  # phase_markdown_score — markdown probability
]

N_ENV_FEATURES = len(ENV_FEATURE_INDICES)   # 38
N_POSITION_FEATURES = 8                     # side, size, entry_dist, ...

# Map from tech_ary column index → offset inside selected feature vector
_EFI_LOOKUP = {col: i for i, col in enumerate(ENV_FEATURE_INDICES)}

# Event column offsets (inside the 38-feature selected vector)
COL_SPRING = _EFI_LOOKUP[35]
COL_UPTHRUST = _EFI_LOOKUP[36]
COL_SC = _EFI_LOOKUP[37]
COL_BC = _EFI_LOOKUP[38]
COL_ABSORPTION = _EFI_LOOKUP[39]
COL_STOPPING = _EFI_LOOKUP[41]

# Phase column offsets
COL_PHASE_ACCUM = _EFI_LOOKUP[53]
COL_PHASE_MARKUP = _EFI_LOOKUP[54]
COL_PHASE_DISTRIB = _EFI_LOOKUP[55]
COL_PHASE_MARKDOWN = _EFI_LOOKUP[56]

# Wave direction column offset
COL_WAVE_DIR = _EFI_LOOKUP[15]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_npz(npz_path: str):
    """Load close prices and tech features from NPZ."""
    if not npz_path or not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    close_ary = data["close_ary"].astype(np.float32)
    tech_ary = data["tech_ary"].astype(np.float32)
    if close_ary.ndim == 2:
        close_ary = close_ary[:, 0]
    return close_ary, tech_ary


# ═══════════════════════════════════════════════════════════════════════════
# Single Environment  (Gymnasium + ElegantRL metadata)
# ═══════════════════════════════════════════════════════════════════════════

class NQWyckoffWeisEnv:
    """
    Single-instance Wyckoff/Weis NQ futures trading environment.

    Observation (state_dim = 46):
      [38 selected features from tech_ary | 8 position-state features]

    Action (discrete, 6):
      hold | enter_long | enter_short | add | reduce | exit

    Reward = PnL_delta + entry_alignment_bonus − invalid_action_penalty

    Compatible with ElegantRL single-env evaluation and Gymnasium wrappers.
    """

    def __init__(
        self,
        npz_path: str = "",
        beg_idx: int = 0,
        end_idx: int = 0,
        max_step: int = 1024,
        commission: float = 1.50,       # $ per side per contract
        slippage_ticks: float = 1.0,
        tick_size: float = 0.25,
        tick_value: float = 5.0,        # NQ: $5/tick → $20/point
        max_position_size: int = 2,
        reward_scale: float = 1.0,
        event_threshold: float = 0.3,   # min score to trigger entry bonus
        entry_bonus_scale: float = 0.50,
        invalid_penalty: float = 0.02,
        mgmt_bonus_scale: float = 0.02,
        mgmt_penalty_scale: float = 0.03,
        overstay_bars: int = 20,
        regime_penalty_scale: float = 0.05,
        idle_penalty: float = 0.05,
        carry_cost: float = 0.0,
        vesting_bars: int = 10,
        pnl_norm: float = 2000.0,
        reward_clip: float = 2.0,
        bar_range: float = 40.0,
        random_start: bool = True,
        feature_indices: list[int] | None = None,
        **kwargs,
    ):
        close_ary, tech_ary = _load_npz(npz_path)
        if end_idx <= 0:
            end_idx = len(close_ary)
        self.close_ary = close_ary[beg_idx:end_idx]
        self.tech_ary = tech_ary[beg_idx:end_idx]

        fi = feature_indices or ENV_FEATURE_INDICES
        self._fi = np.array(fi, dtype=np.intp)
        self.n_features = len(fi)

        # Trading config
        self.commission = commission
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.max_position_size = max_position_size
        self.reward_scale = reward_scale
        self.event_threshold = event_threshold
        self.entry_bonus_scale = entry_bonus_scale
        self.invalid_penalty = invalid_penalty
        self.mgmt_bonus_scale = mgmt_bonus_scale
        self.mgmt_penalty_scale = mgmt_penalty_scale
        self.overstay_bars = overstay_bars
        self.regime_penalty_scale = regime_penalty_scale
        self.idle_penalty = idle_penalty
        self.carry_cost = carry_cost
        self.vesting_bars = vesting_bars
        self.reward_clip = reward_clip
        self.random_start = random_start
        self.bar_range = bar_range
        self.max_episode_bars = max_step

        # PnL normalisation – default $2000 so 40pt 2-lot = 0.8 (no clip)
        self.pnl_norm = pnl_norm

        # Local drift detrending: per-bar rolling mean of prior price changes
        close_flat = self.close_ary.flatten()
        bar_changes = np.diff(close_flat, prepend=close_flat[0])
        import pandas as pd
        self.drift_per_bar = (pd.Series(bar_changes)
                             .rolling(50, min_periods=1).mean()
                             .shift(1).fillna(0.0).values)

        # ElegantRL metadata
        self.state_dim = self.n_features + N_POSITION_FEATURES
        self.action_dim = N_ACTIONS
        self.if_discrete = True
        self.max_step = self.close_ary.shape[0] - 1
        self.num_envs = 1
        self.env_name = "NQWyckoffWeisEnv-v1"
        self.target_return = +np.inf

        # Runtime state (set in reset)
        self._t: int = 0
        self._start: int = 0
        self._end: int = 0
        self._step: int = 0
        # Position
        self._side: int = 0
        self._size: float = 0.0
        self._entry_price: float = 0.0
        self._bars_in_trade: int = 0
        self._unrealized: float = 0.0
        self._realized: float = 0.0
        self._mfe: float = 0.0
        self._mae: float = 0.0
        # Entry bonus vesting
        self._vesting_drip: float = 0.0
        self._vesting_remaining: int = 0
        # Bookkeeping
        self.cumulative_returns = 0.0
        self.total_trades = 0
        self._action_counts = [0] * N_ACTIONS  # per-action histogram

    # ─── Reset ────────────────────────────────────────────────────────────

    def reset(self):
        margin = 50  # need some bars for features to stabilise
        max_start = max(margin, self.close_ary.shape[0] - self.max_episode_bars - 2)
        if self.random_start and max_start > margin:
            self._start = int(np.random.randint(margin, max_start))
        else:
            self._start = margin
        self._end = min(self._start + self.max_episode_bars, self.close_ary.shape[0] - 1)
        self._t = self._start
        self._step = 0
        # Position
        self._side = 0
        self._size = 0.0
        self._entry_price = 0.0
        self._bars_in_trade = 0
        self._unrealized = 0.0
        self._realized = 0.0
        self._mfe = 0.0
        self._mae = 0.0
        self._vesting_drip = 0.0
        self._vesting_remaining = 0
        self.total_trades = 0
        self._action_counts = [0] * N_ACTIONS
        return self._get_obs(), {}

    # ─── Step ─────────────────────────────────────────────────────────────

    def step(self, action: int):
        action = int(action)

        # Snapshot pre-action state for reward gating
        prev_price = float(self.close_ary[self._t])
        prev_side = self._side  # needed to gate entry bonus

        # 1) Execute action at current bar close
        penalty = self._execute(action)

        # 2) Snapshot post-action position for correct credit assignment
        #    (entry actions now get PnL credit on the bar they enter)
        post_side = self._side
        post_size = self._size

        # 3) Advance bar
        self._t += 1
        self._step += 1

        # 4) Mark-to-market
        self._mark_to_market()

        # 5) Reward — dense: detrended position_exposure × price_change
        #    Subtract average drift so unconditional long/short has 0 expected reward
        curr_price = float(self.close_ary[self._t])
        holding_pnl = 0.0
        if post_side != 0 and post_size > 0:
            detrended_change = (curr_price - prev_price) - self.drift_per_bar[self._t]
            ticks = (detrended_change / self.tick_size) * post_side
            holding_pnl = ticks * self.tick_value * post_size
        pnl_delta = holding_pnl / self.pnl_norm

        # Entry bonus: deferred until vesting_bars of holding
        # On entry with valid signal, store the bonus; pay it only after
        # the agent has held for vesting_bars (forfeit if exiting early)
        raw_entry_bonus = self._entry_bonus(action)
        # Gate: only award bonus when entering from FLAT (prevent flip farming)
        if prev_side != 0:
            raw_entry_bonus = 0.0
        if raw_entry_bonus > 0 and self.vesting_bars > 0:
            self._vesting_drip = raw_entry_bonus  # deferred amount
            self._vesting_remaining = self.vesting_bars
        vesting_bonus = 0.0
        if self._vesting_remaining > 0:
            self._vesting_remaining -= 1
            if self._vesting_remaining == 0:  # vesting complete → pay out
                vesting_bonus = self._vesting_drip
                self._vesting_drip = 0.0

        mgmt_bonus = self._management_bonus(action)
        regime_penalty = self._regime_mismatch_penalty(action)

        # Carry cost: per-bar cost scaled by position size (1 lot = 1×, 2 lots = 2×)
        carry = self.carry_cost * post_size

        reward = (pnl_delta + vesting_bonus + mgmt_bonus
                  - penalty - regime_penalty - carry) * self.reward_scale
        if self.reward_clip > 0:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        # 6) Done
        truncated = (self._t >= self._end) or (self._step >= self.max_episode_bars)
        terminated = False

        if truncated and self._side != 0:
            # Flatten at episode end
            px = self._exec_price(is_buy=(self._side < 0))
            self._realized += self._compute_pnl(px) - self.commission * self._size
            self._side = 0
            self._size = 0.0

        if truncated:
            self.cumulative_returns = self._realized

        obs = self._get_obs()
        self._action_counts[action] += 1
        info = {
            "equity": self._realized + self._unrealized,
            "realized_pnl": self._realized,
            "position_side": self._side,
            "total_trades": self.total_trades,
            "action_counts": list(self._action_counts),
        }
        return obs, float(reward), terminated, truncated, info

    # ─── Action execution ─────────────────────────────────────────────────

    def _execute(self, action: int) -> float:
        penalty = 0.0
        is_flat = self._side == 0
        is_positioned = not is_flat

        if action == ACTION_HOLD:
            pass

        elif action == ACTION_ENTER_LONG:
            if is_flat:
                px = self._exec_price(is_buy=True)
                self._open(1, 1.0, px)
            else:
                penalty = self.invalid_penalty

        elif action == ACTION_ENTER_SHORT:
            if is_flat:
                px = self._exec_price(is_buy=False)
                self._open(-1, 1.0, px)
            else:
                penalty = self.invalid_penalty

        elif action == ACTION_ADD:
            if is_positioned and self._size < self.max_position_size:
                px = self._exec_price(is_buy=(self._side > 0))
                self._add(px)
            else:
                penalty = self.invalid_penalty

        elif action == ACTION_REDUCE:
            if is_positioned:
                px = self._exec_price(is_buy=(self._side < 0))
                self._reduce(px)
            else:
                penalty = self.invalid_penalty

        elif action == ACTION_EXIT:
            if is_positioned:
                px = self._exec_price(is_buy=(self._side < 0))
                self._close(px)
            else:
                penalty = self.invalid_penalty
        else:
            penalty = 0.05  # unknown action

        return penalty

    def _exec_price(self, is_buy: bool) -> float:
        px = float(self.close_ary[self._t])
        slip = self.slippage_ticks * self.tick_size
        return px + slip if is_buy else px - slip

    def _open(self, side: int, size: float, price: float):
        self._side = side
        self._size = size
        self._entry_price = price
        self._bars_in_trade = 0
        self._mfe = 0.0
        self._mae = 0.0
        self._realized -= self.commission
        self.total_trades += 1

    def _add(self, price: float):
        new_size = self._size + 1.0
        self._entry_price = (
            self._entry_price * self._size + price
        ) / new_size
        self._size = new_size
        self._realized -= self.commission
        self.total_trades += 1

    def _reduce(self, price: float):
        frac = 1.0 / max(self._size, 1.0)
        pnl = self._compute_pnl(price) * frac
        self._realized += pnl - self.commission
        self._size -= 1.0
        if self._size <= 0:
            self._side = 0
            self._size = 0.0
            self._entry_price = 0.0
            self._bars_in_trade = 0
            self._mfe = 0.0
            self._mae = 0.0
            self._vesting_drip = 0.0
            self._vesting_remaining = 0

    def _close(self, price: float):
        pnl = self._compute_pnl(price)
        self._realized += pnl - self.commission * self._size  # per-contract commission
        self._side = 0
        self._size = 0.0
        self._entry_price = 0.0
        self._bars_in_trade = 0
        self._mfe = 0.0
        self._mae = 0.0
        self._vesting_drip = 0.0
        self._vesting_remaining = 0
        self.total_trades += 1

    def _compute_pnl(self, price: float) -> float:
        ticks = ((price - self._entry_price) / self.tick_size) * self._side
        return ticks * self.tick_value * self._size

    # ─── Mark-to-market ───────────────────────────────────────────────────

    def _mark_to_market(self):
        if self._side == 0:
            self._unrealized = 0.0
            return
        px = float(self.close_ary[self._t])
        self._unrealized = self._compute_pnl(px)
        self._bars_in_trade += 1
        self._mfe = max(self._mfe, self._unrealized)
        self._mae = min(self._mae, self._unrealized)

    # ─── Entry bonus ──────────────────────────────────────────────────────

    def _entry_bonus(self, action: int) -> float:
        feats = self.tech_ary[self._t]  # full 61-feature row
        bonus = 0.0
        thr = self.event_threshold
        scale = self.entry_bonus_scale

        if action == ACTION_ENTER_LONG:
            if feats[35] > thr:           # spring_score
                bonus += scale
            if feats[39] > thr and feats[15] > 0:  # absorption + up-wave
                bonus += scale * 0.6
            if feats[41] > thr and feats[15] > 0:  # stopping_action + up-wave
                bonus += scale * 0.4

        elif action == ACTION_ENTER_SHORT:
            if feats[36] > thr:           # upthrust_score
                bonus += scale
            if feats[39] > thr and feats[15] < 0:  # absorption + down-wave
                bonus += scale * 0.6
            if feats[41] > thr and feats[15] < 0:  # stopping_action + down-wave
                bonus += scale * 0.4

        return min(bonus, scale)  # cap stacking to 1× scale

    # ─── Management bonus ─────────────────────────────────────────────────

    def _has_entry_signal(self) -> bool:
        """Check if a primary Wyckoff event signal (spring/upthrust) is active."""
        feats = self.tech_ary[self._t]
        thr = self.event_threshold
        return bool(
            feats[35] > thr or   # spring  (~1% of bars)
            feats[36] > thr      # upthrust (~4% of bars)
        )

    def _management_bonus(self, action: int) -> float:
        """Reward holding winners, penalise overstaying in losers.
        When flat, penalise only explicit HOLD when a setup exists."""
        if self._side == 0:
            if action == ACTION_HOLD and self._has_entry_signal():
                return -self.idle_penalty
            return 0.0
        if self._unrealized > 0:
            return self.mgmt_bonus_scale
        if self._unrealized < 0 and self._bars_in_trade > self.overstay_bars:
            return -self.mgmt_penalty_scale
        return 0.0

    # ─── Regime mismatch penalty ──────────────────────────────────────────

    def _regime_mismatch_penalty(self, action: int) -> float:
        """Penalise entering or holding against the dominant Wyckoff phase.

        Fires on entry AND per bar while positioned against the dominant
        phase.  The per-bar component uses half the scale so the agent
        has time to exit rather than being crushed immediately.
        """
        feats = self.tech_ary[self._t]
        bullish = max(feats[53], feats[54])   # accum / markup
        bearish = max(feats[55], feats[56])   # distrib / markdown
        scale = self.regime_penalty_scale
        penalty = 0.0

        # On entry
        if action == ACTION_ENTER_SHORT and bullish > 0.7:
            penalty += scale
        if action == ACTION_ENTER_LONG and bearish > 0.7:
            penalty += scale

        # Per bar while positioned against phase (half scale)
        if self._side > 0 and bearish > 0.7:
            penalty += scale * 0.5
        elif self._side < 0 and bullish > 0.7:
            penalty += scale * 0.5

        return penalty

    # ─── Observation ──────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        # Market features (selected columns)
        feats = self.tech_ary[self._t, self._fi]  # (n_features,)

        # Position features
        px = float(self.close_ary[self._t])
        atr = self.bar_range  # configurable range bar size
        if self._side != 0:
            entry_dist = ((px - self._entry_price) / atr) * self._side
        else:
            entry_dist = 0.0

        pos_feats = np.array([
            float(self._side),
            self._size / max(self.max_position_size, 1),
            entry_dist,
            self._unrealized / self.pnl_norm,
            float(np.clip(self._realized / (2.0 * self.pnl_norm), -1.0, 1.0)),
            min(self._bars_in_trade / 50.0, 1.0),
            self._mfe / self.pnl_norm,
            self._mae / self.pnl_norm,
        ], dtype=np.float32)

        obs = np.concatenate([feats, pos_feats]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


# ═══════════════════════════════════════════════════════════════════════════
# GPU-Vectorized Environment  (for PPO training)
# ═══════════════════════════════════════════════════════════════════════════

class NQWyckoffWeisVecEnv:
    """
    GPU-vectorized Wyckoff/Weis discrete-action environment.

    All state is PyTorch tensors on GPU, num_envs episodes run in parallel
    with per-env bar tracking and auto-reset.

    Observation  (state_dim = 46):
      [38 selected features | 8 position features]

    Action (discrete, 6):
      hold | enter_long | enter_short | add | reduce | exit

    Reward:
      PnL_change / norm  +  event_alignment_bonus  −  penalty

    Usage::

        from elegantrl.agents.AgentPPO import AgentDiscretePPO
        from elegantrl.train.config import Config

        env_args = {
            'env_name': 'NQWyckoffWeisVecEnv-v1',
            'num_envs': 256,
            'npz_path': 'wyckoff_nq_40pt.npz',
            'episode_len': 1024,
            'gpu_id': 0,
        }
        args = Config(AgentDiscretePPO, NQWyckoffWeisVecEnv, env_args)
    """

    def __init__(
        self,
        npz_path: str = "",
        num_envs: int = 256,
        gpu_id: int = 0,
        episode_len: int = 1024,
        beg_idx: int = 0,
        end_idx: int = 0,
        commission: float = 1.50,
        slippage_ticks: float = 1.0,
        tick_size: float = 0.25,
        tick_value: float = 5.0,
        max_position_size: int = 2,
        reward_scale: float = 1.0,
        event_threshold: float = 0.3,
        entry_bonus_scale: float = 0.50,
        invalid_penalty: float = 0.02,
        mgmt_bonus_scale: float = 0.02,
        mgmt_penalty_scale: float = 0.03,
        overstay_bars: int = 20,
        regime_penalty_scale: float = 0.05,
        idle_penalty: float = 0.05,
        carry_cost: float = 0.0,
        vesting_bars: int = 10,
        pnl_norm: float = 2000.0,
        reward_clip: float = 2.0,
        bar_range: float = 40.0,
        feature_indices: list[int] | None = None,
        gamma: float = 0.99,            # kept for ElegantRL Config compat
        reward_mode: str = "pnl",       # kept for Config compat
        log_dir: str = "",
        **kwargs,
    ):
        self.device = th.device(
            f"cuda:{gpu_id}" if (th.cuda.is_available() and gpu_id >= 0) else "cpu"
        )

        # ── Load data ────────────────────────────────────────────────────
        close_ary, tech_ary = _load_npz(npz_path)
        if end_idx <= 0:
            end_idx = len(close_ary)
        close_ary = close_ary[beg_idx:end_idx]
        tech_ary = tech_ary[beg_idx:end_idx]

        fi = feature_indices or ENV_FEATURE_INDICES
        n_selected = len(fi)

        # Full tech array on GPU (we index selected columns in get_state)
        self.close_price = th.tensor(close_ary, dtype=th.float32, device=self.device)
        self.tech_factor = th.tensor(tech_ary, dtype=th.float32, device=self.device)
        self.feature_cols = th.tensor(fi, dtype=th.long, device=self.device)
        self.n_features = n_selected

        # ── Trading config ───────────────────────────────────────────────
        self.commission = commission
        self.slip = slippage_ticks * tick_size  # $ slippage per contract
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.max_position_size = float(max_position_size)
        self.reward_scale = reward_scale
        self.event_threshold = event_threshold
        self.entry_bonus_scale = entry_bonus_scale
        self.invalid_penalty_val = invalid_penalty
        self.mgmt_bonus_scale = mgmt_bonus_scale
        self.mgmt_penalty_scale = mgmt_penalty_scale
        self.overstay_bars = overstay_bars
        self.regime_penalty_scale = regime_penalty_scale
        self.idle_penalty = idle_penalty
        self.carry_cost = carry_cost
        self.vesting_bars = vesting_bars
        self.reward_clip = reward_clip
        self.pnl_norm = pnl_norm  # $2000 default – 40pt 2-lot = 0.8 (no clip)
        self.bar_range = bar_range  # range bar size for ATR normalization
        self.gamma = gamma

        # Local drift detrending: per-bar rolling mean of prior price changes.
        # Removes LOCAL trend bias so passive long/short → ~0 expected reward.
        # shift(1) avoids lookahead: drift[t] uses bars [t-50, t-1] only.
        close_flat = close_ary.flatten()
        bar_changes = np.diff(close_flat, prepend=close_flat[0])
        import pandas as pd
        rolling_drift = (pd.Series(bar_changes)
                         .rolling(50, min_periods=1).mean()
                         .shift(1).fillna(0.0).values)
        self.drift_per_bar = th.tensor(rolling_drift, dtype=th.float32, device=self.device)

        # ── ElegantRL metadata ───────────────────────────────────────────
        self.env_name = "NQWyckoffWeisVecEnv-v1"
        self.num_envs = num_envs
        self.max_step = self.close_price.shape[0] - 1
        self.state_dim = n_selected + N_POSITION_FEATURES   # 38 + 8 = 46
        self.action_dim = N_ACTIONS                          # 6
        self.if_discrete = True
        self.target_return = +np.inf

        # Sub-episode config
        self._stagger = episode_len is not None and episode_len < self.max_step
        self._episode_len = episode_len if self._stagger else self.max_step

        # ── Event feature column indices into *full* tech_ary (61 cols) ──
        # Used for reward shaping — index directly into self.tech_factor
        # NOTE: 3 wave features appended at 58-60, original 58 indices preserved
        self._col_spring = 35
        self._col_upthrust = 36
        self._col_absorption = 39
        self._col_stopping = 41
        self._col_wave_dir = 15
        # Phase columns for regime mismatch penalty
        self._col_phase_accum = 53
        self._col_phase_markup = 54
        self._col_phase_distrib = 55
        self._col_phase_markdown = 56

        # ── Per-env state (allocated in reset) ───────────────────────────
        self.day = None             # (ne,) long — bar index
        self.step_count = None      # (ne,) long — steps in current episode
        self.pos_side = None        # (ne,) float: -1, 0, +1
        self.pos_size = None        # (ne,) float: 0 .. max
        self.entry_price = None     # (ne,) float
        self.bars_in_trade = None   # (ne,) long
        self.unrealized_pnl = None  # (ne,) float ($)
        self.realized_pnl = None    # (ne,) float ($)
        self.mfe = None             # (ne,) float ($)
        self.mae = None             # (ne,) float ($)
        # Entry bonus vesting (deferred payout after holding vesting_bars)
        self.vesting_amount = None    # (ne,) float — deferred bonus amount
        self.vesting_remaining = None # (ne,) long — bars until payout
        # Bookkeeping
        self.cumulative_returns = None
        self.total_trades = None

        # Reward component diagnostics (running sums for logging)
        self._diag_pnl = 0.0
        self._diag_entry = 0.0
        self._diag_mgmt = 0.0
        self._diag_penalty = 0.0
        self._diag_regime = 0.0
        self._diag_carry = 0.0
        self._diag_steps = 0
        self._diag_action_counts = th.zeros(N_ACTIONS, dtype=th.long, device=self.device)
        self._diag_episodes_done = 0
        self._diag_episode_pnl_sum = 0.0
        self._diag_episode_pnl_sq = 0.0
        self._diag_episode_trades_sum = 0

        # ── File logger ──────────────────────────────────────────────────
        self._flog = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            _pid = os.getpid()
            _logpath = os.path.join(log_dir, f"env_diag_pid{_pid}.log")
            self._flog = open(_logpath, "a", buffering=1)  # line-buffered
            self._flog.write(
                f"# NQWyckoffWeisVecEnv diag | pid={_pid} num_envs={num_envs} "
                f"episode_len={episode_len} bars={len(close_ary)} "
                f"entry_bonus={entry_bonus_scale} pnl_norm={pnl_norm}\n"
            )
            self._flog.write(
                "step,|pnl|,|entry|,|mgmt|,|pen|,|regime|,"
                "H%,EL%,ES%,A%,R%,X%,"
                "ep_done,ep_pnl_mean,ep_pnl_std,ep_trades_mean,"
                "pos_pct,long_pct,short_pct,avg_unreal\n"
            )

        # Startup banner — confirms new code is loaded
        print(
            f"[VecEnv] pnl_norm={self.pnl_norm} "
            f"drift_per_bar=local_rolling_50 "
            f"drift_mean={self.drift_per_bar.mean().item():.4f}pts "
            f"drift_std={self.drift_per_bar.std().item():.4f}pts "
            f"carry_cost={self.carry_cost} idle_penalty={self.idle_penalty} "
            f"entry_bonus={self.entry_bonus_scale} vesting_bars={self.vesting_bars} "
            f"regime_penalty={self.regime_penalty_scale} "
            f"bars={self.close_price.shape[0]} num_envs={num_envs}",
            flush=True,
        )

    # ─── Reset ────────────────────────────────────────────────────────────

    def reset(self):
        ne = self.num_envs
        dev = self.device

        self.pos_side = th.zeros(ne, dtype=th.float32, device=dev)
        self.pos_size = th.zeros(ne, dtype=th.float32, device=dev)
        self.entry_price = th.zeros(ne, dtype=th.float32, device=dev)
        self.bars_in_trade = th.zeros(ne, dtype=th.long, device=dev)
        self.unrealized_pnl = th.zeros(ne, dtype=th.float32, device=dev)
        self.realized_pnl = th.zeros(ne, dtype=th.float32, device=dev)
        self.mfe = th.zeros(ne, dtype=th.float32, device=dev)
        self.mae = th.zeros(ne, dtype=th.float32, device=dev)
        self.vesting_amount = th.zeros(ne, dtype=th.float32, device=dev)
        self.vesting_remaining = th.zeros(ne, dtype=th.long, device=dev)
        self.total_trades = th.zeros(ne, dtype=th.long, device=dev)
        self.cumulative_returns = [0.0] * ne

        if self._stagger:
            max_start = max(50, self.max_step - self._episode_len)
            self.day = th.randint(50, max_start, (ne,), dtype=th.long, device=dev)
            # Desynchronise via day offset only; step_count starts at 0
            # so every env gets a full first episode (no partial-episode waste)
            self.step_count = th.zeros(ne, dtype=th.long, device=dev)
        else:
            self.day = th.full((ne,), 50, dtype=th.long, device=dev)  # skip warmup
            self.step_count = th.zeros(ne, dtype=th.long, device=dev)

        return self.get_state(), {}

    # ─── Observation ──────────────────────────────────────────────────────

    def get_state(self):
        """Return (num_envs, state_dim) tensor."""
        # Market features
        features = self.tech_factor[self.day][:, self.feature_cols]  # (ne, n_features)

        # Position features
        curr_price = self.close_price[self.day]  # (ne,)
        atr = self.bar_range  # configurable range bar size

        entry_dist = th.where(
            self.pos_side != 0,
            ((curr_price - self.entry_price) / atr) * self.pos_side,
            th.zeros_like(self.pos_side),
        )

        pos_feats = th.stack([
            self.pos_side,
            self.pos_size / max(self.max_position_size, 1.0),
            entry_dist,
            self.unrealized_pnl / self.pnl_norm,
            (self.realized_pnl / (2.0 * self.pnl_norm)).clamp(-1.0, 1.0),
            (self.bars_in_trade.float() / 50.0).clamp(max=1.0),
            self.mfe / self.pnl_norm,
            self.mae / self.pnl_norm,
        ], dim=1)  # (ne, 8)

        state = th.cat([features, pos_feats], dim=1)  # (ne, state_dim)
        return th.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

    # ─── Step ─────────────────────────────────────────────────────────────

    def step(self, action):
        """
        Vectorized step.

        Args:
            action: (num_envs,) int64  — values in [0, 5]
        Returns:
            state:    (num_envs, state_dim)
            reward:   (num_envs,)
            terminal: (num_envs,) bool  — always False (time-limit only)
            done:     (num_envs,) bool
            info:     dict
        """
        if action.dim() == 2:
            action = action[:, 0]
        action = action.long()

        # 1) Snapshot pre-action state for reward gating
        prev_price = self.close_price[self.day]
        prev_side = self.pos_side.clone()  # needed to gate entry bonus

        # 2) Execute discrete actions
        penalty = self._execute_actions(action)

        # 3) Snapshot post-action position for correct credit assignment
        #    (entry actions now get PnL credit on the bar they enter)
        post_side = self.pos_side.clone()
        post_size = self.pos_size.clone()

        # 4) Advance bar
        self.day = th.clamp(self.day + 1, max=self.max_step)
        self.step_count += 1

        # 5) Mark-to-market
        self._mark_to_market()

        # 6) Reward — dense: position_exposure × price_change
        #    For entry bars, use fill price (not bar close) as baseline
        #    to avoid phantom edge from ignoring slippage in the reward.
        curr_price = self.close_price[self.day]
        just_entered = (action == ACTION_ENTER_LONG) | (action == ACTION_ENTER_SHORT)
        was_flat = (prev_price == self.entry_price)  # stale check
        # Entry bar baseline = entry_price (includes slippage), else prev bar close
        baseline = th.where(
            just_entered & (self.pos_side.abs() > 0),
            self.entry_price,
            prev_price,
        )
        price_change = curr_price - baseline
        # Detrend: subtract local rolling drift so passive directional → ~0 reward
        detrended_change = price_change - self.drift_per_bar[self.day]
        ticks = (detrended_change / self.tick_size) * post_side
        holding_pnl = ticks * self.tick_value * post_size
        pnl_delta = holding_pnl / self.pnl_norm

        # Entry bonus: deferred until vesting_bars of holding
        # On entry with valid signal, store the bonus; pay it only after
        # the agent has held for vesting_bars (forfeit if exiting early)
        raw_entry_bonus = self._compute_entry_bonus(action)
        # Gate: only award bonus when entering from FLAT (prevent flip farming)
        was_flat = prev_side == 0
        raw_entry_bonus = th.where(was_flat, raw_entry_bonus, th.zeros_like(raw_entry_bonus))
        has_new_bonus = raw_entry_bonus > 0
        if has_new_bonus.any():
            self.vesting_amount = th.where(has_new_bonus, raw_entry_bonus, self.vesting_amount)
            self.vesting_remaining = th.where(has_new_bonus,
                                              th.full_like(self.vesting_remaining, self.vesting_bars),
                                              self.vesting_remaining)
        # Tick down and pay when vesting completes
        still_vesting = self.vesting_remaining > 0
        self.vesting_remaining = th.where(still_vesting,
                                          self.vesting_remaining - 1,
                                          self.vesting_remaining)
        vesting_done = still_vesting & (self.vesting_remaining == 0)
        entry_bonus = th.where(vesting_done, self.vesting_amount,
                               th.zeros_like(self.vesting_amount))
        self.vesting_amount = th.where(vesting_done,
                                       th.zeros_like(self.vesting_amount),
                                       self.vesting_amount)

        mgmt_bonus = self._compute_management_bonus(action)
        regime_penalty = self._compute_regime_mismatch_penalty(action)

        # Carry cost: per-bar cost scaled by position size (1 lot = 1×, 2 lots = 2×)
        carry = self.carry_cost * post_size

        reward = (pnl_delta + entry_bonus + mgmt_bonus
                  - penalty - regime_penalty - carry) * self.reward_scale
        if self.reward_clip > 0:
            reward = th.clamp(reward, -self.reward_clip, self.reward_clip)

        # Reward component diagnostics
        self._diag_pnl += pnl_delta.abs().mean().item()
        self._diag_entry += entry_bonus.abs().mean().item()
        self._diag_mgmt += mgmt_bonus.abs().mean().item()
        self._diag_penalty += penalty.abs().mean().item()
        self._diag_regime += regime_penalty.abs().mean().item()
        self._diag_carry += carry.mean().item()
        self._diag_steps += 1
        # Action distribution tracking
        for a in range(N_ACTIONS):
            self._diag_action_counts[a] += (action == a).sum()
        if self._diag_steps % 5000 == 0:
            self._log_diagnostics(action)

        # 7) Episode management
        done = (self.step_count >= self._episode_len) | (self.day >= self.max_step)

        # Flatten positions at episode end
        self._flatten_done(done)

        # Auto-reset done envs
        if done.any():
            for i in th.where(done)[0].tolist():
                pnl_i = self.realized_pnl[i].item()
                self.cumulative_returns[i] = pnl_i
                self._diag_episodes_done += 1
                self._diag_episode_pnl_sum += pnl_i
                self._diag_episode_pnl_sq += pnl_i ** 2
                self._diag_episode_trades_sum += self.total_trades[i].item()
            self._auto_reset(done)

        state = self.get_state()
        terminal = th.zeros(self.num_envs, dtype=th.bool, device=self.device)
        return state, reward, terminal, done, {}

    # ─── Action execution (vectorized) ────────────────────────────────────

    def _execute_actions(self, action):
        """Execute discrete actions. Returns penalty (num_envs,)."""
        ne = self.num_envs
        dev = self.device
        penalty = th.zeros(ne, dtype=th.float32, device=dev)

        curr_price = self.close_price[self.day]
        buy_price = curr_price + self.slip
        sell_price = curr_price - self.slip

        is_flat = self.pos_side == 0
        is_positioned = ~is_flat
        is_long = self.pos_side > 0

        # ── ENTER_LONG ──
        enter_long = action == ACTION_ENTER_LONG
        valid_el = enter_long & is_flat
        invalid_el = enter_long & is_positioned

        self.pos_side = th.where(valid_el, th.ones_like(self.pos_side), self.pos_side)
        self.pos_size = th.where(valid_el, th.ones_like(self.pos_size), self.pos_size)
        self.entry_price = th.where(valid_el, buy_price, self.entry_price)
        self.bars_in_trade = th.where(valid_el, th.zeros_like(self.bars_in_trade), self.bars_in_trade)
        self.mfe = th.where(valid_el, th.zeros_like(self.mfe), self.mfe)
        self.mae = th.where(valid_el, th.zeros_like(self.mae), self.mae)
        self.realized_pnl = th.where(valid_el, self.realized_pnl - self.commission, self.realized_pnl)
        self.total_trades = th.where(valid_el, self.total_trades + 1, self.total_trades)

        # ── ENTER_SHORT ──
        enter_short = action == ACTION_ENTER_SHORT
        valid_es = enter_short & is_flat
        invalid_es = enter_short & is_positioned

        self.pos_side = th.where(valid_es, -th.ones_like(self.pos_side), self.pos_side)
        self.pos_size = th.where(valid_es, th.ones_like(self.pos_size), self.pos_size)
        self.entry_price = th.where(valid_es, sell_price, self.entry_price)
        self.bars_in_trade = th.where(valid_es, th.zeros_like(self.bars_in_trade), self.bars_in_trade)
        self.mfe = th.where(valid_es, th.zeros_like(self.mfe), self.mfe)
        self.mae = th.where(valid_es, th.zeros_like(self.mae), self.mae)
        self.realized_pnl = th.where(valid_es, self.realized_pnl - self.commission, self.realized_pnl)
        self.total_trades = th.where(valid_es, self.total_trades + 1, self.total_trades)

        # ── ADD ──
        add = action == ACTION_ADD
        can_add = is_positioned & (self.pos_size < self.max_position_size)
        valid_add = add & can_add
        invalid_add = add & (~can_add)

        add_price = th.where(is_long, buy_price, sell_price)
        old_size = self.pos_size
        new_size = old_size + 1.0
        # Average entry price: weighted avg of old entry + new add price
        new_avg = (self.entry_price * old_size + add_price) / new_size.clamp(min=1.0)
        self.entry_price = th.where(valid_add, new_avg, self.entry_price)
        self.pos_size = th.where(valid_add, new_size, self.pos_size)
        self.realized_pnl = th.where(valid_add, self.realized_pnl - self.commission, self.realized_pnl)
        self.total_trades = th.where(valid_add, self.total_trades + 1, self.total_trades)

        # ── REDUCE ──
        reduce = action == ACTION_REDUCE
        valid_reduce = reduce & is_positioned
        invalid_reduce = reduce & is_flat

        # PnL for 1 contract (fractional share of total position PnL)
        reduce_price = th.where(is_long, sell_price, buy_price)
        frac = (1.0 / self.pos_size.clamp(min=1.0))
        reduce_pnl = self._compute_pnl(reduce_price) * frac
        self.realized_pnl = th.where(
            valid_reduce, self.realized_pnl + reduce_pnl - self.commission, self.realized_pnl
        )
        new_size_r = (self.pos_size - 1.0).clamp(min=0.0)
        goes_flat = valid_reduce & (new_size_r <= 0)
        self.pos_size = th.where(valid_reduce, new_size_r, self.pos_size)
        # Clear position if fully reduced
        self.pos_side = th.where(goes_flat, th.zeros_like(self.pos_side), self.pos_side)
        self.entry_price = th.where(goes_flat, th.zeros_like(self.entry_price), self.entry_price)
        self.vesting_amount = th.where(goes_flat, th.zeros_like(self.vesting_amount), self.vesting_amount)
        self.vesting_remaining = th.where(goes_flat, th.zeros_like(self.vesting_remaining), self.vesting_remaining)
        self.total_trades = th.where(valid_reduce, self.total_trades + 1, self.total_trades)

        # ── EXIT ──
        exit_ = action == ACTION_EXIT
        valid_exit = exit_ & is_positioned
        invalid_exit = exit_ & is_flat

        exit_price = th.where(is_long, sell_price, buy_price)
        exit_pnl = self._compute_pnl(exit_price)
        exit_comm = self.commission * self.pos_size  # per-contract commission
        self.realized_pnl = th.where(
            valid_exit, self.realized_pnl + exit_pnl - exit_comm, self.realized_pnl
        )
        self.pos_side = th.where(valid_exit, th.zeros_like(self.pos_side), self.pos_side)
        self.pos_size = th.where(valid_exit, th.zeros_like(self.pos_size), self.pos_size)
        self.entry_price = th.where(valid_exit, th.zeros_like(self.entry_price), self.entry_price)
        self.vesting_amount = th.where(valid_exit, th.zeros_like(self.vesting_amount), self.vesting_amount)
        self.vesting_remaining = th.where(valid_exit, th.zeros_like(self.vesting_remaining), self.vesting_remaining)
        self.total_trades = th.where(valid_exit, self.total_trades + 1, self.total_trades)

        # ── Penalties ──
        invalid = invalid_el | invalid_es | invalid_add | invalid_reduce | invalid_exit
        penalty = th.where(invalid, th.full_like(penalty, self.invalid_penalty_val), penalty)

        return penalty

    # ─── PnL helpers ──────────────────────────────────────────────────────

    def _compute_pnl(self, price):
        """Compute $ PnL for full position at given price. Shape (ne,)."""
        ticks = ((price - self.entry_price) / self.tick_size) * self.pos_side
        return ticks * self.tick_value * self.pos_size

    def _mark_to_market(self):
        """Update unrealized PnL, bars_in_trade, MFE/MAE."""
        positioned = self.pos_side != 0
        px = self.close_price[self.day]
        new_pnl = self._compute_pnl(px)

        self.unrealized_pnl = th.where(positioned, new_pnl, th.zeros_like(new_pnl))
        self.bars_in_trade = th.where(
            positioned, self.bars_in_trade + 1, self.bars_in_trade
        )
        self.mfe = th.where(positioned, th.maximum(self.mfe, new_pnl), self.mfe)
        self.mae = th.where(positioned, th.minimum(self.mae, new_pnl), self.mae)

    # ─── Reward shaping ───────────────────────────────────────────────────

    def _compute_entry_bonus(self, action):
        """
        Wyckoff event-aligned entry bonus.

        Awards a small bonus when the agent enters on a recognised Wyckoff
        setup (spring, upthrust, absorption, stopping action).
        """
        ne = self.num_envs
        bonus = th.zeros(ne, dtype=th.float32, device=self.device)
        thr = self.event_threshold
        scale = self.entry_bonus_scale

        # Read event features for current bar (full 58 cols)
        spring   = self.tech_factor[self.day, self._col_spring]     # (ne,)
        upthrust = self.tech_factor[self.day, self._col_upthrust]
        absorb   = self.tech_factor[self.day, self._col_absorption]
        stopping = self.tech_factor[self.day, self._col_stopping]
        wave_dir = self.tech_factor[self.day, self._col_wave_dir]

        # ── Long entries ──
        is_long_entry = action == ACTION_ENTER_LONG
        # Spring + up-wave → best long setup
        spring_match = is_long_entry & (spring > thr)
        bonus = th.where(spring_match, bonus + scale, bonus)
        # Absorption in up-wave
        absorb_long = is_long_entry & (absorb > thr) & (wave_dir > 0)
        bonus = th.where(absorb_long, bonus + scale * 0.6, bonus)
        # Stopping action in up-wave
        stop_long = is_long_entry & (stopping > thr) & (wave_dir > 0)
        bonus = th.where(stop_long, bonus + scale * 0.4, bonus)

        # ── Short entries ──
        is_short_entry = action == ACTION_ENTER_SHORT
        # Upthrust + down-wave → best short setup
        upthrust_match = is_short_entry & (upthrust > thr)
        bonus = th.where(upthrust_match, bonus + scale, bonus)
        # Absorption in down-wave
        absorb_short = is_short_entry & (absorb > thr) & (wave_dir < 0)
        bonus = th.where(absorb_short, bonus + scale * 0.6, bonus)
        # Stopping action in down-wave
        stop_short = is_short_entry & (stopping > thr) & (wave_dir < 0)
        bonus = th.where(stop_short, bonus + scale * 0.4, bonus)

        return th.clamp(bonus, max=scale)  # cap stacking to 1× scale

    # ─── Management bonus (vectorized) ─────────────────────────────────────

    def _compute_management_bonus(self, action):
        """
        Reward holding winners, penalise overstaying in losers.
        When flat, penalise only explicit HOLD when a setup exists.

        -idle_penalty      when flat + HOLD + setup active.
        +mgmt_bonus_scale  per bar while in a winning position.
        -mgmt_penalty_scale per bar when in a losing position too long.
        """
        ne = self.num_envs
        bonus = th.zeros(ne, dtype=th.float32, device=self.device)

        is_flat = self.pos_side == 0
        positioned = ~is_flat
        winning = positioned & (self.unrealized_pnl > 0)
        losing_overstay = (positioned
                           & (self.unrealized_pnl < 0)
                           & (self.bars_in_trade > self.overstay_bars))

        # Conditional idle penalty: only penalise HOLD when a primary setup exists
        thr = self.event_threshold
        spring   = self.tech_factor[self.day, self._col_spring] > thr
        upthrust = self.tech_factor[self.day, self._col_upthrust] > thr
        setup_present = spring | upthrust  # ~5% of bars (was 37.5% with absorption+stopping)
        hold_on_setup = is_flat & (action == ACTION_HOLD) & setup_present

        bonus = th.where(hold_on_setup, bonus - self.idle_penalty, bonus)
        bonus = th.where(winning, bonus + self.mgmt_bonus_scale, bonus)
        bonus = th.where(losing_overstay, bonus - self.mgmt_penalty_scale, bonus)
        return bonus

    # ─── Regime mismatch penalty (vectorized) ─────────────────────────────

    def _compute_regime_mismatch_penalty(self, action):
        """Penalise entering or holding against the dominant Wyckoff phase.

        Fires on entry AND per bar while positioned against the dominant
        phase.  The per-bar component uses half the scale so the agent
        has time to exit rather than being crushed immediately.
        """
        ne = self.num_envs
        penalty = th.zeros(ne, dtype=th.float32, device=self.device)
        scale = self.regime_penalty_scale

        # Phase scores from full tech_ary
        phase_accum  = self.tech_factor[self.day, self._col_phase_accum]
        phase_markup = self.tech_factor[self.day, self._col_phase_markup]
        phase_distrib  = self.tech_factor[self.day, self._col_phase_distrib]
        phase_markdown = self.tech_factor[self.day, self._col_phase_markdown]

        bullish = th.maximum(phase_accum, phase_markup)
        bearish = th.maximum(phase_distrib, phase_markdown)

        # On entry
        short_vs_bull = (action == ACTION_ENTER_SHORT) & (bullish > 0.7)
        penalty = th.where(short_vs_bull, penalty + scale, penalty)
        long_vs_bear = (action == ACTION_ENTER_LONG) & (bearish > 0.7)
        penalty = th.where(long_vs_bear, penalty + scale, penalty)

        # Per bar while positioned against phase (half scale)
        long_in_bear = (self.pos_side > 0) & (bearish > 0.7)
        penalty = th.where(long_in_bear, penalty + scale * 0.5, penalty)
        short_in_bull = (self.pos_side < 0) & (bullish > 0.7)
        penalty = th.where(short_in_bull, penalty + scale * 0.5, penalty)

        return penalty

    # ─── Episode management ───────────────────────────────────────────────

    def _flatten_done(self, done):
        """Flatten open positions for done envs."""
        has_pos = done & (self.pos_side.abs() > 1e-6)
        if not has_pos.any():
            return
        is_long = self.pos_side > 0
        exit_price = th.where(
            is_long,
            self.close_price[self.day] - self.slip,
            self.close_price[self.day] + self.slip,
        )
        exit_pnl = self._compute_pnl(exit_price)
        flatten_comm = self.commission * self.pos_size  # per-contract commission
        self.realized_pnl = th.where(
            has_pos, self.realized_pnl + exit_pnl - flatten_comm, self.realized_pnl
        )
        self.pos_side = th.where(done, th.zeros_like(self.pos_side), self.pos_side)
        self.pos_size = th.where(done, th.zeros_like(self.pos_size), self.pos_size)

    def _auto_reset(self, mask):
        """Reset done envs to new random starting positions."""
        if self._stagger:
            n = mask.sum().item()
            max_start = max(50, self.max_step - self._episode_len)
            self.day[mask] = th.randint(50, max_start, (n,), dtype=th.long, device=self.device)
        else:
            self.day[mask] = 50

        self.step_count[mask] = 0
        self.pos_side[mask] = 0.0
        self.pos_size[mask] = 0.0
        self.entry_price[mask] = 0.0
        self.bars_in_trade[mask] = 0
        self.unrealized_pnl[mask] = 0.0
        self.realized_pnl[mask] = 0.0
        self.mfe[mask] = 0.0
        self.mae[mask] = 0.0
        self.vesting_amount[mask] = 0.0
        self.vesting_remaining[mask] = 0
        self.total_trades[mask] = 0

    # ─── Diagnostics ──────────────────────────────────────────────────────

    def _log_diagnostics(self, action):
        """Write comprehensive diagnostics to log file (and brief summary to stdout)."""
        n = self._diag_steps
        ne = self.num_envs

        # Reward components (running averages)
        r_pnl = self._diag_pnl / n
        r_entry = self._diag_entry / n
        r_mgmt = self._diag_mgmt / n
        r_pen = self._diag_penalty / n
        r_regime = self._diag_regime / n
        r_carry = self._diag_carry / n

        # Action distribution (%)
        total_actions = self._diag_action_counts.sum().item()
        if total_actions > 0:
            act_pct = (self._diag_action_counts.float() / total_actions * 100).tolist()
        else:
            act_pct = [0.0] * N_ACTIONS

        # Position snapshot
        pos_pct = (self.pos_side != 0).float().mean().item() * 100
        long_pct = (self.pos_side > 0).float().mean().item() * 100
        short_pct = (self.pos_side < 0).float().mean().item() * 100
        avg_unreal = self.unrealized_pnl.mean().item()

        # Episode stats
        ep_done = self._diag_episodes_done
        if ep_done > 0:
            ep_pnl_mean = self._diag_episode_pnl_sum / ep_done
            ep_pnl_var = self._diag_episode_pnl_sq / ep_done - ep_pnl_mean ** 2
            ep_pnl_std = ep_pnl_var ** 0.5 if ep_pnl_var > 0 else 0.0
            ep_trades_mean = self._diag_episode_trades_sum / ep_done
        else:
            ep_pnl_mean = ep_pnl_std = ep_trades_mean = 0.0

        # Brief stdout
        print(
            f"[Diag] s={n} |pnl|={r_pnl:.4f} |ent|={r_entry:.4f} carry={r_carry:.4f} "
            f"act=[H{act_pct[0]:.0f} EL{act_pct[1]:.0f} ES{act_pct[2]:.0f} "
            f"A{act_pct[3]:.0f} R{act_pct[4]:.0f} X{act_pct[5]:.0f}] "
            f"pos={pos_pct:.0f}% ep_pnl=${ep_pnl_mean:.0f}±{ep_pnl_std:.0f} "
            f"trades={ep_trades_mean:.1f}",
            flush=True
        )

        # Detailed CSV to file
        if self._flog:
            self._flog.write(
                f"{n},{r_pnl:.6f},{r_entry:.6f},{r_mgmt:.6f},{r_pen:.6f},{r_regime:.6f},"
                f"{act_pct[0]:.1f},{act_pct[1]:.1f},{act_pct[2]:.1f},"
                f"{act_pct[3]:.1f},{act_pct[4]:.1f},{act_pct[5]:.1f},"
                f"{ep_done},{ep_pnl_mean:.2f},{ep_pnl_std:.2f},{ep_trades_mean:.1f},"
                f"{pos_pct:.1f},{long_pct:.1f},{short_pct:.1f},{avg_unreal:.2f}\n"
            )

    def close(self):
        if self._flog:
            self._flog.close()
            self._flog = None
