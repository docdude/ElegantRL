"""
Transformer-compatible Wyckoff Trading Environment (GPU-vectorized).

Key difference from NQWyckoffWeisVecEnv:
  - Observation is (seq_len × n_bar_features) + (n_position_features,),
    flattened into a 1D vector per env so ElegantRL's buffer can store it.
  - The actor's _split_state() reshapes it back to (batch, seq_len, n_features).
  - Each bar, the env maintains a sliding window of the last `seq_len` bars
    of selected features, providing temporal context to the transformer.

Trading mechanics (action execution, PnL, rewards) are inherited directly
from NQWyckoffWeisVecEnv to ensure identical backtest semantics.
"""

import os
import numpy as np
import torch as th

from .config_v2 import TransformerConfig, TRANSFORMER_FEATURE_INDICES

# Action constants (same as NQWyckoffWeisEnv)
ACTION_HOLD = 0
ACTION_ENTER_LONG = 1
ACTION_ENTER_SHORT = 2
ACTION_ADD = 3
ACTION_REDUCE = 4
ACTION_EXIT = 5
N_ACTIONS = 6
N_POSITION_FEATURES = 8


def _load_npz(npz_path: str):
    if npz_path is None or not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    close_ary = data['close_ary'].astype(np.float32)
    tech_ary = data['tech_ary'].astype(np.float32)
    if close_ary.ndim == 2:
        close_ary = close_ary[:, 0]
    return close_ary, tech_ary


class WyckoffTransformerVecEnv:
    """
    GPU-vectorized env that provides sliding-window observations
    for the transformer encoder.

    Observation layout (flattened):
      [seq_len × n_bar_features | 8 position features]

    The actor's _split_state() reshapes the first part to
    (batch, seq_len, n_bar_features) for the transformer.

    Parameters
    ----------
    config : TransformerConfig
        Architecture config (seq_len, feature indices, etc.)
    npz_path : str
        Path to NPZ with close_ary and tech_ary.
    num_envs : int
        Number of parallel environments.
    gpu_id : int
        GPU device index (-1 for CPU).
    episode_len : int
        Bars per sub-episode.
    commission, tick_size, tick_value, etc.
        Trading parameters (same as NQWyckoffWeisVecEnv).
    """

    def __init__(
        self,
        config: TransformerConfig | None = None,
        npz_path: str = "",
        num_envs: int = 256,
        gpu_id: int = 0,
        episode_len: int = 512,
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
        carry_cost: float = 0.0,
        carry_multiplier: float = 0.3,
        vesting_bars: int = 10,
        pnl_norm: float = 2000.0,
        reward_clip: float = 2.0,
        bar_range: float = 40.0,
        reward_mode: str = "dense_pnl",
        sign_flip: bool = False,
        gamma: float = 0.99,
        feat_mean: np.ndarray | None = None,
        feat_std: np.ndarray | None = None,
        **kwargs,
    ):
        if config is None:
            config = TransformerConfig()
        self.config = config

        self.device = th.device(
            f"cuda:{gpu_id}" if (th.cuda.is_available() and gpu_id >= 0) else "cpu"
        )
        self.reward_mode = reward_mode
        self.sign_flip = sign_flip

        # Load data
        close_ary, tech_ary = _load_npz(npz_path)
        if end_idx <= 0:
            end_idx = len(close_ary)
        close_ary = close_ary[beg_idx:end_idx]
        tech_ary = tech_ary[beg_idx:end_idx]

        # Feature selection for transformer input
        fi = list(config.feature_indices) if config.feature_indices else TRANSFORMER_FEATURE_INDICES
        self._fi = th.tensor(fi, dtype=th.long, device=self.device)
        self.n_bar_features = len(fi)
        self.seq_len = config.seq_len

        # Full arrays on GPU
        self.close_price = th.tensor(close_ary, dtype=th.float32, device=self.device)
        self.tech_factor = th.tensor(tech_ary, dtype=th.float32, device=self.device)

        # Pre-select features and normalize using pre-training stats
        selected = self.tech_factor[:, self._fi]  # (n_bars, n_bar_features)
        if feat_mean is not None and feat_std is not None:
            _mean = th.tensor(feat_mean.flatten(), dtype=th.float32, device=self.device)
            _std = th.tensor(feat_std.flatten(), dtype=th.float32, device=self.device)
            selected = (selected - _mean.unsqueeze(0)) / _std.unsqueeze(0)
            print("  [TransformerVecEnv] Applied pre-training feature normalization")
        pad = th.zeros(self.seq_len - 1, self.n_bar_features, dtype=th.float32, device=self.device)
        self._padded_features = th.cat([pad, selected], dim=0)  # (pad + n_bars, n_features)

        # Trading config
        self.commission = commission
        self.slip = slippage_ticks * tick_size
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.max_position_size = float(max_position_size)
        self.reward_scale = reward_scale
        self.event_threshold = event_threshold
        self.entry_bonus_scale = entry_bonus_scale
        self.invalid_penalty_val = invalid_penalty
        self.carry_cost = carry_cost
        self.carry_multiplier = carry_multiplier
        self.vesting_bars = vesting_bars
        self.pnl_norm = pnl_norm
        self.reward_clip = reward_clip
        self.bar_range = bar_range
        self.gamma = gamma

        # Dense PnL adjustments
        if self.reward_mode == 'dense_pnl' and pnl_norm > 1000:
            self.pnl_norm = 500.0
        if self.reward_mode == 'dense_pnl' and reward_clip < 5.0:
            self.reward_clip = 5.0

        # Auto-calibrate carry cost
        if self.carry_multiplier > 0 and self.reward_mode == 'dense_pnl':
            bar_changes = np.diff(close_ary.flatten(), prepend=close_ary.flatten()[0])
            mean_abs_change = float(np.abs(bar_changes).mean())
            expected_abs_pnl = mean_abs_change * (tick_value / tick_size)
            self.carry_cost = (expected_abs_pnl / self.pnl_norm) * self.carry_multiplier

        # Drift detrending
        import pandas as pd
        close_flat = close_ary.flatten()
        bar_changes = np.diff(close_flat, prepend=close_flat[0])
        rolling_drift = (pd.Series(bar_changes)
                         .rolling(50, min_periods=1).mean()
                         .shift(1).fillna(0.0).values)
        self.drift_per_bar = th.tensor(rolling_drift, dtype=th.float32, device=self.device)

        # Event feature columns (into full tech_ary for reward shaping)
        self._col_spring = 35
        self._col_upthrust = 36
        self._col_absorption = 39
        self._col_stopping = 41
        self._col_wave_dir = 15
        self._col_wave_vol_vs_prev = 21
        self._col_large_wave = 34
        self._col_phase_accum = 53
        self._col_phase_markup = 54
        self._col_phase_distrib = 55
        self._col_phase_markdown = 56

        # CiB thresholds
        self._cib_min_pb_vol = 0.3
        self._cib_max_pb_vol = 0.85
        self._cib_max_large_wave = 0.5

        # ElegantRL metadata
        self.env_name = "WyckoffTransformerVecEnv-v1"
        self.num_envs = num_envs
        self.max_step = self.close_price.shape[0] - 1
        self.state_dim = self.seq_len * self.n_bar_features + N_POSITION_FEATURES
        self.action_dim = N_ACTIONS
        self.if_discrete = True
        self.target_return = +np.inf

        # Sub-episode config
        min_start = self.seq_len  # need seq_len bars of history
        self._stagger = episode_len is not None and episode_len < self.max_step
        self._episode_len = episode_len if self._stagger else self.max_step
        self._min_start = min_start

        # Direction flip
        self.direction_flip = None

        # Per-env state (allocated in reset)
        self.day = None
        self.step_count = None
        self.pos_side = None
        self.pos_size = None
        self.entry_price = None
        self.bars_in_trade = None
        self.unrealized_pnl = None
        self.realized_pnl = None
        self.mfe = None
        self.mae = None
        self.vesting_amount = None
        self.vesting_remaining = None
        self.cumulative_returns = None
        self.total_trades = None

        print(
            f"[TransformerVecEnv] seq_len={self.seq_len} n_bar_features={self.n_bar_features} "
            f"state_dim={self.state_dim} reward_mode={self.reward_mode} "
            f"episode_len={episode_len} bars={len(close_ary)} num_envs={num_envs}",
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

        # Direction flip
        if self.sign_flip:
            self.direction_flip = th.where(
                th.rand(ne, device=dev) < 0.5,
                th.ones(ne, device=dev), -th.ones(ne, device=dev),
            )
        else:
            self.direction_flip = th.ones(ne, dtype=th.float32, device=dev)

        if self._stagger:
            max_start = max(self._min_start, self.max_step - self._episode_len)
            self.day = th.randint(self._min_start, max_start, (ne,), dtype=th.long, device=dev)
            self.step_count = th.zeros(ne, dtype=th.long, device=dev)
        else:
            self.day = th.full((ne,), self._min_start, dtype=th.long, device=dev)
            self.step_count = th.zeros(ne, dtype=th.long, device=dev)

        return self.get_state(), {}

    # ─── Observation ──────────────────────────────────────────────────────

    def get_state(self):
        """
        Return (num_envs, state_dim) tensor.

        First part: sliding window of selected features, flattened.
        Last 8: position features.

        The sliding window uses _padded_features where the first (seq_len-1)
        rows are zeros, so day=0 in the original data corresponds to index
        (seq_len-1) in padded, and the window [day:day+seq_len] in padded
        is exactly [day-seq_len+1:day+1] in original (zero-padded at start).
        """
        ne = self.num_envs
        sl = self.seq_len

        # Build window indices: for each env, gather [day_i .. day_i+seq_len) from padded
        # padded index = day (original) since we prepended (seq_len-1) zeros
        #   day=0 → padded[0:seq_len] = [zeros | bar0]
        #   day=5 → padded[5:5+seq_len] = [bars before day5 ... day5]
        offsets = th.arange(sl, device=self.device).unsqueeze(0)  # (1, seq_len)
        indices = self.day.unsqueeze(1) + offsets                  # (ne, seq_len)
        # Gather: (ne, seq_len, n_bar_features)
        windows = self._padded_features[indices]
        # Flatten: (ne, seq_len * n_bar_features)
        windows_flat = windows.reshape(ne, sl * self.n_bar_features)

        # Position features (8)
        curr_price = self.close_price[self.day]
        atr = self.bar_range

        entry_dist = th.where(
            self.pos_side != 0,
            ((curr_price - self.entry_price) / atr) * self.pos_side,
            th.zeros_like(self.pos_side),
        )

        obs_side = self.pos_side * self.direction_flip

        pos_feats = th.stack([
            obs_side,
            self.pos_size / max(self.max_position_size, 1.0),
            entry_dist,
            self.unrealized_pnl / self.pnl_norm,
            (self.realized_pnl / (2.0 * self.pnl_norm)).clamp(-1.0, 1.0),
            (self.bars_in_trade.float() / 50.0).clamp(max=1.0),
            self.mfe / self.pnl_norm,
            self.mae / self.pnl_norm,
        ], dim=1)  # (ne, 8)

        state = th.cat([windows_flat, pos_feats], dim=1)  # (ne, state_dim)
        return th.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

    # ─── Step ─────────────────────────────────────────────────────────────

    def step(self, action):
        if action.dim() == 2:
            action = action[:, 0]
        action = action.long()

        # Direction flip
        if self.sign_flip:
            flipped = self.direction_flip < 0
            is_el = action == ACTION_ENTER_LONG
            is_es = action == ACTION_ENTER_SHORT
            action = th.where(flipped & is_el, th.full_like(action, ACTION_ENTER_SHORT), action)
            action = th.where(flipped & is_es, th.full_like(action, ACTION_ENTER_LONG), action)

        prev_price = self.close_price[self.day]
        prev_side = self.pos_side.clone()
        prev_realized = self.realized_pnl.clone()
        prev_unrealized = self.unrealized_pnl.clone()

        # Execute actions
        penalty = self._execute_actions(action)

        post_side = self.pos_side.clone()
        post_size = self.pos_size.clone()

        # Advance bar
        self.day = th.clamp(self.day + 1, max=self.max_step)
        self.step_count += 1

        # Mark-to-market
        self._mark_to_market()

        # Reward computation
        curr_price = self.close_price[self.day]

        if self.reward_mode == 'dense_pnl':
            prev_equity = prev_realized + prev_unrealized
            curr_equity = self.realized_pnl + self.unrealized_pnl
            equity_change = curr_equity - prev_equity

            drift_pts = self.drift_per_bar[self.day]
            drift_dollars = (drift_pts / self.tick_size) * self.tick_value * post_side * post_size
            detrended_change = equity_change - drift_dollars
            per_contract = detrended_change / post_size.clamp(min=1.0)
            pnl_delta = per_contract / self.pnl_norm

            carry = self.carry_cost * post_size

            # Entry bonus with vesting
            raw_entry_bonus = self._compute_entry_bonus(action)
            was_flat = prev_side == 0
            raw_entry_bonus = th.where(was_flat, raw_entry_bonus, th.zeros_like(raw_entry_bonus))
            entry_bonus = self._vest_bonus(raw_entry_bonus)

            reward = (pnl_delta + entry_bonus - penalty - carry) * self.reward_scale
        else:
            exit_pnl = (self.realized_pnl - prev_realized) / self.pnl_norm

            raw_entry_bonus = self._compute_entry_bonus(action)
            was_flat = prev_side == 0
            raw_entry_bonus = th.where(was_flat, raw_entry_bonus, th.zeros_like(raw_entry_bonus))
            entry_bonus = self._vest_bonus(raw_entry_bonus)

            carry = self.carry_cost * post_size
            reward = (exit_pnl + entry_bonus - penalty - carry) * self.reward_scale

        if self.reward_clip > 0:
            reward = th.clamp(reward, -self.reward_clip, self.reward_clip)

        # Episode management
        done = (self.step_count >= self._episode_len) | (self.day >= self.max_step)
        self._flatten_done(done)

        if done.any():
            for i in th.where(done)[0].tolist():
                self.cumulative_returns[i] = self.realized_pnl[i].item()
            self._auto_reset(done)

        state = self.get_state()
        terminal = th.zeros(self.num_envs, dtype=th.bool, device=self.device)
        return state, reward, terminal, done, {}

    # ─── Vesting helper ───────────────────────────────────────────────────

    def _vest_bonus(self, raw_bonus):
        has_new = raw_bonus > 0
        if has_new.any():
            self.vesting_amount = th.where(has_new, raw_bonus, self.vesting_amount)
            self.vesting_remaining = th.where(
                has_new,
                th.full_like(self.vesting_remaining, self.vesting_bars),
                self.vesting_remaining,
            )
        still_vesting = self.vesting_remaining > 0
        # Spread bonus evenly over vesting period (not cliff at end)
        per_bar = self.vesting_amount / max(self.vesting_bars, 1)
        bonus = th.where(still_vesting, per_bar, th.zeros_like(self.vesting_amount))
        self.vesting_remaining = th.where(
            still_vesting, self.vesting_remaining - 1, self.vesting_remaining
        )
        vesting_done = still_vesting & (self.vesting_remaining == 0)
        self.vesting_amount = th.where(vesting_done, th.zeros_like(self.vesting_amount), self.vesting_amount)
        return bonus

    # ─── Action execution ─────────────────────────────────────────────────

    def _execute_actions(self, action):
        ne = self.num_envs
        dev = self.device
        penalty = th.zeros(ne, dtype=th.float32, device=dev)

        curr_price = self.close_price[self.day]
        buy_price = curr_price + self.slip
        sell_price = curr_price - self.slip

        is_flat = self.pos_side == 0
        is_positioned = ~is_flat
        is_long = self.pos_side > 0

        # ENTER_LONG
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

        # ENTER_SHORT
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

        # ADD
        add = action == ACTION_ADD
        can_add = is_positioned & (self.pos_size < self.max_position_size)
        valid_add = add & can_add
        invalid_add = add & (~can_add)

        add_price = th.where(is_long, buy_price, sell_price)
        old_size = self.pos_size
        new_size = old_size + 1.0
        new_avg = (self.entry_price * old_size + add_price) / new_size.clamp(min=1.0)
        self.entry_price = th.where(valid_add, new_avg, self.entry_price)
        self.pos_size = th.where(valid_add, new_size, self.pos_size)
        self.realized_pnl = th.where(valid_add, self.realized_pnl - self.commission, self.realized_pnl)
        self.total_trades = th.where(valid_add, self.total_trades + 1, self.total_trades)

        # REDUCE
        reduce = action == ACTION_REDUCE
        valid_reduce = reduce & is_positioned
        invalid_reduce = reduce & is_flat

        reduce_price = th.where(is_long, sell_price, buy_price)
        frac = 1.0 / self.pos_size.clamp(min=1.0)
        reduce_pnl = self._compute_pnl(reduce_price) * frac
        self.realized_pnl = th.where(
            valid_reduce, self.realized_pnl + reduce_pnl - self.commission, self.realized_pnl
        )
        new_size_r = (self.pos_size - 1.0).clamp(min=0.0)
        goes_flat = valid_reduce & (new_size_r <= 0)
        self.pos_size = th.where(valid_reduce, new_size_r, self.pos_size)
        self.pos_side = th.where(goes_flat, th.zeros_like(self.pos_side), self.pos_side)
        self.entry_price = th.where(goes_flat, th.zeros_like(self.entry_price), self.entry_price)
        self.vesting_amount = th.where(goes_flat, th.zeros_like(self.vesting_amount), self.vesting_amount)
        self.vesting_remaining = th.where(goes_flat, th.zeros_like(self.vesting_remaining), self.vesting_remaining)
        self.total_trades = th.where(valid_reduce, self.total_trades + 1, self.total_trades)

        # EXIT
        exit_ = action == ACTION_EXIT
        valid_exit = exit_ & is_positioned
        invalid_exit = exit_ & is_flat

        exit_price = th.where(is_long, sell_price, buy_price)
        exit_pnl = self._compute_pnl(exit_price)
        exit_comm = self.commission * self.pos_size
        self.realized_pnl = th.where(
            valid_exit, self.realized_pnl + exit_pnl - exit_comm, self.realized_pnl
        )
        self.pos_side = th.where(valid_exit, th.zeros_like(self.pos_side), self.pos_side)
        self.pos_size = th.where(valid_exit, th.zeros_like(self.pos_size), self.pos_size)
        self.entry_price = th.where(valid_exit, th.zeros_like(self.entry_price), self.entry_price)
        self.vesting_amount = th.where(valid_exit, th.zeros_like(self.vesting_amount), self.vesting_amount)
        self.vesting_remaining = th.where(valid_exit, th.zeros_like(self.vesting_remaining), self.vesting_remaining)
        self.total_trades = th.where(valid_exit, self.total_trades + 1, self.total_trades)

        # Penalties
        invalid = invalid_el | invalid_es | invalid_add | invalid_reduce | invalid_exit
        penalty = th.where(invalid, th.full_like(penalty, self.invalid_penalty_val), penalty)

        return penalty

    # ─── PnL helpers ──────────────────────────────────────────────────────

    def _compute_pnl(self, price):
        ticks = ((price - self.entry_price) / self.tick_size) * self.pos_side
        return ticks * self.tick_value * self.pos_size

    def _mark_to_market(self):
        positioned = self.pos_side != 0
        px = self.close_price[self.day]
        new_pnl = self._compute_pnl(px)
        self.unrealized_pnl = th.where(positioned, new_pnl, th.zeros_like(new_pnl))
        self.bars_in_trade = th.where(positioned, self.bars_in_trade + 1, self.bars_in_trade)
        self.mfe = th.where(positioned, th.max(self.mfe, new_pnl), self.mfe)
        self.mae = th.where(positioned, th.min(self.mae, new_pnl), self.mae)

    def _flatten_done(self, done):
        done_has_pos = done & (self.pos_side.abs() > 0)
        is_long = self.pos_side > 0
        exit_price = th.where(
            is_long,
            self.close_price[self.day] - self.slip,
            self.close_price[self.day] + self.slip,
        )
        exit_pnl = self._compute_pnl(exit_price)
        exit_comm = self.commission * self.pos_size
        self.realized_pnl = th.where(
            done_has_pos, self.realized_pnl + exit_pnl - exit_comm, self.realized_pnl
        )
        self.pos_side = th.where(done, th.zeros_like(self.pos_side), self.pos_side)
        self.pos_size = th.where(done, th.zeros_like(self.pos_size), self.pos_size)

    def _auto_reset(self, mask):
        n_reset = mask.sum().item()
        if self._stagger:
            max_start = max(self._min_start, self.max_step - self._episode_len)
            new_starts = th.randint(self._min_start, max_start, (n_reset,), dtype=th.long, device=self.device)
            self.day[mask] = new_starts
        else:
            self.day[mask] = self._min_start

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

        if self.sign_flip:
            n = mask.sum().item()
            self.direction_flip[mask] = th.where(
                th.rand(n, device=self.device) < 0.5,
                th.ones(n, device=self.device), -th.ones(n, device=self.device),
            )

    # ─── Entry bonus (reward shaping) ─────────────────────────────────────

    def _compute_entry_bonus(self, action):
        """Reward entering on valid Wyckoff signals."""
        feats = self.tech_factor[self.day]  # (ne, 72) full features
        ne = self.num_envs
        bonus = th.zeros(ne, dtype=th.float32, device=self.device)
        scale = self.entry_bonus_scale
        thr = self.event_threshold

        # Springs and upthrusts
        spring = feats[:, self._col_spring]
        upthrust = feats[:, self._col_upthrust]
        absorption = feats[:, self._col_absorption]
        stopping = feats[:, self._col_stopping]
        wave_dir = feats[:, self._col_wave_dir]

        # CiB features
        vol_ratio = feats[:, self._col_wave_vol_vs_prev]
        large_wave = feats[:, self._col_large_wave]
        cib_active = ((vol_ratio > self._cib_min_pb_vol)
                      & (vol_ratio < self._cib_max_pb_vol)
                      & (large_wave < self._cib_max_large_wave))

        is_el = action == ACTION_ENTER_LONG
        is_es = action == ACTION_ENTER_SHORT

        # Long entry bonuses
        long_bonus = th.zeros_like(bonus)
        long_bonus += th.where(spring > thr, th.full_like(bonus, scale), th.zeros_like(bonus))
        long_bonus += th.where(
            (absorption > thr) & (wave_dir > 0),
            th.full_like(bonus, scale * 0.6), th.zeros_like(bonus)
        )
        long_bonus += th.where(
            (stopping > thr) & (wave_dir > 0),
            th.full_like(bonus, scale * 0.4), th.zeros_like(bonus)
        )
        long_bonus += th.where(
            (wave_dir < 0) & cib_active,
            th.full_like(bonus, scale), th.zeros_like(bonus)
        )

        # Short entry bonuses
        short_bonus = th.zeros_like(bonus)
        short_bonus += th.where(upthrust > thr, th.full_like(bonus, scale), th.zeros_like(bonus))
        short_bonus += th.where(
            (absorption > thr) & (wave_dir < 0),
            th.full_like(bonus, scale * 0.6), th.zeros_like(bonus)
        )
        short_bonus += th.where(
            (stopping > thr) & (wave_dir < 0),
            th.full_like(bonus, scale * 0.4), th.zeros_like(bonus)
        )
        short_bonus += th.where(
            (wave_dir > 0) & cib_active,
            th.full_like(bonus, scale), th.zeros_like(bonus)
        )

        bonus = th.where(is_el, long_bonus.clamp(max=scale), bonus)
        bonus = th.where(is_es, short_bonus.clamp(max=scale), bonus)

        return bonus
