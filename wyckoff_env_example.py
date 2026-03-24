import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass


ACTION_HOLD = 0
ACTION_ENTER_LONG = 1
ACTION_ENTER_SHORT = 2
ACTION_ADD = 3
ACTION_REDUCE = 4
ACTION_EXIT = 5


@dataclass
class PositionState:
    side: int = 0                  # -1 short, 0 flat, +1 long
    size: float = 0.0
    entry_price: float = 0.0
    bars_in_trade: int = 0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0


@dataclass
class WaveState:
    direction: int = 0
    start_price: float = 0.0
    end_price: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0
    delta: float = 0.0
    bars: int = 0


class NQWyckoffWeisEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data: dict,
        env_name: str = "NQWyckoffWeisEnv",
        num_envs: int = 1,
        max_step: int = 512,
        state_dim: int | None = None,
        action_dim: int = 6,
        if_discrete: bool = True,
        commission: float = 1.50,
        slippage_ticks: float = 1.0,
        tick_size: float = 0.25,
        tick_value: float = 5.0,
        max_position_size: int = 2,
        wave_reversal_ticks: int = 8,
        range_lookback: int = 80,
        reward_scale: float = 1.0,
        random_start: bool = True,
        session_indices: np.ndarray | None = None,
        **kwargs,
    ):
        super().__init__()

        # -------- ElegantRL-required metadata --------
        self.env_name = env_name
        self.num_envs = num_envs          # keep 1 here; ElegantRL VecEnv handles batching
        self.max_step = max_step
        self.action_dim = action_dim
        self.if_discrete = if_discrete

        # -------- market data --------
        self.data = data
        self.n_rows = len(self.data["close"])
        self.session_indices = session_indices
        self.random_start = random_start

        # -------- trading config --------
        self.commission = commission
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.max_position_size = max_position_size
        self.wave_reversal_ticks = wave_reversal_ticks
        self.range_lookback = range_lookback
        self.reward_scale = reward_scale

        # -------- runtime state --------
        self.position = PositionState()
        self.curr_wave = WaveState()
        self.prev_wave = WaveState()
        self.t = 0
        self.start_idx = 0
        self.end_idx = 0
        self.episode_step = 0

        # Build one sample observation to infer state_dim if needed
        tmp_obs_dim = 29  # update if you change _get_observation()
        self.state_dim = tmp_obs_dim if state_dim is None else state_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.action_dim)

    # ------------------------------------------------------------------
    # Reset / Step: keep this API exactly clean for ElegantRL wrapping
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.position = PositionState()
        self.curr_wave = WaveState()
        self.prev_wave = WaveState()
        self.episode_step = 0

        self.start_idx = self._sample_start_index()
        self.end_idx = min(self.start_idx + self.max_step, self.n_rows - 1)
        self.t = self.start_idx

        self._update_wave_state(self.t)
        self._update_position_mark_to_market(self.t)

        obs = self._get_observation(self.t)
        info = {
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
        }
        return obs, info

    def step(self, action):
        action = int(action)

        equity_before = self._equity(self.t)
        invalid_penalty = self._execute_action(action)

        self.t += 1
        self.episode_step += 1

        self._update_wave_state(self.t)
        self._update_position_mark_to_market(self.t)
        event_flags = self._compute_event_flags(self.t)

        equity_after = self._equity(self.t)
        pnl_component = (equity_after - equity_before) / (10.0 * self.tick_value)

        entry_bonus = self._entry_bonus(action, event_flags)
        overtrade_penalty = self._overtrade_penalty(action)

        reward = (
            pnl_component
            + entry_bonus
            - invalid_penalty
            - overtrade_penalty
        ) * self.reward_scale

        terminated = False
        truncated = (self.t >= self.end_idx) or (self.episode_step >= self.max_step)

        obs = self._get_observation(self.t)
        info = {
            "equity": equity_after,
            "realized_pnl": self.position.realized_pnl,
            "unrealized_pnl": self.position.unrealized_pnl,
            "position_side": self.position.side,
            "position_size": self.position.size,
            "event_flags": event_flags,
        }
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------
    def _get_observation(self, t: int) -> np.ndarray:
        atr_like = self._atr_like(t)

        o = float(self.data["open"][t])
        h = float(self.data["high"][t])
        l = float(self.data["low"][t])
        c = float(self.data["close"][t])
        v = float(self.data["volume"][t])

        c1 = float(self.data["close"][max(0, t - 1)])
        c3 = float(self.data["close"][max(0, t - 3)])
        c8 = float(self.data["close"][max(0, t - 8)])

        lo = max(0, t - self.range_lookback + 1)
        local_high = float(np.max(self.data["high"][lo:t + 1]))
        local_low = float(np.min(self.data["low"][lo:t + 1]))
        range_width = max(local_high - local_low, 1e-6)
        range_pos = 2.0 * ((c - local_low) / range_width) - 1.0

        curr_move = self.curr_wave.end_price - self.curr_wave.start_price
        prev_move = self.prev_wave.end_price - self.prev_wave.start_price
        curr_vol = self.curr_wave.volume
        prev_vol = self.prev_wave.volume

        flags = self._compute_event_flags(t)

        if self.position.side == 0:
            entry_dist = 0.0
        else:
            entry_dist = ((c - self.position.entry_price) / atr_like) * self.position.side

        obs = np.array([
            # bar features
            (c - c1) / atr_like,
            (c - c3) / atr_like,
            (c - c8) / atr_like,
            (h - l) / atr_like,
            (c - o) / atr_like,
            (h - max(o, c)) / atr_like,
            (min(o, c) - l) / atr_like,
            np.log1p(v),

            # wave features
            float(self.curr_wave.direction),
            curr_move / atr_like,
            np.log1p(curr_vol),
            min(self.curr_wave.bars / 20.0, 3.0),
            curr_vol / (abs(curr_move) + 1e-6),
            curr_move / (abs(prev_move) + 1e-6),
            curr_vol / (prev_vol + 1e-6),

            # context
            range_pos,
            (c - local_low) / atr_like,
            (local_high - c) / atr_like,
            self._trend_slope(t, 20) / atr_like,
            *self._vol_regime_one_hot(t),
            flags["spring_long"],
            flags["upthrust_short"],
            flags["lps_long"],
            flags["lps_short"],

            # position state
            float(self.position.side),
            self.position.size / max(1.0, self.max_position_size),
            entry_dist,
            self.position.unrealized_pnl / (10.0 * self.tick_value),
            self.position.realized_pnl / (20.0 * self.tick_value),
            min(self.position.bars_in_trade / 50.0, 1.0),
            self.position.mfe / (10.0 * self.tick_value),
            self.position.mae / (10.0 * self.tick_value),
        ], dtype=np.float32)

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    # ------------------------------------------------------------------
    # Trading logic
    # ------------------------------------------------------------------
    def _execute_action(self, action: int) -> float:
        penalty = 0.0
        px = self._execution_price(self.t, action)

        if action == ACTION_HOLD:
            return 0.0

        if action == ACTION_ENTER_LONG:
            if self.position.side == 0:
                self._open_position(side=1, size=1.0, price=px)
            else:
                penalty += 0.02

        elif action == ACTION_ENTER_SHORT:
            if self.position.side == 0:
                self._open_position(side=-1, size=1.0, price=px)
            else:
                penalty += 0.02

        elif action == ACTION_ADD:
            if self.position.side != 0 and self.position.size < self.max_position_size:
                self._add_position(px, 1.0)
            else:
                penalty += 0.02

        elif action == ACTION_REDUCE:
            if self.position.side != 0:
                self._reduce_position(px, 1.0)
            else:
                penalty += 0.02

        elif action == ACTION_EXIT:
            if self.position.side != 0:
                self._close_position(px)
            else:
                penalty += 0.02

        else:
            penalty += 0.05

        return penalty

    def _open_position(self, side: int, size: float, price: float):
        self.position.side = side
        self.position.size = size
        self.position.entry_price = price
        self.position.bars_in_trade = 0
        self.position.unrealized_pnl = 0.0
        self.position.mfe = 0.0
        self.position.mae = 0.0
        self.position.realized_pnl -= self.commission

    def _add_position(self, price: float, add_size: float):
        new_size = self.position.size + add_size
        avg_price = (
            self.position.entry_price * self.position.size + price * add_size
        ) / new_size
        self.position.entry_price = avg_price
        self.position.size = new_size
        self.position.realized_pnl -= self.commission

    def _reduce_position(self, price: float, reduce_size: float):
        reduce_size = min(reduce_size, self.position.size)
        pnl = self._position_pnl(price) * (reduce_size / self.position.size)
        self.position.realized_pnl += pnl - self.commission
        self.position.size -= reduce_size

        if self.position.size <= 0:
            carry_realized = self.position.realized_pnl
            self.position = PositionState(realized_pnl=carry_realized)

    def _close_position(self, price: float):
        pnl = self._position_pnl(price)
        carry_realized = self.position.realized_pnl + pnl - self.commission
        self.position = PositionState(realized_pnl=carry_realized)

    def _position_pnl(self, price: float) -> float:
        ticks = ((price - self.position.entry_price) / self.tick_size) * self.position.side
        return ticks * self.tick_value * self.position.size

    def _update_position_mark_to_market(self, t: int):
        if self.position.side == 0:
            self.position.unrealized_pnl = 0.0
            return

        px = float(self.data["close"][t])
        self.position.unrealized_pnl = self._position_pnl(px)
        self.position.bars_in_trade += 1
        self.position.mfe = max(self.position.mfe, self.position.unrealized_pnl)
        self.position.mae = min(self.position.mae, self.position.unrealized_pnl)

    def _equity(self, t: int) -> float:
        return self.position.realized_pnl + self.position.unrealized_pnl

    def _execution_price(self, t: int, action: int) -> float:
        c = float(self.data["close"][t])
        slip = self.slippage_ticks * self.tick_size

        if action in (ACTION_ENTER_LONG, ACTION_ADD):
            return c + slip
        if action in (ACTION_ENTER_SHORT,):
            return c - slip
        if self.position.side > 0:
            return c - slip
        if self.position.side < 0:
            return c + slip
        return c

    # ------------------------------------------------------------------
    # Wave / Wyckoff logic
    # ------------------------------------------------------------------
    def _update_wave_state(self, t: int):
        close_px = float(self.data["close"][t])
        high_px = float(self.data["high"][t])
        low_px = float(self.data["low"][t])
        vol = float(self.data["volume"][t])

        if self.curr_wave.bars == 0:
            self.curr_wave = WaveState(
                direction=0,
                start_price=close_px,
                end_price=close_px,
                high=high_px,
                low=low_px,
                volume=vol,
                delta=0.0,
                bars=1,
            )
            return

        prev_end = self.curr_wave.end_price
        move = close_px - prev_end
        reversal_amt = self.wave_reversal_ticks * self.tick_size

        if self.curr_wave.direction == 0:
            if move > 0:
                self.curr_wave.direction = 1
            elif move < 0:
                self.curr_wave.direction = -1

        continue_wave = (
            self.curr_wave.direction == 0
            or np.sign(move) == self.curr_wave.direction
            or abs(close_px - prev_end) < reversal_amt
        )

        if continue_wave:
            self.curr_wave.end_price = close_px
            self.curr_wave.high = max(self.curr_wave.high, high_px)
            self.curr_wave.low = min(self.curr_wave.low, low_px)
            self.curr_wave.volume += vol
            self.curr_wave.bars += 1
        else:
            self.prev_wave = self.curr_wave
            self.curr_wave = WaveState(
                direction=1 if move > 0 else -1,
                start_price=prev_end,
                end_price=close_px,
                high=high_px,
                low=low_px,
                volume=vol,
                delta=0.0,
                bars=1,
            )

    def _compute_event_flags(self, t: int) -> dict:
        lo = max(1, t - self.range_lookback + 1)
        local_high_prev = float(np.max(self.data["high"][lo:t]))
        local_low_prev = float(np.min(self.data["low"][lo:t]))

        h = float(self.data["high"][t])
        l = float(self.data["low"][t])
        c = float(self.data["close"][t])

        local_high = float(np.max(self.data["high"][lo:t + 1]))
        local_low = float(np.min(self.data["low"][lo:t + 1]))
        range_width = max(local_high - local_low, 1e-6)
        range_pos = 2.0 * ((c - local_low) / range_width) - 1.0
        trend_slope = self._trend_slope(t, 20)

        spring_long = int(
            l < local_low_prev and c > local_low_prev and self.curr_wave.direction == 1
        )
        upthrust_short = int(
            h > local_high_prev and c < local_high_prev and self.curr_wave.direction == -1
        )
        lps_long = int(
            trend_slope > 0 and range_pos > -0.3 and self.curr_wave.volume < max(self.prev_wave.volume, 1e-6)
        )
        lps_short = int(
            trend_slope < 0 and range_pos < 0.3 and self.curr_wave.volume < max(self.prev_wave.volume, 1e-6)
        )

        return {
            "spring_long": spring_long,
            "upthrust_short": upthrust_short,
            "lps_long": lps_long,
            "lps_short": lps_short,
        }

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------
    def _entry_bonus(self, action: int, flags: dict) -> float:
        bonus = 0.0
        if action == ACTION_ENTER_LONG and flags["spring_long"]:
            bonus += 0.25
        if action == ACTION_ENTER_LONG and flags["lps_long"]:
            bonus += 0.15
        if action == ACTION_ENTER_SHORT and flags["upthrust_short"]:
            bonus += 0.25
        if action == ACTION_ENTER_SHORT and flags["lps_short"]:
            bonus += 0.15
        return bonus

    def _overtrade_penalty(self, action: int) -> float:
        if action != ACTION_HOLD and self.episode_step < 3:
            return 0.01
        return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_start_index(self) -> int:
        min_idx = max(self.range_lookback + 8, 32)
        max_idx = self.n_rows - self.max_step - 2
        if not self.random_start or max_idx <= min_idx:
            return min_idx
        return np.random.randint(min_idx, max_idx)

    def _atr_like(self, t: int, lookback: int = 20) -> float:
        lo = max(1, t - lookback + 1)
        highs = self.data["high"][lo:t + 1]
        lows = self.data["low"][lo:t + 1]
        prev_closes = self.data["close"][lo - 1:t]
        tr = np.maximum(
            highs - lows,
            np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes))
        )
        return float(np.mean(tr) + 1e-6)

    def _trend_slope(self, t: int, lookback: int = 20) -> float:
        lo = max(0, t - lookback + 1)
        y = self.data["close"][lo:t + 1].astype(np.float64)
        x = np.arange(len(y), dtype=np.float64)
        if len(y) < 2:
            return 0.0
        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2) + 1e-12
        slope = np.sum((x - x_mean) * (y - y_mean)) / denom
        return float(slope)

    def _vol_regime_one_hot(self, t: int) -> tuple[float, float, float]:
        lo = max(5, t - 50)
        rets = np.diff(self.data["close"][lo:t + 1])
        rv = float(np.std(rets)) if len(rets) > 1 else 0.0
        if rv < 3.0:
            return (1.0, 0.0, 0.0)
        if rv < 6.0:
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)