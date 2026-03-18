"""
Single-instrument Wyckoff Range-Bar Trading Environment.

Designed for NQ futures (or any single instrument) with Wyckoff features.
Unlike StockTradingEnv (multi-stock portfolio), this env trades ONE asset
with discrete position states: short (-1), flat (0), long (+1).

Data format (NPZ):
    close_ary : (n_bars, 1) — range-bar close prices
    tech_ary  : (n_bars, n_features) — Wyckoff feature matrix

Action space: continuous [-1, 1]
    Discretized to: -1 (short), 0 (flat), +1 (long)
    Thresholds at ±0.33

State: [position, unrealized_pnl_norm, cash_norm, *tech_features]

Reward: configurable via reward_mode parameter
    "pnl"       — realized + unrealized PnL change (default)
    "log_ret"   — log return of portfolio value
    "sharpe"    — differential Sharpe (Moody & Saffell 1998)
    "sortino"   — differential Sortino
"""

import os
import numpy as np
import torch as th
from typing import Tuple

ARY = np.ndarray

# Transaction cost for NQ futures: ~$4.50 round-trip per contract
# At NQ ~20000, 1 point = $20, so cost ≈ 0.01125 points per side
# We use a conservative 0.5 point per side as default (covers slippage)
DEFAULT_COST_PER_TRADE = 0.5  # points per side


class WyckoffTradingEnv:
    """
    Single-instrument trading env for Wyckoff range bars.

    Parameters
    ----------
    initial_amount : float
        Starting capital in points (e.g. 1000 NQ points).
    cost_per_trade : float
        Transaction cost per side in price points.
    gamma : float
        Discount factor for terminal reward shaping.
    reward_mode : str
        One of: "pnl", "log_ret", "sharpe", "sortino"
    reward_scale : float
        Multiplier applied to raw reward.
    beg_idx : int
        Start index into the data arrays.
    end_idx : int
        End index into the data arrays.
    npz_path : str
        Path to NPZ file with close_ary and tech_ary.
    """

    def __init__(
        self,
        initial_amount: float = 1000.0,
        cost_per_trade: float = DEFAULT_COST_PER_TRADE,
        gamma: float = 0.99,
        reward_mode: str = "pnl",
        reward_scale: float = 1.0,
        beg_idx: int = 0,
        end_idx: int = 0,
        npz_path: str = None,
        **kwargs,  # absorb extra kwargs from build_env
    ):
        self.npz_path = npz_path
        self.close_ary, self.tech_ary = self._load_data(npz_path)

        # Slice to requested range
        if end_idx <= 0:
            end_idx = len(self.close_ary)
        self.close_ary = self.close_ary[beg_idx:end_idx]
        self.tech_ary = self.tech_ary[beg_idx:end_idx]

        self.initial_amount = initial_amount
        self.cost_per_trade = cost_per_trade
        self.gamma = gamma
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale

        # Position tracking
        self.position = 0        # -1, 0, +1
        self.entry_price = 0.0
        self.cash = 0.0          # accumulated PnL in points
        self.day = 0
        self.rewards = []
        self.total_asset = 0.0
        self.cumulative_returns = 0.0
        self.if_random_reset = False

        # Differential Sharpe state
        self._A = 0.0  # EMA of returns
        self._B = 0.0  # EMA of squared returns
        self._eta = 0.005  # EMA decay for diff Sharpe

        # Env metadata (ElegantRL interface)
        n_features = self.tech_ary.shape[1]
        self.env_name = 'WyckoffTradingEnv-v1'
        # state = [position(1), unrealized_pnl(1), cash_norm(1), tech_features(n)]
        self.state_dim = 3 + n_features
        self.action_dim = 1
        self.if_discrete = False
        self.max_step = self.close_ary.shape[0] - 1
        self.target_return = +np.inf
        self.num_envs = 1

    def _load_data(self, npz_path: str) -> Tuple[ARY, ARY]:
        if npz_path is None or not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ not found: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        close_ary = data['close_ary'].astype(np.float32)
        tech_ary = data['tech_ary'].astype(np.float32)
        # Ensure close is 1D for single instrument
        if close_ary.ndim == 2:
            close_ary = close_ary[:, 0]
        return close_ary, tech_ary

    def reset(self) -> Tuple[ARY, dict]:
        self.day = 0
        self.position = 0
        self.entry_price = 0.0
        self.cash = 0.0
        self.rewards = []
        self.total_asset = self.initial_amount
        self.cumulative_returns = 0.0
        self._A = 0.0
        self._B = 0.0
        return self.get_state(), {}

    def get_state(self) -> ARY:
        price = self.close_ary[self.day]

        # Unrealized PnL normalized by initial capital
        if self.position != 0:
            unrealized = self.position * (price - self.entry_price)
        else:
            unrealized = 0.0

        state = np.concatenate([
            np.array([
                float(self.position),
                np.tanh(unrealized / self.initial_amount),
                np.tanh(self.cash / self.initial_amount),
            ], dtype=np.float32),
            self.tech_ary[self.day],
        ])
        return state

    def step(self, action) -> Tuple[ARY, float, bool, bool, dict]:
        # Decode action: continuous [-1, 1] → discrete {-1, 0, +1}
        if isinstance(action, np.ndarray):
            action = action.item() if action.size == 1 else action[0]
        if action > 0.33:
            target_pos = 1
        elif action < -0.33:
            target_pos = -1
        else:
            target_pos = 0

        prev_price = self.close_ary[self.day]
        self.day += 1
        curr_price = self.close_ary[self.day]

        # Execute position change
        old_position = self.position
        if target_pos != old_position:
            # Close existing position
            if old_position != 0:
                pnl = old_position * (curr_price - self.entry_price)
                self.cash += pnl - self.cost_per_trade
            # Open new position
            if target_pos != 0:
                self.entry_price = curr_price
                self.cash -= self.cost_per_trade
            else:
                self.entry_price = 0.0
            self.position = target_pos

        # Current portfolio value
        unrealized = 0.0
        if self.position != 0:
            unrealized = self.position * (curr_price - self.entry_price)
        new_total = self.initial_amount + self.cash + unrealized
        prev_total = self.total_asset

        # Compute reward
        reward = self._compute_reward(new_total, prev_total, curr_price, prev_price)
        self.rewards.append(reward)
        self.total_asset = new_total

        # Terminal
        terminal = self.day == self.max_step
        if terminal:
            # Force close any open position
            if self.position != 0:
                pnl = self.position * (curr_price - self.entry_price)
                self.cash += pnl - self.cost_per_trade
                self.position = 0
                self.entry_price = 0.0
                self.total_asset = self.initial_amount + self.cash

            reward += 1 / (1 - self.gamma) * np.mean(self.rewards)
            self.cumulative_returns = self.total_asset / self.initial_amount * 100

        return self.get_state(), float(reward * self.reward_scale), terminal, False, {}

    def _compute_reward(
        self, new_total: float, prev_total: float,
        curr_price: float, prev_price: float,
    ) -> float:
        if self.reward_mode == "pnl":
            # Normalized PnL change
            return (new_total - prev_total) / self.initial_amount

        elif self.reward_mode == "log_ret":
            if prev_total > 0 and new_total > 0:
                return np.log(new_total / prev_total)
            return 0.0

        elif self.reward_mode == "sharpe":
            # Differential Sharpe ratio (Moody & Saffell 1998)
            r = (new_total - prev_total) / max(prev_total, 1e-8)
            dA = r - self._A
            dB = r * r - self._B
            if self._B - self._A ** 2 > 1e-12:
                denom = (self._B - self._A ** 2) ** 1.5
                dsr = (self._B * dA - 0.5 * self._A * dB) / denom
            else:
                dsr = 0.0
            self._A += self._eta * dA
            self._B += self._eta * dB
            return dsr

        elif self.reward_mode == "sortino":
            r = (new_total - prev_total) / max(prev_total, 1e-8)
            dA = r - self._A
            # Only penalize downside
            down_r2 = r * r if r < 0 else 0.0
            dB = down_r2 - self._B
            if self._B > 1e-12:
                denom = self._B ** 1.5
                dsr = (self._B * dA - 0.5 * self._A * dB) / denom
            else:
                dsr = 0.0
            self._A += self._eta * dA
            self._B += self._eta * dB
            return dsr

        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")


# ═══════════════════════════════════════════════════════════════════════════
# GPU-Vectorized Version (for high-throughput training)
# ═══════════════════════════════════════════════════════════════════════════

class WyckoffTradingVecEnv:
    """
    GPU-vectorized Wyckoff trading env for parallel episode rollouts.

    All state is PyTorch tensors on GPU, num_envs episodes run in parallel
    via batched tensor operations with per-env day tracking.

    When episode_len is set (training mode):
      - Each env runs sub-episodes of episode_len bars
      - Starting positions are staggered randomly across the data
      - Per-env auto-reset on done for desynchronized terminal signals
      - This enables PPO to get frequent done signals within each horizon

    When episode_len is None (eval mode):
      - All envs walk through data from start to end (synchronized)
      - Compatible with ElegantRL evaluator expectations

    Position: -1 (short), 0 (flat), +1 (long)
    Action: continuous [-1, 1] → discretized via ±0.33 threshold
    State: [position, unrealized_pnl_norm, cash_norm, *tech_features]
    """

    def __init__(
        self,
        initial_amount: float = 1000.0,
        cost_per_trade: float = 0.5,
        gamma: float = 0.99,
        reward_mode: str = "pnl",
        reward_scale: float = 1.0,
        beg_idx: int = 0,
        end_idx: int = 0,
        num_envs: int = 256,
        gpu_id: int = 0,
        npz_path: str = None,
        episode_len: int = None,
        **kwargs,
    ):
        self.device = th.device(
            f"cuda:{gpu_id}" if (th.cuda.is_available() and gpu_id >= 0) else "cpu"
        )

        # Load data to GPU
        close_ary, tech_ary = self._load_data(npz_path)
        if end_idx <= 0:
            end_idx = len(close_ary)
        close_ary = close_ary[beg_idx:end_idx]
        tech_ary = tech_ary[beg_idx:end_idx]
        self.close_price = th.tensor(close_ary, dtype=th.float32, device=self.device)
        self.tech_factor = th.tensor(tech_ary, dtype=th.float32, device=self.device)

        self.initial_amount = initial_amount
        self.cost_per_trade = cost_per_trade
        self.gamma = gamma
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self.if_random_reset = True

        # Differential Sharpe EMA decay
        self._eta = 0.005

        # Env metadata
        n_features = self.tech_factor.shape[1]
        self.env_name = 'WyckoffTradingVecEnv-v1'
        self.num_envs = num_envs
        self.max_step = self.close_price.shape[0] - 1
        self.state_dim = 3 + n_features
        self.action_dim = 1
        self.if_discrete = False
        self.target_return = +np.inf

        # Sub-episode config: stagger when episode_len is shorter than full data
        self._stagger = episode_len is not None and episode_len < self.max_step
        self._episode_len = episode_len if self._stagger else self.max_step

        # vmap functions (following StockTradingVecEnv pattern from the paper)
        self.vmap_get_state = th.vmap(
            func=lambda pos, unreal, cash, techs: th.cat((pos, unreal, cash, techs)),
            in_dims=(0, 0, 0, 0), out_dims=0)

        # State tracking (set in reset)
        self.day = None           # (num_envs,) long — per-env position in data
        self.step_count = None    # (num_envs,) long — steps within current sub-episode
        self.position = None      # (num_envs,) int: -1, 0, +1
        self.entry_price = None   # (num_envs,)
        self.cash = None          # (num_envs,) accumulated PnL
        self.total_asset = None   # (num_envs,)
        self._A = None            # (num_envs,) EMA of returns
        self._B = None            # (num_envs,) EMA of squared returns
        self.reward_sum = None    # (num_envs,) per-episode reward accumulator
        self.reward_count = None  # (num_envs,) per-episode step counter
        self.cumulative_returns = None

    def _load_data(self, npz_path: str):
        if npz_path is None or not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ not found: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        close_ary = data['close_ary'].astype(np.float32)
        tech_ary = data['tech_ary'].astype(np.float32)
        if close_ary.ndim == 2:
            close_ary = close_ary[:, 0]
        return close_ary, tech_ary

    def reset(self):
        ne = self.num_envs
        dev = self.device

        self.position = th.zeros(ne, dtype=th.int32, device=dev)
        self.entry_price = th.zeros(ne, dtype=th.float32, device=dev)
        self.cash = th.zeros(ne, dtype=th.float32, device=dev)
        self.total_asset = th.full((ne,), self.initial_amount, dtype=th.float32, device=dev)
        self._A = th.zeros(ne, dtype=th.float32, device=dev)
        self._B = th.zeros(ne, dtype=th.float32, device=dev)
        self.reward_sum = th.zeros(ne, dtype=th.float32, device=dev)
        self.reward_count = th.zeros(ne, dtype=th.long, device=dev)
        self.cumulative_returns = [0.0] * ne

        if self._stagger:
            # Random starting positions, leave room for at least one episode
            max_start = max(1, self.max_step - self._episode_len)
            self.day = th.randint(0, max_start, (ne,), dtype=th.long, device=dev)
            # Random intra-episode offset → desynchronizes dones across envs
            offset = th.randint(0, self._episode_len, (ne,), dtype=th.long, device=dev)
            self.step_count = offset
            self.day = th.clamp(self.day + offset, max=self.max_step)
        else:
            self.day = th.zeros(ne, dtype=th.long, device=dev)
            self.step_count = th.zeros(ne, dtype=th.long, device=dev)

        if self.if_random_reset:
            rand_factor = th.rand(ne, dtype=th.float32, device=dev) * 0.1 + 0.95
            self.total_asset = self.total_asset * rand_factor
            self.cash = self.total_asset - self.initial_amount

        return self.get_state(), {}

    def get_state(self):
        """Return (num_envs, state_dim) tensor using vmap."""
        price = self.close_price[self.day]           # (num_envs,) per-env price
        pos_f = self.position.float().unsqueeze(1)   # (num_envs, 1)
        unrealized = (pos_f * (price - self.entry_price).unsqueeze(1))
        return self.vmap_get_state(
            pos_f,
            (unrealized / self.initial_amount).tanh(),
            (self.cash / self.initial_amount).tanh().unsqueeze(1),
            self.tech_factor[self.day])              # (num_envs, n_features)

    def step(self, action):
        """
        Branchless vectorized step with per-env day tracking and auto-reset.
        Uses th.where instead of if-branches to avoid GPU synchronization.
        """
        # Decode action → target position {-1, 0, +1}
        if action.dim() == 2:
            action = action[:, 0]
        target_pos = th.zeros_like(action, dtype=th.int32)
        target_pos[action > 0.33] = 1
        target_pos[action < -0.33] = -1

        prev_price = self.close_price[self.day]                 # (num_envs,)
        self.day = th.clamp(self.day + 1, max=self.max_step)
        self.step_count += 1
        curr_price = self.close_price[self.day]                 # (num_envs,)

        old_pos = self.position
        pos_changed = (target_pos != old_pos)
        pos_f = old_pos.float()

        # Branchless position close: PnL realized where position changed AND had position
        had_pos = pos_changed & (old_pos != 0)
        close_pnl = pos_f * (curr_price - self.entry_price) - self.cost_per_trade
        self.cash = th.where(had_pos, self.cash + close_pnl, self.cash)

        # Branchless position open: set entry price and deduct cost where opening new position
        opening = pos_changed & (target_pos != 0)
        self.entry_price = th.where(opening, curr_price, self.entry_price)
        self.cash = th.where(opening, self.cash - self.cost_per_trade, self.cash)

        # Branchless flat: clear entry price where going flat
        going_flat = pos_changed & (target_pos == 0)
        self.entry_price = th.where(going_flat, th.zeros_like(self.entry_price), self.entry_price)

        self.position = target_pos

        # Portfolio value (branchless)
        new_pos_f = target_pos.float()
        unrealized = new_pos_f * (curr_price - self.entry_price)
        unrealized = th.where(target_pos == 0, th.zeros_like(unrealized), unrealized)
        new_total = self.initial_amount + self.cash + unrealized
        prev_total = self.total_asset

        # Compute reward (vectorized)
        reward = self._compute_reward(new_total, prev_total, curr_price, prev_price)
        self.reward_sum += reward
        self.reward_count += 1
        self.total_asset = new_total

        # Per-env done: sub-episode finished OR hit end of data
        done = (self.step_count >= self._episode_len) | (self.day >= self.max_step)
        done_f = done.float()

        # Branchless terminal handling: force-close positions for done envs
        done_has_pos = done & (self.position != 0)
        term_pnl = self.position.float() * (curr_price - self.entry_price) - self.cost_per_trade
        self.cash = th.where(done_has_pos, self.cash + term_pnl, self.cash)
        self.position = th.where(done, th.zeros_like(self.position), self.position)
        self.entry_price = th.where(done, th.zeros_like(self.entry_price), self.entry_price)
        self.total_asset = th.where(done, self.initial_amount + self.cash, self.total_asset)

        # Terminal reward bonus: mean_reward / (1 - gamma)
        safe_count = th.clamp(self.reward_count.float(), min=1)
        mean_r = self.reward_sum / safe_count
        reward = reward + done_f * mean_r / (1.0 - self.gamma)

        # Save cumulative returns (list for evaluator compatibility)
        done_returns = (self.total_asset / self.initial_amount * 100)
        # Update only done envs — this is a small CPU-side update, runs rarely
        if done.any():
            for i in th.where(done)[0].tolist():
                self.cumulative_returns[i] = done_returns[i].item()
            self._auto_reset(done)

        state = self.get_state()
        truncate = th.zeros(self.num_envs, dtype=th.bool, device=self.device)
        return state, reward, done, truncate, {}

    def _auto_reset(self, mask):
        """Reset only the envs indicated by mask to new random starting positions."""
        n_reset = mask.sum().item()

        if self._stagger:
            max_start = max(1, self.max_step - self._episode_len)
            new_starts = th.randint(0, max_start, (n_reset,), dtype=th.long, device=self.device)
            self.day[mask] = new_starts
        else:
            self.day[mask] = 0

        self.step_count[mask] = 0
        self.position[mask] = 0
        self.entry_price[mask] = 0.0
        self.cash[mask] = 0.0
        self.total_asset[mask] = self.initial_amount
        self._A[mask] = 0.0
        self._B[mask] = 0.0
        self.reward_sum[mask] = 0.0
        self.reward_count[mask] = 0

    def _compute_reward(self, new_total, prev_total, curr_price, prev_price):
        """Vectorized reward computation. Returns (num_envs,) tensor."""
        if self.reward_mode == "pnl":
            return (new_total - prev_total) / self.initial_amount * self.reward_scale

        elif self.reward_mode == "log_ret":
            safe_prev = th.clamp(prev_total, min=1e-8)
            safe_new = th.clamp(new_total, min=1e-8)
            return th.log(safe_new / safe_prev) * self.reward_scale

        elif self.reward_mode == "sharpe":
            r = (new_total - prev_total) / th.clamp(prev_total, min=1e-8)
            dA = r - self._A
            dB = r * r - self._B
            var = self._B - self._A ** 2
            denom = th.clamp(var, min=1e-12) ** 1.5
            dsr = (self._B * dA - 0.5 * self._A * dB) / denom
            dsr[var < 1e-12] = 0.0
            self._A = self._A + self._eta * dA
            self._B = self._B + self._eta * dB
            return dsr * self.reward_scale

        elif self.reward_mode == "sortino":
            r = (new_total - prev_total) / th.clamp(prev_total, min=1e-8)
            dA = r - self._A
            down_r2 = th.where(r < 0, r * r, th.zeros_like(r))
            dB = down_r2 - self._B
            denom = th.clamp(self._B, min=1e-12) ** 1.5
            dsr = (self._B * dA - 0.5 * self._A * dB) / denom
            dsr[self._B < 1e-12] = 0.0
            self._A = self._A + self._eta * dA
            self._B = self._B + self._eta * dB
            return dsr * self.reward_scale

        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    def close(self):
        pass
