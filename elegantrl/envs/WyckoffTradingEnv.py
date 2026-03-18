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

    Mirrors StockTradingVecEnv pattern: all state is PyTorch tensors on GPU,
    num_envs episodes run in parallel via batched tensor operations.

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

        # State tracking (set in reset)
        self.day = None
        self.position = None      # (num_envs,) int: -1, 0, +1
        self.entry_price = None   # (num_envs,)
        self.cash = None          # (num_envs,) accumulated PnL
        self.total_asset = None   # (num_envs,)
        self._A = None            # (num_envs,) EMA of returns
        self._B = None            # (num_envs,) EMA of squared returns
        self.rewards = None
        self.cumulative_returns = None

        # Env metadata
        n_features = self.tech_factor.shape[1]
        self.env_name = 'WyckoffTradingVecEnv-v1'
        self.num_envs = num_envs
        self.max_step = self.close_price.shape[0] - 1
        self.state_dim = 3 + n_features
        self.action_dim = 1
        self.if_discrete = False
        self.target_return = +np.inf

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
        self.day = 0
        ne = self.num_envs
        dev = self.device

        self.position = th.zeros(ne, dtype=th.int32, device=dev)
        self.entry_price = th.zeros(ne, dtype=th.float32, device=dev)
        self.cash = th.zeros(ne, dtype=th.float32, device=dev)
        self.total_asset = th.full((ne,), self.initial_amount, dtype=th.float32, device=dev)
        self._A = th.zeros(ne, dtype=th.float32, device=dev)
        self._B = th.zeros(ne, dtype=th.float32, device=dev)
        self.rewards = []

        if self.if_random_reset:
            # Slight randomization of starting cash for diversity
            rand_factor = th.rand(ne, dtype=th.float32, device=dev) * 0.1 + 0.95
            self.total_asset = self.total_asset * rand_factor
            self.cash = self.total_asset - self.initial_amount

        return self.get_state(), {}

    def get_state(self):
        """Return (num_envs, state_dim) tensor."""
        price = self.close_price[self.day]  # scalar
        pos_f = self.position.float()

        # Unrealized PnL: position * (current_price - entry_price)
        unrealized = pos_f * (price - self.entry_price)

        # State: [position, tanh(unrealized_pnl_norm), tanh(cash_norm), tech_features...]
        state = th.zeros(self.num_envs, self.state_dim, dtype=th.float32, device=self.device)
        state[:, 0] = pos_f
        state[:, 1] = th.tanh(unrealized / self.initial_amount)
        state[:, 2] = th.tanh(self.cash / self.initial_amount)
        state[:, 3:] = self.tech_factor[self.day].unsqueeze(0).expand(self.num_envs, -1)
        return state

    def step(self, action):
        """
        Vectorized step: action shape (num_envs, 1) or (num_envs,).

        Returns: state (num_envs, state_dim), reward (num_envs,),
                 done (num_envs,), truncate (num_envs,), info dict
        """
        import torch as th

        # Decode action → target position {-1, 0, +1}
        if action.dim() == 2:
            action = action[:, 0]
        target_pos = th.zeros_like(action, dtype=th.int32)
        target_pos[action > 0.33] = 1
        target_pos[action < -0.33] = -1

        prev_price = self.close_price[self.day]
        self.day += 1
        curr_price = self.close_price[self.day]

        old_pos = self.position
        pos_changed = target_pos != old_pos
        pos_f = old_pos.float()

        # Close existing position where changed and had a position
        had_pos = pos_changed & (old_pos != 0)
        if had_pos.any():
            pnl = pos_f[had_pos] * (curr_price - self.entry_price[had_pos])
            self.cash[had_pos] += pnl - self.cost_per_trade

        # Open new position where changed and target != 0
        opening = pos_changed & (target_pos != 0)
        if opening.any():
            self.entry_price[opening] = curr_price
            self.cash[opening] -= self.cost_per_trade

        # Clear entry price for flat positions
        going_flat = pos_changed & (target_pos == 0)
        if going_flat.any():
            self.entry_price[going_flat] = 0.0

        self.position = target_pos

        # Portfolio value
        new_pos_f = target_pos.float()
        unrealized = new_pos_f * (curr_price - self.entry_price)
        # Zero unrealized for flat positions (entry_price=0 but be safe)
        unrealized[target_pos == 0] = 0.0
        new_total = self.initial_amount + self.cash + unrealized
        prev_total = self.total_asset

        # Compute reward (vectorized)
        reward = self._compute_reward(new_total, prev_total, curr_price, prev_price)

        self.rewards.append(reward)
        self.total_asset = new_total

        # Terminal
        done = self.day == self.max_step
        if done:
            # Force close all positions
            has_pos = self.position != 0
            if has_pos.any():
                pos_f = self.position[has_pos].float()
                pnl = pos_f * (curr_price - self.entry_price[has_pos])
                self.cash[has_pos] += pnl - self.cost_per_trade
                self.position[has_pos] = 0
                self.entry_price[has_pos] = 0.0
                self.total_asset = self.initial_amount + self.cash

            mean_reward = th.stack(self.rewards).mean(dim=0)
            reward = reward + mean_reward * (1.0 / (1.0 - self.gamma))
            self.cumulative_returns = (self.total_asset / self.initial_amount * 100).cpu().tolist()

        state = self.reset()[0] if done else self.get_state()
        done_t = th.tensor(done, dtype=th.bool, device=self.device).expand(self.num_envs)
        truncate_t = th.zeros(self.num_envs, dtype=th.bool, device=self.device)
        return state, reward, done_t, truncate_t, {}

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
