"""
Single-instrument Wyckoff Range-Bar Trading Environment.

Designed for NQ futures (or any single instrument) with Wyckoff features.
Unlike StockTradingEnv (multi-stock portfolio), this env trades ONE asset
with configurable position sizing.

Position sizing modes (controlled by continuous_sizing parameter):
    False (default) — Binary {-1, 0, +1} via ±0.33 thresholds
        Models a 1-contract prop trader. Realistic for NQ futures.
    True — Continuous [-1, +1] position sizing
        Models a multi-contract trader who sizes in/out based on conviction.

Data format (NPZ):
    close_ary : (n_bars, 1) — range-bar close prices
    tech_ary  : (n_bars, n_features) — Wyckoff feature matrix

Action space: continuous [-1, 1]
    Binary mode: discretized to {-1, 0, +1} via ±0.33 thresholds
    Continuous mode: maps directly to position size

PnL model: mark-to-market each bar
    step_pnl = position * (close[t] - close[t-1])
    cost: binary mode — flat cost_per_trade per position change
          continuous mode — cost proportional to |position_change|

State: [position, last_step_pnl_norm, cash_norm, *tech_features]

Reward: configurable via reward_mode parameter
    "pnl"       — mark-to-market PnL change (default)
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
        window_size: int = 1,
        feature_indices: list = None,
        continuous_sizing: bool = False,
        **kwargs,  # absorb extra kwargs from build_env
    ):
        self.npz_path = npz_path
        self.close_ary, self.tech_ary = self._load_data(npz_path)

        # Slice to requested range
        if end_idx <= 0:
            end_idx = len(self.close_ary)
        self.close_ary = self.close_ary[beg_idx:end_idx]
        self.tech_ary = self.tech_ary[beg_idx:end_idx]

        # Feature selection: keep only selected columns
        if feature_indices is not None:
            self.tech_ary = self.tech_ary[:, feature_indices]

        self.initial_amount = initial_amount
        self.cost_per_trade = cost_per_trade
        self.gamma = gamma
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale

        # Sliding window
        self.window_size = window_size

        # Position sizing mode
        self.continuous_sizing = continuous_sizing

        # Position tracking (continuous sizing)
        self.position = 0.0      # float in [-1, +1]
        self.cash = 0.0          # accumulated M2M PnL minus costs
        self.last_step_pnl = 0.0 # PnL from last bar (for state)
        self.day = 0
        self.rewards = []
        self.total_asset = 0.0
        self.cumulative_returns = 0.0
        self.if_random_reset = False

        # Trade tracking
        self.total_trades = 0
        self.total_turnover = 0.0  # sum of |position_change|

        # Differential Sharpe state
        self._A = 0.0  # EMA of returns
        self._B = 0.0  # EMA of squared returns
        self._eta = 0.005  # EMA decay for diff Sharpe

        # Env metadata (ElegantRL interface)
        n_features = self.tech_ary.shape[1]
        self.n_features = n_features
        self.env_name = 'WyckoffTradingEnv-v1'
        # state = [position(1), unrealized_pnl(1), cash_norm(1), window(W*F)]
        self.state_dim = 3 + window_size * n_features
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
        self.position = 0.0
        self.cash = 0.0
        self.last_step_pnl = 0.0
        self.rewards = []
        self.total_asset = self.initial_amount
        self.cumulative_returns = 0.0
        self.total_trades = 0
        self.total_turnover = 0.0
        self._A = 0.0
        self._B = 0.0
        return self.get_state(), {}

    def get_state(self) -> ARY:
        agent_state = np.array([
            self.position,
            np.tanh(self.last_step_pnl / self.initial_amount),
            np.tanh(self.cash / self.initial_amount),
        ], dtype=np.float32)

        # Sliding window: tech_ary[day-W+1 : day+1], zero-padded at start
        W = self.window_size
        if W <= 1:
            window_flat = self.tech_ary[self.day]
        else:
            start = self.day - W + 1
            if start >= 0:
                window_flat = self.tech_ary[start:self.day + 1].flatten()
            else:
                # Zero-pad the beginning
                pad_rows = -start
                valid = self.tech_ary[0:self.day + 1]  # (day+1, F)
                pad = np.zeros((pad_rows, self.n_features), dtype=np.float32)
                window_flat = np.concatenate([pad, valid], axis=0).flatten()

        return np.concatenate([agent_state, window_flat])

    def step(self, action) -> Tuple[ARY, float, bool, bool, dict]:
        if isinstance(action, np.ndarray):
            action = action.item() if action.size == 1 else action[0]

        # Decode action based on sizing mode
        if self.continuous_sizing:
            target_pos = float(np.clip(action, -1.0, 1.0))
        else:
            # Binary: discretize to {-1, 0, +1}
            if action > 0.33:
                target_pos = 1.0
            elif action < -0.33:
                target_pos = -1.0
            else:
                target_pos = 0.0

        prev_price = self.close_ary[self.day]
        self.day += 1
        curr_price = self.close_ary[self.day]

        # Mark-to-market: PnL from holding old position during this bar
        old_position = self.position
        step_pnl = old_position * (curr_price - prev_price)

        # Transaction cost
        pos_change = abs(target_pos - old_position)
        if self.continuous_sizing:
            cost = pos_change * self.cost_per_trade
        else:
            # Binary: flat cost per position change event
            cost = self.cost_per_trade if pos_change > 1e-6 else 0.0

        # Trade tracking
        if pos_change > 1e-6:
            self.total_trades += 1
            self.total_turnover += pos_change

        # Update accounting
        self.cash += step_pnl - cost
        self.position = target_pos
        self.last_step_pnl = step_pnl

        new_total = self.initial_amount + self.cash
        prev_total = self.total_asset

        # Compute reward
        reward = self._compute_reward(new_total, prev_total, curr_price, prev_price)
        self.rewards.append(reward)
        self.total_asset = new_total

        # Terminal
        terminal = self.day == self.max_step
        if terminal:
            # Charge cost to flatten at episode end
            if abs(self.position) > 1e-6:
                if self.continuous_sizing:
                    self.cash -= abs(self.position) * self.cost_per_trade
                else:
                    self.cash -= self.cost_per_trade
                self.total_trades += 1
                self.total_turnover += abs(self.position)
                self.position = 0.0
                self.total_asset = self.initial_amount + self.cash

            self.cumulative_returns = self.total_asset / self.initial_amount * 100

        return self.get_state(), float(reward * self.reward_scale), False, terminal, {}

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

    Continuous position sizing: action [-1, +1] maps directly to position.
    PnL is mark-to-market each bar: pnl = position * (close[t] - close[t-1]).
    Transaction cost is proportional to position change.

    When episode_len is set (training mode):
      - Each env runs sub-episodes of episode_len bars
      - Starting positions are staggered randomly across the data
      - Per-env auto-reset on done for desynchronized terminal signals

    When episode_len is None (eval mode):
      - All envs walk through data from start to end (synchronized)

    Position: float in [-1, +1] (continuous sizing)
    Action: continuous [-1, 1] → position size directly
    State: [position_size, last_step_pnl_norm, cash_norm, *tech_features]
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
        window_size: int = 1,
        feature_indices: list = None,
        continuous_sizing: bool = False,
        **kwargs,
    ):
        self.device = th.device(
            f"cuda:{gpu_id}" if (th.cuda.is_available() and gpu_id >= 0) else "cpu"
        )

        # Position sizing mode
        self.continuous_sizing = continuous_sizing

        # Load data to GPU
        close_ary, tech_ary = self._load_data(npz_path)
        if end_idx <= 0:
            end_idx = len(close_ary)
        close_ary = close_ary[beg_idx:end_idx]
        tech_ary = tech_ary[beg_idx:end_idx]

        # Feature selection: keep only selected columns
        if feature_indices is not None:
            tech_ary = tech_ary[:, feature_indices]

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

        # Sliding window
        self.window_size = window_size

        # Env metadata
        n_features = self.tech_factor.shape[1]
        self.n_features = n_features
        self.env_name = 'WyckoffTradingVecEnv-v1'
        self.num_envs = num_envs
        self.max_step = self.close_price.shape[0] - 1
        self.state_dim = 3 + window_size * n_features
        self.action_dim = 1
        self.if_discrete = False
        self.target_return = +np.inf

        # Sub-episode config: stagger when episode_len is shorter than full data
        self._stagger = episode_len is not None and episode_len < self.max_step
        self._episode_len = episode_len if self._stagger else self.max_step

        # Pre-pad tech_factor with zeros for sliding window (avoids runtime branching)
        # Shape: (window_size-1 + n_bars, n_features)
        if window_size > 1:
            pad = th.zeros(window_size - 1, n_features, dtype=th.float32, device=self.device)
            self._padded_tech = th.cat([pad, self.tech_factor], dim=0)  # (W-1+N, F)
        else:
            self._padded_tech = self.tech_factor

        # Trade-level reward: bonus at position close proportional to trade PnL
        self.trade_reward_weight = kwargs.get('trade_reward_weight', 0.5)
        self.entry_price = None   # (num_envs,) price when position was opened
        self.entry_cash = None    # (num_envs,) cash when position was opened

        # State tracking (set in reset)
        self.day = None           # (num_envs,) long — per-env position in data
        self.step_count = None    # (num_envs,) long — steps within current sub-episode
        self.position = None      # (num_envs,) float: continuous in [-1, +1]
        self.cash = None          # (num_envs,) accumulated M2M PnL minus costs
        self.last_step_pnl = None # (num_envs,) PnL from last bar (for state)
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

        self.position = th.zeros(ne, dtype=th.float32, device=dev)
        self.cash = th.zeros(ne, dtype=th.float32, device=dev)
        self.last_step_pnl = th.zeros(ne, dtype=th.float32, device=dev)
        self.total_asset = th.full((ne,), self.initial_amount, dtype=th.float32, device=dev)
        self._A = th.zeros(ne, dtype=th.float32, device=dev)
        self._B = th.zeros(ne, dtype=th.float32, device=dev)
        self.reward_sum = th.zeros(ne, dtype=th.float32, device=dev)
        self.reward_count = th.zeros(ne, dtype=th.long, device=dev)
        self.total_trades = th.zeros(ne, dtype=th.long, device=dev)
        self.total_turnover = th.zeros(ne, dtype=th.float32, device=dev)
        self.cumulative_returns = [0.0] * ne
        self.entry_price = th.zeros(ne, dtype=th.float32, device=dev)
        self.entry_cash = th.zeros(ne, dtype=th.float32, device=dev)

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
        """Return (num_envs, state_dim) tensor with sliding window."""
        agent_state = th.cat([
            self.position.unsqueeze(1),
            (self.last_step_pnl / self.initial_amount).tanh().unsqueeze(1),
            (self.cash / self.initial_amount).tanh().unsqueeze(1),
        ], dim=1)  # (num_envs, 3)

        W = self.window_size
        if W <= 1:
            # No window: just current bar features
            window_flat = self.tech_factor[self.day]  # (num_envs, F)
        else:
            # Sliding window via padded tech array (W-1 zero rows prepended)
            # day[i] in original data → day[i] + (W-1) in padded → window is [day[i]:day[i]+W]
            # Build index matrix: (num_envs, W) where each row is [d, d+1, ..., d+W-1]
            offsets = th.arange(W, device=self.device).unsqueeze(0)  # (1, W)
            indices = self.day.unsqueeze(1) + offsets                # (num_envs, W)
            # Gather from padded tech: (num_envs, W, F)
            window = self._padded_tech[indices]  # advanced indexing
            # Flatten to (num_envs, W*F)
            window_flat = window.reshape(self.num_envs, W * self.n_features)

        return th.cat([agent_state, window_flat], dim=1)  # (num_envs, 3 + W*F)

    def step(self, action):
        """
        Branchless vectorized step with per-env day tracking and auto-reset.
        Supports both binary {-1,0,+1} and continuous [-1,+1] position sizing.
        """
        # Decode action based on sizing mode
        if action.dim() == 2:
            action = action[:, 0]
        if self.continuous_sizing:
            target_pos = action.clamp(-1.0, 1.0)
        else:
            # Binary: discretize to {-1, 0, +1}
            target_pos = th.zeros_like(action)
            target_pos[action > 0.33] = 1.0
            target_pos[action < -0.33] = -1.0

        prev_price = self.close_price[self.day]                 # (num_envs,)
        self.day = th.clamp(self.day + 1, max=self.max_step)
        self.step_count += 1
        curr_price = self.close_price[self.day]                 # (num_envs,)

        # Mark-to-market: PnL from holding old position during this bar
        old_pos = self.position
        step_pnl = old_pos * (curr_price - prev_price)

        # Transaction cost
        pos_change = (target_pos - old_pos).abs()
        changed = pos_change > 1e-6
        if self.continuous_sizing:
            cost = pos_change * self.cost_per_trade
        else:
            # Binary: flat cost per position change event
            cost = th.where(changed, th.full_like(pos_change, self.cost_per_trade),
                           th.zeros_like(pos_change))

        # Trade tracking (accumulated per-env, reset on auto_reset)
        self.total_trades += changed.long()
        self.total_turnover += pos_change

        # Update accounting
        self.cash = self.cash + step_pnl - cost
        self.position = target_pos
        self.last_step_pnl = step_pnl

        new_total = self.initial_amount + self.cash
        prev_total = self.total_asset

        # Compute bar-level reward (vectorized)
        reward = self._compute_reward(new_total, prev_total, curr_price, prev_price)

        # Trade-level reward bonus: concentrated signal at position changes
        if self.trade_reward_weight > 0:
            reward = reward + self._trade_reward_bonus(old_pos, target_pos, curr_price)

        self.reward_sum += reward
        self.reward_count += 1
        self.total_asset = new_total

        # Per-env done: sub-episode finished OR hit end of data
        done = (self.step_count >= self._episode_len) | (self.day >= self.max_step)

        # Terminal: charge cost to flatten for done envs
        done_has_pos = done & (self.position.abs() > 1e-6)
        if self.continuous_sizing:
            flatten_cost = self.position.abs() * self.cost_per_trade
        else:
            flatten_cost = th.where(done_has_pos,
                                    th.full_like(self.position, self.cost_per_trade),
                                    th.zeros_like(self.position))
        self.cash = th.where(done_has_pos, self.cash - flatten_cost, self.cash)
        self.position = th.where(done, th.zeros_like(self.position), self.position)
        self.total_asset = th.where(done, self.initial_amount + self.cash, self.total_asset)

        # Save cumulative returns (list for evaluator compatibility)
        done_returns = (self.total_asset / self.initial_amount * 100)
        # Update only done envs — this is a small CPU-side update, runs rarely
        if done.any():
            for i in th.where(done)[0].tolist():
                self.cumulative_returns[i] = done_returns[i].item()
            self._auto_reset(done)

        state = self.get_state()
        # All episode endings are time-limited (not natural MDP termination).
        # Return as truncation so GAE bootstraps V(s) instead of assuming V=0.
        terminal = th.zeros(self.num_envs, dtype=th.bool, device=self.device)
        return state, reward, terminal, done, {}

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
        self.position[mask] = 0.0
        self.cash[mask] = 0.0
        self.last_step_pnl[mask] = 0.0
        self.total_asset[mask] = self.initial_amount
        self._A[mask] = 0.0
        self._B[mask] = 0.0
        self.reward_sum[mask] = 0.0
        self.reward_count[mask] = 0
        self.total_trades[mask] = 0
        self.total_turnover[mask] = 0.0
        self.entry_price[mask] = 0.0
        self.entry_cash[mask] = 0.0

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

    def _trade_reward_bonus(self, old_pos, new_pos, curr_price):
        """
        Concentrated trade-level reward at position changes.

        Fires when:
        - Position closes (full or partial): delivers realized PnL as bonus
        - Position opens: records entry price for later PnL computation

        This gives the agent a clear credit assignment signal at the entry/exit
        decision point, rather than diffusing it across bar-level M2M PnL.
        """
        bonus = th.zeros_like(old_pos)

        # Detect position closing (reducing or flipping)
        # closing_fraction: how much of the old position was closed
        # e.g. old=1.0, new=0.0 → closed 1.0; old=1.0, new=-1.0 → closed 1.0
        was_positioned = old_pos.abs() > 1e-6
        closed_portion = th.clamp(old_pos.abs() - new_pos * old_pos.sign(), min=0.0)
        closed_portion = th.clamp(closed_portion, max=old_pos.abs())  # can't close more than held
        is_closing = was_positioned & (closed_portion > 1e-6)

        if is_closing.any():
            # Realized PnL on the closed portion
            trade_pnl = old_pos.sign() * (curr_price - self.entry_price) * closed_portion
            trade_ret = trade_pnl / self.initial_amount
            bonus[is_closing] = trade_ret[is_closing] * self.reward_scale * self.trade_reward_weight

        # Detect position opening (was flat or adding)
        is_opening = (old_pos.abs() < 1e-6) & (new_pos.abs() > 1e-6)
        if is_opening.any():
            self.entry_price[is_opening] = curr_price[is_opening]
            self.entry_cash[is_opening] = self.cash[is_opening]

        # Detect position flip (close + open in one step)
        is_flipping = was_positioned & (new_pos * old_pos < -1e-6)
        if is_flipping.any():
            self.entry_price[is_flipping] = curr_price[is_flipping]
            self.entry_cash[is_flipping] = self.cash[is_flipping]

        return bonus

    def close(self):
        pass
