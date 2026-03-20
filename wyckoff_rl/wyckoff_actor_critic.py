"""
Wyckoff CNN Actor / Critic for temporal pattern recognition.

These replace the standard MLP ActorPPO / CriticPPO with a 1D CNN
that processes a sliding window of Wyckoff features before the MLP head.

State layout (flat): [agent_state(3), window(W*F)]
  - agent_state: [position, unrealized_pnl_norm, cash_norm]
  - window: W bars × F features, flattened in row-major order

The CNN reads the (W, F) window as a 1D sequence with F channels,
learns temporal filters (e.g. "spring followed by declining volume"),
then the MLP head combines the CNN embedding with agent state.

Interface matches ActorPPO / CriticPPO exactly so ElegantRL's
AgentPPO can use them as drop-in replacements.
"""

import torch as th
from torch import nn

TEN = th.Tensor


class TemporalEncoder(nn.Module):
    """1D CNN encoder for (batch, W, F) feature windows."""

    def __init__(self, n_features: int, window_size: int, embed_dim: int = 64):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size

        # 1D conv: treat features as channels, time as the spatial dimension
        # Input: (batch, F, W) → Output: (batch, embed_dim, ?)
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # Global average pool → fixed-size (batch, 32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Project to embed_dim
        self.proj = nn.Linear(32, embed_dim) if embed_dim != 32 else nn.Identity()
        self.embed_dim = embed_dim

    def forward(self, window: TEN) -> TEN:
        """
        Parameters
        ----------
        window : (batch, W, F) tensor

        Returns
        -------
        (batch, embed_dim) tensor
        """
        # Conv1d expects (batch, channels, length) = (batch, F, W)
        x = window.transpose(1, 2)             # (batch, F, W)
        x = self.conv(x)                       # (batch, 32, W)
        x = self.pool(x).squeeze(-1)           # (batch, 32)
        return self.proj(x)                    # (batch, embed_dim)


class ActorPPO_Wyckoff(nn.Module):
    """PPO Actor with 1D CNN temporal encoder for Wyckoff sliding window."""

    def __init__(self, net_dims: list[int], state_dim: int, action_dim: int,
                 n_features: int = 0, window_size: int = 1):
        super().__init__()

        # Infer n_features and window_size from state_dim if not provided
        agent_state_dim = 3
        if n_features > 0 and window_size > 1:
            self.n_features = n_features
            self.window_size = window_size
        else:
            # Fallback: assume flat state (backward compat with window_size=1)
            self.n_features = state_dim - agent_state_dim
            self.window_size = 1

        self.agent_state_dim = agent_state_dim

        if self.window_size > 1:
            embed_dim = 64
            self.encoder = TemporalEncoder(self.n_features, self.window_size, embed_dim)
            mlp_input_dim = agent_state_dim + embed_dim
        else:
            # No window: standard MLP (backward compatible)
            self.encoder = None
            mlp_input_dim = state_dim

        # MLP head: [mlp_input_dim, *net_dims, action_dim]
        dims = [mlp_input_dim, *net_dims, action_dim]
        net_list = []
        for i in range(len(dims) - 1):
            net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.GELU()])
        del net_list[-1]  # remove last activation → raw output
        self.net = nn.Sequential(*net_list)
        # Orthogonal init on output layer (PPO convention)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

        self.action_std_log = nn.Parameter(th.zeros((1, action_dim)), requires_grad=True)
        self.ActionDist = th.distributions.normal.Normal

        # State normalization parameters (updated by ElegantRL's running stats)
        self.state_avg = nn.Parameter(th.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(th.ones((state_dim,)), requires_grad=False)

    def state_norm(self, state: TEN) -> TEN:
        return (state - self.state_avg) / (self.state_std + 1e-4)

    def _encode(self, state: TEN) -> TEN:
        """Normalize state, split into agent_state + window, encode → MLP input.

        Handles both 2D (batch, state_dim) and 3D (horizon, num_envs, state_dim)
        inputs.  The latter occurs in AgentPPO.update_net when the critic is
        called on slices of the 3D replay buffer.
        """
        leading = state.shape[:-1]  # () for 1D, (B,) for 2D, (H, E) for 3D
        if state.dim() >= 3:
            state = state.reshape(-1, state.shape[-1])  # flatten to (B*, state_dim)

        state = self.state_norm(state)
        if self.encoder is not None:
            agent_state = state[:, :self.agent_state_dim]  # (B*, 3)
            window_flat = state[:, self.agent_state_dim:]  # (B*, W*F)
            window = window_flat.reshape(-1, self.window_size, self.n_features)
            embedding = self.encoder(window)
            mlp_input = th.cat([agent_state, embedding], dim=1)
        else:
            mlp_input = state

        if len(leading) >= 2:  # restore 3-D shape for downstream nn.Linear
            mlp_input = mlp_input.reshape(*leading, -1)
        return mlp_input

    def forward(self, state: TEN) -> TEN:
        mlp_input = self._encode(state)
        action = self.net(mlp_input)
        return self.convert_action_for_env(action)

    def get_action(self, state: TEN) -> tuple[TEN, TEN]:
        mlp_input = self._encode(state)
        action_avg = self.net(mlp_input)
        action_std = self.action_std_log.exp()
        dist = self.ActionDist(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        return action, logprob

    def get_logprob_entropy(self, state: TEN, action: TEN) -> tuple[TEN, TEN]:
        mlp_input = self._encode(state)
        action_avg = self.net(mlp_input)
        action_std = self.action_std_log.exp()
        dist = self.ActionDist(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: TEN) -> TEN:
        return action.tanh()


class CriticPPO_Wyckoff(nn.Module):
    """PPO Critic with 1D CNN temporal encoder for Wyckoff sliding window."""

    def __init__(self, net_dims: list[int], state_dim: int, action_dim: int,
                 n_features: int = 0, window_size: int = 1):
        super().__init__()
        assert isinstance(action_dim, int)

        agent_state_dim = 3
        if n_features > 0 and window_size > 1:
            self.n_features = n_features
            self.window_size = window_size
        else:
            self.n_features = state_dim - agent_state_dim
            self.window_size = 1

        self.agent_state_dim = agent_state_dim

        if self.window_size > 1:
            embed_dim = 64
            self.encoder = TemporalEncoder(self.n_features, self.window_size, embed_dim)
            mlp_input_dim = agent_state_dim + embed_dim
        else:
            self.encoder = None
            mlp_input_dim = state_dim

        # MLP head: [mlp_input_dim, *net_dims, 1]
        dims = [mlp_input_dim, *net_dims, 1]
        net_list = []
        for i in range(len(dims) - 1):
            net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.GELU()])
        del net_list[-1]
        self.net = nn.Sequential(*net_list)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.5)
        nn.init.zeros_(self.net[-1].bias)

        self.state_avg = nn.Parameter(th.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(th.ones((state_dim,)), requires_grad=False)

    def state_norm(self, state: TEN) -> TEN:
        return (state - self.state_avg) / (self.state_std + 1e-4)

    def _encode(self, state: TEN) -> TEN:
        """Same 2-D / 3-D handling as ActorPPO_Wyckoff._encode."""
        leading = state.shape[:-1]
        if state.dim() >= 3:
            state = state.reshape(-1, state.shape[-1])

        state = self.state_norm(state)
        if self.encoder is not None:
            agent_state = state[:, :self.agent_state_dim]
            window_flat = state[:, self.agent_state_dim:]
            window = window_flat.reshape(-1, self.window_size, self.n_features)
            embedding = self.encoder(window)
            mlp_input = th.cat([agent_state, embedding], dim=1)
        else:
            mlp_input = state

        if len(leading) >= 2:
            mlp_input = mlp_input.reshape(*leading, -1)
        return mlp_input

    def forward(self, state: TEN) -> TEN:
        mlp_input = self._encode(state)
        return self.net(mlp_input)
