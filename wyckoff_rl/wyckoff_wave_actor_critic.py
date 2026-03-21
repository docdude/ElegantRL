"""
Wyckoff WaveNet Actor / Critic for long-range temporal pattern recognition.

Replaces the 2-layer CNN TemporalEncoder with a WaveNet-style encoder using
dilated causal convolutions with gated activations (tanh * sigmoid).

Key advantages over plain CNN:
  - Exponential receptive field: dilation [1,2,4,8,16] → 31-bar receptive
    field with just 5 layers, covering the full 30-bar window
  - Gated activations handle sparse Wyckoff events (Springs, Upthrusts)
    better than ReLU/GELU — the sigmoid gate learns to pass or block
  - Skip connections from every dilation level let the MLP head attend
    to patterns at multiple timescales simultaneously
  - Causal padding ensures no future information leaks into past bars

State layout (flat): [agent_state(3), window(W*F)]
  Same as wyckoff_actor_critic.py — drop-in replacement.

Interface matches ActorPPO / CriticPPO exactly so ElegantRL's
AgentPPO can use them as drop-in replacements.
"""

import torch as th
from torch import nn

TEN = th.Tensor


# ─────────────────────────────────────────────────────────────────────────────
# WaveNet building blocks (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """Conv1d with left-padding so output[t] depends only on input[<=t]."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation)

    def forward(self, x: TEN) -> TEN:
        # x: (B, C, T)
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class WaveNetResidualBlock(nn.Module):
    """Gated causal convolution block with skip connection.

    Core WaveNet building block:
      tanh(causal_conv(x)) * sigmoid(causal_conv(x))
    followed by 1x1 convs for residual and skip outputs.
    """

    def __init__(self, n_channels: int, kernel_size: int = 3,
                 dilation: int = 1):
        super().__init__()
        self.tanh_conv = CausalConv1d(n_channels, n_channels,
                                       kernel_size, dilation)
        self.sigm_conv = CausalConv1d(n_channels, n_channels,
                                       kernel_size, dilation)
        self.residual_conv = nn.Conv1d(n_channels, n_channels, 1)
        self.skip_conv = nn.Conv1d(n_channels, n_channels, 1)

    def forward(self, x: TEN) -> tuple[TEN, TEN]:
        """Returns (residual_out, skip_out)."""
        gated = th.tanh(self.tanh_conv(x)) * th.sigmoid(self.sigm_conv(x))
        residual = x + self.residual_conv(gated)
        skip = self.skip_conv(gated)
        return residual, skip


class WaveNetEncoder(nn.Module):
    """WaveNet temporal encoder for (batch, W, F) feature windows.

    Architecture:
      1. Input projection: F channels → n_channels via 1x1 conv
      2. N stacks of dilated causal conv blocks (dilation [1,2,4,8,16])
         with gated activations and skip connections
      3. Sum all skip outputs → GELU → 1x1 conv → global avg pool
      4. Project to embed_dim

    With n_stacks=2 and dilation_rates=[1,2,4,8,16]:
      Receptive field = n_stacks * sum(dilation_rates) * (kernel_size-1) + 1
                      = 2 * 31 * 2 + 1 = 125 (well beyond the 30-bar window)
      Total conv layers: 2 * 5 * 2 = 20 (but only ~10x params vs plain CNN
        because 1x1 convs are cheap)
    """

    def __init__(self, n_features: int, window_size: int,
                 embed_dim: int = 64, n_channels: int = 32,
                 n_stacks: int = 2, kernel_size: int = 3,
                 dilation_rates: tuple[int, ...] = (1, 2, 4, 8, 16)):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.embed_dim = embed_dim

        # Input projection: (B, F, W) → (B, n_channels, W)
        self.input_proj = nn.Conv1d(n_features, n_channels, 1)

        # Stacks of dilated causal conv blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_stacks):
            for d in dilation_rates:
                self.blocks.append(
                    WaveNetResidualBlock(n_channels, kernel_size, dilation=d)
                )

        # Post-processing: sum skips → activation → projection
        self.post_conv = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(n_channels, n_channels, 1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(n_channels, embed_dim)

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
        x = window.transpose(1, 2)          # (B, F, W)
        x = self.input_proj(x)              # (B, n_channels, W)

        # Accumulate skip connections from all blocks
        skip_sum = th.zeros_like(x)
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip

        # Post-process and pool
        x = self.post_conv(skip_sum)         # (B, n_channels, W)
        x = self.pool(x).squeeze(-1)        # (B, n_channels)
        return self.proj(x)                  # (B, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Actor / Critic with WaveNet encoder
# ─────────────────────────────────────────────────────────────────────────────

class ActorPPO_WaveNet(nn.Module):
    """PPO Actor with WaveNet temporal encoder for Wyckoff sliding window."""

    def __init__(self, net_dims: list[int], state_dim: int, action_dim: int,
                 n_features: int = 0, window_size: int = 1,
                 shared_encoder: WaveNetEncoder = None):
        super().__init__()

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
            self.encoder = shared_encoder or WaveNetEncoder(
                self.n_features, self.window_size, embed_dim)
            mlp_input_dim = agent_state_dim + self.encoder.embed_dim
        else:
            self.encoder = None
            mlp_input_dim = state_dim

        # MLP head: [mlp_input_dim, *net_dims, action_dim]
        dims = [mlp_input_dim, *net_dims, action_dim]
        net_list = []
        for i in range(len(dims) - 1):
            net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.GELU()])
        del net_list[-1]  # remove last activation → raw output
        self.net = nn.Sequential(*net_list)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

        self.action_std_log = nn.Parameter(
            th.zeros((1, action_dim)), requires_grad=True)
        self.ActionDist = th.distributions.normal.Normal

        self.state_avg = nn.Parameter(
            th.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(
            th.ones((state_dim,)), requires_grad=False)

    def state_norm(self, state: TEN) -> TEN:
        return (state - self.state_avg) / (self.state_std + 1e-4)

    def _encode(self, state: TEN) -> TEN:
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


class CriticPPO_WaveNet(nn.Module):
    """PPO Critic with WaveNet temporal encoder for Wyckoff sliding window."""

    def __init__(self, net_dims: list[int], state_dim: int, action_dim: int,
                 n_features: int = 0, window_size: int = 1,
                 shared_encoder: WaveNetEncoder = None):
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
            self.encoder = shared_encoder or WaveNetEncoder(
                self.n_features, self.window_size, embed_dim)
            mlp_input_dim = agent_state_dim + self.encoder.embed_dim
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

        self.state_avg = nn.Parameter(
            th.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(
            th.ones((state_dim,)), requires_grad=False)

    def state_norm(self, state: TEN) -> TEN:
        return (state - self.state_avg) / (self.state_std + 1e-4)

    def _encode(self, state: TEN) -> TEN:
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
