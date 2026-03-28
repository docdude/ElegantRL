"""
Layer 3: RL Actor and Critic operating on transformer encoder latent state.

The actor receives the encoder's last-bar latent vector concatenated with
position features and outputs discrete action logits. The critic does the
same but outputs a scalar value estimate.

This separates representation learning (encoder) from decision-making (policy).
The encoder can be pre-trained with supervised phase/event labels, then
the full stack is fine-tuned with PPO.

Interface is compatible with ElegantRL's PPO agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .encoder import WyckoffTransformerEncoder
from .heads import PhaseHead, EventHead
from .config import TransformerConfig


class ActorDiscreteTransformer(nn.Module):
    """
    Discrete PPO actor with transformer encoder backbone.

    Flow:
      bar_sequence (batch, seq_len, n_features)
        → encoder → latent (batch, d_model)
        → concat with position_features (batch, n_pos)
        → MLP → 6 action logits → Categorical

    The encoder is shared and can be pre-trained.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Shared encoder (Layer 1)
        self.encoder = WyckoffTransformerEncoder(config)

        # Supervised heads (Layer 2) — auxiliary losses during training
        self.phase_head = PhaseHead(config)
        self.event_head = EventHead(config)

        # Policy head (Layer 3)
        policy_input_dim = config.d_model + config.n_position_features
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.n_actions),
        )

        self.action_dim = config.n_actions
        self.softmax = nn.Softmax(dim=-1)

        # State normalization (running stats, updated by PPO)
        self.state_avg = None
        self.state_std = None

    @property
    def state_dim(self) -> int:
        """Total state dim for compatibility — seq_len * n_features + n_position."""
        return self.config.seq_len * self.config.n_bar_features + self.config.n_position_features

    def _split_state(self, state: torch.Tensor):
        """
        Split flat state vector into bar sequence and position features.

        The environment flattens (seq_len, n_features) into a 1D vector,
        with position features appended at the end.

        Parameters
        ----------
        state : (batch, seq_len * n_features + n_position)

        Returns
        -------
        bar_seq : (batch, seq_len, n_features)
        pos_feats : (batch, n_position)
        """
        n_bar = self.config.seq_len * self.config.n_bar_features
        bar_flat = state[:, :n_bar]
        pos_feats = state[:, n_bar:]
        bar_seq = bar_flat.reshape(-1, self.config.seq_len, self.config.n_bar_features)
        return bar_seq, pos_feats

    def _encode(self, state: torch.Tensor):
        """
        Encode state → (latent, pos_feats).

        Returns
        -------
        latent : (batch, d_model) — last-bar encoder output
        pos_feats : (batch, n_position) — position features
        full_latent : (batch, seq_len, d_model) — full sequence latent (for aux heads)
        """
        bar_seq, pos_feats = self._split_state(state)
        full_latent = self.encoder(bar_seq)
        latent = full_latent[:, -1, :]  # Last bar's representation
        return latent, pos_feats, full_latent

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action selection (argmax).

        Parameters
        ----------
        state : (batch, state_dim) — flattened bar sequence + position features

        Returns
        -------
        action : (batch,) — integer action indices
        """
        latent, pos_feats, _ = self._encode(state)
        policy_input = torch.cat([latent, pos_feats], dim=-1)
        logits = self.policy_head(policy_input)
        return logits.argmax(dim=-1)

    def get_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Stochastic action selection for training.

        Returns
        -------
        action : (batch,) — sampled action
        logprob : (batch,) — log probability of sampled action
        """
        latent, pos_feats, _ = self._encode(state)
        policy_input = torch.cat([latent, pos_feats], dim=-1)
        logits = self.policy_head(policy_input)
        probs = self.softmax(logits)
        dist = Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob

    def get_logprob_entropy(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probability and entropy for PPO objective.

        Returns
        -------
        logprob : (batch,)
        entropy : (batch,)
        """
        latent, pos_feats, _ = self._encode(state)
        policy_input = torch.cat([latent, pos_feats], dim=-1)
        logits = self.policy_head(policy_input)
        probs = self.softmax(logits)
        dist = Categorical(probs)
        return dist.log_prob(action), dist.entropy()

    def get_aux_losses(
        self,
        state: torch.Tensor,
        phase_targets: torch.Tensor | None = None,
        event_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute auxiliary supervised losses from the classification heads.
        These provide gradient signal to the encoder during RL training.

        Returns dict of named losses (only for provided targets).
        """
        latent, _, _ = self._encode(state)
        losses = {}

        if phase_targets is not None:
            losses["phase"] = self.phase_head.loss(latent, phase_targets)
        if event_targets is not None:
            losses["event"] = self.event_head.loss(latent, event_targets)

        return losses

    def get_phase_prediction(self, state: torch.Tensor) -> torch.Tensor:
        """Predict current Wyckoff phase."""
        latent, _, _ = self._encode(state)
        return self.phase_head.predict(latent)

    def get_event_prediction(self, state: torch.Tensor) -> torch.Tensor:
        """Predict active Wyckoff events."""
        latent, _, _ = self._encode(state)
        return self.event_head.predict(latent)


class CriticTransformer(nn.Module):
    """
    Value function critic with transformer encoder backbone.

    Shares the same architecture as the actor but with a scalar value output.
    In practice, the encoder weights can be shared or separate.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Separate encoder for critic (no shared weights — PPO standard)
        self.encoder = WyckoffTransformerEncoder(config)

        # Value head
        value_input_dim = config.d_model + config.n_position_features
        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

        # State normalization
        self.state_avg = None
        self.state_std = None

    @property
    def state_dim(self) -> int:
        return self.config.seq_len * self.config.n_bar_features + self.config.n_position_features

    def _split_state(self, state: torch.Tensor):
        n_bar = self.config.seq_len * self.config.n_bar_features
        bar_flat = state[:, :n_bar]
        pos_feats = state[:, n_bar:]
        bar_seq = bar_flat.reshape(-1, self.config.seq_len, self.config.n_bar_features)
        return bar_seq, pos_feats

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : (batch, state_dim)

        Returns
        -------
        value : (batch,) — scalar value estimate
        """
        bar_seq, pos_feats = self._split_state(state)
        full_latent = self.encoder(bar_seq)
        latent = full_latent[:, -1, :]
        value_input = torch.cat([latent, pos_feats], dim=-1)
        return self.value_head(value_input).squeeze(-1)
