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
from torch.distributions import Categorical

from .encoder import WyckoffTransformerEncoder
from .heads_v2 import PhaseHead, EventHead
from .config_v2 import TransformerConfig


class ActorDiscreteTransformer(nn.Module):
    """
    Discrete PPO actor with transformer encoder backbone.

    Flow:
      bar_sequence (batch, seq_len, n_features)
        -> encoder -> latent (batch, d_model)
        -> concat with position_features (batch, n_pos)
        -> MLP -> action logits -> Categorical
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Shared encoder
        self.encoder = WyckoffTransformerEncoder(config)

        # Auxiliary supervised heads
        self.phase_head = PhaseHead(config)
        self.event_head = EventHead(config)

        # Policy head
        policy_input_dim = config.d_model + config.n_position_features
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.n_actions),
        )

        self.action_dim = config.n_actions

        # Optional running normalization hooks for compatibility
        self.state_avg = None
        self.state_std = None

    @property
    def state_dim(self) -> int:
        return self.config.seq_len * self.config.n_bar_features + self.config.n_position_features

    def _split_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split flat state into:
        - bar sequence: (batch, seq_len, n_bar_features)
        - position features: (batch, n_position_features)
        """
        n_bar = self.config.seq_len * self.config.n_bar_features
        bar_flat = state[:, :n_bar]
        pos_feats = state[:, n_bar:]
        bar_seq = bar_flat.reshape(-1, self.config.seq_len, self.config.n_bar_features)
        return bar_seq, pos_feats

    def _encode(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        latent : (batch, d_model)
            Last-bar encoder representation
        pos_feats : (batch, n_position_features)
        full_latent : (batch, seq_len, d_model)
        """
        bar_seq, pos_feats = self._split_state(state)
        full_latent = self.encoder(bar_seq)
        latent = full_latent[:, -1, :]
        return latent, pos_feats, full_latent

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action selection (argmax).
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
        action : (batch,)
        logprob : (batch,)
        """
        latent, pos_feats, _ = self._encode(state)
        policy_input = torch.cat([latent, pos_feats], dim=-1)
        logits = self.policy_head(policy_input)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob

    def get_logprob_entropy(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probability and entropy for PPO objective.
        """
        latent, pos_feats, _ = self._encode(state)
        policy_input = torch.cat([latent, pos_feats], dim=-1)
        logits = self.policy_head(policy_input)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy()

    def get_aux_losses(
        self,
        state: torch.Tensor,
        phase_targets: torch.Tensor | None = None,
        event_targets: torch.Tensor | None = None,
        phase_weight: torch.Tensor | None = None,
        event_weight: torch.Tensor | None = None,
        phase_class_weight: torch.Tensor | None = None,
        event_pos_weight: torch.Tensor | None = None,
        phase_label_smoothing: float = 0.0,
        event_focal_gamma: float = 1.5,
        negative_event_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Compute auxiliary supervised losses.

        phase_weight
            Per-sample confidence for phase labels.
            Ignored bars are excluded inside PhaseHead.loss via PHASE_IGNORE_INDEX.

        event_weight
            Positive-event confidence matrix with same shape as event_targets.
            Negative labels automatically receive weight=negative_event_weight.
        """
        latent, _, _ = self._encode(state)
        losses: dict[str, torch.Tensor] = {}

        if phase_targets is not None:
            losses["phase"] = self.phase_head.loss(
                latent,
                phase_targets,
                class_weight=phase_class_weight,
                sample_weight=phase_weight,
                label_smoothing=phase_label_smoothing,
            )

        if event_targets is not None:
            event_sample_weight = None
            if event_weight is not None:
                pos_conf = event_weight.float()
                event_sample_weight = torch.where(
                    event_targets > 0.5,
                    pos_conf.clamp_min(0.25),
                    torch.full_like(event_targets, float(negative_event_weight)),
                )

            losses["event"] = self.event_head.loss(
                latent,
                event_targets,
                pos_weight=event_pos_weight,
                sample_weight=event_sample_weight,
                focal_gamma=event_focal_gamma,
            )

        return losses

    def get_phase_prediction(self, state: torch.Tensor) -> torch.Tensor:
        latent, _, _ = self._encode(state)
        return self.phase_head.predict(latent)

    def get_event_prediction(self, state: torch.Tensor) -> torch.Tensor:
        latent, _, _ = self._encode(state)
        return self.event_head.predict(latent)


class CriticTransformer(nn.Module):
    """
    Value function critic with transformer encoder backbone.
    Uses a separate encoder, which is standard for PPO.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.encoder = WyckoffTransformerEncoder(config)

        value_input_dim = config.d_model + config.n_position_features
        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

        self.state_avg = None
        self.state_std = None

    @property
    def state_dim(self) -> int:
        return self.config.seq_len * self.config.n_bar_features + self.config.n_position_features

    def _split_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_bar = self.config.seq_len * self.config.n_bar_features
        bar_flat = state[:, :n_bar]
        pos_feats = state[:, n_bar:]
        bar_seq = bar_flat.reshape(-1, self.config.seq_len, self.config.n_bar_features)
        return bar_seq, pos_feats

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        bar_seq, pos_feats = self._split_state(state)
        full_latent = self.encoder(bar_seq)
        latent = full_latent[:, -1, :]
        value_input = torch.cat([latent, pos_feats], dim=-1)
        return self.value_head(value_input).squeeze(-1)
