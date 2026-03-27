"""
Layer 2: Supervised classification heads for Wyckoff phase and event detection.

These heads sit on top of the transformer encoder's latent representations
and are trained with supervised or weakly-supervised labels.

PhaseHead:  Classifies the current market regime (accumulation, distribution, etc.)
EventHead:  Multi-label detection of Wyckoff events (spring, upthrust, test, etc.)

Both operate on per-bar latent vectors from the encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import TransformerConfig


class PhaseHead(nn.Module):
    """
    Wyckoff phase classification head.

    Input:  (batch, d_model) — encoder latent for current bar
    Output: (batch, n_phases) — logits over phase labels

    Trained with cross-entropy against weak phase labels derived from
    wave structure analysis.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.n_phases),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        latent : Tensor, shape (batch, d_model) or (batch, seq_len, d_model)

        Returns
        -------
        Tensor, shape (batch, n_phases) or (batch, seq_len, n_phases) — logits
        """
        return self.head(latent)

    def predict(self, latent: torch.Tensor) -> torch.Tensor:
        """Return predicted phase index."""
        return self.forward(latent).argmax(dim=-1)

    def loss(self, latent: torch.Tensor, targets: torch.Tensor,
             weights: torch.Tensor | None = None) -> torch.Tensor:
        """
        Cross-entropy loss for phase classification.

        Parameters
        ----------
        latent : (batch, d_model) or (batch, seq_len, d_model)
        targets : (batch,) or (batch, seq_len) — integer phase labels
        weights : optional class weights tensor

        Returns
        -------
        Scalar loss
        """
        logits = self.forward(latent)
        if logits.dim() == 3:
            # (batch, seq, n_phases) → (batch*seq, n_phases)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
        return F.cross_entropy(logits, targets, weight=weights)


class EventHead(nn.Module):
    """
    Wyckoff event detection head (multi-label).

    Input:  (batch, d_model) — encoder latent for current bar
    Output: (batch, n_events) — independent logits per event type

    Trained with binary cross-entropy: multiple events can co-occur.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.n_events),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        latent : (batch, d_model) or (batch, seq_len, d_model)

        Returns
        -------
        Tensor — logits, same leading dims + n_events
        """
        return self.head(latent)

    def predict(self, latent: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary event activations."""
        return (torch.sigmoid(self.forward(latent)) > threshold).float()

    def loss(self, latent: torch.Tensor, targets: torch.Tensor,
             pos_weight: torch.Tensor | None = None) -> torch.Tensor:
        """
        Binary cross-entropy loss for multi-label event detection.

        Parameters
        ----------
        latent : (batch, d_model) or (batch, seq_len, d_model)
        targets : (batch, n_events) or (batch, seq_len, n_events) — binary labels
        pos_weight : optional positive class weights (events are rare)

        Returns
        -------
        Scalar loss
        """
        logits = self.forward(latent)
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1, targets.size(-1))
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=pos_weight
        )


class ExcursionHead(nn.Module):
    """
    Expected excursion prediction head.

    Predicts expected favorable/adverse excursion from current bar,
    normalized by bar range. Helps RL layer gauge risk/reward.

    Input:  (batch, d_model)
    Output: (batch, 2) — [expected_favorable, expected_adverse]
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 2),
            nn.Softplus(),  # Excursions are non-negative
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.head(latent)

    def loss(self, latent: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Smooth L1 loss for excursion prediction."""
        preds = self.forward(latent)
        if preds.dim() == 3:
            preds = preds.reshape(-1, 2)
            targets = targets.reshape(-1, 2)
        return F.smooth_l1_loss(preds, targets)
