"""
Layer 2: Supervised classification heads for Wyckoff phase and event detection.

PhaseHead:
    Single-label classification of the current structural regime.

EventHead:
    Multi-label event detection (spring, upthrust, SOS, SOW, etc.).

Both operate on per-bar latent vectors from the encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config_v2 import TransformerConfig, PHASE_IGNORE_INDEX


class PhaseHead(nn.Module):
    """
    Phase / regime classification head.

    Input:
        (batch, d_model) or (batch, seq_len, d_model)

    Output:
        logits over phase labels
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
        return self.head(latent)

    def predict(self, latent: torch.Tensor) -> torch.Tensor:
        return self.forward(latent).argmax(dim=-1)

    def loss(
        self,
        latent: torch.Tensor,
        targets: torch.Tensor,
        class_weight: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        ignore_index: int = PHASE_IGNORE_INDEX,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """
        Weighted cross-entropy with ignore-index support.
        """
        logits = self.forward(latent)

        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            if sample_weight is not None:
                sample_weight = sample_weight.reshape(-1)

        loss = F.cross_entropy(
            logits,
            targets,
            weight=class_weight,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction="none",
        )

        valid = (targets != ignore_index).float()
        if sample_weight is None:
            w = valid
        else:
            w = sample_weight.float() * valid

        return (loss * w).sum() / w.sum().clamp_min(1.0)


class EventHead(nn.Module):
    """
    Multi-label event detection head.

    Input:
        (batch, d_model) or (batch, seq_len, d_model)

    Output:
        independent logits per event type
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
        return self.head(latent)

    def predict(self, latent: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (torch.sigmoid(self.forward(latent)) > threshold).float()

    def loss(
        self,
        latent: torch.Tensor,
        targets: torch.Tensor,
        pos_weight: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        focal_gamma: float = 2.0,
    ) -> torch.Tensor:
        """
        Weighted focal BCE loss for multi-label event detection.
        """
        logits = self.forward(latent)

        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1, targets.size(-1))
            if sample_weight is not None:
                sample_weight = sample_weight.reshape(-1, sample_weight.shape[-1])

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=pos_weight,
            reduction="none",
        )

        if focal_gamma > 0:
            probs = torch.sigmoid(logits)
            p_t = probs * targets.float() + (1.0 - probs) * (1.0 - targets.float())
            bce = ((1.0 - p_t) ** focal_gamma) * bce

        if sample_weight is None:
            w = torch.ones_like(targets, dtype=logits.dtype)
        else:
            w = sample_weight.float()

        return (bce * w).sum() / w.sum().clamp_min(1.0)


class ExcursionHead(nn.Module):
    """
    Expected excursion prediction head.
    Predicts expected favorable/adverse excursion from the current bar.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 2),
            nn.Softplus(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.head(latent)

    def loss(self, latent: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = self.forward(latent)
        if preds.dim() == 3:
            preds = preds.reshape(-1, 2)
            targets = targets.reshape(-1, 2)
        return F.smooth_l1_loss(preds, targets)
