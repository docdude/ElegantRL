"""
Layer 1: Causal Transformer Encoder for bar sequences.

Reads a sequence of bar-level feature vectors and produces contextualized
latent representations at each timestep. Causal masking ensures each bar
can only attend to itself and past bars — compatible with real-time inference.

Architecture:
  Input embedding: Linear(n_features → d_model) + learned positional encoding
  N × TransformerEncoderLayer with causal mask
  Output: (batch, seq_len, d_model) — latent state per bar
"""

import math
import torch
import torch.nn as nn
from .config_v2 import TransformerConfig


class WyckoffTransformerEncoder(nn.Module):
    """
    Causal transformer encoder over bar-level feature sequences.

    Input:  (batch, seq_len, n_bar_features) — raw per-bar features
    Output: (batch, seq_len, d_model) — contextualized latent per bar
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.seq_len = config.seq_len

        # Input projection: per-bar features → d_model
        self.input_proj = nn.Linear(config.n_bar_features, config.d_model)

        # Learned positional embeddings
        self.pos_embed = nn.Embedding(config.seq_len, config.d_model)

        # Layer norm after embedding (pre-norm architecture)
        self.embed_norm = nn.LayerNorm(config.d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
        )

        # Final layer norm
        self.output_norm = nn.LayerNorm(config.d_model)

        # Register causal mask as buffer (not a parameter)
        self._register_causal_mask(config.seq_len)

        self._init_weights()

    def _register_causal_mask(self, size: int):
        """Create and register causal attention mask."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def _init_weights(self):
        """Xavier uniform for projections, normal for embeddings."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, n_bar_features)

        Returns
        -------
        Tensor, shape (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project features to d_model
        h = self.input_proj(x)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=x.device)
        h = h + self.pos_embed(positions)

        # Pre-norm
        h = self.embed_norm(h)

        # Get causal mask for current sequence length
        if seq_len <= self.seq_len:
            mask = self.causal_mask[:seq_len, :seq_len]
        else:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        # Transformer encoding with causal mask
        h = self.encoder(h, mask=mask)

        # Final norm
        h = self.output_norm(h)

        return h

    def get_last_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience: encode full sequence, return only the last timestep.

        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, n_bar_features)

        Returns
        -------
        Tensor, shape (batch, d_model)
        """
        return self.forward(x)[:, -1, :]

    @property
    def output_dim(self) -> int:
        return self.d_model
