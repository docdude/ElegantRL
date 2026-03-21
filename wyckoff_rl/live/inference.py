"""
Model Inference Engine — loads actor checkpoint and produces actions.

Handles:
  - Loading the saved ActorPPO model (full object, not just state_dict)
  - Building the sliding-window state vector
  - Forward pass → action → target position
"""

from __future__ import annotations

import os
import json
import numpy as np
import torch

from .live_features import N_TRAINING_FEATURES


class InferenceEngine:
    """
    Loads a trained ActorPPO and maps bar features → target position.

    Parameters
    ----------
    checkpoint_path : str
        Path to actor .pt file (saved as full model object via torch.save).
    window_size : int
        Sliding window of bars (must match training = 30).
    n_features : int
        Number of selected features per bar (must match training = 36).
    continuous_sizing : bool
        If False (default), discretize to {-1, 0, +1} via ±0.33 thresholds.
        If True, use raw continuous action in [-1, +1].
    initial_amount : float
        For agent state normalization (PnL / initial_amount).
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(
        self,
        checkpoint_path: str,
        window_size: int = 30,
        n_features: int = N_TRAINING_FEATURES,
        continuous_sizing: bool = False,
        initial_amount: float = 1000.0,
        device: str = "cpu",
    ):
        self.window_size = window_size
        self.n_features = n_features
        self.continuous_sizing = continuous_sizing
        self.initial_amount = initial_amount
        self.device = device
        self.state_dim = 3 + window_size * n_features  # 1083

        # Feature window buffer: (window_size, n_features), zero-padded initially
        self._window = np.zeros((window_size, n_features), dtype=np.float32)
        self._n_bars_seen = 0

        # Load model
        self.actor = self._load_actor(checkpoint_path)

    def _load_actor(self, path: str):
        """Load the full ActorPPO model object."""
        model = torch.load(path, map_location=self.device, weights_only=False)
        model.eval()
        model.to(self.device)
        return model

    def push_features(self, features: np.ndarray):
        """
        Push a new bar's feature vector into the sliding window.

        Parameters
        ----------
        features : np.ndarray, shape (n_features,)
        """
        assert features.shape == (self.n_features,), \
            f"Expected ({self.n_features},), got {features.shape}"
        # Shift window left, append new features at the end
        self._window[:-1] = self._window[1:]
        self._window[-1] = features
        self._n_bars_seen += 1

    def get_action(
        self,
        position: float,
        unrealized_pnl: float,
        cash: float,
    ) -> tuple[float, float]:
        """
        Build state and run inference.

        Parameters
        ----------
        position : float
            Current position (-1, 0, or +1 for binary; continuous otherwise).
        unrealized_pnl : float
            Current unrealized PnL in points.
        cash : float
            Accumulated realized PnL in points.

        Returns
        -------
        target_position : float
            {-1, 0, +1} in binary mode, or [-1, +1] in continuous mode.
        raw_action : float
            The raw tanh-bounded action from the actor.
        """
        # Agent state: [position, tanh(pnl/initial), tanh(cash/initial)]
        agent_state = np.array([
            position,
            np.tanh(unrealized_pnl / self.initial_amount),
            np.tanh(cash / self.initial_amount),
        ], dtype=np.float32)

        # Flatten window: (W, F) → (W*F,) in row-major order
        window_flat = self._window.flatten()

        # Full state
        state = np.concatenate([agent_state, window_flat])
        assert state.shape == (self.state_dim,), \
            f"State dim mismatch: {state.shape} vs {self.state_dim}"

        # Inference
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor)  # forward() includes tanh
        raw_action = action.squeeze().item()

        # Map to target position
        if self.continuous_sizing:
            target = raw_action
        else:
            # Binary: ±0.33 thresholds
            if raw_action > 0.33:
                target = 1.0
            elif raw_action < -0.33:
                target = -1.0
            else:
                target = 0.0

        return target, raw_action

    @property
    def ready(self) -> bool:
        """True if we have at least window_size bars in the buffer."""
        return self._n_bars_seen >= self.window_size

    @property
    def bars_seen(self) -> int:
        return self._n_bars_seen

    def reset(self):
        """Reset the sliding window."""
        self._window[:] = 0.0
        self._n_bars_seen = 0

    @classmethod
    def from_config(cls, config_dir: str, **overrides) -> "InferenceEngine":
        """
        Create engine from a run_config.json directory.

        Looks for run_config.json to extract env_params, then finds the
        best actor checkpoint.
        """
        config_path = os.path.join(config_dir, "run_config.json")
        with open(config_path) as f:
            config = json.load(f)

        env = config.get("env_params", {})
        kwargs = {
            "window_size": env.get("window_size", 30),
            "n_features": len(env.get("feature_indices", list(range(36)))),
            "continuous_sizing": env.get("continuous_sizing", False),
            "initial_amount": env.get("initial_amount", 1000.0),
        }
        kwargs.update(overrides)

        # Find checkpoint in config_dir (expects checkpoint_path override)
        return cls(**kwargs)
