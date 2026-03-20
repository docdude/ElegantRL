"""
AgentPPO_Wyckoff — PPO agent with CNN temporal encoder.

Inherits all PPO logic from AgentPPO, only replaces actor/critic
with the CNN-equipped Wyckoff versions. No canonical ElegantRL changes.
"""

import torch as th
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.train import Config

from wyckoff_rl.wyckoff_actor_critic import (
    ActorPPO_Wyckoff, CriticPPO_Wyckoff, TemporalEncoder,
)


class AgentPPO_Wyckoff(AgentPPO):
    """PPO with CNN temporal encoder for Wyckoff sliding window observations.

    Actor and critic share a single TemporalEncoder so the CNN temporal
    features are trained by both the policy and value gradients, which
    stabilises learning and halves the CNN parameters.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int,
                 gpu_id: int = 0, args: Config = Config()):
        # Let parent init everything (creates MLP actor/critic + optimizers)
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)

        # Read window config from args (set by function_train_test)
        n_features = getattr(args, 'n_features', 0)
        window_size = getattr(args, 'window_size', 1)

        # Create one shared encoder for both actor and critic
        shared_encoder = None
        if window_size > 1 and n_features > 0:
            shared_encoder = TemporalEncoder(n_features, window_size, embed_dim=64)

        # Replace actor and critic with CNN versions, sharing the encoder
        self.act = ActorPPO_Wyckoff(
            net_dims=net_dims, state_dim=state_dim, action_dim=action_dim,
            n_features=n_features, window_size=window_size,
            shared_encoder=shared_encoder,
        ).to(self.device)

        self.cri = CriticPPO_Wyckoff(
            net_dims=net_dims, state_dim=state_dim, action_dim=action_dim,
            n_features=n_features, window_size=window_size,
            shared_encoder=shared_encoder,
        ).to(self.device)

        # Rebuild optimizers — shared encoder params appear in both act and cri.
        # Both optimizers update the shared encoder so it learns from both
        # the policy gradient (actor) and value regression (critic).
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)
