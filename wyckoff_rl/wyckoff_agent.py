"""
AgentPPO_Wyckoff — PPO agent with CNN temporal encoder.

Inherits all PPO logic from AgentPPO, only replaces actor/critic
with the CNN-equipped Wyckoff versions. No canonical ElegantRL changes.
"""

import torch as th
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.train import Config

from wyckoff_rl.wyckoff_actor_critic import ActorPPO_Wyckoff, CriticPPO_Wyckoff


class AgentPPO_Wyckoff(AgentPPO):
    """PPO with CNN temporal encoder for Wyckoff sliding window observations."""

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int,
                 gpu_id: int = 0, args: Config = Config()):
        # Let parent init everything (creates MLP actor/critic + optimizers)
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)

        # Read window config from args (set by function_train_test)
        n_features = getattr(args, 'n_features', 0)
        window_size = getattr(args, 'window_size', 1)

        # Replace actor and critic with CNN versions
        self.act = ActorPPO_Wyckoff(
            net_dims=net_dims, state_dim=state_dim, action_dim=action_dim,
            n_features=n_features, window_size=window_size,
        ).to(self.device)

        self.cri = CriticPPO_Wyckoff(
            net_dims=net_dims, state_dim=state_dim, action_dim=action_dim,
            n_features=n_features, window_size=window_size,
        ).to(self.device)

        # Rebuild optimizers for the new parameters
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)
