"""
AgentPPO_WaveNet — PPO agent with WaveNet temporal encoder.

Inherits all PPO logic from AgentPPO, only replaces actor/critic
with the WaveNet-equipped versions. Drop-in alongside AgentPPO_Wyckoff.
"""

import torch as th
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.train import Config

from wyckoff_rl.wyckoff_wave_actor_critic import (
    ActorPPO_WaveNet, CriticPPO_WaveNet, WaveNetEncoder,
)


class AgentPPO_WaveNet(AgentPPO):
    """PPO with WaveNet temporal encoder for Wyckoff sliding window observations.

    Actor and critic share a single WaveNetEncoder so the dilated causal
    convolutions are trained by both policy and value gradients.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int,
                 gpu_id: int = 0, args: Config = Config()):
        # Let parent init everything (creates MLP actor/critic + optimizers)
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)

        n_features = getattr(args, 'n_features', 0)
        window_size = getattr(args, 'window_size', 1)

        # WaveNet encoder hyperparams (from args or defaults)
        wn_channels = getattr(args, 'wavenet_channels', 32)
        wn_stacks = getattr(args, 'wavenet_stacks', 2)
        wn_kernel = getattr(args, 'wavenet_kernel_size', 3)
        wn_dilations = getattr(args, 'wavenet_dilations', (1, 2, 4, 8, 16))

        # Create one shared encoder for both actor and critic
        shared_encoder = None
        if window_size > 1 and n_features > 0:
            shared_encoder = WaveNetEncoder(
                n_features, window_size, embed_dim=64,
                n_channels=wn_channels, n_stacks=wn_stacks,
                kernel_size=wn_kernel, dilation_rates=wn_dilations,
            )

        # Replace actor and critic with WaveNet versions
        self.act = ActorPPO_WaveNet(
            net_dims=net_dims, state_dim=state_dim, action_dim=action_dim,
            n_features=n_features, window_size=window_size,
            shared_encoder=shared_encoder,
        ).to(self.device)

        self.cri = CriticPPO_WaveNet(
            net_dims=net_dims, state_dim=state_dim, action_dim=action_dim,
            n_features=n_features, window_size=window_size,
            shared_encoder=shared_encoder,
        ).to(self.device)

        # Rebuild optimizers with shared encoder params
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)
