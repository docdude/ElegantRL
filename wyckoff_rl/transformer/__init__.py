"""
Transformer-based Wyckoff trading architecture.

Three-layer stack:
  Layer 1: Causal Transformer encoder over bar sequences
  Layer 2: Supervised phase/event classification heads  
  Layer 3: RL policy (PPO) on encoder latent state
"""

from .encoder import WyckoffTransformerEncoder
from .heads import PhaseHead, EventHead, ExcursionHead
from .actor import ActorDiscreteTransformer, CriticTransformer
from .env import WyckoffTransformerVecEnv
from .config import TransformerConfig
