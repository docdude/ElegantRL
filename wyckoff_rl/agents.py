"""
Wyckoff RL — Fixed Discrete PPO Agent.

Subclasses ElegantRL's AgentDiscretePPO to restore the correct PPO
clipped surrogate objective (the canonical version was modified with
a broken one-sided clip that removes the trust region constraint).

Also supports optional asymmetric advantage weighting (loss_weight > 1
penalises negative advantages more, Sortino-style) applied *on top* of
the proper min(surrogate1, surrogate2) formulation.
"""

import torch as th
import torch.nn.functional as F
from elegantrl.agents.AgentPPO import AgentDiscretePPO
from elegantrl.train import Config

TEN = th.Tensor


class AgentDiscreteWyckoffPPO(AgentDiscretePPO):
    """AgentDiscretePPO with the correct PPO clipped surrogate restored.

    Optionally applies a KL divergence penalty against a frozen BC reference
    policy (set via bc_ref_act / bc_kl_coeff by _load_bc_checkpoint).
    """

    def update_objectives(self, buffer: tuple[TEN, ...], update_t: int) -> tuple[float, float, float]:
        states, actions, unmasks, logprobs, advantages, reward_sums = buffer

        sample_len = states.shape[0]
        num_seqs = states.shape[1]
        ids = th.randint(sample_len * num_seqs, size=(self.batch_size,), requires_grad=False, device=self.device)
        ids0 = th.fmod(ids, sample_len)
        ids1 = th.div(ids, sample_len, rounding_mode='floor')

        state = states[ids0, ids1]
        action = actions[ids0, ids1]
        unmask = unmasks[ids0, ids1]
        logprob = logprobs[ids0, ids1]
        advantage = advantages[ids0, ids1]
        reward_sum = reward_sums[ids0, ids1]

        # ── Critic update ────────────────────────────────────────────────
        value = self.cri(state).squeeze(1)
        obj_critic = (self.criterion(value, reward_sum) * unmask).mean()
        self.optimizer_backward(self.cri_optimizer, obj_critic)

        # ── Actor update — correct PPO clipped surrogate ─────────────────
        new_logprob, entropy = self.act.get_logprob_entropy(state, action)
        ratio = (new_logprob - logprob.detach()).exp()

        # Asymmetric advantage weighting (loss_weight=1.0 → standard PPO)
        adv_weight = th.where(advantage.lt(0), self.loss_weight, 1.0)
        weighted_adv = advantage * adv_weight

        # Standard PPO clipped surrogate with trust region
        surrogate1 = weighted_adv * ratio
        surrogate2 = weighted_adv * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
        surrogate = th.min(surrogate1, surrogate2)

        obj_surrogate = (surrogate * unmask).mean()
        obj_entropy = (entropy * unmask).mean()
        obj_actor_full = obj_surrogate - obj_entropy * self.lambda_entropy

        # ── KL penalty against frozen BC reference policy ────────────
        bc_ref = getattr(self, 'bc_ref_act', None)
        bc_kl = getattr(self, 'bc_kl_coeff', 0.0)
        if bc_ref is not None and bc_kl > 0:
            with th.no_grad():
                bc_logits = bc_ref.net(bc_ref.state_norm(state))
                bc_log_probs = F.log_softmax(bc_logits, dim=-1)
            cur_logits = self.act.net(self.act.state_norm(state))
            cur_log_probs = F.log_softmax(cur_logits, dim=-1)
            # KL(current || BC) — penalise divergence from expert
            kl_div = (cur_log_probs.exp() * (cur_log_probs - bc_log_probs)).sum(dim=-1)
            obj_kl = (kl_div * unmask).mean()
            obj_actor_full = obj_actor_full - bc_kl * obj_kl

        self.optimizer_backward(self.act_optimizer, -obj_actor_full)

        return obj_critic.item(), obj_surrogate.item(), obj_entropy.item()
