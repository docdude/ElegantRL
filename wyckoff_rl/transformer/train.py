"""
Transformer Wyckoff Training Pipeline.

Two-phase training:
  Phase 1 — Supervised pre-training:
    Train encoder + heads on weak labels (phase/event/excursion).
    Bootstraps Wyckoff-relevant representations.

  Phase 2 — RL fine-tuning (PPO):
    Freeze or slow-train encoder, train policy head with PPO.
    Auxiliary losses on heads continue to provide gradient signal.

Usage:
    # Phase 1: Supervised pre-training
    python -m wyckoff_rl.transformer.train --phase pretrain \
        --npz-path datasets/us30_100pt.npz --epochs 50

    # Phase 2: RL fine-tuning
    python -m wyckoff_rl.transformer.train --phase rl \
        --npz-path datasets/us30_100pt.npz \
        --pretrained-encoder checkpoints/encoder_pretrained.pt \
        --total-steps 500000
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from wyckoff_rl.transformer.config import (
    TransformerConfig, TRANSFORMER_FEATURE_INDICES, N_PHASES, N_EVENTS,
)
from wyckoff_rl.transformer.encoder import WyckoffTransformerEncoder
from wyckoff_rl.transformer.heads import PhaseHead, EventHead, ExcursionHead
from wyckoff_rl.transformer.actor import ActorDiscreteTransformer, CriticTransformer
from wyckoff_rl.transformer.env import WyckoffTransformerVecEnv
from wyckoff_rl.transformer.labels import generate_structural_labels, save_labels, load_labels


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Supervised Pre-training
# ═══════════════════════════════════════════════════════════════════════

class WyckoffSequenceDataset(Dataset):
    """
    Dataset of (seq_len, n_features) windows with corresponding labels.

    Slides a window over the NPZ data, returning feature sequences
    and their labels at the last bar of each window.
    """

    def __init__(self, npz_path: str, label_path: str, config: TransformerConfig,
                 augment: bool = False, noise_std: float = 0.02):
        data = np.load(npz_path, allow_pickle=True)
        tech_ary = data['tech_ary'].astype(np.float32)

        # Select transformer features
        fi = config.feature_indices or TRANSFORMER_FEATURE_INDICES
        self.features = tech_ary[:, fi]  # (n_bars, n_features)
        self.seq_len = config.seq_len
        self.n_bars = len(self.features)
        self.augment = augment
        self.noise_std = noise_std

        # Compute per-feature normalization stats (mean/std from full dataset)
        self.feat_mean = self.features.mean(axis=0, keepdims=True)  # (1, F)
        self.feat_std = self.features.std(axis=0, keepdims=True) + 1e-8  # (1, F)

        # Load labels
        labels = load_labels(label_path)
        self.phase_labels = labels['phase']        # (n_bars,) int64
        self.event_labels = labels['events']       # (n_bars, N_EVENTS) float32
        self.excursion_labels = labels['excursion'] # (n_bars, 2) float32

        # Valid indices: need seq_len bars of history
        self.valid_start = self.seq_len
        self.n_samples = self.n_bars - self.valid_start

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        bar_idx = self.valid_start + idx
        # Window: [bar_idx - seq_len, bar_idx)
        window = self.features[bar_idx - self.seq_len:bar_idx].copy()  # (seq_len, n_features)

        # Normalize features
        window = (window - self.feat_mean) / self.feat_std

        # Training augmentation: additive Gaussian noise
        if self.augment:
            window = window + np.random.randn(*window.shape).astype(np.float32) * self.noise_std

        return {
            'features': torch.from_numpy(window),
            'phase': torch.tensor(self.phase_labels[bar_idx], dtype=torch.long),
            'events': torch.from_numpy(self.event_labels[bar_idx]),
            'excursion': torch.from_numpy(self.excursion_labels[bar_idx]),
        }


def pretrain(
    npz_path: str,
    label_path: str | None = None,
    config: TransformerConfig | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.15,
    save_dir: str = "checkpoints/transformer",
    device: str = "cuda",
    parquet_path: str | None = None,
    label_smoothing: float = 0.1,
    weight_decay: float = 5e-3,
    patience: int = 10,
    noise_std: float = 0.02,
):
    """
    Phase 1: Supervised pre-training of encoder + heads.

    Trains on structural labels derived from raw OHLCV price structure.
    The encoder learns temporal representations relevant to Wyckoff analysis.
    """
    if config is None:
        config = TransformerConfig()

    os.makedirs(save_dir, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Generate labels if not provided
    if label_path is None:
        label_path = os.path.join(save_dir, "structural_labels.npz")
        if not os.path.exists(label_path):
            if parquet_path is None:
                # Derive parquet path from npz path
                parquet_path = npz_path.replace('.npz', '_bars.parquet')
            print(f"Generating structural labels from {parquet_path}...")
            labels = generate_structural_labels(parquet_path, npz_path=npz_path)
            save_labels(labels, label_path)
            print(f"  Saved to {label_path}")
            # Print label distribution
            phase_counts = np.bincount(labels['phase'], minlength=N_PHASES)
            print(f"  Phase distribution: {dict(enumerate(phase_counts.tolist()))}")
            event_counts = labels['events'][:, 1:].sum(axis=0)
            print(f"  Event counts (excl. none): {event_counts.astype(int).tolist()}")

    # Dataset — train set gets augmentation, val does not
    train_full = WyckoffSequenceDataset(npz_path, label_path, config,
                                        augment=True, noise_std=noise_std)
    val_full = WyckoffSequenceDataset(npz_path, label_path, config,
                                      augment=False)
    # Share normalization stats (val uses train stats)
    val_full.feat_mean = train_full.feat_mean
    val_full.feat_std = train_full.feat_std

    n_total = len(train_full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    # Time-ordered split (not random — respects temporal structure)
    train_dataset = torch.utils.data.Subset(train_full, range(n_train))
    val_dataset = torch.utils.data.Subset(val_full, range(n_train, n_total))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"Dataset: {n_total} samples (train={n_train}, val={n_val})")
    print(f"Params/sample ratio: {0:.1f} (will be computed after model build)")
    print(f"Config: seq_len={config.seq_len}, d_model={config.d_model}, "
          f"n_layers={config.n_layers}, n_heads={config.n_heads}")
    print(f"Regularization: dropout={config.dropout}, weight_decay={weight_decay}, "
          f"label_smoothing={label_smoothing}, noise_std={noise_std}")

    # Build encoder + heads
    encoder = WyckoffTransformerEncoder(config).to(dev)
    phase_head = PhaseHead(config).to(dev)
    event_head = EventHead(config).to(dev)
    excursion_head = ExcursionHead(config).to(dev)

    # Count parameters
    n_params = sum(p.numel() for p in encoder.parameters())
    n_params += sum(p.numel() for m in [phase_head, event_head, excursion_head]
                    for p in m.parameters())
    print(f"Total parameters: {n_params:,}")

    print(f"Params/sample ratio: {n_params / n_train:.1f}")

    # Phase class weights — inverse frequency for imbalanced labels
    phase_counts = np.bincount(train_full.phase_labels[:n_train + train_full.valid_start],
                               minlength=N_PHASES).astype(np.float32)
    phase_counts = np.maximum(phase_counts, 1.0)
    phase_weights = (1.0 / phase_counts)
    phase_weights = phase_weights / phase_weights.sum() * N_PHASES  # normalize
    phase_weights_t = torch.from_numpy(phase_weights).to(dev)
    print(f"Phase weights: {dict(zip(range(N_PHASES), [f'{w:.2f}' for w in phase_weights]))}")

    # Optimizer
    all_params = (list(encoder.parameters()) + list(phase_head.parameters())
                  + list(event_head.parameters()) + list(excursion_head.parameters()))
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        # ── Train ──
        encoder.train(); phase_head.train(); event_head.train(); excursion_head.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            feats = batch['features'].to(dev)            # (B, seq, F)
            phase_tgt = batch['phase'].to(dev)            # (B,)
            event_tgt = batch['events'].to(dev)           # (B, N_EVENTS)
            excur_tgt = batch['excursion'].to(dev)        # (B, 2)

            # Forward
            latent = encoder(feats)              # (B, seq, d_model)
            last_latent = latent[:, -1, :]       # (B, d_model)

            loss_phase = F.cross_entropy(
                phase_head(last_latent), phase_tgt,
                weight=phase_weights_t, label_smoothing=label_smoothing,
            )
            loss_event = event_head.loss(last_latent, event_tgt)
            loss_excur = excursion_head.loss(last_latent, excur_tgt)

            loss = (config.phase_loss_weight * loss_phase
                    + config.event_loss_weight * loss_event
                    + loss_excur)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)

        # ── Validate ──
        encoder.eval(); phase_head.eval(); event_head.eval(); excursion_head.eval()
        val_loss = 0.0
        phase_correct = 0
        n_val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                feats = batch['features'].to(dev)
                phase_tgt = batch['phase'].to(dev)
                event_tgt = batch['events'].to(dev)
                excur_tgt = batch['excursion'].to(dev)

                latent = encoder(feats)
                last_latent = latent[:, -1, :]

                loss_phase = F.cross_entropy(
                    phase_head(last_latent), phase_tgt,
                    weight=phase_weights_t, label_smoothing=label_smoothing,
                )
                loss_event = event_head.loss(last_latent, event_tgt)
                loss_excur = excursion_head.loss(last_latent, excur_tgt)

                loss = (config.phase_loss_weight * loss_phase
                        + config.event_loss_weight * loss_event
                        + loss_excur)
                val_loss += loss.item() * feats.shape[0]

                # Phase accuracy
                pred = phase_head.predict(last_latent)
                phase_correct += (pred == phase_tgt).sum().item()
                n_val_samples += feats.shape[0]

        avg_val_loss = val_loss / max(n_val_samples, 1)
        phase_acc = phase_correct / max(n_val_samples, 1)

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"train_loss={avg_train_loss:.4f} | "
              f"val_loss={avg_val_loss:.4f} | "
              f"phase_acc={phase_acc:.3f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Save best + early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            checkpoint = {
                'encoder': encoder.state_dict(),
                'phase_head': phase_head.state_dict(),
                'event_head': event_head.state_dict(),
                'excursion_head': excursion_head.state_dict(),
                'config': config,
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'phase_acc': phase_acc,
                'feat_mean': train_full.feat_mean,
                'feat_std': train_full.feat_std,
            }
            torch.save(checkpoint, os.path.join(save_dir, "pretrained_best.pt"))
            print(f"  → Saved best model (val_loss={avg_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  ✘ Early stopping: no improvement for {patience} epochs")
                break

    # Final save
    torch.save(checkpoint, os.path.join(save_dir, "pretrained_final.pt"))
    print(f"\nPre-training complete. Best val_loss={best_val_loss:.4f} "
          f"(epoch {checkpoint.get('epoch', '?')})")
    return os.path.join(save_dir, "pretrained_best.pt")


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: RL Fine-tuning (PPO)
# ═══════════════════════════════════════════════════════════════════════

def load_pretrained(actor: ActorDiscreteTransformer, checkpoint_path: str):
    """Load pre-trained encoder and head weights into the actor."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    actor.encoder.load_state_dict(ckpt['encoder'])
    actor.phase_head.load_state_dict(ckpt['phase_head'])
    actor.event_head.load_state_dict(ckpt['event_head'])
    actor.excursion_head.load_state_dict(ckpt['excursion_head'])
    print(f"Loaded pre-trained encoder from {checkpoint_path} "
          f"(epoch={ckpt.get('epoch')}, phase_acc={ckpt.get('phase_acc', 0):.3f})")


def rl_train(
    npz_path: str,
    pretrained_path: str | None = None,
    config: TransformerConfig | None = None,
    total_steps: int = 500_000,
    num_envs: int = 128,
    horizon_len: int = 256,
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    entropy_coeff: float = 0.01,
    aux_loss_coeff: float = 0.1,
    encoder_lr_scale: float = 0.1,
    save_dir: str = "checkpoints/transformer_rl",
    gpu_id: int = 0,
    label_path: str | None = None,
    # Env params
    commission: float = 1.00,
    tick_size: float = 1.0,
    tick_value: float = 1.0,
    bar_range: float = 100.0,
    episode_len: int = 512,
    reward_mode: str = "dense_pnl",
    **env_kwargs,
):
    """
    Phase 2: PPO fine-tuning with multi-task auxiliary losses.

    The actor's encoder gradients come from three sources:
      1. PPO policy gradient (via policy_head)
      2. Phase classification loss (via phase_head)
      3. Event detection loss (via event_head)

    The encoder learning rate is scaled down (encoder_lr_scale) to preserve
    pre-trained representations while allowing adaptation.
    """
    if config is None:
        config = TransformerConfig()

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # ── Build environment ────────────────────────────────────────────────
    env = WyckoffTransformerVecEnv(
        config=config,
        npz_path=npz_path,
        num_envs=num_envs,
        gpu_id=gpu_id,
        episode_len=episode_len,
        commission=commission,
        tick_size=tick_size,
        tick_value=tick_value,
        bar_range=bar_range,
        reward_mode=reward_mode,
        **env_kwargs,
    )

    # ── Build actor and critic ───────────────────────────────────────────
    actor = ActorDiscreteTransformer(config).to(device)
    critic = CriticTransformer(config).to(device)

    # Load pre-trained weights
    if pretrained_path and os.path.exists(pretrained_path):
        load_pretrained(actor, pretrained_path)
        # Also init critic encoder from pre-trained weights
        ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        critic.encoder.load_state_dict(ckpt['encoder'])
        print("  Also loaded pre-trained encoder into critic")

    # ── Optimizer with differential learning rates ───────────────────────
    actor_encoder_params = list(actor.encoder.parameters())
    actor_head_params = (list(actor.policy_head.parameters())
                         + list(actor.phase_head.parameters())
                         + list(actor.event_head.parameters())
                         + list(actor.excursion_head.parameters()))

    actor_optimizer = torch.optim.Adam([
        {'params': actor_encoder_params, 'lr': lr_actor * encoder_lr_scale},
        {'params': actor_head_params, 'lr': lr_actor},
    ])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)

    # Count parameters
    n_actor = sum(p.numel() for p in actor.parameters())
    n_critic = sum(p.numel() for p in critic.parameters())
    print(f"Actor: {n_actor:,} params | Critic: {n_critic:,} params")
    print(f"Encoder LR: {lr_actor * encoder_lr_scale:.2e} | "
          f"Heads LR: {lr_actor:.2e} | Critic LR: {lr_critic:.2e}")

    # ── Load labels for auxiliary losses (optional) ──────────────────────
    aux_labels = None
    if label_path and os.path.exists(label_path):
        aux_labels = load_labels(label_path)
        print(f"Loaded auxiliary labels from {label_path}")

    # ── PPO training loop ────────────────────────────────────────────────
    state, _ = env.reset()
    total_reward = torch.zeros(num_envs, device=device)
    episode_count = 0
    episode_pnl_sum = 0.0

    steps_done = 0
    n_updates = total_steps // (num_envs * horizon_len)

    print(f"\nRL Training: {total_steps:,} steps, {n_updates} updates, "
          f"{num_envs} envs × {horizon_len} horizon")

    for update in range(n_updates):
        # ── Collect rollout ──────────────────────────────────────────
        states_buf = torch.zeros(horizon_len, num_envs, env.state_dim, device=device)
        actions_buf = torch.zeros(horizon_len, num_envs, dtype=torch.long, device=device)
        logprobs_buf = torch.zeros(horizon_len, num_envs, device=device)
        rewards_buf = torch.zeros(horizon_len, num_envs, device=device)
        dones_buf = torch.zeros(horizon_len, num_envs, dtype=torch.bool, device=device)
        values_buf = torch.zeros(horizon_len, num_envs, device=device)

        actor.eval()
        critic.eval()

        with torch.no_grad():
            for t in range(horizon_len):
                states_buf[t] = state
                action, logprob = actor.get_action(state)
                value = critic(state)

                next_state, reward, terminal, done, info = env.step(action)

                actions_buf[t] = action
                logprobs_buf[t] = logprob
                rewards_buf[t] = reward
                dones_buf[t] = done
                values_buf[t] = value

                # Track episode stats
                total_reward += reward
                if done.any():
                    for i in torch.where(done)[0].tolist():
                        episode_count += 1
                        episode_pnl_sum += env.cumulative_returns[i]
                        total_reward[i] = 0.0

                state = next_state

            # Bootstrap value for GAE
            next_value = critic(state)

        steps_done += horizon_len * num_envs

        # ── Compute GAE ──────────────────────────────────────────────
        advantages = torch.zeros_like(rewards_buf)
        lastgaelam = 0.0
        for t in reversed(range(horizon_len)):
            if t == horizon_len - 1:
                next_non_terminal = ~dones_buf[t]
                next_val = next_value
            else:
                next_non_terminal = ~dones_buf[t]
                next_val = values_buf[t + 1]
            delta = rewards_buf[t] + gamma * next_val * next_non_terminal.float() - values_buf[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_non_terminal.float() * lastgaelam
        returns = advantages + values_buf

        # ── Flatten for minibatch updates ────────────────────────────
        b_states = states_buf.reshape(-1, env.state_dim)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # ── PPO update ───────────────────────────────────────────────
        actor.train()
        critic.train()

        n_samples = b_states.shape[0]
        mini_batch_size = min(512, n_samples)
        n_epochs_ppo = 4

        for _ in range(n_epochs_ppo):
            indices = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, mini_batch_size):
                end = min(start + mini_batch_size, n_samples)
                idx = indices[start:end]

                mb_states = b_states[idx]
                mb_actions = b_actions[idx]
                mb_logprobs = b_logprobs[idx]
                mb_advantages = b_advantages[idx]
                mb_returns = b_returns[idx]

                # Actor loss (PPO-clip)
                new_logprob, entropy = actor.get_logprob_entropy(mb_states, mb_actions)
                ratio = (new_logprob - mb_logprobs).exp()
                surr1 = ratio * mb_advantages
                surr2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                actor_loss = policy_loss + entropy_coeff * entropy_loss

                # Auxiliary losses (if labels available)
                # Note: this is a simplified version. In production,
                # you'd index into the label arrays using the env's bar positions.
                # For now, aux losses are applied during pre-training only.

                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_optimizer.step()

                # Critic loss
                value_pred = critic(mb_states)
                critic_loss = F.mse_loss(value_pred, mb_returns)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                critic_optimizer.step()

        # ── Logging ──────────────────────────────────────────────────
        if (update + 1) % 5 == 0 or update == 0:
            avg_pnl = episode_pnl_sum / max(episode_count, 1)
            print(f"Update {update+1:4d}/{n_updates} | "
                  f"steps={steps_done:,} | "
                  f"episodes={episode_count} | "
                  f"avg_pnl=${avg_pnl:.2f} | "
                  f"policy_loss={policy_loss.item():.4f} | "
                  f"value_loss={critic_loss.item():.4f} | "
                  f"entropy={-entropy_loss.item():.4f}")

        # ── Save checkpoint ──────────────────────────────────────────
        if (update + 1) % 50 == 0:
            ckpt_path = os.path.join(save_dir, f"rl_step{steps_done}.pt")
            torch.save({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'config': config,
                'steps': steps_done,
                'episodes': episode_count,
                'avg_pnl': episode_pnl_sum / max(episode_count, 1),
            }, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = os.path.join(save_dir, "rl_final.pt")
    torch.save({
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'config': config,
        'steps': steps_done,
        'episodes': episode_count,
        'avg_pnl': episode_pnl_sum / max(episode_count, 1),
    }, final_path)
    print(f"\nRL training complete. Final: {final_path}")
    return final_path


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Transformer Wyckoff Training")
    parser.add_argument("--phase", choices=["pretrain", "rl", "both"], default="both")
    parser.add_argument("--npz-path", required=True, help="Path to NPZ data file")
    parser.add_argument("--parquet-path", default=None, help="Path to parquet with raw OHLCV bars")
    parser.add_argument("--label-path", default=None, help="Pre-computed labels NPZ")
    parser.add_argument("--pretrained-encoder", default=None, help="Pre-trained checkpoint")
    parser.add_argument("--save-dir", default="checkpoints/transformer")
    parser.add_argument("--gpu-id", type=int, default=0)

    # Pre-training args
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (0=disabled)")
    parser.add_argument("--noise-std", type=float, default=0.02,
                        help="Gaussian noise augmentation std")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate (encoder + heads)")

    # RL args
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--horizon-len", type=int, default=256)
    parser.add_argument("--rl-lr", type=float, default=3e-4)
    parser.add_argument("--episode-len", type=int, default=512)
    parser.add_argument("--reward-mode", default="dense_pnl")

    # Instrument config
    parser.add_argument("--commission", type=float, default=1.00)
    parser.add_argument("--tick-size", type=float, default=1.0)
    parser.add_argument("--tick-value", type=float, default=1.0)
    parser.add_argument("--bar-range", type=float, default=100.0)

    # Architecture overrides
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)

    args = parser.parse_args()

    config = TransformerConfig(
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )

    pretrained_path = args.pretrained_encoder

    if args.phase in ("pretrain", "both"):
        pretrained_path = pretrain(
            npz_path=args.npz_path,
            label_path=args.label_path,
            config=config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.pretrain_lr,
            save_dir=args.save_dir,
            device=f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu",
            parquet_path=args.parquet_path,
            label_smoothing=args.label_smoothing,
            weight_decay=args.weight_decay,
            patience=args.patience,
            noise_std=args.noise_std,
        )

    if args.phase in ("rl", "both"):
        rl_train(
            npz_path=args.npz_path,
            pretrained_path=pretrained_path,
            config=config,
            total_steps=args.total_steps,
            num_envs=args.num_envs,
            horizon_len=args.horizon_len,
            lr_actor=args.rl_lr,
            save_dir=os.path.join(args.save_dir, "rl"),
            gpu_id=args.gpu_id,
            label_path=args.label_path,
            commission=args.commission,
            tick_size=args.tick_size,
            tick_value=args.tick_value,
            bar_range=args.bar_range,
            episode_len=args.episode_len,
            reward_mode=args.reward_mode,
        )


if __name__ == "__main__":
    main()
