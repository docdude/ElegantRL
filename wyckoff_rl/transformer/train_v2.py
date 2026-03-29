"""
Transformer Wyckoff Training Pipeline.

Two-phase training:
  Phase 1 — Supervised pre-training:
    Train encoder + heads on structural weak labels.
    Bootstraps Wyckoff-relevant representations.

  Phase 2 — RL fine-tuning (PPO):
    Fine-tune actor/critic with PPO while optionally retaining
    auxiliary supervised losses for phase/event heads.
"""

import os
import sys
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from wyckoff_rl.transformer.config_v2 import (  # noqa: E402
    TransformerConfig,
    TRANSFORMER_FEATURE_INDICES,
    N_REGIMES,
    N_EVENTS,
    PHASE_IGNORE_INDEX,
)
from wyckoff_rl.transformer.encoder import WyckoffTransformerEncoder  # noqa: E402
from wyckoff_rl.transformer.heads_v2 import PhaseHead, EventHead  # noqa: E402
from wyckoff_rl.transformer.actor_v2 import ActorDiscreteTransformer, CriticTransformer  # noqa: E402
from wyckoff_rl.transformer.env import WyckoffTransformerVecEnv  # noqa: E402
from wyckoff_rl.transformer.labels_v2 import generate_structural_labels, save_labels, load_labels  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class WyckoffSequenceDataset(Dataset):
    """
    Sliding-window dataset of transformer inputs and labels.

    Returns:
        features      : (seq_len, n_features)
        phase         : scalar int64
        phase_weight  : scalar float32
        events        : (n_events,) float32
        event_weight  : (n_events,) float32
    """

    def __init__(
        self,
        npz_path: str,
        label_path: str,
        config: TransformerConfig,
        augment: bool = False,
        noise_std: float = 0.02,
        end_idx: int = 0,
    ):
        data = np.load(npz_path, allow_pickle=True)
        tech_ary = data["tech_ary"].astype(np.float32)

        fi = config.feature_indices or TRANSFORMER_FEATURE_INDICES
        features_full = tech_ary[:, fi]

        if end_idx > 0:
            features_full = features_full[:end_idx]

        self.features = features_full
        self.seq_len = config.seq_len
        self.n_bars = len(self.features)
        self.augment = augment
        self.noise_std = noise_std

        if self.n_bars <= self.seq_len:
            raise ValueError(
                f"Not enough bars ({self.n_bars}) for seq_len={self.seq_len}."
            )

        self.feat_mean = self.features.mean(axis=0, keepdims=True)
        self.feat_std = self.features.std(axis=0, keepdims=True) + 1e-8

        labels = load_labels(label_path)

        phase_all = labels["phase"]
        phase_weight_all = labels["phase_weight"]
        event_all = labels["events"]
        event_weight_all = labels["event_weight"]

        if end_idx > 0:
            phase_all = phase_all[:end_idx]
            phase_weight_all = phase_weight_all[:end_idx]
            event_all = event_all[:end_idx]
            event_weight_all = event_weight_all[:end_idx]

        self.phase_labels = phase_all.astype(np.int64)
        self.phase_weight = phase_weight_all.astype(np.float32)
        self.event_labels = event_all.astype(np.float32)
        self.event_weight = event_weight_all.astype(np.float32)

        if self.event_labels.ndim != 2 or self.event_labels.shape[1] != config.n_events:
            raise ValueError(
                f"Label file has event shape {self.event_labels.shape}, "
                f"but config expects (*, {config.n_events}). Regenerate labels."
            )

        valid_phase = self.phase_labels[self.phase_labels != PHASE_IGNORE_INDEX]
        if len(valid_phase) > 0 and valid_phase.max() >= config.n_phases:
            raise ValueError(
                f"Label file has phase index {valid_phase.max()} but config expects "
                f"0..{config.n_phases - 1}. Regenerate labels."
            )

        self.valid_start = self.seq_len
        self.n_samples = self.n_bars - self.valid_start

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        bar_idx = self.valid_start + idx

        window = self.features[bar_idx - self.seq_len:bar_idx].copy()
        window = (window - self.feat_mean) / self.feat_std

        if self.augment:
            noise = np.random.randn(*window.shape).astype(np.float32) * self.noise_std
            window = window + noise

        return {
            "features": torch.from_numpy(window),
            "phase": torch.tensor(self.phase_labels[bar_idx], dtype=torch.long),
            "phase_weight": torch.tensor(self.phase_weight[bar_idx], dtype=torch.float32),
            "events": torch.from_numpy(self.event_labels[bar_idx]).float(),
            "event_weight": torch.from_numpy(self.event_weight[bar_idx]).float(),
        }


# ═══════════════════════════════════════════════════════════════════════
# Loss helpers
# ═══════════════════════════════════════════════════════════════════════

def weighted_phase_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: torch.Tensor,
    class_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    loss = F.cross_entropy(
        logits,
        targets,
        weight=class_weight,
        ignore_index=PHASE_IGNORE_INDEX,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    valid = (targets != PHASE_IGNORE_INDEX).float()
    w = sample_weight.float() * valid
    return (loss * w).sum() / w.sum().clamp_min(1.0)


def weighted_event_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_confidence: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
    focal_gamma: float = 1.5,
    negative_weight: float = 1.0,
) -> torch.Tensor:
    """
    Event loss where:
    - positive labels use positive_confidence
    - negative labels use weight=negative_weight
    """
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

    if positive_confidence is None:
        w = torch.ones_like(targets, dtype=logits.dtype)
    else:
        w = torch.where(
            targets > 0.5,
            positive_confidence.float().clamp_min(0.25),
            torch.full_like(targets, float(negative_weight)),
        )

    return (bce * w).sum() / w.sum().clamp_min(1.0)


def _print_label_summary(labels: dict):
    phase_arr = labels["phase"]
    valid_phase = phase_arr[phase_arr != PHASE_IGNORE_INDEX]
    if len(valid_phase) > 0:
        phase_counts = np.bincount(valid_phase, minlength=N_REGIMES)
        print(f"  Regime distribution (valid only): {dict(enumerate(phase_counts.tolist()))}")
    print(f"  Ignored phase bars: {(phase_arr == PHASE_IGNORE_INDEX).sum()}")

    event_counts = (labels["events"] > 0.5).sum(axis=0)
    print(f"  Event counts: {event_counts.astype(int).tolist()}")


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Supervised pre-training
# ═══════════════════════════════════════════════════════════════════════

def pretrain(
    npz_path: str,
    label_path: str | None = None,
    pretrained_path: str | None = None,
    config: TransformerConfig | None = None,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 4e-5,
    val_split: float = 0.15,
    save_dir: str = "checkpoints/transformer",
    device: str = "cuda",
    parquet_path: str | None = None,
    label_smoothing: float = 0.12,
    weight_decay: float = 0.08,
    patience: int = 20,
    noise_std: float = 0.03,
    train_end_bar: int = 0,
    warmup_epochs: int = 6,
):
    if config is None:
        config = TransformerConfig()

    os.makedirs(save_dir, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Resolve / generate labels
    if label_path is None:
        label_path = os.path.join(save_dir, "structural_labels.npz")

    if not os.path.exists(label_path):
        if parquet_path is None:
            parquet_path = npz_path.replace(".npz", "_bars.parquet")
        os.makedirs(os.path.dirname(label_path) or ".", exist_ok=True)

        print(f"Generating structural labels from {parquet_path}...")
        labels = generate_structural_labels(parquet_path, npz_path=npz_path)
        save_labels(labels, label_path)
        print(f"  Saved to {label_path}")
        _print_label_summary(labels)

    # Datasets
    train_full = WyckoffSequenceDataset(
        npz_path,
        label_path,
        config,
        augment=True,
        noise_std=noise_std,
        end_idx=train_end_bar,
    )
    val_full = WyckoffSequenceDataset(
        npz_path,
        label_path,
        config,
        augment=False,
        end_idx=train_end_bar,
    )

    if train_end_bar > 0:
        total_bars = np.load(npz_path, allow_pickle=True)["tech_ary"].shape[0]
        print(
            f"Holdout: training on bars 0-{train_end_bar}, "
            f"holding out {total_bars - train_end_bar} bars for test"
        )

    # Share train normalization with val
    val_full.feat_mean = train_full.feat_mean
    val_full.feat_std = train_full.feat_std

    n_total = len(train_full)
    if n_total < 2:
        raise ValueError(f"Not enough training samples: {n_total}")

    n_val = max(1, int(n_total * val_split))
    n_val = min(n_val, n_total - 1)
    n_train = n_total - n_val

    train_dataset = torch.utils.data.Subset(train_full, range(n_train))
    val_dataset = torch.utils.data.Subset(val_full, range(n_train, n_total))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Dataset: {n_total} samples (train={n_train}, val={n_val})")
    print(
        f"Config: seq_len={config.seq_len}, d_model={config.d_model}, "
        f"n_layers={config.n_layers}, n_heads={config.n_heads}"
    )
    print(
        f"Regularization: dropout={config.dropout}, weight_decay={weight_decay}, "
        f"label_smoothing={label_smoothing}, noise_std={noise_std}"
    )

    # Build model
    encoder = WyckoffTransformerEncoder(config).to(dev)
    phase_head = PhaseHead(config).to(dev)
    event_head = EventHead(config).to(dev)

    if pretrained_path and os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        phase_head.load_state_dict(ckpt["phase_head"])
        event_head.load_state_dict(ckpt["event_head"])
        print(
            f"Loaded pre-trained weights for supervised warm start from "
            f"{pretrained_path} (epoch={ckpt.get('epoch')}, "
            f"phase_acc={ckpt.get('phase_acc', 0):.3f})"
        )

    n_params = sum(p.numel() for p in encoder.parameters())
    n_params += sum(p.numel() for p in phase_head.parameters())
    n_params += sum(p.numel() for p in event_head.parameters())
    print(f"Total parameters: {n_params:,}")
    print(f"Params/sample ratio: {n_params / max(n_train, 1):.1f}")

    # Use only actual train target bars for class statistics
    target_start = train_full.valid_start
    target_end = target_start + n_train

    # Phase class weights
    phase_targets_for_weight = train_full.phase_labels[target_start:target_end]
    valid_phase_mask = phase_targets_for_weight != PHASE_IGNORE_INDEX

    if valid_phase_mask.any():
        phase_counts = np.bincount(
            phase_targets_for_weight[valid_phase_mask],
            minlength=N_REGIMES,
        ).astype(np.float32)
    else:
        phase_counts = np.ones(N_REGIMES, dtype=np.float32)

    phase_counts = np.maximum(phase_counts, 1.0)
    phase_weights = phase_counts.sum() / (N_REGIMES * phase_counts)
    phase_weights = phase_weights / phase_weights.mean()
    phase_weights_t = torch.tensor(phase_weights, dtype=torch.float32, device=dev)

    print(f"Regime weights: {dict(zip(range(N_REGIMES), [f'{w:.2f}' for w in phase_weights]))}")
    print(f"Valid phase samples: {int(valid_phase_mask.sum())} / {len(phase_targets_for_weight)}")

    # Event pos_weight -- IMPORTANT: count all negatives, not only weighted positives
    event_labels_train = train_full.event_labels[target_start:target_end]
    event_pos = (event_labels_train > 0.5).sum(axis=0).astype(np.float32)
    event_neg = event_labels_train.shape[0] - event_pos

    event_pos = np.maximum(event_pos, 1.0)
    event_neg = np.maximum(event_neg, 1.0)

    event_pw = np.clip(event_neg / event_pos, 1.0, 25.0)
    event_pos_weight = torch.tensor(event_pw, dtype=torch.float32, device=dev)

    print(f"Event pos_weight: {[f'{w:.1f}' for w in event_pw]}")

    all_params = list(encoder.parameters()) + list(phase_head.parameters()) + list(event_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch: int):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    checkpoint = None

    for epoch in range(epochs):
        # Train
        encoder.train()
        phase_head.train()
        event_head.train()

        train_loss = 0.0
        train_loss_phase = 0.0
        train_loss_event = 0.0
        n_batches = 0

        for batch in train_loader:
            feats = batch["features"].to(dev)
            phase_tgt = batch["phase"].to(dev)
            phase_wt = batch["phase_weight"].to(dev)
            event_tgt = batch["events"].to(dev)
            event_wt = batch["event_weight"].to(dev)

            latent = encoder(feats)
            last_latent = latent[:, -1, :]

            phase_logits = phase_head(last_latent)
            event_logits = event_head(last_latent)

            loss_phase = weighted_phase_loss(
                phase_logits,
                phase_tgt,
                phase_wt,
                class_weight=phase_weights_t,
                label_smoothing=label_smoothing,
            )
            loss_event = weighted_event_loss(
                event_logits,
                event_tgt,
                positive_confidence=event_wt,
                pos_weight=event_pos_weight,
                focal_gamma=1.5,
                negative_weight=1.0,
            )

            loss = (
                config.phase_loss_weight * loss_phase
                + config.event_loss_weight * loss_event
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_loss_phase += loss_phase.item()
            train_loss_event += loss_event.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)
        avg_train_phase = train_loss_phase / max(n_batches, 1)
        avg_train_event = train_loss_event / max(n_batches, 1)

        # Validate
        encoder.eval()
        phase_head.eval()
        event_head.eval()

        val_loss = 0.0
        val_loss_phase = 0.0
        val_loss_event = 0.0
        n_val_rows = 0
        phase_correct = 0
        n_val_phase = 0

        with torch.no_grad():
            for batch in val_loader:
                feats = batch["features"].to(dev)
                phase_tgt = batch["phase"].to(dev)
                phase_wt = batch["phase_weight"].to(dev)
                event_tgt = batch["events"].to(dev)
                event_wt = batch["event_weight"].to(dev)

                latent = encoder(feats)
                last_latent = latent[:, -1, :]

                phase_logits = phase_head(last_latent)
                event_logits = event_head(last_latent)

                loss_phase = weighted_phase_loss(
                    phase_logits,
                    phase_tgt,
                    phase_wt,
                    class_weight=phase_weights_t,
                    label_smoothing=label_smoothing,
                )
                loss_event = weighted_event_loss(
                    event_logits,
                    event_tgt,
                    positive_confidence=event_wt,
                    pos_weight=event_pos_weight,
                    focal_gamma=1.5,
                    negative_weight=1.0,
                )

                loss = (
                    config.phase_loss_weight * loss_phase
                    + config.event_loss_weight * loss_event
                )

                val_loss += loss.item() * feats.shape[0]
                val_loss_phase += loss_phase.item() * feats.shape[0]
                val_loss_event += loss_event.item() * feats.shape[0]
                n_val_rows += feats.shape[0]

                pred = phase_logits.argmax(dim=1)
                valid_phase = phase_tgt != PHASE_IGNORE_INDEX
                phase_correct += ((pred == phase_tgt) & valid_phase).sum().item()
                n_val_phase += valid_phase.sum().item()

        avg_val_loss = val_loss / max(n_val_rows, 1)
        avg_val_phase = val_loss_phase / max(n_val_rows, 1)
        avg_val_event = val_loss_event / max(n_val_rows, 1)
        phase_acc = phase_correct / max(n_val_phase, 1)

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} (ph={avg_train_phase:.4f} ev={avg_train_event:.4f}) | "
            f"val_loss={avg_val_loss:.4f} (ph={avg_val_phase:.4f} ev={avg_val_event:.4f}) | "
            f"phase_acc={phase_acc:.3f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            checkpoint = {
                "encoder": encoder.state_dict(),
                "phase_head": phase_head.state_dict(),
                "event_head": event_head.state_dict(),
                "config": config,
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "phase_acc": phase_acc,
                "feat_mean": train_full.feat_mean,
                "feat_std": train_full.feat_std,
            }
            torch.save(checkpoint, os.path.join(save_dir, "pretrained_best.pt"))
            print(f"  -> Saved best model (val_loss={avg_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"  x Early stopping: no improvement for {patience} epochs")
                break

    if checkpoint is None:
        checkpoint = {
            "encoder": encoder.state_dict(),
            "phase_head": phase_head.state_dict(),
            "event_head": event_head.state_dict(),
            "config": config,
            "epoch": epochs,
            "val_loss": best_val_loss,
            "phase_acc": 0.0,
            "feat_mean": train_full.feat_mean,
            "feat_std": train_full.feat_std,
        }

    torch.save(checkpoint, os.path.join(save_dir, "pretrained_final.pt"))
    print(
        f"\nPre-training complete. Best val_loss={best_val_loss:.4f} "
        f"(epoch {checkpoint.get('epoch', '?')})"
    )
    return os.path.join(save_dir, "pretrained_best.pt")


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: RL fine-tuning (PPO)
# ═══════════════════════════════════════════════════════════════════════

def load_pretrained(actor: ActorDiscreteTransformer, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor.encoder.load_state_dict(ckpt["encoder"])
    actor.phase_head.load_state_dict(ckpt["phase_head"])
    actor.event_head.load_state_dict(ckpt["event_head"])
    print(
        f"Loaded pre-trained encoder from {checkpoint_path} "
        f"(epoch={ckpt.get('epoch')}, phase_acc={ckpt.get('phase_acc', 0):.3f})"
    )


def rl_train(
    npz_path: str,
    pretrained_path: str | None = None,
    config: TransformerConfig | None = None,
    total_steps: int = 50_000_000,
    num_envs: int = 512,
    horizon_len: int = 128,
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    entropy_coeff: float = 0.05,
    target_entropy: float = 0.4,
    aux_loss_coeff: float = 0.1,
    encoder_lr_scale: float = 0.1,
    save_dir: str = "checkpoints/transformer_rl",
    gpu_id: int = 0,
    label_path: str | None = None,
    commission: float = 1.00,
    tick_size: float = 1.0,
    tick_value: float = 1.0,
    bar_range: float = 100.0,
    episode_len: int = 512,
    reward_mode: str = "dense_pnl",
    train_end_bar: int = 0,
    **env_kwargs,
):
    if config is None:
        config = TransformerConfig()

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Pull normalization stats from pretrained checkpoint if available
    feat_mean, feat_std = None, None
    if pretrained_path and os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        feat_mean = ckpt.get("feat_mean")
        feat_std = ckpt.get("feat_std")
        if feat_mean is not None and feat_std is not None:
            print("  Extracted feat_mean/feat_std from pretrained checkpoint")

    env_end_idx = train_end_bar if train_end_bar > 0 else 0
    if env_end_idx > 0:
        print(f"RL env: using bars 0-{env_end_idx} (holdout from bar {env_end_idx})")

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
        end_idx=env_end_idx,
        feat_mean=feat_mean,
        feat_std=feat_std,
        **env_kwargs,
    )

    actor = ActorDiscreteTransformer(config).to(device)
    critic = CriticTransformer(config).to(device)

    if pretrained_path and os.path.exists(pretrained_path):
        load_pretrained(actor, pretrained_path)
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        critic.encoder.load_state_dict(ckpt["encoder"])
        print("  Also loaded pre-trained encoder into critic")

    actor_encoder_params = list(actor.encoder.parameters())
    actor_head_params = (
        list(actor.policy_head.parameters())
        + list(actor.phase_head.parameters())
        + list(actor.event_head.parameters())
    )

    actor_optimizer = torch.optim.Adam(
        [
            {"params": actor_encoder_params, "lr": lr_actor * encoder_lr_scale},
            {"params": actor_head_params, "lr": lr_actor},
        ]
    )
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)

    n_actor = sum(p.numel() for p in actor.parameters())
    n_critic = sum(p.numel() for p in critic.parameters())
    print(f"Actor: {n_actor:,} params | Critic: {n_critic:,} params")
    print(
        f"Encoder LR: {lr_actor * encoder_lr_scale:.2e} | "
        f"Heads LR: {lr_actor:.2e} | Critic LR: {lr_critic:.2e}"
    )

    # Optional auxiliary labels
    aux_labels = None
    aux_phase_class_weight_t = None
    aux_event_pos_weight_t = None

    if label_path and os.path.exists(label_path):
        aux_labels = load_labels(label_path)
        aux_end = env_end_idx if env_end_idx > 0 else len(aux_labels["phase"])

        aux_phase = aux_labels["phase"][:aux_end]
        valid_phase = aux_phase != PHASE_IGNORE_INDEX
        if valid_phase.any():
            phase_counts = np.bincount(
                aux_phase[valid_phase],
                minlength=config.n_phases,
            ).astype(np.float32)
            phase_counts = np.maximum(phase_counts, 1.0)
            phase_w = phase_counts.sum() / (config.n_phases * phase_counts)
            phase_w = phase_w / phase_w.mean()
            aux_phase_class_weight_t = torch.tensor(phase_w, dtype=torch.float32, device=device)
        else:
            phase_w = None

        aux_events = aux_labels["events"][:aux_end].astype(np.float32)
        event_pos = (aux_events > 0.5).sum(axis=0).astype(np.float32)
        event_neg = aux_events.shape[0] - event_pos
        event_pos = np.maximum(event_pos, 1.0)
        event_neg = np.maximum(event_neg, 1.0)
        event_pw = np.clip(event_neg / event_pos, 1.0, 100.0)
        aux_event_pos_weight_t = torch.tensor(event_pw, dtype=torch.float32, device=device)

        print(f"Loaded auxiliary labels from {label_path}")
        if phase_w is not None:
            print(f"  Aux phase weights: {[f'{x:.2f}' for x in phase_w]}")
        print(f"  Aux event pos_weight: {[f'{x:.1f}' for x in event_pw]}")

    state, _ = env.reset()
    total_reward = torch.zeros(num_envs, device=device)
    episode_count = 0
    episode_pnl_sum = 0.0

    steps_done = 0
    n_updates = total_steps // (num_envs * horizon_len)

    print(
        f"\nRL Training: {total_steps:,} steps, {n_updates} updates, "
        f"{num_envs} envs x {horizon_len} horizon"
    )

    for update in range(n_updates):
        # Rollout buffers
        states_buf = torch.zeros(horizon_len, num_envs, env.state_dim, device=device)
        actions_buf = torch.zeros(horizon_len, num_envs, dtype=torch.long, device=device)
        logprobs_buf = torch.zeros(horizon_len, num_envs, device=device)
        rewards_buf = torch.zeros(horizon_len, num_envs, device=device)
        dones_buf = torch.zeros(horizon_len, num_envs, dtype=torch.bool, device=device)
        values_buf = torch.zeros(horizon_len, num_envs, device=device)
        bar_idx_buf = torch.zeros(horizon_len, num_envs, dtype=torch.long, device=device)

        actor.eval()
        critic.eval()

        with torch.no_grad():
            for t in range(horizon_len):
                states_buf[t] = state
                bar_idx_buf[t] = env.day.clone()

                action, logprob = actor.get_action(state)
                value = critic(state)

                next_state, reward, terminal, done, info = env.step(action)

                actions_buf[t] = action
                logprobs_buf[t] = logprob
                rewards_buf[t] = reward
                dones_buf[t] = done
                values_buf[t] = value

                total_reward += reward
                if done.any():
                    for i in torch.where(done)[0].tolist():
                        episode_count += 1
                        episode_pnl_sum += env.cumulative_returns[i]
                        total_reward[i] = 0.0

                state = next_state

            next_value = critic(state)

        steps_done += horizon_len * num_envs

        # GAE
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
            advantages[t] = lastgaelam = (
                delta + gamma * gae_lambda * next_non_terminal.float() * lastgaelam
            )

        returns = advantages + values_buf

        # Flatten
        b_states = states_buf.reshape(-1, env.state_dim)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_bar_idx = bar_idx_buf.reshape(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

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

                # Actor loss
                new_logprob, entropy = actor.get_logprob_entropy(mb_states, mb_actions)
                ratio = (new_logprob - mb_logprobs).exp()

                surr1 = ratio * mb_advantages
                surr2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                entropy_loss = -entropy.mean()
                cur_entropy = -entropy_loss.item()

                if cur_entropy < target_entropy:
                    ent_coeff = entropy_coeff * (
                        1.0 + 3.0 * (target_entropy - cur_entropy) / target_entropy
                    )
                else:
                    ent_coeff = entropy_coeff

                actor_loss = policy_loss + ent_coeff * entropy_loss

                # Auxiliary supervised losses
                if aux_labels is not None:
                    mb_bar = b_bar_idx[idx].detach().cpu().numpy()
                    mb_bar = np.clip(mb_bar, 0, len(aux_labels["phase"]) - 1)

                    phase_tgt = torch.tensor(
                        aux_labels["phase"][mb_bar],
                        dtype=torch.long,
                        device=device,
                    )
                    phase_wt = torch.tensor(
                        aux_labels["phase_weight"][mb_bar],
                        dtype=torch.float32,
                        device=device,
                    )

                    event_tgt = torch.tensor(
                        aux_labels["events"][mb_bar],
                        dtype=torch.float32,
                        device=device,
                    )
                    event_conf = torch.tensor(
                        aux_labels["event_weight"][mb_bar],
                        dtype=torch.float32,
                        device=device,
                    )

                    a_losses = actor.get_aux_losses(
                        mb_states,
                        phase_targets=phase_tgt,
                        event_targets=event_tgt,
                        phase_weight=phase_wt,
                        event_weight=event_conf,
                        phase_class_weight=aux_phase_class_weight_t,
                        event_pos_weight=aux_event_pos_weight_t,
                        phase_label_smoothing=0.0,
                        event_focal_gamma=1.5,
                        negative_event_weight=1.0,
                    )

                    if a_losses:
                        aux_total = torch.stack(list(a_losses.values())).sum()
                        actor_loss = actor_loss + aux_loss_coeff * aux_total

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

        if (update + 1) % 1 == 0 or update == 0:
            avg_pnl = episode_pnl_sum / max(episode_count, 1)
            print(
                f"Update {update + 1:4d}/{n_updates} | "
                f"steps={steps_done:,} | "
                f"episodes={episode_count} | "
                f"avg_pnl=${avg_pnl:.2f} | "
                f"policy_loss={policy_loss.item():.4f} | "
                f"value_loss={critic_loss.item():.4f} | "
                f"entropy={-entropy_loss.item():.4f} | "
                f"ent_c={ent_coeff:.4f}"
            )

        if (update + 1) % 25 == 0:
            ckpt_path = os.path.join(save_dir, f"rl_step{steps_done}.pt")
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "config": config,
                    "steps": steps_done,
                    "episodes": episode_count,
                    "avg_pnl": episode_pnl_sum / max(episode_count, 1),
                },
                ckpt_path,
            )
            print(f"  -> Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(save_dir, "rl_final.pt")
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "config": config,
            "steps": steps_done,
            "episodes": episode_count,
            "avg_pnl": episode_pnl_sum / max(episode_count, 1),
        },
        final_path,
    )
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pretrain-lr", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.12)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--warmup-epochs", type=int, default=6)

    # RL args
    parser.add_argument("--total-steps", type=int, default=50_000_000)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--horizon-len", type=int, default=128)
    parser.add_argument("--rl-lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coeff", type=float, default=0.05)
    parser.add_argument("--episode-len", type=int, default=512)
    parser.add_argument("--reward-mode", default="dense_pnl")

    # Instrument config
    parser.add_argument("--commission", type=float, default=1.00)
    parser.add_argument("--tick-size", type=float, default=1.0)
    parser.add_argument("--tick-value", type=float, default=1.0)
    parser.add_argument("--bar-range", type=float, default=100.0)

    # Data split
    parser.add_argument(
        "--train-end-bar",
        type=int,
        default=0,
        help="Bar index for train/test split (0=use all). "
             "Bars 0..N-1 for train+val, N..end for holdout test.",
    )

    # Architecture overrides
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=None, help="FF dim (default: 3*d_model)")
    parser.add_argument("--event-loss-weight", type=float, default=0.15)

    args = parser.parse_args()

    config = TransformerConfig(
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff if args.d_ff else args.d_model * 3,
        dropout=args.dropout,
        event_loss_weight=args.event_loss_weight,
    )

    resolved_label_path = args.label_path
    if resolved_label_path is None:
        auto_label_path = os.path.join(args.save_dir, "structural_labels.npz")
        if os.path.exists(auto_label_path):
            resolved_label_path = auto_label_path

    pretrained_path = args.pretrained_encoder

    if args.phase in ("pretrain", "both"):
        pretrained_path = pretrain(
            npz_path=args.npz_path,
            label_path=resolved_label_path,
            pretrained_path=args.pretrained_encoder,
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
            train_end_bar=args.train_end_bar,
            warmup_epochs=args.warmup_epochs,
        )

        if resolved_label_path is None:
            auto_label_path = os.path.join(args.save_dir, "structural_labels.npz")
            if os.path.exists(auto_label_path):
                resolved_label_path = auto_label_path

    if args.phase in ("rl", "both"):
        if pretrained_path is None:
            auto_ckpt = os.path.join(args.save_dir, "pretrained_best.pt")
            if os.path.exists(auto_ckpt):
                pretrained_path = auto_ckpt
                print(f"Auto-discovered pretrained checkpoint: {auto_ckpt}")

        if resolved_label_path is None:
            auto_label_path = os.path.join(args.save_dir, "structural_labels.npz")
            if os.path.exists(auto_label_path):
                resolved_label_path = auto_label_path

        rl_train(
            npz_path=args.npz_path,
            pretrained_path=pretrained_path,
            config=config,
            total_steps=args.total_steps,
            num_envs=args.num_envs,
            horizon_len=args.horizon_len,
            lr_actor=args.rl_lr,
            entropy_coeff=args.entropy_coeff,
            save_dir=os.path.join(args.save_dir, "rl"),
            gpu_id=args.gpu_id,
            label_path=resolved_label_path,
            commission=args.commission,
            tick_size=args.tick_size,
            tick_value=args.tick_value,
            bar_range=args.bar_range,
            episode_len=args.episode_len,
            reward_mode=args.reward_mode,
            train_end_bar=args.train_end_bar,
        )


if __name__ == "__main__":
    main()
