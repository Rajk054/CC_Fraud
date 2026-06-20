"""
Focal Loss binary classifier for fraud detection.

Why Focal Loss?
  Standard BCE collapses when negatives outnumber positives 500:1 — the model
  can minimise loss by always predicting "legit."  Focal Loss down-weights
  easy negatives so the optimiser concentrates on hard/minority examples.

  FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

  γ (focusing parameter): higher γ → stronger down-weighting of easy examples.
    γ=0 recovers weighted BCE; typical range 0.5–5; default 2 works well.
  α (class weight for positive class): usually set to 1 − (prior fraud rate).

Architecture:
  Residual MLP with skip connections for training stability on deep nets.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary Focal Loss.

    Args:
        alpha: Weight for the positive class.  If None, computed from
               class frequencies passed to the forward() call.
        gamma: Focusing parameter (≥ 0).  γ=2 is the standard default.
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  Raw model outputs (pre-sigmoid), shape (N,)
            targets: Binary labels {0, 1}, shape (N,)
        """
        targets = targets.float()

        # Binary cross-entropy (numerically stable via logits form)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t: probability assigned to the TRUE class
        p_t = torch.exp(-bce)  # = sigmoid(logit) when target=1, else 1-sigmoid

        # α_t weighting
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        # Focal weight
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ── Residual Block ────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Two-layer residual block with BN and GELU activation."""

    def __init__(self, dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


# ── Classifier ───────────────────────────────────────────────────────────────

class FraudClassifier(nn.Module):
    """
    Residual MLP binary classifier.

    Architecture:
      input_dim → stem → [ResidualBlock × n_res_blocks] → head → logit(1)

    Args:
        input_dim:    Number of input features.
        hidden_dim:   Width of the stem and residual blocks.
        n_res_blocks: Number of residual blocks.
        dropout:      Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_res_blocks: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Stem: project to hidden_dim
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_res_blocks)]
        )

        # Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logit, shape (N,)."""
        h = self.stem(x)
        h = self.res_blocks(h)
        return self.head(h).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns fraud probability in [0, 1], shape (N,)."""
        return torch.sigmoid(self.forward(x))


# ── Training ─────────────────────────────────────────────────────────────────

def train_classifier(
    model: FraudClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 60,
    batch_size: int = 512,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    focal_alpha: Optional[float] = None,   # If None, auto-computed
    focal_gamma: float = 2.0,
    patience: int = 10,
    device: str = "cpu",
) -> dict:
    """
    Train the focal-loss classifier.

    focal_alpha defaults to (1 - fraud_rate), which upweights the minority class.
    """
    device = torch.device(device)
    model.to(device)

    if focal_alpha is None:
        fraud_rate = y_train.mean()
        focal_alpha = float(1.0 - fraud_rate)
        logger.info(f"Auto focal_alpha = {focal_alpha:.4f} (fraud rate = {fraud_rate:.4%})")

    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    X_tr = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_tr = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_vl = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_vl = torch.from_numpy(y_val.astype(np.float32)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 10,
        steps_per_epoch=max(1, int(np.ceil(len(X_tr) / batch_size))),
        epochs=epochs,
    )

    train_losses, val_losses = [], []
    best_val, best_state, stale = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        perm = torch.randperm(len(X_tr))
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, len(X_tr), batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]

            if len(xb) < 2:
                continue

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_logits = model(X_vl)
            val_loss = criterion(val_logits, y_vl).item()

        train_losses.append(epoch_loss / max(n_batches, 1))
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch:03d}/{epochs}  "
                f"train_loss={epoch_loss/max(n_batches,1):.5f}  "
                f"val_loss={val_loss:.5f}"
            )

        if stale >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
    }


def predict_classifier(
    model: FraudClassifier,
    X: np.ndarray,
    threshold: float = 0.5,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (fraud_probabilities, binary_predictions)."""
    device = torch.device(device)
    model.eval().to(device)
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)

    with torch.no_grad():
        probs = model.predict_proba(X_t).cpu().numpy()

    preds = (probs >= threshold).astype(int)
    return probs, preds
