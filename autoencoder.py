"""
Autoencoder-based anomaly detector.

Training philosophy:
  - Train ONLY on legitimate (isFraud==0) transactions.
  - The autoencoder learns a compact representation of "normal" behaviour.
  - At inference, reconstruction error (MSE per sample) is the anomaly score:
    fraudulent transactions deviate from the learned distribution and produce
    higher error.

Architecture:
  Input  →  [FC → BN → LeakyReLU → Dropout] × n  →  Bottleneck
         →  [FC → BN → LeakyReLU] × n             →  Reconstruction

Threshold selection:
  The decision threshold is set on a held-out normal set so that a target
  FPR (e.g. 2%) is achieved — or via cost-matrix optimisation (see metrics.py).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class FraudAutoencoder(nn.Module):
    """
    Symmetric autoencoder with configurable depth and bottleneck size.

    Args:
        input_dim:     Number of input features.
        hidden_dims:   Encoder hidden layer widths (decoder is mirrored).
                       e.g. [256, 128, 64]  →  bottleneck at 64,
                       decoder is [64, 128, 256].
        bottleneck:    Width of the compressed representation.
        dropout:       Dropout rate (applied in encoder only).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        bottleneck: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # ── Encoder ─────────────────────────────────────────────────────────
        enc_layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ]
            prev = h
        enc_layers += [nn.Linear(prev, bottleneck)]
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder ─────────────────────────────────────────────────────────
        dec_layers: List[nn.Module] = []
        prev = bottleneck
        for h in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.1),
            ]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, embedding)."""
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error (used as anomaly score)."""
        recon, _ = self.forward(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=1)


# ── Training ─────────────────────────────────────────────────────────────────

def train_autoencoder(
    model: FraudAutoencoder,
    X_train: np.ndarray,           # legitimate transactions ONLY
    X_val: np.ndarray,             # legitimate + fraud for threshold selection
    y_val: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 7,
    device: str = "cpu",
) -> dict:
    """
    Train the autoencoder on legitimate transactions.

    Returns a dict with:
      - 'train_losses': list of per-epoch train loss
      - 'val_losses':   list of per-epoch val recon loss (legit only)
      - 'best_epoch':   epoch at which val loss was lowest
    """
    device = torch.device(device)
    model.to(device)

    X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
    X_val_t   = torch.from_numpy(X_val.astype(np.float32)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    train_losses, val_losses = [], []
    best_val, best_state, stale = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        perm = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(X_train_t), batch_size):
            idx = perm[start : start + batch_size]
            xb = X_train_t[idx]

            # BatchNorm cannot estimate variance from a singleton batch.
            if len(xb) < 2:
                continue

            optimizer.zero_grad()
            recon, _ = model(xb)
            loss = F.mse_loss(recon, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)

        # ── Validate (on legit val only) ──────────────────────────────────
        model.eval()
        with torch.no_grad():
            legit_mask = y_val == 0
            X_val_legit = X_val_t[torch.from_numpy(legit_mask)]
            recon_val, _ = model(X_val_legit)
            val_loss = F.mse_loss(recon_val, X_val_legit).item()

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        if epoch % 5 == 0:
            logger.info(
                f"Epoch {epoch:03d}/{epochs}  "
                f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}"
            )

        if stale >= patience:
            logger.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
    }


def compute_threshold(
    model: FraudAutoencoder,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_fpr: float = 0.02,
    device: str = "cpu",
) -> float:
    """
    Set reconstruction-error threshold so FPR ≤ target_fpr on the val set.
    This keeps false alarms at an acceptable level while we optimise recall.
    """
    device = torch.device(device)
    model.eval().to(device)
    X_t = torch.from_numpy(X_val.astype(np.float32)).to(device)

    with torch.no_grad():
        errors = model.reconstruction_error(X_t).cpu().numpy()

    # Compute threshold on LEGITIMATE transactions only
    legit_errors = errors[y_val == 0]
    threshold = float(np.quantile(legit_errors, 1.0 - target_fpr))
    logger.info(f"Autoencoder threshold (FPR≤{target_fpr:.0%}): {threshold:.6f}")
    return threshold


def predict_autoencoder(
    model: FraudAutoencoder,
    X: np.ndarray,
    threshold: float,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (anomaly_scores, binary_predictions).
    anomaly_scores can be used as continuous fraud probabilities for ranking.
    """
    device = torch.device(device)
    model.eval().to(device)
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)

    with torch.no_grad():
        scores = model.reconstruction_error(X_t).cpu().numpy()

    preds = (scores > threshold).astype(int)
    return scores, preds
