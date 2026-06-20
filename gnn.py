"""
Graph Neural Network for fraud ring detection using PyTorch Geometric.

Why GNNs catch what point-wise models miss:
  A fraud ring may involve many accounts, cards, and devices that look
  individually normal but form suspicious cliques in the transaction graph.
  GNNs propagate messages across edges, so node j's representation is
  informed by its fraudulent neighbour i even before j is labeled.

Architecture: GraphSAGE
  - Chosen over GCN because SAGEConv supports inductive learning — we can
    embed new transactions at inference without retraining.
  - 3 SAGE layers with skip connections and BatchNorm.
  - Edge attributes (which entity type connects two nodes) are incorporated
    via gated aggregation.

Usage pattern:
  1.  build_transaction_graph()  →  (x, edge_index, edge_attr, y)
  2.  FraudGNN(input_dim=…)
  3.  train_gnn(…)
  4.  predict_gnn(…)

MEMORY NOTE
Full-batch training on large graphs does not fit in GPU memory.
We use PyG's NeighborLoader for mini-batch sampling — each mini-batch
is a subgraph of ≤num_neighbors neighbours per node per layer.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# PyTorch Geometric imports (install: pip install torch-geometric)
try:
    from torch_geometric.nn import SAGEConv, BatchNorm
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logger.warning(
        "torch-geometric not installed.  GNN model unavailable.  "
        "Install with: pip install torch-geometric torch-scatter torch-sparse"
    )


# ── Model ────────────────────────────────────────────────────────────────────

class FraudGNN(nn.Module):
    """
    GraphSAGE-based node classifier for fraud detection.

    Args:
        input_dim:   Node feature dimension (F).
        hidden_dim:  Hidden layer width.
        n_layers:    Number of SAGE convolution layers.
        dropout:     Dropout rate applied to hidden representations.
        edge_dim:    Edge attribute dimension K (from graph_builder).
                     If > 0, a linear gate projects edge features and
                     multiplies them into the aggregated messages.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.3,
        edge_dim: int = 0,
    ) -> None:
        super().__init__()
        assert PYG_AVAILABLE, "torch-geometric must be installed to use FraudGNN."

        self.n_layers = n_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # SAGEConv layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Optional edge-feature gate
        self.edge_gate: Optional[nn.Linear] = None
        if edge_dim > 0:
            self.edge_gate = nn.Linear(edge_dim, hidden_dim, bias=False)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns raw logits, shape (N,)."""
        h = self.input_proj(x)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h_new = conv(h, edge_index)

            # Optional: modulate messages by edge type
            if self.edge_gate is not None and edge_attr is not None:
                # Aggregate edge gate signals per node (destination)
                gate = torch.sigmoid(self.edge_gate(edge_attr))  # (E, hidden)
                # Mean-pool edge gates into destination nodes
                dst = edge_index[1]                              # (E,)
                node_gate = torch.zeros_like(h_new)
                node_gate.scatter_reduce_(0, dst.unsqueeze(1).expand_as(gate),
                                          gate, reduce="mean", include_self=False)
                h_new = h_new * (1.0 + node_gate)

            h_new = bn(h_new)
            h_new = F.gelu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # residual

        return self.head(h).squeeze(-1)

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(x, edge_index, edge_attr))


# ── Training ─────────────────────────────────────────────────────────────────

def train_gnn(
    model: "FraudGNN",
    data: "Data",                      # PyG Data with x, edge_index, edge_attr, y, train_mask, val_mask
    *,
    epochs: int = 100,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    focal_alpha: Optional[float] = None,
    focal_gamma: float = 2.0,
    num_neighbors: List[int] = [15, 10, 5],  # per-layer neighbour samples
    batch_size: int = 512,
    patience: int = 15,
    device: str = "cpu",
) -> dict:
    """
    Mini-batch training via NeighborLoader.

    focal_alpha and focal_gamma mirror the tabular classifier for consistency.
    """
    assert PYG_AVAILABLE, "torch-geometric not installed."

    device = torch.device(device)
    model.to(device)
    # Class weight for focal loss
    from classifier import FocalLoss
    if focal_alpha is None:
        train_rate = float(data.y[data.train_mask].float().mean())
        focal_alpha = 1.0 - train_rate
        logger.info(f"GNN focal_alpha={focal_alpha:.4f} (fraud rate={train_rate:.4%})")
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # NeighborLoader: samples subgraph around seed nodes for scalable training
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=data.train_mask,
        shuffle=True,
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size * 2,
        input_nodes=data.val_mask,
        shuffle=False,
    )

    train_losses, val_losses = [], []
    best_val, best_state, stale = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        epoch_loss, n_batches = 0.0, 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch.x, batch.edge_index,
                           getattr(batch, "edge_attr", None))
            # Only compute loss on seed nodes (first batch_size nodes)
            n_seed = batch.batch_size
            loss = criterion(logits[:n_seed], batch.y[:n_seed].float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index,
                               getattr(batch, "edge_attr", None))
                n_seed = batch.batch_size
                loss = criterion(logits[:n_seed], batch.y[:n_seed].float())
                val_loss += loss.item()
                val_batches += 1

        epoch_val = val_loss / max(val_batches, 1)
        train_losses.append(epoch_loss / max(n_batches, 1))
        val_losses.append(epoch_val)

        if epoch_val < best_val:
            best_val = epoch_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        if epoch % 10 == 0:
            logger.info(
                f"GNN Epoch {epoch:03d}/{epochs}  "
                f"train_loss={epoch_loss/max(n_batches,1):.5f}  "
                f"val_loss={epoch_val:.5f}"
            )

        if stale >= patience:
            logger.info(f"GNN early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    return {"train_losses": train_losses, "val_losses": val_losses, "best_epoch": best_epoch}


def predict_gnn(
    model: "FraudGNN",
    data: "Data",
    mask: torch.Tensor,
    *,
    threshold: float = 0.5,
    num_neighbors: List[int] = [15, 10, 5],
    batch_size: int = 1024,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (fraud_probabilities, binary_predictions) for nodes in mask."""
    assert PYG_AVAILABLE
    device = torch.device(device)
    model.eval().to(device)
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=mask,
        shuffle=False,
    )

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probs = model.predict_proba(batch.x, batch.edge_index,
                                        getattr(batch, "edge_attr", None))
            all_probs.append(probs[: batch.batch_size].cpu().numpy())

    probs_np = np.concatenate(all_probs)
    preds_np = (probs_np >= threshold).astype(int)
    return probs_np, preds_np
