"""
Graph construction for the GNN model.

Strategy — shared-entity edges:
  Two transactions share an edge if they have at least one entity value in common
  (same card1, same device, same email domain, etc.).

This graph structure lets the GNN propagate information across fraud rings:
  if transaction A is fraudulent and shares a card with B, the GNN can learn
  to up-weight B even without direct labels on B.

We build:
  - node_features : (N, F) float32 tensor  — one row per transaction
  - edge_index    : (2, E) long tensor      — COO format adjacency
  - edge_attr     : (E, K) float32 tensor  — which entity types are shared
  - y             : (N,)  long tensor       — fraud label

SCALABILITY NOTE
For very large datasets the naive edge enumeration below is O(E) but can
produce O(N²) edges for a single high-cardinality entity (e.g. a shared
email domain used by millions).  We cap per-entity-value fan-out at
MAX_DEGREE neighbours (sampled randomly); the GNN's neighbourhood sampling
then further limits the actual computation graph at training time.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger


MAX_DEGREE = 50  # Max edges per shared-entity value to avoid super-nodes


def build_transaction_graph(
    df: pd.DataFrame,
    feature_cols: List[str],
    entity_cols: List[str],
    label_col: str,
    max_degree: int = MAX_DEGREE,
    rng_seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a PyG-compatible graph from a transaction DataFrame.

    Args:
        df:            DataFrame (already feature-engineered, sorted by time).
        feature_cols:  Which columns to use as node features.
        entity_cols:   Columns that define shared-entity edges.
        label_col:     Binary fraud label column.
        max_degree:    Maximum edges per entity-value bucket.

    Returns:
        x          : (N, F)   node feature tensor
        edge_index : (2, E)   causal edge tensor (earlier transaction -> later)
        edge_attr  : (E, K)   one-hot edge type tensor (K = len(entity_cols))
        y          : (N,)     label tensor
    """
    # Retained in the signature for reproducible future sampling strategies.
    del rng_seed
    df = df.reset_index(drop=True)
    N = len(df)

    # ── Node features ────────────────────────────────────────────────────────
    feat_matrix = df[feature_cols].values.astype(np.float32)
    # Replace any remaining NaN / Inf
    feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.from_numpy(feat_matrix)

    # ── Labels ───────────────────────────────────────────────────────────────
    y = torch.from_numpy(df[label_col].values.astype(np.int64))

    # ── Edge construction ────────────────────────────────────────────────────
    src_list, dst_list, attr_list = [], [], []
    K = len(entity_cols)

    for k, entity in enumerate(entity_cols):
        if entity not in df.columns:
            logger.warning(f"Entity column '{entity}' not in DataFrame — skipping.")
            continue

        entity_type_vec = np.zeros(K, dtype=np.float32)
        entity_type_vec[k] = 1.0

        # Group transaction indices by entity value
        for entity_val, grp_idx in df.groupby(entity).groups.items():
            idx = grp_idx.to_numpy()
            if len(idx) < 2:
                continue
            # Connect every transaction to at most max_degree earlier peers.
            # This preserves all nodes while keeping edge growth O(N * degree).
            src_parts, dst_parts = [], []
            for pos in range(1, len(idx)):
                prior = idx[max(0, pos - max_degree):pos]
                src_parts.append(prior)
                dst_parts.append(np.full(len(prior), idx[pos], dtype=idx.dtype))
            src = np.concatenate(src_parts)
            dst = np.concatenate(dst_parts)

            src_list.append(src)
            dst_list.append(dst)
            attr_list.append(
                np.tile(entity_type_vec, (len(src), 1))
            )

    if not src_list:
        # No edges — return empty tensors (model degrades to MLP)
        logger.warning("No graph edges found. Check entity_cols.")
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, K), dtype=torch.float)
    else:
        src_all = np.concatenate(src_list)
        dst_all = np.concatenate(dst_list)
        attrs_all = np.concatenate(attr_list)

        # Rows are time-ordered, so edges flow from past to future. This avoids
        # training nodes aggregating information from validation/test nodes.
        pairs = np.stack([src_all, dst_all], axis=1)
        unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
        attrs_merged = np.zeros((len(unique_pairs), K), dtype=np.float32)
        np.add.at(attrs_merged, inverse, attrs_all)
        attrs_merged = np.clip(attrs_merged, 0.0, 1.0)

        edge_index = torch.from_numpy(
            unique_pairs.T.astype(np.int64)
        )
        edge_attr = torch.from_numpy(attrs_merged)

    logger.info(
        f"Graph: {N:,} nodes | {edge_index.shape[1]:,} edges "
        f"| {x.shape[1]} node features | {K} edge types"
    )
    return x, edge_index, edge_attr, y


def split_graph_masks(
    y: torch.Tensor,
    train_frac: float = 0.75,
    val_frac: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return boolean masks for train/val/test on the node tensor.
    Uses a temporal (first-N) split — matches the tabular split.
    """
    N = len(y)
    train_end = int(N * train_frac)
    val_end = int(N * (train_frac + val_frac))

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)

    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True

    return train_mask, val_mask, test_mask
