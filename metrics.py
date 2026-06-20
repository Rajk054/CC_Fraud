"""
Evaluation suite: PR-AUC, cost-weighted metrics, and threshold optimisation.

Why PR-AUC instead of ROC-AUC?
  With severe class imbalance (0.3% fraud), ROC-AUC is misleading:
  a model that flags 5% of legit transactions as fraud can still show
  ROC-AUC > 0.95.  PR-AUC (area under the Precision-Recall curve) directly
  measures how many of our flags are real fraud at each recall level.
  A random classifier scores at the fraud rate (~0.003); good models >0.5.

Cost matrix (configurable):
  ┌─────────────────────────────────────────────────────────────────┐
  │                 Predicted Legit   Predicted Fraud               │
  │  Actual Legit       $0              $C_fp (friction)            │
  │  Actual Fraud      $C_fn (loss)     $0                          │
  └─────────────────────────────────────────────────────────────────┘

  C_fn = expected loss from undetected fraud (typically $200–$5,000)
  C_fp = customer friction cost: support calls, churn, interchange fees
         (typically $2–$15 per false decline)

  optimal_threshold() finds the threshold that minimises total cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ── Cost Matrix ──────────────────────────────────────────────────────────────

@dataclass
class CostMatrix:
    """
    Business cost parameters.

    Attributes:
        cost_fn:  Cost of a missed fraud (False Negative).
        cost_fp:  Cost of a false alarm (False Positive — customer friction).
        avg_tx_amount: Mean transaction amount, used to scale C_fn if None.
    """
    cost_fn: float = 400.0   # $400 per missed fraud
    cost_fp: float = 5.0     # $5 per false decline

    def total_cost(self, tn: int, fp: int, fn: int, tp: int) -> float:
        return fn * self.cost_fn + fp * self.cost_fp

    def normalized_cost(self, tn: int, fp: int, fn: int, tp: int) -> float:
        """Cost per transaction (makes costs comparable across dataset sizes)."""
        n = tn + fp + fn + tp
        return self.total_cost(tn, fp, fn, tp) / n if n > 0 else 0.0


# ── Threshold Optimisation ───────────────────────────────────────────────────

def optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    cost_matrix: CostMatrix,
    n_thresholds: int = 500,
) -> Tuple[float, float]:
    """
    Find the decision threshold that minimises total cost.

    Returns:
        (best_threshold, best_cost_per_transaction)
    """
    del n_thresholds  # retained for backward compatibility
    y_true = np.asarray(y_true, dtype=np.int64)
    y_scores = np.asarray(y_scores, dtype=float)
    if len(y_true) == 0:
        raise ValueError("Cannot optimise a threshold on an empty validation set.")

    order = np.argsort(-y_scores, kind="stable")
    scores_sorted = y_scores[order]
    labels_sorted = y_true[order]
    cum_tp = np.cumsum(labels_sorted)
    cum_fp = np.cumsum(1 - labels_sorted)
    group_ends = np.flatnonzero(np.r_[scores_sorted[:-1] != scores_sorted[1:], True])

    tp = cum_tp[group_ends]
    fp = cum_fp[group_ends]
    positives = int(labels_sorted.sum())
    fn = positives - tp
    costs = (fn * cost_matrix.cost_fn + fp * cost_matrix.cost_fp) / len(y_true)

    # Include the valid policy of flagging nothing.
    no_flag_cost = positives * cost_matrix.cost_fn / len(y_true)
    best_idx = int(np.argmin(costs))
    if no_flag_cost < costs[best_idx]:
        return float(np.nextafter(scores_sorted[0], np.inf)), float(no_flag_cost)
    return float(scores_sorted[group_ends[best_idx]]), float(costs[best_idx])


# ── Full Evaluation ──────────────────────────────────────────────────────────

@dataclass
class ModelEvaluation:
    """Holds all evaluation results for a single model."""
    model_name: str
    pr_auc: float
    roc_auc: float
    threshold: float
    precision: float
    recall: float
    f1: float
    total_cost: float
    cost_per_tx: float
    confusion: np.ndarray        # shape (2, 2)
    precisions: np.ndarray       # PR curve
    recalls: np.ndarray
    thresholds_curve: np.ndarray


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    cost_matrix: CostMatrix,
    threshold: Optional[float] = None,   # If None, use optimal threshold
) -> ModelEvaluation:
    """
    Run the full evaluation suite for one model.

    Args:
        model_name:   Human-readable name for reporting.
        y_true:       Ground truth binary labels.
        y_scores:     Continuous fraud probability / score (higher = more fraudulent).
        cost_matrix:  Cost parameters.
        threshold:    Fixed decision threshold.  If None, cost-optimised.

    Returns:
        ModelEvaluation dataclass.
    """
    pr_auc = average_precision_score(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    precisions, recalls, thresh_curve = precision_recall_curve(y_true, y_scores)

    # Threshold selection
    if threshold is None:
        threshold, _ = optimal_threshold(y_true, y_scores, cost_matrix)

    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    return ModelEvaluation(
        model_name=model_name,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        threshold=threshold,
        precision=prec,
        recall=rec,
        f1=f1,
        total_cost=cost_matrix.total_cost(tn, fp, fn, tp),
        cost_per_tx=cost_matrix.normalized_cost(tn, fp, fn, tp),
        confusion=np.array([[tn, fp], [fn, tp]]),
        precisions=precisions,
        recalls=recalls,
        thresholds_curve=thresh_curve,
    )


def compare_models(
    results: List[ModelEvaluation],
) -> None:
    """Pretty-print a comparison table."""
    header = f"{'Model':<22} {'PR-AUC':>8} {'ROC-AUC':>8} {'Prec':>7} {'Recall':>7} {'F1':>7} {'$/tx':>8} {'Threshold':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: -x.pr_auc):
        print(
            f"{r.model_name:<22} {r.pr_auc:>8.4f} {r.roc_auc:>8.4f} "
            f"{r.precision:>7.4f} {r.recall:>7.4f} {r.f1:>7.4f} "
            f"${r.cost_per_tx:>7.4f} {r.threshold:>10.4f}"
        )
    print("=" * len(header) + "\n")


# ── Ensemble ─────────────────────────────────────────────────────────────────

def ensemble_scores(
    score_dict: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Weighted average ensemble.

    Args:
        score_dict: {'autoencoder': scores_ae, 'classifier': scores_cls, 'gnn': scores_gnn}
        weights:    Per-model weights (defaults to equal weighting).
                    Should be set proportional to each model's val PR-AUC.

    Returns:
        Ensemble probability scores, shape (N,).
    """
    models = list(score_dict.keys())
    if weights is None:
        weights = {m: 1.0 / len(models) for m in models}

    total_weight = sum(weights[m] for m in models)
    ensemble = sum(
        weights[m] / total_weight * score_dict[m] for m in models
    )
    return ensemble


def normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise so all models output on the same [0,1] scale."""
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)
