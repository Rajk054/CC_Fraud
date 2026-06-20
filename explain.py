"""
Explainability for flagged transactions.

Two complementary techniques are provided:

1. SHAP (SHapley Additive exPlanations) — model-agnostic, game-theory based.
   We use shap.GradientExplainer for neural net models and DeepExplainer as
   an alternative.  For the autoencoder the "score" to explain is the
   reconstruction error (a scalar per sample).

2. Integrated Gradients (Captum) — attribution method that integrates
   gradients along a straight-line path from a baseline to the input.
   More faithful to the model's actual gradient flow than SHAP for NNs.

Output:
   Each flagged transaction gets a dict:
   {
     "feature_attributions": {"feature_name": attribution_value, ...},
     "top_positive": [("feat", val), ...],   # features pushing toward fraud
     "top_negative": [("feat", val), ...],   # features pushing away from fraud
   }
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed. SHAP explanations unavailable.")

# Captum (Integrated Gradients)
try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    logger.warning("captum not installed. Integrated Gradients unavailable.")


# ── SHAP Explainer ────────────────────────────────────────────────────────────

class SHAPExplainer:
    """
    Gradient-based SHAP explainer for PyTorch models.

    Usage:
        explainer = SHAPExplainer(classifier_model, X_background, feature_names)
        attributions = explainer.explain(X_flagged)
    """

    def __init__(
        self,
        model: nn.Module,
        background: np.ndarray,       # Representative sample of legit txns
        feature_names: List[str],
        device: str = "cpu",
        n_background: int = 200,
    ) -> None:
        assert SHAP_AVAILABLE, "pip install shap"
        self.device = torch.device(device)
        self.feature_names = feature_names
        self.model = model.eval().to(self.device)

        # Subsample background for speed
        idx = np.random.choice(len(background), min(n_background, len(background)), replace=False)
        bg_t = torch.from_numpy(background[idx].astype(np.float32)).to(self.device)

        # Wrap model so it returns a scalar per sample (sigmoid output)
        class _Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                out = self.m(x)
                if out.dim() == 1:
                    return torch.sigmoid(out).unsqueeze(1)
                return torch.sigmoid(out)

        wrapper = _Wrapper(model).to(self.device)
        self.explainer = shap.GradientExplainer(wrapper, bg_t)

    def explain(
        self,
        X: np.ndarray,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Returns a list of attribution dicts, one per input row.
        """
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        shap_values = self.explainer.shap_values(X_t)  # (N, F) or [(N,F)]

        if isinstance(shap_values, list):
            sv = shap_values[0]  # positive class
        else:
            sv = shap_values

        if hasattr(sv, "numpy"):
            sv = sv.numpy()
        sv = np.array(sv).squeeze()

        if sv.ndim == 1:
            sv = sv[np.newaxis, :]

        results = []
        for i in range(len(sv)):
            attribs = dict(zip(self.feature_names, sv[i].tolist()))
            sorted_attribs = sorted(attribs.items(), key=lambda x: -abs(x[1]))
            results.append({
                "feature_attributions": attribs,
                "top_positive": [(k, v) for k, v in sorted_attribs if v > 0][:top_k],
                "top_negative": [(k, v) for k, v in sorted_attribs if v < 0][:top_k],
                "method": "SHAP-GradientExplainer",
            })
        return results


# ── Integrated Gradients Explainer ───────────────────────────────────────────

class IGExplainer:
    """
    Integrated Gradients explainer via Captum.

    Baseline: zeros (represents a "missing feature" baseline).
    For reconstruction error (autoencoder), we wrap the model to return
    a single scalar = mean reconstruction error per sample.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        device: str = "cpu",
        baseline: Optional[np.ndarray] = None,  # defaults to zeros
    ) -> None:
        assert CAPTUM_AVAILABLE, "pip install captum"
        self.device = torch.device(device)
        self.feature_names = feature_names
        self.model = model.eval().to(self.device)
        self.baseline = baseline

        # For models that return logits, wrap to ensure scalar output
        class _Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                out = self.m(x)
                # Handle (recon, z) tuple from autoencoder
                if isinstance(out, tuple):
                    recon, _ = out
                    return ((recon - x) ** 2).mean(dim=1, keepdim=True)
                if out.dim() == 1:
                    return torch.sigmoid(out).unsqueeze(1)
                return torch.sigmoid(out)

        self.wrapper = _Wrapper(model).to(self.device)
        self.ig = IntegratedGradients(self.wrapper)

    def explain(
        self,
        X: np.ndarray,
        n_steps: int = 50,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Returns attribution dicts with Integrated Gradients attributions.
        n_steps: number of interpolation steps (higher = more accurate).
        """
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        X_t.requires_grad_(True)

        if self.baseline is None:
            baseline = torch.zeros_like(X_t)
        else:
            baseline = torch.from_numpy(
                self.baseline.astype(np.float32)
            ).to(self.device)
            baseline = baseline.expand_as(X_t)

        attributions, delta = self.ig.attribute(
            X_t,
            baselines=baseline,
            n_steps=n_steps,
            return_convergence_delta=True,
        )
        attribs_np = attributions.detach().cpu().numpy()
        delta_np = delta.detach().cpu().numpy()

        results = []
        for i in range(len(attribs_np)):
            attribs = dict(zip(self.feature_names, attribs_np[i].tolist()))
            sorted_attribs = sorted(attribs.items(), key=lambda x: -abs(x[1]))
            results.append({
                "feature_attributions": attribs,
                "top_positive": [(k, v) for k, v in sorted_attribs if v > 0][:top_k],
                "top_negative": [(k, v) for k, v in sorted_attribs if v < 0][:top_k],
                "convergence_delta": float(delta_np[i]) if i < len(delta_np) else None,
                "method": "IntegratedGradients",
            })
        return results


# ── Combined Explanation ──────────────────────────────────────────────────────

def explain_flagged(
    model: nn.Module,
    X_flagged: np.ndarray,
    feature_names: List[str],
    background: np.ndarray,
    method: str = "ig",              # "shap" | "ig" | "both"
    device: str = "cpu",
) -> List[Dict]:
    """
    Convenience function — runs the appropriate explainer.

    Returns a list of explanation dicts, one per flagged transaction.
    Each dict has: feature_attributions, top_positive, top_negative, method.
    """
    if method == "shap" or method == "both":
        if not SHAP_AVAILABLE:
            raise ImportError("pip install shap")
        shap_exp = SHAPExplainer(model, background, feature_names, device)
        shap_results = shap_exp.explain(X_flagged)

    if method == "ig" or method == "both":
        if not CAPTUM_AVAILABLE:
            raise ImportError("pip install captum")
        ig_exp = IGExplainer(model, feature_names, device)
        ig_results = ig_exp.explain(X_flagged)

    if method == "both":
        # Merge: average attributions from both methods for robustness
        merged = []
        for s, g in zip(shap_results, ig_results):
            avg_attribs = {
                feat: (s["feature_attributions"][feat] + g["feature_attributions"][feat]) / 2
                for feat in feature_names
            }
            sorted_avg = sorted(avg_attribs.items(), key=lambda x: -abs(x[1]))
            merged.append({
                "feature_attributions": avg_attribs,
                "top_positive": [(k, v) for k, v in sorted_avg if v > 0][:10],
                "top_negative": [(k, v) for k, v in sorted_avg if v < 0][:10],
                "shap_detail": s,
                "ig_detail": g,
                "method": "SHAP+IG-average",
            })
        return merged

    return shap_results if method == "shap" else ig_results


def format_explanation_for_human(explanation: Dict, transaction_id: str) -> str:
    """Render an explanation as human-readable text for an analyst dashboard."""
    lines = [
        f"=== Explanation for Transaction {transaction_id} ===",
        f"Method: {explanation.get('method', 'unknown')}",
        "",
        "Top features INCREASING fraud risk:",
    ]
    for feat, val in explanation.get("top_positive", [])[:5]:
        lines.append(f"  ▲  {feat:<35} contribution = {val:+.4f}")

    lines.append("\nTop features DECREASING fraud risk:")
    for feat, val in explanation.get("top_negative", [])[:5]:
        lines.append(f"  ▼  {feat:<35} contribution = {val:+.4f}")

    return "\n".join(lines)
