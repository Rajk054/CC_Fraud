"""
FastAPI scoring service with:
  - Live transaction scoring (ensemble of all three models)
  - Integrated Gradients / SHAP explanations for flagged transactions
  - Drift monitoring via Population Stability Index (PSI)
  - Simulated transaction stream endpoint for demos

Run:
    uvicorn src.api.app:app --reload --port 8000

Endpoints:
    POST /score              — Score a single transaction
    POST /score/batch        — Score a batch
    GET  /stream             — SSE stream of simulated transactions
    GET  /drift/report       — Current feature drift report
    GET  /health             — Health check
"""

from __future__ import annotations

import asyncio
import json
import time
import threading
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler

# Project imports (adjust paths as needed)
from autoencoder import FraudAutoencoder, predict_autoencoder
from classifier import FraudClassifier, predict_classifier
from metrics import CostMatrix
from explain import explain_flagged


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DRIFT_WINDOW = 1000                  # transactions before drift is re-computed
PSI_ALERT_THRESHOLD = 0.20          # PSI > 0.2 = significant drift
COST_MATRIX = CostMatrix(cost_fn=400.0, cost_fp=5.0)
DEFAULT_THRESHOLD = 0.40            # fallback before trained metadata is available


# ── Application State ─────────────────────────────────────────────────────────

class AppState:
    """Mutable state shared across requests."""

    def __init__(self):
        self.autoencoder: Optional[FraudAutoencoder] = None
        self.classifier: Optional[FraudClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.medians: Optional[np.ndarray] = None
        self.lower: Optional[np.ndarray] = None
        self.upper: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.ae_threshold: float = 0.5
        self.ae_score_low: float = 0.0
        self.ae_score_high: float = 1.0
        self.ensemble_weights: Dict[str, float] = {"autoencoder": 0.5, "classifier": 0.5}
        self.deployable_mode: str = "ensemble"
        self.threshold: float = DEFAULT_THRESHOLD
        self.ae_background: Optional[np.ndarray] = None

        # Drift monitoring
        self.reference_distribution: Optional[np.ndarray] = None  # (N_ref, F)
        self.live_window: deque = deque(maxlen=DRIFT_WINDOW)
        self.drift_scores: Dict[str, float] = {}
        self._lock = threading.Lock()

        # Score history for simulated stream
        self.score_history: deque = deque(maxlen=500)


state = AppState()


# ── Drift Monitoring ──────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index for a single feature.

    PSI < 0.10 : no significant shift
    PSI < 0.20 : moderate shift — monitor
    PSI >= 0.20 : significant shift — alert / retrain
    """
    eps = 1e-6
    # Use quantiles of expected to define bins
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] -= eps
    bins[-1] += eps

    exp_hist, _ = np.histogram(expected, bins=bins)
    act_hist, _ = np.histogram(actual, bins=bins)

    exp_pct = exp_hist / (exp_hist.sum() + eps)
    act_pct = act_hist / (act_hist.sum() + eps)

    psi = np.sum((act_pct - exp_pct) * np.log((act_pct + eps) / (exp_pct + eps)))
    return float(psi)


def update_drift_report() -> Dict[str, float]:
    """Compute PSI for each feature in the live window vs reference."""
    with state._lock:
        if state.reference_distribution is None or len(state.live_window) < 100:
            return {}
        live = np.array(list(state.live_window))  # (W, F)

    drift = {}
    for i, fname in enumerate(state.feature_names):
        ref_col = state.reference_distribution[:, i]
        live_col = live[:, i]
        drift[fname] = compute_psi(ref_col, live_col)

    state.drift_scores = drift
    n_alerts = sum(1 for v in drift.values() if v > PSI_ALERT_THRESHOLD)
    if n_alerts > 0:
        logger.warning(f"DRIFT ALERT: {n_alerts} features exceed PSI={PSI_ALERT_THRESHOLD}")
    return drift


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    """
    Incoming transaction.  Fields match IEEE-CIS column names after feature
    engineering; extend with velocity features from the feature pipeline.
    Clients can POST either raw fields (if a feature-engineering sidecar
    computes velocities) or pre-computed feature vectors.
    """
    transaction_id: str
    features: Dict[str, float] = Field(
        ..., description="Key-value map of raw engineered feature values."
    )
    amount: float = Field(default=0.0, description="Original transaction amount (for cost calc).")


class ExplanationResponse(BaseModel):
    method: str
    top_positive: List[List[Any]]
    top_negative: List[List[Any]]


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    flagged: bool
    selected_model: str
    threshold: float
    model_scores: Dict[str, float]
    explanation: Optional[ExplanationResponse] = None
    drift_alert: bool
    latency_ms: float


class BatchScoreRequest(BaseModel):
    transactions: List[TransactionRequest]


class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse]
    batch_size: int
    total_latency_ms: float
    flagged_count: int


class DriftReport(BaseModel):
    feature_psi: Dict[str, float]
    n_alerts: int
    alert_features: List[str]
    window_size: int
    alert_threshold: float


# ── App Lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup."""
    logger.info("Loading models …")
    try:
        _load_models()
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Model loading failed: {e}. Service will return 503 until models are available.")
    yield
    logger.info("Shutting down.")


def _load_models() -> None:
    """Load serialised models from MODEL_DIR."""
    meta_path = MODEL_DIR / "metadata.json"
    if not meta_path.exists():
        logger.warning(f"No metadata.json in {MODEL_DIR}. Run train_all.py first.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    state.feature_names = meta["feature_names"]
    state.ae_threshold = meta.get("ae_threshold", 0.5)
    state.ae_score_low = meta.get("ae_score_low", 0.0)
    state.ae_score_high = meta.get("ae_score_high", 1.0)
    state.ensemble_weights = meta.get(
        "deployable_ensemble_weights", {"autoencoder": 0.5, "classifier": 0.5}
    )
    state.deployable_mode = meta.get("deployable_mode", "ensemble")
    state.threshold = meta.get("deployable_threshold", DEFAULT_THRESHOLD)
    F = len(state.feature_names)

    # Autoencoder
    ae = FraudAutoencoder(input_dim=F, **meta.get("ae_config", {}))
    ae.load_state_dict(torch.load(MODEL_DIR / "autoencoder.pt", map_location=DEVICE))
    state.autoencoder = ae.eval()

    # Classifier
    clf = FraudClassifier(input_dim=F, **meta.get("clf_config", {}))
    clf.load_state_dict(torch.load(MODEL_DIR / "classifier.pt", map_location=DEVICE))
    state.classifier = clf.eval()

    # Scaler
    import joblib
    state.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    prep_path = MODEL_DIR / "preprocessor.pkl"
    if prep_path.exists():
        prep = joblib.load(prep_path)
        state.medians = np.asarray(prep["medians"], dtype=np.float32)
        state.lower = np.asarray(prep["lower"], dtype=np.float32)
        state.upper = np.asarray(prep["upper"], dtype=np.float32)

    # Reference distribution for drift (stored as numpy array)
    ref_path = MODEL_DIR / "reference_distribution.npy"
    if ref_path.exists():
        state.reference_distribution = np.load(ref_path)

    # SHAP background
    bg_path = MODEL_DIR / "shap_background.npy"
    if bg_path.exists():
        state.ae_background = np.load(bg_path)


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring with autoencoder + focal-loss classifier ensemble.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Scoring Logic ─────────────────────────────────────────────────────────────

def _build_feature_vector(tx: TransactionRequest) -> np.ndarray:
    """Convert request dict → ordered numpy feature vector."""
    defaults = state.medians if state.medians is not None else np.zeros(len(state.feature_names))
    vec = np.array(
        [tx.features.get(f, defaults[i]) for i, f in enumerate(state.feature_names)],
        dtype=np.float32,
    )
    invalid = ~np.isfinite(vec)
    vec[invalid] = defaults[invalid]
    if state.lower is not None and state.upper is not None:
        vec = np.clip(vec, state.lower, state.upper)
    return vec


def _score_single(
    tx: TransactionRequest,
    explain: bool = False,
) -> ScoreResponse:
    t0 = time.perf_counter()

    if state.classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    # Feature vector
    raw_vec = _build_feature_vector(tx)
    X = raw_vec[np.newaxis, :]  # (1, F)

    if state.scaler is not None:
        X = state.scaler.transform(X)

    # Autoencoder score
    ae_scores, _ = predict_autoencoder(
        state.autoencoder, X, state.ae_threshold, device=DEVICE
    )
    span = max(state.ae_score_high - state.ae_score_low, 1e-12)
    ae_score_norm = float(np.clip((ae_scores[0] - state.ae_score_low) / span, 0.0, 1.0))

    # Classifier score
    clf_probs, _ = predict_classifier(state.classifier, X, device=DEVICE)
    clf_prob = float(clf_probs[0])

    # Ensemble (equal weight without GNN; adjust if GNN available)
    wa = state.ensemble_weights.get("autoencoder", 0.5)
    wc = state.ensemble_weights.get("classifier", 0.5)
    ensemble = (wa * ae_score_norm + wc * clf_prob) / max(wa + wc, 1e-12)

    selected_score = {
        "autoencoder": ae_score_norm,
        "classifier": clf_prob,
        "ensemble": ensemble,
    }.get(state.deployable_mode, ensemble)
    flagged = selected_score >= state.threshold

    # Update live window for drift monitoring
    with state._lock:
        state.live_window.append(raw_vec.copy())

    # Periodically refresh drift report
    if len(state.live_window) % 100 == 0:
        asyncio.get_event_loop().run_in_executor(None, update_drift_report)

    any_drift = any(v > PSI_ALERT_THRESHOLD for v in state.drift_scores.values())

    # Explanation (only for flagged, on demand)
    explanation_resp = None
    if explain and flagged and state.ae_background is not None:
        try:
            exp = explain_flagged(
                state.autoencoder if state.deployable_mode == "autoencoder" else state.classifier,
                X, state.feature_names,
                state.ae_background, method="ig", device=DEVICE
            )
            e = exp[0]
            explanation_resp = ExplanationResponse(
                method=e["method"],
                top_positive=[[k, round(v, 5)] for k, v in e["top_positive"]],
                top_negative=[[k, round(v, 5)] for k, v in e["top_negative"]],
            )
        except Exception as ex:
            logger.warning(f"Explanation failed: {ex}")

    latency_ms = (time.perf_counter() - t0) * 1000

    result = ScoreResponse(
        transaction_id=tx.transaction_id,
        fraud_probability=round(selected_score, 6),
        flagged=flagged,
        selected_model=state.deployable_mode,
        threshold=state.threshold,
        model_scores={
            "autoencoder": round(ae_score_norm, 6),
            "classifier": round(clf_prob, 6),
            "ensemble": round(ensemble, 6),
        },
        explanation=explanation_resp,
        drift_alert=any_drift,
        latency_ms=round(latency_ms, 2),
    )

    state.score_history.append({
        "transaction_id": tx.transaction_id,
        "fraud_probability": selected_score,
        "flagged": flagged,
        "timestamp": time.time(),
    })
    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": state.classifier is not None,
        "live_window_size": len(state.live_window),
        "drift_alerts": sum(1 for v in state.drift_scores.values() if v > PSI_ALERT_THRESHOLD),
    }


@app.post("/score", response_model=ScoreResponse)
async def score(tx: TransactionRequest, explain: bool = False):
    """Score a single transaction. Pass ?explain=true for feature attributions."""
    return _score_single(tx, explain=explain)


@app.post("/score/batch", response_model=BatchScoreResponse)
async def score_batch(req: BatchScoreRequest):
    """Score a batch of transactions (more efficient than calling /score N times)."""
    t0 = time.perf_counter()
    results = [_score_single(tx) for tx in req.transactions]
    total_ms = (time.perf_counter() - t0) * 1000
    return BatchScoreResponse(
        results=results,
        batch_size=len(results),
        total_latency_ms=round(total_ms, 2),
        flagged_count=sum(1 for r in results if r.flagged),
    )


@app.get("/drift/report", response_model=DriftReport)
async def drift_report():
    """Return current PSI drift report across all features."""
    drift = update_drift_report()
    alert_feats = [f for f, v in drift.items() if v > PSI_ALERT_THRESHOLD]
    return DriftReport(
        feature_psi={k: round(v, 5) for k, v in drift.items()},
        n_alerts=len(alert_feats),
        alert_features=alert_feats,
        window_size=len(state.live_window),
        alert_threshold=PSI_ALERT_THRESHOLD,
    )


@app.get("/stream")
async def stream_transactions():
    """
    Server-Sent Events endpoint that simulates a live transaction stream.
    Each event is a scored transaction with a small probability of fraud.
    Useful for demos and dashboard testing.
    """
    async def generate():
        rng = np.random.default_rng()
        tx_id = 0
        while True:
            # Simulate a transaction feature vector
            is_fraud = rng.random() < 0.005   # 0.5% base fraud rate
            n_feats = len(state.feature_names) if state.feature_names else 20
            feats = rng.normal(0, 1, size=n_feats)
            if is_fraud:
                # Fraud transactions: higher amounts, unusual velocity
                feats += rng.normal(2, 1, size=n_feats) * 0.3

            features = {
                f: float(feats[i]) for i, f in enumerate(
                    state.feature_names or [f"f{i}" for i in range(n_feats)]
                )
            }

            tx = TransactionRequest(
                transaction_id=f"SIM_{tx_id:08d}",
                features=features,
                amount=float(rng.lognormal(4.5, 1.2)),
            )
            tx_id += 1

            try:
                result = _score_single(tx)
                data = result.model_dump()
                data["simulated_true_label"] = int(is_fraud)
                yield f"data: {json.dumps(data)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

            await asyncio.sleep(0.2)   # 5 transactions/second

    return StreamingResponse(generate(), media_type="text/event-stream")
