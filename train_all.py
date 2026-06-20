"""Train, evaluate, and persist the fraud detection models.

Examples:
    python train_all.py --dataset paysim --data-dir data --quick
    python train_all.py --dataset paysim --data-dir data --epochs 30
    python train_all.py --dataset ieee --data-dir data/ieee --epochs 30
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from autoencoder import FraudAutoencoder, compute_threshold, predict_autoencoder, train_autoencoder
from classifier import FraudClassifier, predict_classifier, train_classifier
from graph_builder import build_transaction_graph, split_graph_masks
from load import (
    IEEE_AMOUNT_COL,
    IEEE_ENTITY_COLS,
    IEEE_LABEL_COL,
    IEEE_TIME_COL,
    PAYSIM_AMOUNT_COL,
    PAYSIM_ENTITY_COLS,
    PAYSIM_LABEL_COL,
    load_ieee,
    load_paysim,
    temporal_split,
)
from metrics import CostMatrix, ensemble_scores, evaluate_model, optimal_threshold
from velocity import engineer_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["paysim", "ieee"], default="paysim")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--nrows", type=int)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--skip-gnn", action="store_true")
    p.add_argument("--quick", action="store_true", help="Small CPU smoke run (20k rows, 2 epochs).")
    p.add_argument("--cost-fn", type=float, default=400.0)
    p.add_argument("--cost-fp", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_engineer(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], str, str, list[str]]:
    nrows = 20_000 if args.quick and args.nrows is None else args.nrows
    if args.dataset == "paysim":
        df = load_paysim(args.data_dir, nrows)
        entities = PAYSIM_ENTITY_COLS
        time_col, amount_col, label_col = "TransactionDT", PAYSIM_AMOUNT_COL, PAYSIM_LABEL_COL
    else:
        df = load_ieee(args.data_dir, nrows)
        entities = [c for c in IEEE_ENTITY_COLS if c in df.columns]
        time_col, amount_col, label_col = IEEE_TIME_COL, IEEE_AMOUNT_COL, IEEE_LABEL_COL

    # All rolling features exclude the current transaction. Computing them
    # before the temporal split therefore uses past context without future leakage.
    windows = [1, 6] if args.quick else [1, 6, 24, 168]
    df = engineer_all(df, entities, time_col, amount_col, windows)
    excluded = {"TransactionID", label_col, time_col, "step", "isFlaggedFraud", *entities}
    feature_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    return df, feature_cols, time_col, label_col, entities


def fit_preprocessor(
    train: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.Series, pd.Series, pd.Series, StandardScaler]:
    medians = train[feature_cols].median().fillna(0.0)
    lower = train[feature_cols].quantile(0.001).fillna(medians)
    upper = train[feature_cols].quantile(0.999).fillna(medians)
    scaler = StandardScaler().fit(train[feature_cols].fillna(medians).clip(lower, upper, axis=1))
    return medians, lower, upper, scaler


def transform(
    frame: pd.DataFrame,
    feature_cols: list[str],
    medians: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    scaler: StandardScaler,
) -> np.ndarray:
    clean = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(medians)
    clean = clean.clip(lower, upper, axis=1)
    return scaler.transform(clean).astype(np.float32)


def serialise_evaluation(result: Any) -> dict[str, Any]:
    d = asdict(result)
    for key in ("confusion", "precisions", "recalls", "thresholds_curve"):
        d[key] = np.asarray(d[key]).tolist()
    for key, value in d.items():
        if isinstance(value, np.generic):
            d[key] = value.item()
    return d


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    epochs = 2 if args.quick else args.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.artifact_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    df, features, time_col, label_col, entities = load_and_engineer(args)
    train, val, test = temporal_split(df, time_col)
    if train[label_col].nunique() < 2 or val[label_col].nunique() < 2 or test[label_col].nunique() < 2:
        raise ValueError("Every temporal split must contain fraud and legitimate rows; increase --nrows.")

    medians, lower, upper, scaler = fit_preprocessor(train, features)
    X_train = transform(train, features, medians, lower, upper, scaler)
    X_val = transform(val, features, medians, lower, upper, scaler)
    X_test = transform(test, features, medians, lower, upper, scaler)
    y_train = train[label_col].to_numpy(np.int64)
    y_val = val[label_col].to_numpy(np.int64)
    y_test = test[label_col].to_numpy(np.int64)

    ae_config = {"hidden_dims": [128, 64], "bottleneck": 24, "dropout": 0.15}
    clf_config = {"hidden_dim": 128, "n_res_blocks": 3, "dropout": 0.25}
    ae = FraudAutoencoder(len(features), **ae_config)
    train_autoencoder(
        ae, X_train[y_train == 0], X_val, y_val, epochs=epochs,
        batch_size=args.batch_size, device=device,
    )
    ae_threshold = compute_threshold(ae, X_val, y_val, device=device)
    ae_val_raw, _ = predict_autoencoder(ae, X_val, ae_threshold, device)
    ae_test_raw, _ = predict_autoencoder(ae, X_test, ae_threshold, device)
    ae_low, ae_high = np.quantile(ae_val_raw[y_val == 0], [0.01, 0.99])
    ae_span = max(float(ae_high - ae_low), 1e-12)
    ae_val = np.clip((ae_val_raw - ae_low) / ae_span, 0, 1)
    ae_test = np.clip((ae_test_raw - ae_low) / ae_span, 0, 1)

    clf = FraudClassifier(len(features), **clf_config)
    train_classifier(
        clf, X_train, y_train, X_val, y_val, epochs=epochs,
        batch_size=args.batch_size, device=device,
    )
    clf_val, _ = predict_classifier(clf, X_val, device=device)
    clf_test, _ = predict_classifier(clf, X_test, device=device)

    val_scores: dict[str, np.ndarray] = {"autoencoder": ae_val, "classifier": clf_val}
    test_scores: dict[str, np.ndarray] = {"autoencoder": ae_test, "classifier": clf_test}
    gnn_config = {"hidden_dim": 96, "n_layers": 3, "dropout": 0.3, "edge_dim": len(entities)}

    if not args.skip_gnn:
        try:
            from torch_geometric.data import Data
            from gnn import FraudGNN, predict_gnn, train_gnn

            graph_df = pd.concat([train, val, test], ignore_index=True)
            graph_X = np.vstack([X_train, X_val, X_test])
            graph_df.loc[:, features] = graph_X
            x, edge_index, edge_attr, y = build_transaction_graph(
                graph_df, features, entities, label_col, max_degree=25 if args.quick else 50
            )
            train_mask, val_mask, test_mask = split_graph_masks(
                y, train_frac=len(train) / len(graph_df), val_frac=len(val) / len(graph_df)
            )
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
            gnn = FraudGNN(len(features), **gnn_config)
            train_gnn(gnn, data, epochs=epochs, batch_size=args.batch_size, device=device)
            gnn_val, _ = predict_gnn(gnn, data, val_mask, device=device)
            gnn_test, _ = predict_gnn(gnn, data, test_mask, device=device)
            val_scores["gnn"] = gnn_val
            test_scores["gnn"] = gnn_test
            torch.save(gnn.state_dict(), out / "gnn.pt")
        except (ImportError, OSError) as exc:
            print(f"GNN skipped because PyTorch Geometric is unavailable: {exc}")

    # Validation PR-AUC supplies transparent ensemble weights. Thresholds are
    # selected on validation only, then locked for the final test evaluation.
    weights = {name: max(average_precision_score(y_val, score), 1e-6)
               for name, score in val_scores.items()}
    val_scores["ensemble"] = ensemble_scores(val_scores, weights)
    test_scores["ensemble"] = ensemble_scores(test_scores, weights)
    deploy_names = ["autoencoder", "classifier"]
    deploy_weights = {k: weights[k] for k in deploy_names}
    deploy_val = ensemble_scores({k: val_scores[k] for k in deploy_names}, deploy_weights)

    costs = CostMatrix(args.cost_fn, args.cost_fp)
    thresholds = {name: optimal_threshold(y_val, score, costs)[0]
                  for name, score in val_scores.items()}
    evaluations = {
        name: serialise_evaluation(evaluate_model(name, y_test, score, costs, thresholds[name]))
        for name, score in test_scores.items()
    }
    deploy_candidates = {
        "autoencoder": ae_val,
        "classifier": clf_val,
        "ensemble": deploy_val,
    }
    deploy_choices = {
        name: optimal_threshold(y_val, score, costs)
        for name, score in deploy_candidates.items()
    }
    deploy_mode = min(deploy_choices, key=lambda name: deploy_choices[name][1])
    deploy_threshold = deploy_choices[deploy_mode][0]

    torch.save(ae.state_dict(), out / "autoencoder.pt")
    torch.save(clf.state_dict(), out / "classifier.pt")
    joblib.dump(scaler, out / "scaler.pkl")
    joblib.dump({"medians": medians, "lower": lower, "upper": upper}, out / "preprocessor.pkl")
    reference = train[features].replace([np.inf, -np.inf], np.nan).fillna(medians)
    reference = reference.clip(lower, upper, axis=1).to_numpy(np.float32)
    np.save(out / "reference_distribution.npy", reference[: min(len(reference), 10_000)])
    np.save(out / "shap_background.npy", X_train[y_train == 0][:500])

    metadata = {
        "dataset": args.dataset,
        "feature_names": features,
        "entity_columns": entities,
        "ae_config": ae_config,
        "clf_config": clf_config,
        "gnn_config": gnn_config,
        "ae_threshold": ae_threshold,
        "ae_score_low": float(ae_low),
        "ae_score_high": float(ae_high),
        "ensemble_weights": {k: float(v) for k, v in weights.items()},
        "deployable_ensemble_weights": {k: float(v) for k, v in deploy_weights.items()},
        "deployable_mode": deploy_mode,
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "deployable_threshold": float(deploy_threshold),
        "cost_matrix": {"false_negative": args.cost_fn, "false_positive": args.cost_fp},
        "best_test_model": min(evaluations, key=lambda k: evaluations[k]["cost_per_tx"]),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out / "evaluation.json").write_text(json.dumps(evaluations, indent=2))

    print("\nFinal untouched-test comparison")
    print(f"{'model':<16} {'PR-AUC':>9} {'recall':>9} {'precision':>10} {'cost/tx':>10}")
    for name, result in sorted(evaluations.items(), key=lambda item: item[1]["cost_per_tx"]):
        print(f"{name:<16} {result['pr_auc']:>9.4f} {result['recall']:>9.4f} "
              f"{result['precision']:>10.4f} {result['cost_per_tx']:>10.4f}")
    print(f"\nArtifacts written to {out}")


if __name__ == "__main__":
    main()
