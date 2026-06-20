# Fraud detection: tabular, anomaly, and graph models

This project compares three PyTorch approaches on entity-aware fraud data:

1. An autoencoder trained only on legitimate transactions. Reconstruction error is the anomaly score.
2. A residual MLP trained with focal loss for severe class imbalance.
3. A GraphSAGE node classifier whose transaction nodes are connected by shared entities. Edges are causal (past to future), so test transactions cannot leak information into training nodes.

The final comparison includes a validation-weighted ensemble. Model selection uses PR-AUC and a business cost matrix, not ROC-AUC alone.

## 1. Environment

Use Python 3.11 or 3.12. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `NeighborLoader` reports that no sampling backend is installed, install `pyg-lib` or the matching `torch-sparse` wheel using the [official PyTorch Geometric instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your PyTorch/CUDA version.

## 2. Data

PaySim is the simplest path. Put `PS_20174392719_1491204439457_log.csv` in `data/`. Its `nameOrig` and `nameDest` fields provide the entity IDs required by the graph. Merchant destinations are represented by IDs beginning with `M`.

IEEE-CIS is also supported. Put `train_transaction.csv` and `train_identity.csv` in one directory. Its card, address, device, and email fields become shared entities.

Do not substitute the ULB credit-card dataset: it has no stable entity identifiers for graph construction.

## 3. Train and compare

First run a small integration check:

```bash
python train_all.py --dataset paysim --data-dir data --quick --skip-gnn
```

Then train all models:

```bash
python train_all.py \
  --dataset paysim \
  --data-dir data \
  --epochs 30 \
  --cost-fn 400 \
  --cost-fp 5
```

For IEEE-CIS, change `--dataset ieee` and point `--data-dir` at its files. Use `--nrows 200000` while iterating before training on the full data.

The pipeline performs these operations:

- chronological train/validation/test split (75%/10%/15%);
- past-only velocity counts, sums, means, and time-since-last-transaction features;
- train-only median, clipping, and scaling statistics;
- autoencoder training on legitimate training rows only;
- focal-loss classifier and GraphSAGE training;
- validation-only ensemble weighting and cost-optimal threshold selection;
- one final evaluation on the untouched test period.

Outputs are written to `artifacts/`:

- model weights and preprocessing artifacts;
- `metadata.json`, containing feature order, thresholds, costs, and ensemble weights;
- `evaluation.json`, containing PR-AUC, precision, recall, confusion matrices, and cost per transaction.

## 4. Explanations

The API returns Integrated Gradients explanations for flagged transactions when called with `?explain=true`. The saved legitimate background sample can also be used with `SHAPExplainer` in `explain.py` for offline analyst reports.

## 5. Run the API

After training:

```bash
uvicorn app:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs`. Important endpoints are:

- `POST /score?explain=true` for one engineered transaction;
- `POST /score/batch` for multiple transactions;
- `GET /stream` for a simulated server-sent-event stream;
- `GET /drift/report` for feature-level Population Stability Index values;
- `GET /health` for artifact readiness.

The request `features` must contain the raw engineered feature values named in `artifacts/metadata.json`; the service applies the saved scaler. In production, maintain per-entity rolling state in a feature store so velocity values at serving time use the same definitions as training.

## 6. Threshold decision

The default cost assumptions are $400 for missed fraud and $5 for unnecessary customer friction. `train_all.py` chooses the threshold on validation data that minimizes:

```text
total cost = false negatives × 400 + false positives × 5
```

This ratio is a business input, not a model constant. Re-estimate missed-fraud cost from actual loss and recovery rates, and false-positive cost from decline abandonment, support contacts, and churn. Re-run threshold selection when those economics or the fraud base rate change. Report recall and customer-friction volume alongside cost so a numerically optimal threshold does not conceal an unacceptable customer experience.

## Production boundary

The point-wise ensemble can score one transaction immediately. A GNN requires a context graph, so production GNN inference should attach the new transaction to a rolling graph or graph feature service before scoring. The included API is a demonstration service for the deployable autoencoder/classifier ensemble; the offline comparison still evaluates all three models and the full ensemble.
