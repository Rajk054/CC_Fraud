# Transaction Fraud Detection with PyTorch

I started this project to compare three different ways of looking at the same fraud problem: unusual transactions, known fraud patterns, and relationships between accounts. Most examples only train a classifier on individual rows. Here I also wanted to test whether a transaction graph can pick up coordinated behaviour that a row-by-row model misses.

The project currently supports PaySim and IEEE-CIS. I am using PaySim for the first full experiment because it includes account identifiers that can be used to build a graph.

> **Project status:** the pipeline and API are implemented. I am currently running the first reproducible PaySim benchmark on Kaggle. I will add the measured results and plots here once that run is complete.

## What I am comparing

| Approach | What it is meant to catch |
| --- | --- |
| Autoencoder | Transactions that do not look like legitimate behaviour seen during training |
| Focal-loss MLP | Fraud patterns learned directly from labelled examples |
| GraphSAGE GNN | Groups of transactions connected through shared accounts, cards, devices or merchants |
| Weighted ensemble | Cases where the models provide complementary signals |

The autoencoder is trained only on legitimate transactions. The supervised models use focal loss because fraud is rare and ordinary binary cross-entropy can be dominated by easy negative examples.

## Features

Alongside the original transaction fields, the pipeline calculates past-only behavioural features for each entity:

- transaction count over 1, 6, 24 and 168 hours
- amount sum and mean over the same windows
- time since the previous transaction
- log amount and whole-number amount flag
- cyclical hour-of-day and day-of-week features

The current transaction is added to an entity's history only after its features are calculated. This keeps it from contributing to its own velocity values.

## Graph construction

Each transaction is a node. Transactions are connected when they share an entity such as an origin account, destination account, card or device. Edges point from earlier transactions to later ones so training nodes cannot aggregate information from the future.

Popular entities can create an impractically dense graph, so each transaction is connected to a limited number of earlier neighbours. GraphSAGE is then trained with neighbour sampling rather than full-graph batches.

## Evaluation

The split is chronological: 75% train, 10% validation and 15% test. Preprocessing is fitted on the training period, ensemble weights and thresholds are chosen on validation, and the test period is held back for the final comparison.

I use PR-AUC as the main ranking metric. Accuracy and ROC-AUC can look strong on an imbalanced dataset even when the model produces too many false alerts.

The operating threshold is selected using a simple business cost model:

```text
total cost = missed fraud × false-negative cost
           + false alerts × false-positive cost
```

The defaults are $400 for missed fraud and $5 for customer friction. These are assumptions for the experiment, not universal values.

## Running the project

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download PaySim's `PS_20174392719_1491204439457_log.csv` into `data/`, then run a small smoke test:

```bash
python train_all.py --dataset paysim --data-dir data --quick --skip-gnn
```

Run the full comparison:

```bash
python train_all.py \
  --dataset paysim \
  --data-dir data \
  --epochs 30 \
  --cost-fn 400 \
  --cost-fp 5
```

Model weights, preprocessing objects and evaluation reports are written to `artifacts/`. The dataset and generated artifacts are intentionally excluded from Git.

## API

After training, start the FastAPI service with:

```bash
uvicorn app:app --reload --port 8000
```

The service includes single and batch scoring, optional Integrated Gradients explanations, a simulated transaction stream, health checks and PSI-based feature drift monitoring. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

A single request can use the autoencoder/classifier ensemble immediately. GNN inference needs graph context, so a production version would connect each new transaction to a rolling transaction graph before scoring it.

## Repository map

- `train_all.py` — end-to-end training and evaluation
- `velocity.py` — temporal and transaction-velocity features
- `autoencoder.py` — reconstruction-based anomaly detector
- `classifier.py` — residual MLP and focal loss
- `graph_builder.py` — causal shared-entity transaction graph
- `gnn.py` — GraphSAGE model and neighbour-sampled training
- `metrics.py` — PR-AUC, cost metrics and threshold selection
- `explain.py` — Integrated Gradients and SHAP helpers
- `app.py` — FastAPI scoring and drift-monitoring service

## Known limitations

PaySim is simulated data, so results should not be treated as evidence of real-world fraud performance. The API also expects engineered features; a production system would need an online feature store and a maintained graph service. Those boundaries are intentional rather than hidden behind the demo.
