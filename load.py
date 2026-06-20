"""
Data loading for IEEE-CIS Fraud Detection and PaySim datasets.

IEEE-CIS: https://www.kaggle.com/c/ieee-fraud-detection
  - transaction_train.csv + identity_train.csv
  - Entity columns: card1-card6, addr1/addr2, DeviceType, DeviceInfo
  - TransactionDT: seconds since reference time (NOT real timestamps)

PaySim: https://www.kaggle.com/datasets/ealaxi/paysim1
  - PS_20174392719_1491204439457_log.csv
  - Entity columns: nameOrig, nameDest  (explicit account IDs)
  - step: hour of simulation
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from loguru import logger


# ── IEEE-CIS ────────────────────────────────────────────────────────────────

IEEE_ENTITY_COLS = ["card1", "card2", "addr1", "DeviceInfo", "P_emaildomain"]
IEEE_CAT_COLS = [
    "ProductCD", "card4", "card6", "P_emaildomain",
    "R_emaildomain", "DeviceType", "DeviceInfo", "M1", "M2", "M3",
    "M4", "M5", "M6", "M7", "M8", "M9",
]
IEEE_ID_COL = "TransactionID"
IEEE_TIME_COL = "TransactionDT"
IEEE_AMOUNT_COL = "TransactionAmt"
IEEE_LABEL_COL = "isFraud"


def load_ieee(
    data_dir: str | Path,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load and merge IEEE-CIS transaction + identity files.

    Args:
        data_dir: Directory containing train_transaction.csv and train_identity.csv
        nrows:    Optional row limit for quick iteration

    Returns:
        Merged DataFrame with label column 'isFraud'
    """
    data_dir = Path(data_dir)
    tx_path = data_dir / "train_transaction.csv"
    id_path = data_dir / "train_identity.csv"

    logger.info(f"Loading IEEE-CIS transactions from {tx_path}")
    tx = pd.read_csv(tx_path, nrows=nrows)

    logger.info(f"Loading IEEE-CIS identities from {id_path}")
    # Do not truncate identity independently: its rows are not guaranteed to
    # align one-for-one with the transaction file.
    ids = pd.read_csv(id_path)

    df = tx.merge(ids, on=IEEE_ID_COL, how="left")
    logger.info(f"Merged shape: {df.shape}  |  Fraud rate: {df[IEEE_LABEL_COL].mean():.4%}")

    # Encode categoricals as integer codes
    for col in IEEE_CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes  # -1 = NaN

    return df


def get_ieee_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns (drop IDs, label, raw categoricals)."""
    drop = {IEEE_ID_COL, IEEE_LABEL_COL, IEEE_TIME_COL}
    return [c for c in df.columns if c not in drop and df[c].dtype != object]


# ── PaySim ──────────────────────────────────────────────────────────────────

PAYSIM_ENTITY_COLS = ["nameOrig", "nameDest"]
PAYSIM_CAT_COLS = ["type"]
PAYSIM_ID_COL = "step"          # hour; no true unique TX id — we add one
PAYSIM_TIME_COL = "step"        # hours since start of simulation
PAYSIM_AMOUNT_COL = "amount"
PAYSIM_LABEL_COL = "isFraud"


def load_paysim(
    data_dir: str | Path,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load PaySim dataset.

    Args:
        data_dir: Directory containing PS_20174392719_1491204439457_log.csv
        nrows:    Optional row limit

    Returns:
        DataFrame with added 'TransactionID' column and encoded categoricals.
    """
    data_dir = Path(data_dir)
    candidates = list(data_dir.glob("PS_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No PaySim CSV found in {data_dir}. "
            "Download from https://www.kaggle.com/datasets/ealaxi/paysim1"
        )

    path = candidates[0]
    logger.info(f"Loading PaySim from {path}")
    df = pd.read_csv(path, nrows=nrows)

    # Add synthetic unique transaction ID
    df.insert(0, "TransactionID", range(len(df)))

    # Filter only TRANSFER and CASH_OUT — PaySim fraud only occurs there
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

    for col in PAYSIM_CAT_COLS:
        df[col] = df[col].astype("category").cat.codes

    # Convert step (hours) to seconds so velocity code is dataset-agnostic
    df["TransactionDT"] = df["step"] * 3600

    logger.info(
        f"PaySim shape after type filter: {df.shape}  "
        f"|  Fraud rate: {df[PAYSIM_LABEL_COL].mean():.4%}"
    )
    return df


def get_paysim_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"TransactionID", PAYSIM_LABEL_COL, "step", "nameOrig", "nameDest", "isFlaggedFraud"}
    return [c for c in df.columns if c not in drop and df[c].dtype != object]


# ── Shared Utilities ─────────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    time_col: str,
    val_frac: float = 0.10,
    test_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by time so train always precedes val/test — mimics production.
    Avoids data leakage that random splits would introduce.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    logger.info(
        f"Split sizes — train: {len(train):,} | val: {len(val):,} | test: {len(test):,}"
    )
    logger.info(
        f"Fraud rates — train: {train[IEEE_LABEL_COL].mean():.4%} | "
        f"val: {val[IEEE_LABEL_COL].mean():.4%} | "
        f"test: {test[IEEE_LABEL_COL].mean():.4%}"
    )
    return train, val, test


def fill_and_clip(
    df: pd.DataFrame,
    feature_cols: list[str],
    clip_quantile: float = 0.999,
) -> pd.DataFrame:
    """Fill NaN with median; clip upper outliers to reduce autoencoder distortion."""
    df = df.copy()
    for col in feature_cols:
        med = df[col].median()
        upper = df[col].quantile(clip_quantile)
        df[col] = df[col].fillna(med).clip(upper=upper)
    return df
