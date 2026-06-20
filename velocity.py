"""
Transaction velocity & temporal feature engineering.

For each entity column (card, device, address …) we compute:
  - tx_cnt_Xh       : number of transactions by this entity in last X hours
  - tx_sum_Xh       : total amount transacted in last X hours
  - tx_mean_Xh      : mean amount in last X hours
  - time_since_last : seconds elapsed since the entity's previous transaction
                      (-1 for first-ever transaction, so the model can learn
                       that "never seen before" is itself a signal)

All windows use `closed='left'` so the current transaction is EXCLUDED —
this prevents target leakage.

IMPORTANT: sort by time before calling; the rolling logic assumes ascending
           TransactionDT.  temporal_split() in load.py already does this.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm


REF_TIMESTAMP = pd.Timestamp("2017-12-01")  # arbitrary; only offsets matter


def _to_datetimeindex(series: pd.Series) -> pd.DatetimeIndex:
    """Convert integer seconds since REF_TIMESTAMP to DatetimeIndex."""
    return REF_TIMESTAMP + pd.to_timedelta(series, unit="s")


def add_velocity_features(
    df: pd.DataFrame,
    entity_cols: List[str],
    time_col: str,
    amount_col: str,
    window_hours: List[int] = [1, 6, 24, 168],
) -> pd.DataFrame:
    """
    Main entry point.  Adds rolling velocity + time-since-last columns.

    Args:
        df:           DataFrame sorted ascending by time_col.
        entity_cols:  e.g. ['card1', 'addr1', 'DeviceInfo']
        time_col:     Integer seconds since reference (TransactionDT)
        amount_col:   e.g. 'TransactionAmt'
        window_hours: Rolling window widths in hours

    Returns:
        df with new feature columns appended (in place copy).
    """
    df = df.copy()
    # Ensure rows are ordered by time
    df = df.sort_values(time_col).reset_index(drop=True)

    # Build a real DatetimeIndex column for rolling
    df["__dt"] = _to_datetimeindex(df[time_col])

    for entity in tqdm(entity_cols, desc="Velocity features"):
        logger.debug(f"  Processing entity: {entity}")

        # ── Time-since-last ──────────────────────────────────────────────────
        prev_time = df.groupby(entity)[time_col].shift(1)
        df[f"{entity}_time_since_last"] = (df[time_col] - prev_time).fillna(-1).astype(float)

        # ── Rolling counts / sums per window ────────────────────────────────
        for wh in window_hours:
            cnt_col = f"{entity}_cnt_{wh}h"
            sum_col = f"{entity}_sum_{wh}h"
            mean_col = f"{entity}_mean_{wh}h"

            cnts = np.zeros(len(df), dtype=np.float32)
            sums = np.zeros(len(df), dtype=np.float64)
            histories = defaultdict(deque)
            running_sums = defaultdict(float)
            window_seconds = wh * 3600
            repeated_values = set(df[entity].value_counts(dropna=True).loc[lambda s: s > 1].index)

            # One chronological pass is O(N) per entity/window and avoids a
            # Python group operation for millions of mostly-unique accounts.
            entity_values = df[entity].to_numpy()
            times = df[time_col].to_numpy()
            amounts = df[amount_col].to_numpy(dtype=float)
            for row, (entity_val, tx_time, amount) in enumerate(
                zip(entity_values, times, amounts)
            ):
                if pd.isna(entity_val) or entity_val not in repeated_values:
                    continue
                history = histories[entity_val]
                cutoff = tx_time - window_seconds
                while history and history[0][0] < cutoff:
                    _, expired_amount = history.popleft()
                    running_sums[entity_val] -= expired_amount
                cnts[row] = len(history)
                sums[row] = running_sums[entity_val]
                # Add only after reading state: the current transaction cannot
                # contribute to its own velocity features.
                history.append((tx_time, amount))
                running_sums[entity_val] += amount

            df[cnt_col] = cnts
            df[sum_col] = sums
            df[mean_col] = np.where(df[cnt_col] > 0, df[sum_col] / df[cnt_col], 0.0)

    df = df.drop(columns=["__dt"])
    return df


def add_global_temporal_features(
    df: pd.DataFrame,
    time_col: str,
) -> pd.DataFrame:
    """
    Day-of-week and hour-of-day derived from the integer-second offset.
    These capture time-of-day fraud patterns without knowing the calendar date.
    """
    df = df.copy()
    seconds_in_day = 86_400
    seconds_in_week = 604_800

    df["hour_of_day"] = (df[time_col] % seconds_in_day) // 3600
    df["day_of_week"] = (df[time_col] % seconds_in_week) // seconds_in_day

    # Cyclical encoding so hour 23 is close to hour 0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def add_amount_features(df: pd.DataFrame, amount_col: str) -> pd.DataFrame:
    """Log-transform + amount bucketing to stabilise heavy-tail distribution."""
    df = df.copy()
    df["log_amount"] = np.log1p(df[amount_col])
    df["amount_rounded"] = (df[amount_col] % 1 == 0).astype(int)  # whole-dollar flag
    return df


def engineer_all(
    df: pd.DataFrame,
    entity_cols: List[str],
    time_col: str,
    amount_col: str,
    window_hours: List[int] = [1, 6, 24, 168],
) -> pd.DataFrame:
    """Convenience wrapper: runs all feature engineering in correct order."""
    logger.info("Engineering amount features …")
    df = add_amount_features(df, amount_col)

    logger.info("Engineering temporal features …")
    df = add_global_temporal_features(df, time_col)

    logger.info("Engineering velocity features (this may take a few minutes) …")
    df = add_velocity_features(df, entity_cols, time_col, amount_col, window_hours)

    return df
