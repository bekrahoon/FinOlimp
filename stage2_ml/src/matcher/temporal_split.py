"""Shared chronological split helpers for run-local Stage 2 training."""
from __future__ import annotations

import math

import pandas as pd


def select_cutoff(date_values, validation_fraction: float = 0.2) -> pd.Timestamp:
    """Pick a day boundary so the latest fraction of unique dates is validation."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    dates = pd.to_datetime(pd.Series(date_values), errors="coerce").dropna()
    unique_dates = pd.DatetimeIndex(dates.dt.normalize().unique()).sort_values()
    if len(unique_dates) < 2:
        raise ValueError("A chronological train/validation split needs at least 2 dates")
    split_at = math.floor(len(unique_dates) * (1 - validation_fraction))
    split_at = min(max(split_at, 1), len(unique_dates) - 1)
    return pd.Timestamp(unique_dates[split_at])


def select_month_cutoff(date_values, validation_months: int) -> pd.Timestamp:
    """Pick a month-aligned boundary: validation = last N full calendar months.

    Returns midnight of the first day of the earliest validation month, so the
    cutoff is always a round "1st of month" date. Train = strictly before it.
    """
    if int(validation_months) < 1:
        raise ValueError("validation_months must be at least 1")
    months = int(validation_months)
    dates = pd.to_datetime(pd.Series(date_values), errors="coerce").dropna()
    if dates.empty:
        raise ValueError("A calendar-month split needs at least one valid date")
    latest_month = dates.dt.normalize().max().replace(day=1)
    cutoff = (latest_month - pd.DateOffset(months=months - 1)).normalize()
    earliest_month = dates.dt.normalize().min().replace(day=1)
    if cutoff <= earliest_month:
        raise ValueError(
            "Not enough calendar months for "
            f"validation_months={months}: data starts {earliest_month:%Y-%m-%d}")
    return pd.Timestamp(cutoff)


def date_masks(date_values, cutoff: str | pd.Timestamp
               ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return train, validation, and unknown-date masks for operation dates."""
    dates = pd.to_datetime(pd.Series(date_values), errors="coerce").dt.normalize()
    boundary = pd.Timestamp(cutoff).normalize()
    return dates.lt(boundary), dates.ge(boundary), dates.isna()


def pair_masks(frame: pd.DataFrame, cutoff: str | pd.Timestamp
               ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keep pairs wholly on one side of the boundary; discard cross-period pairs."""
    a_dates = pd.to_datetime(frame["a_date"], errors="coerce").dt.normalize()
    e_dates = pd.to_datetime(frame["e_date"], errors="coerce").dt.normalize()
    boundary = pd.Timestamp(cutoff).normalize()
    train = a_dates.lt(boundary) & e_dates.lt(boundary)
    validation = a_dates.ge(boundary) & e_dates.ge(boundary)
    excluded = ~(train | validation)
    return train, validation, excluded
