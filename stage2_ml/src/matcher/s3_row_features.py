"""Raw-only row-level features for Stage 2 input admission."""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


STATUS_FEATURES = [
    "status_is_blank", "status_is_draft", "status_is_reissue",
    "group_has_reissue", "group_draft_count", "group_reissue_count",
    "group_row_count",
]


def act_status_features(acts: pd.DataFrame) -> pd.DataFrame:
    """Raw status features with group context computed strictly from prior dates."""
    status = (acts.get("act_status", pd.Series("", index=acts.index))
              .fillna("").astype(str).str.casefold().str.strip())
    folder = (acts.get("folder", pd.Series("", index=acts.index))
              .fillna("").astype(str).str.casefold().str.strip())
    period = (acts.get("period_start", pd.Series("", index=acts.index))
              .fillna("").astype(str).str.strip())
    group = folder + "\x1f" + period
    date_values = acts.get("date_norm", acts.get("date", pd.Series(index=acts.index)))
    operation_date = _parse_operation_dates(date_values).dt.normalize()
    is_draft = status.str.contains("черновик", regex=False)
    is_reissue = status.str.contains("переиздан", regex=False)
    group_frame = pd.DataFrame({
        "group": group,
        "operation_date": operation_date,
        "draft": is_draft.astype(int),
        "reissue": is_reissue.astype(int),
    }, index=acts.index)
    dated = group_frame[group_frame["operation_date"].notna()]
    daily = dated.groupby(["group", "operation_date"], sort=True).agg(
        draft=("draft", "sum"), reissue=("reissue", "sum"),
        row_count=("draft", "size"),
    )
    prior = daily.groupby(level=0)[["draft", "reissue", "row_count"]].cumsum()
    prior = prior - daily[["draft", "reissue", "row_count"]]
    prior.columns = ["group_draft_count", "group_reissue_count", "group_row_count"]
    prior["group_has_reissue"] = prior["group_reissue_count"].gt(0).astype(int)
    history = prior.to_dict("index")
    mapped = [history.get((str(g), pd.Timestamp(d)), {}) if pd.notna(d) else {}
              for g, d in zip(group, operation_date)]
    out = pd.DataFrame(index=acts.index)
    out["status_is_blank"] = status.eq("").astype(int)
    out["status_is_draft"] = is_draft.astype(int)
    out["status_is_reissue"] = is_reissue.astype(int)
    for col in ("group_has_reissue", "group_draft_count", "group_reissue_count",
                "group_row_count"):
        out[col] = [values.get(col, 0) for values in mapped]
    return out[STATUS_FEATURES].fillna(0).astype(float)


def _parse_operation_dates(values: pd.Series) -> pd.Series:
    """Parse supported export date shapes without interpreting ISO as day-first."""
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    remaining = values.notna()
    for date_format in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
                        "%d.%m.%Y %H:%M:%S", "%d.%m.%Y",
                        "%d/%m/%Y", "%Y/%m/%d"):
        if not remaining.any():
            break
        attempt = pd.to_datetime(values[remaining], format=date_format, errors="coerce")
        hits = attempt.notna()
        parsed.loc[attempt[hits].index] = attempt[hits]
        remaining &= parsed.isna()
    if remaining.any():
        attempt = pd.to_datetime(values[remaining], errors="coerce", dayfirst=True)
        hits = attempt.notna()
        parsed.loc[attempt[hits].index] = attempt[hits]
    return parsed


def fit_status_model(features, labels, seed: int = 13):
    """Fit the keep/drop classifier with balanced class weights."""
    labels = np.asarray(labels, dtype=int)
    if not len(labels):
        raise ValueError("Status training table is empty")
    if len(np.unique(labels)) < 2:
        model = DummyClassifier(strategy="constant", constant=int(labels[0]))
        model.fit(np.asarray(features, dtype=float), labels)
        return model
    counts = np.bincount(labels, minlength=2)
    weights = np.array([len(labels) / max(2 * counts[label], 1)
                        for label in labels])
    model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08,
                                           max_leaf_nodes=7, min_samples_leaf=5,
                                           random_state=seed)
    model.fit(np.asarray(features, dtype=float), labels, sample_weight=weights)
    return model
