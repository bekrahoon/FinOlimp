"""Source classifier fit on the explicit run-local temporal training fold."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from source_features_s3 import SOURCE_FEATURES

SOURCE_CLASSES = ["1С", "агент", "бот", "выяснить"]
SEED = 31


def fit_source_model(train: pd.DataFrame):
    """Fit only the rows already assigned to the current run's train fold."""
    if train.empty:
        raise ValueError("Source training fold is empty")
    labels = train["label_source"].astype(str).to_numpy()
    if len(set(labels)) < 2:
        model = DummyClassifier(strategy="constant", constant=labels[0])
    else:
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                               max_leaf_nodes=15, random_state=SEED)
    model.fit(train[SOURCE_FEATURES].fillna(-1).values, labels)
    return model


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(y_true, y_pred, labels=SOURCE_CLASSES,
                                   output_dict=True, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6)
        if len(y_true) else None,
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=SOURCE_CLASSES,
                                          average="macro", zero_division=0)), 6)
        if len(y_true) else None,
        "per_class": {cls: {
            "precision": round(float(report[cls]["precision"]), 6),
            "recall": round(float(report[cls]["recall"]), 6),
            "f1": round(float(report[cls]["f1-score"]), 6),
            "support": int(report[cls]["support"]),
        } for cls in SOURCE_CLASSES},
        "confusion": confusion_matrix(y_true, y_pred,
                                      labels=SOURCE_CLASSES).tolist()
        if len(y_true) else [],
    }
