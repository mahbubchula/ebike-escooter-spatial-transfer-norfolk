"""Shared helpers: paths, data loading and metrics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, log_loss,
                             matthews_corrcoef, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
SPL = ROOT / "data" / "processed" / "splits"
OUT = ROOT / "outputs"
FS = json.loads((ROOT / "configs" / "features.json").read_text())
SEED, TARGET = 42, "e_bike"
for sub in ("models", "tables", "reports", "logs", "predictions", "xai"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)


def load(split, cols=None, extra=("start_dt", "day", "o_tract", "d_tract", "o_block")):
    cols = list(dict.fromkeys([TARGET, *extra, *(cols or FS["main_plus_history"])]))
    return pd.read_parquet(SPL / f"{split}.parquet", columns=cols)


def cal_slope_intercept(y, p, eps=1e-6):
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def metrics(y, p, threshold=0.5):
    y, p = np.asarray(y), np.asarray(p)
    yh = (p >= threshold).astype(int)
    sl, ic = cal_slope_intercept(y, p)
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), "brier": brier_score_loss(y, p),
            "balanced_acc": balanced_accuracy_score(y, yh), "f1": f1_score(y, yh, zero_division=0),
            "mcc": matthews_corrcoef(y, yh), "cal_slope": sl, "cal_intercept": ic, "mean_pred": float(p.mean()),
            "observed_rate": float(y.mean()), "threshold": threshold, "n": int(len(y))}
