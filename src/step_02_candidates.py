"""Step 02: untuned candidate models, trained on inner_train and compared on validation (2025).

Models: prior-only dummy; spline logistic regression; logistic regression with origin-tract fixed effects (spatial
benchmark: every tract its own intercept); LightGBM (with and without class weighting); XGBoost; CatBoost.
The test split is not loaded.

Run:  python src/step_02_candidates.py [--set main]
"""
import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FS, OUT, SEED, TARGET, load, metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="main", choices=list(FS))
A = ap.parse_args()
F = FS[A.set]
tr, va = load("inner_train", F), load("valid", F)
assert tr.start_dt.max() < va.start_dt.min()
print(f"[{A.set}] inner_train {tr.shape}, valid {va.shape}, e-bike {tr[TARGET].mean():.4f} / {va[TARGET].mean():.4f}")
y, yv = tr[TARGET].values, va[TARGET].values
rows, preds = [], {}


def rec(name, p, t0, note=""):
    m = metrics(yv, p)
    m.update(model=name, fit_seconds=round(time.time() - t0, 1), note=note)
    rows.append(m)
    preds[name] = p
    print(f"{name:22s} PR-AUC {m['pr_auc']:.4f}  AUC {m['roc_auc']:.4f}  logloss {m['log_loss']:.4f}  "
          f"Brier {m['brier']:.4f}  slope {m['cal_slope']:.3f}  int {m['cal_intercept']:+.3f}  ({m['fit_seconds']}s)", flush=True)


t0 = time.time()
rec("Dummy", np.full(len(va), y.mean()), t0)

binary = [f for f in F if set(pd.unique(tr[f].dropna())) <= {0, 1}]
contin = [f for f in F if f not in binary]
sub = tr.sample(n=min(600_000, len(tr)), random_state=SEED)
t0 = time.time()
pre = ColumnTransformer([("cont", Pipeline([("imp", SimpleImputer(strategy="median")),
                                            ("spl", SplineTransformer(n_knots=5, degree=3)), ("sc", StandardScaler())]), contin),
                         ("bin", SimpleImputer(strategy="most_frequent"), binary)])
lr = Pipeline([("pre", pre), ("lr", LogisticRegression(C=1.0, max_iter=3000))]).fit(sub[F], sub[TARGET])
rec("LogReg (splines)", lr.predict_proba(va[F])[:, 1], t0, "600k subsample")

t0 = time.time()
pre_fe = ColumnTransformer([("cont", Pipeline([("imp", SimpleImputer(strategy="median")),
                                               ("spl", SplineTransformer(n_knots=5, degree=3)), ("sc", StandardScaler())]),
                             [c for c in contin if not c.startswith("o_")]),
                            ("bin", SimpleImputer(strategy="most_frequent"), binary),
                            ("tract", OneHotEncoder(handle_unknown="ignore", min_frequency=200), ["o_tract"])])
lrfe = Pipeline([("pre", pre_fe), ("lr", LogisticRegression(C=1.0, max_iter=3000))])
lrfe.fit(sub[F + ["o_tract"]], sub[TARGET])
rec("LogReg + tract FE", lrfe.predict_proba(va[F + ["o_tract"]])[:, 1], t0, "origin-tract fixed effects, 600k subsample")

dtr = lgb.Dataset(tr[F], y, free_raw_data=False)
dva = lgb.Dataset(va[F], yv, reference=dtr, free_raw_data=False)
for name, extra in (("LightGBM", {}), ("LightGBM (weighted)", {"is_unbalance": True})):
    t0 = time.time()
    p = dict(objective="binary", metric="binary_logloss", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=SEED, verbosity=-1, **extra)
    m = lgb.train(p, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
    rec(name, m.predict(va[F], num_iteration=m.best_iteration), t0, f"best_iter={m.best_iteration}")

t0 = time.time()
xg = XGBClassifier(n_estimators=5000, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8,
                   tree_method="hist", eval_metric="logloss", early_stopping_rounds=150, random_state=SEED, n_jobs=-1)
xg.fit(tr[F], y, eval_set=[(va[F], yv)], verbose=False)
rec("XGBoost", xg.predict_proba(va[F])[:, 1], t0, f"best_iter={xg.best_iteration}")

t0 = time.time()
cb = CatBoostClassifier(iterations=5000, learning_rate=0.08, depth=7, loss_function="Logloss", random_seed=SEED,
                        od_type="Iter", od_wait=150, verbose=False, thread_count=-1)
cb.fit(tr[F], y, eval_set=(va[F], yv), use_best_model=True)
rec("CatBoost", cb.predict_proba(va[F])[:, 1], t0, f"best_iter={cb.get_best_iteration()}")

res = pd.DataFrame(rows).set_index("model")
res.to_csv(OUT / "tables" / f"T04_candidates_valid_{A.set}.csv")
pd.DataFrame(preds).assign(y=yv).astype("float32").to_parquet(OUT / "predictions" / f"valid_candidates_{A.set}.parquet")
print(res[["pr_auc", "roc_auc", "log_loss", "brier", "cal_slope", "cal_intercept", "mean_pred", "observed_rate"]].round(4))
