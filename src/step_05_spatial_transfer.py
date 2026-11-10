"""Step 05: spatial transferability (RQ2), validation phase (inner_train -> 2025 validation).

For each spatial block b (k-means blocks of tract centroids, step 01):
  seen    model: trained on inner-train trips from all blocks
  unseen  model: trained on inner-train trips with NEITHER origin NOR destination in block b (strict hold-out)
  matched model: trained on a random subsample of ALL inner-train trips of the same size as the unseen training set
                 (size-matched control: matched - unseen isolates spatial novelty from the loss of training data)
Both are evaluated on 2025 validation trips that start in block b. Same hyperparameters and number of trees
(selected in step 03) for LightGBM; the spline logistic regression is refitted on a 600k subsample in both cases.
Day-block bootstrap (200 replicates) gives 95% CIs for metric(seen) - metric(unseen) per block.
The test split is NOT used.

Outputs: outputs/tables/T06_spatial_transfer_valid.csv, outputs/predictions/spatial_transfer_valid.parquet

Run:  python src/step_05_spatial_transfer.py
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FS, OUT, SEED, TARGET, load, metrics  # noqa: E402

SEL = json.loads((OUT / "reports" / "03_selection_main.json").read_text())
F, PARAMS, NT = SEL["features"], SEL["lgbm_params"], SEL["n_trees"]
tr, va = load("inner_train", F), load("valid", F)
blocks = sorted(tr.o_block.unique())
d_block = pd.read_csv(OUT / "tables" / "T03b_tract_blocks.csv", dtype={"GEOID": str}).set_index("GEOID")["block"]
tr["d_block"] = tr.d_tract.map(d_block)
binary = [f for f in F if set(pd.unique(tr[f].dropna())) <= {0, 1}]
contin = [f for f in F if f not in binary]
rng = np.random.default_rng(SEED)


def fit_lgb(df):
    return lgb.train(PARAMS, lgb.Dataset(df[F], df[TARGET], params={"feature_pre_filter": False}), NT)


def fit_lr(df):
    pre = ColumnTransformer([("cont", Pipeline([("imp", SimpleImputer(strategy="median")),
                                                ("spl", SplineTransformer(n_knots=5, degree=3)),
                                                ("sc", StandardScaler())]), contin),
                             ("bin", SimpleImputer(strategy="most_frequent"), binary)])
    s = df.sample(n=min(600_000, len(df)), random_state=SEED)
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=1.0, max_iter=3000))]).fit(s[F], s[TARGET])


def boot_diff(y, p_seen, p_unseen, day, reps=200):
    days = np.unique(day)
    idx = pd.Series(np.arange(len(day))).groupby(day).apply(np.array)
    out = {"pr_auc": [], "roc_auc": [], "log_loss": []}
    for _ in range(reps):
        ii = np.concatenate(idx.loc[rng.choice(days, len(days), replace=True)].values)
        yb = y[ii]
        if yb.sum() == 0:
            continue
        out["pr_auc"].append(average_precision_score(yb, p_seen[ii]) - average_precision_score(yb, p_unseen[ii]))
        out["roc_auc"].append(roc_auc_score(yb, p_seen[ii]) - roc_auc_score(yb, p_unseen[ii]))
        out["log_loss"].append(log_loss(yb, np.clip(p_unseen[ii], 1e-6, 1 - 1e-6)) - log_loss(yb, np.clip(p_seen[ii], 1e-6, 1 - 1e-6)))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


t0 = time.time()
seen = {"LightGBM": fit_lgb(tr), "LogReg": fit_lr(tr)}
print(f"seen models fitted ({(time.time() - t0) / 60:.1f} min)", flush=True)
rows, preds = [], []
for b in blocks:
    tb = tr[(tr.o_block != b) & (tr.d_block != b)]
    unseen = {"LightGBM": fit_lgb(tb), "LogReg": fit_lr(tb)}
    tm = tr.sample(n=len(tb), random_state=SEED + int(b))
    matched = {"LightGBM": fit_lgb(tm), "LogReg": fit_lr(tm)}
    vb = va[va.o_block == b]
    y, day = vb[TARGET].values, vb["day"].values
    for name in ("LightGBM", "LogReg"):
        def pr(m):
            return m.predict(vb[F]) if name == "LightGBM" else m.predict_proba(vb[F])[:, 1]
        ps, pu, pm = pr(seen[name]), pr(unseen[name]), pr(matched[name])
        ci = boot_diff(y, pm, pu, day)
        ci_full = boot_diff(y, ps, pu, day)
        for cond, p, n_ in (("seen", ps, len(tr)), ("matched", pm, len(tm)), ("unseen", pu, len(tb))):
            m = metrics(y, p)
            m.update(block=int(b), model=name, condition=cond, train_rows=n_,
                     pr_lift=m["pr_auc"] / max(m["observed_rate"], 1e-9))
            rows.append(m)
        rows[-1].update(diff_matched_pr_auc_ci=ci["pr_auc"], diff_matched_roc_auc_ci=ci["roc_auc"],
                        diff_matched_logloss_ci=ci["log_loss"], diff_seen_pr_auc_ci=ci_full["pr_auc"],
                        diff_seen_roc_auc_ci=ci_full["roc_auc"])
        preds.append(pd.DataFrame({"block": b, "model": name, "y": y, "p_seen": ps.astype("float32"),
                                   "p_matched": pm.astype("float32"), "p_unseen": pu.astype("float32"),
                                   "o_tract": vb.o_tract.values}))
    print(f"block {b}: valid trips {len(vb):,}, e-bike {y.mean():.3f}, unseen train rows {len(tb):,} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT / "tables" / "T06_spatial_transfer_valid.csv", index=False)
pd.concat(preds).to_parquet(OUT / "predictions" / "spatial_transfer_valid.parquet", index=False)
piv = res.pivot_table(index=["model", "block"], columns="condition", values=["pr_auc", "roc_auc", "log_loss", "cal_intercept"])
print(piv.round(4).to_string())
