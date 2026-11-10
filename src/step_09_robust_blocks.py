"""Step 09: sensitivity of the spatial-transferability result to the number of spatial blocks (K = 4, 8).

Same protocol as step 05 (inner_train -> 2025 validation; size-matched vs strict unseen; identical model settings),
with blocks re-derived from tract centroids by k-means (seed 42) for K = 4 and K = 8. The test split is NOT used.
Reported: per-block AUC / PR-AUC penalty (matched - unseen) and the trip-weighted mean penalty per K and model.

Output: outputs/tables/T15_block_sensitivity.csv

Run:  python src/step_09_robust_blocks.py
"""
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, SEED, TARGET, load, metrics  # noqa: E402

SEL = json.loads((OUT / "reports" / "03_selection_main.json").read_text())
F, PARAMS, NT = SEL["features"], SEL["lgbm_params"], SEL["n_trees"]
tr, va = load("inner_train", F), load("valid", F)
binary = [f for f in F if set(pd.unique(tr[f].dropna())) <= {0, 1}]
contin = [f for f in F if f not in binary]
tracts = gpd.read_file(ROOT / "data" / "external" / "tiger" / "tracts_norfolk.gpkg").to_crs(32618)
tracts = tracts[tracts.GEOID.isin(set(tr.o_tract) | set(tr.d_tract) | set(va.o_tract))].reset_index(drop=True)
xy = np.column_stack([tracts.geometry.centroid.x, tracts.geometry.centroid.y])


def fit_lgb(df):
    return lgb.train(PARAMS, lgb.Dataset(df[F], df[TARGET], params={"feature_pre_filter": False}), NT)


def fit_lr(df):
    pre = ColumnTransformer([("cont", Pipeline([("imp", SimpleImputer(strategy="median")),
                                                ("spl", SplineTransformer(n_knots=5, degree=3)),
                                                ("sc", StandardScaler())]), contin),
                             ("bin", SimpleImputer(strategy="most_frequent"), binary)])
    s = df.sample(n=min(600_000, len(df)), random_state=SEED)
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=1.0, max_iter=3000))]).fit(s[F], s[TARGET])


t0 = time.time()
rows = []
for K in (4, 8):
    lab = KMeans(K, random_state=SEED, n_init=50).fit_predict(xy)
    bmap = pd.Series(lab + 1, index=tracts.GEOID.values)
    ob, db, vb_ = tr.o_tract.map(bmap), tr.d_tract.map(bmap), va.o_tract.map(bmap)
    for b in range(1, K + 1):
        tb = tr[(ob != b) & (db != b)]
        tm = tr.sample(n=len(tb), random_state=SEED + 100 * K + b)
        vb = va[vb_ == b]
        if len(vb) < 2000 or vb[TARGET].sum() < 50:
            continue
        for name, fit in (("LightGBM", fit_lgb), ("LogReg", fit_lr)):
            mu, mm = fit(tb), fit(tm)
            pr = (lambda m: m.predict(vb[F])) if name == "LightGBM" else (lambda m: m.predict_proba(vb[F])[:, 1])
            a, c = metrics(vb[TARGET].values, pr(mm)), metrics(vb[TARGET].values, pr(mu))
            rows.append(dict(K=K, block=b, model=name, n_valid=len(vb), ebike=vb[TARGET].mean(),
                             share_train_removed=1 - len(tb) / len(tr), auc_matched=a["roc_auc"], auc_unseen=c["roc_auc"],
                             pr_matched=a["pr_auc"], pr_unseen=c["pr_auc"], int_matched=a["cal_intercept"],
                             int_unseen=c["cal_intercept"]))
        print(f"K={K} block {b}: valid {len(vb):,} ({(time.time() - t0) / 60:.1f} min)", flush=True)
res = pd.DataFrame(rows)
res["auc_penalty"] = res.auc_matched - res.auc_unseen
res["pr_penalty"] = res.pr_matched - res.pr_unseen
res.to_csv(OUT / "tables" / "T15_block_sensitivity.csv", index=False)
k6 = pd.read_csv(OUT / "tables" / "T06_spatial_transfer_valid.csv").pivot_table(
    index=["model", "block"], columns="condition", values=["roc_auc", "n"]).reset_index()
summ = res.groupby(["K", "model"]).apply(lambda g: pd.Series({
    "blocks": len(g), "blocks_with_penalty": int((g.auc_penalty > 0).sum()),
    "auc_penalty_weighted": np.average(g.auc_penalty, weights=g.n_valid),
    "auc_penalty_max": g.auc_penalty.max(), "pr_penalty_weighted": np.average(g.pr_penalty, weights=g.n_valid)}))
print(summ.round(4).to_string())
