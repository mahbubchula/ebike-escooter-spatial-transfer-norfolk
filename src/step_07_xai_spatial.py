"""Step 07: explanations of the LOCKED main models and their spatial structure (RQ1, RQ3).

Sample: 2026 test trips, stratified by origin tract (up to 1,500 per tract), so every tract has a stable profile.
Explanations (log-odds scale):
  LightGBM : exact TreeSHAP (pred_contrib).
  Spline LR: exact additive decomposition: the model is logit = b0 + sum_j g_j(x_j), so feature j contributes
             g_j(x_j) - mean_background g_j (interventional SHAP of an additive model).
Global : mean |SHAP| by feature and group with day-block bootstrap CI; agreement between the two models.
Spatial: tract mean SHAP per feature/group; global Moran's I (queen contiguity, 999 permutations) for tract mean
         SHAP and tract residuals (observed - predicted, all 2026 trips); LISA clusters; k-means "zones" of tract
         SHAP profiles (k chosen by silhouette, 3-6).
Effects: ALE (LightGBM) for top continuous features; spline shape functions g_j (LR).
Interactions: TreeSHAP interaction values on 2,000 trips.
Nothing here changes a model. SHAP describes model behaviour, not causal effects.

Run:  python src/step_07_xai_spatial.py
"""
import json
import sys
import time
from pathlib import Path

import esda
import geopandas as gpd
import joblib
import lightgbm as lgb
import libpysal
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FS, OUT, ROOT, SEED, TARGET, load  # noqa: E402

XAI, TAB, MOD = OUT / "xai", OUT / "tables", OUT / "models"
F = FS["main"]
rng = np.random.default_rng(SEED)
t0 = time.time()


def tic(m):
    print(f"[{(time.time() - t0) / 60:5.1f} min] {m}", flush=True)


def group_of(f):
    if f in ("hour", "is_weekend", "is_holiday", "is_daylight"):
        return "Time"
    if f in ("od_dist_km", "same_tract", "o_dist_cbd_km", "d_dist_cbd_km"):
        return "Trip geometry"
    if f.startswith("o_"):
        return "Origin area"
    if f.startswith("d_"):
        return "Destination area"
    if f.startswith("city_"):
        return "Fleet supply"
    return "Weather"


te = load("test", F)
shuf = te.sample(frac=1.0, random_state=SEED)
smp = shuf[shuf.groupby("o_tract").cumcount() < 1500].reset_index(drop=True)
assert smp.o_tract.notna().all() and smp.groupby("o_tract").size().max() <= 1500
# design weights so global summaries represent all 2026 trips (stratified sample: up to 1,500 per tract)
N_t, n_t = te.o_tract.value_counts(), smp.o_tract.value_counts()
smp["w"] = smp.o_tract.map(N_t / n_t).astype(float)
assert abs(smp.w.sum() - len(te)) < 1e-6 * len(te)
tic(f"sample {len(smp):,} trips from {smp.o_tract.nunique()} tracts (design-weighted to {len(te):,} trips)")

# ------------------------------------------------------------------ SHAP: LightGBM
bst = lgb.Booster(model_file=str(MOD / "lgbm_main.txt"))
c = bst.predict(smp[F], pred_contrib=True)
assert np.abs(c.sum(1) - bst.predict(smp[F], raw_score=True)).max() < 1e-8
sh_lgb = pd.DataFrame(c[:, :-1], columns=F)
base_lgb = c[0, -1]

# ------------------------------------------------------------------ exact additive decomposition: spline LR
lr = joblib.load(MOD / "logreg_main.joblib")
pre, est = lr.named_steps["pre"], lr.named_steps["lr"]
Z = pre.transform(smp[F])
names = pre.get_feature_names_out()
coef = est.coef_[0]
bg = pre.transform(load("final_train", F).sample(100_000, random_state=SEED)[F])
cols_of = {}
cont = pre.transformers_[0][2]
binr = pre.transformers_[1][2]
for j, f in enumerate(cont):
    cols_of[f] = [i for i, n in enumerate(names) if n.startswith(f"cont__{f}_sp_")]
for f in binr:
    cols_of[f] = [i for i, n in enumerate(names) if n == f"bin__{f}"]
assert sorted(sum(cols_of.values(), [])) == list(range(len(names)))
sh_lr = pd.DataFrame({f: Z[:, ix] @ coef[ix] - (bg[:, ix] @ coef[ix]).mean() for f, ix in cols_of.items()})[F]
raw_lr = est.decision_function(Z)
base_lr = raw_lr.mean() - sh_lr.sum(1).mean()
assert np.abs(base_lr + sh_lr.sum(1).values - raw_lr).max() < 1e-6
tic("SHAP computed for both models (additivity verified)")

meta = smp[["start_dt", "day", "o_tract", "d_tract", "o_block", TARGET, "w"]].copy()
meta["p_lgbm"] = 1 / (1 + np.exp(-(base_lgb + sh_lgb.sum(1))))
meta["p_lr"] = 1 / (1 + np.exp(-raw_lr))
pd.concat([meta, smp[F].add_prefix("x_"), sh_lgb.add_prefix("lgb_"), sh_lr.add_prefix("lr_")], axis=1).to_parquet(
    XAI / "shap_test_sample.parquet", index=False)

# ------------------------------------------------------------------ global importance with day-block bootstrap
ud, inv = np.unique(meta.day.values, return_inverse=True)
rows = []
for mname, S in (("LightGBM", sh_lgb), ("LogReg", sh_lr)):
    A = np.abs(S.values)
    G = {g: [i for i, f in enumerate(F) if group_of(f) == g] for g in dict.fromkeys(map(group_of, F))}
    AG = np.column_stack([np.abs(S.values[:, ix].sum(1)) for ix in G.values()])
    wt = smp.w.values
    Sd, Sg = np.zeros((len(ud), A.shape[1])), np.zeros((len(ud), AG.shape[1]))
    Nd = np.bincount(inv, weights=wt).astype(float)
    np.add.at(Sd, inv, A * wt[:, None])
    np.add.at(Sg, inv, AG * wt[:, None])
    W = np.array([np.bincount(rng.integers(0, len(ud), len(ud)), minlength=len(ud)) for _ in range(300)])
    bF, bG = (W @ Sd) / (W @ Nd)[:, None], (W @ Sg) / (W @ Nd)[:, None]
    for lab, est_, bb, kind in ((F, np.average(A, axis=0, weights=wt), bF, "feature"),
                                (list(G), np.average(AG, axis=0, weights=wt), bG, "group")):
        for i, n in enumerate(lab):
            rows.append(dict(model=mname, kind=kind, name=n, group=group_of(n) if kind == "feature" else n,
                             mean_abs_shap=est_[i], lo=np.percentile(bb[:, i], 2.5), hi=np.percentile(bb[:, i], 97.5)))
gi = pd.DataFrame(rows)
gi["rank"] = gi.groupby(["model", "kind"])["mean_abs_shap"].rank(ascending=False).astype(int)
gi.to_csv(TAB / "T11_shap_importance.csv", index=False)
fi = gi[gi.kind == "feature"].pivot(index="name", columns="model", values="mean_abs_shap")
from scipy.stats import spearmanr  # noqa: E402
agree = {"spearman_importance": float(spearmanr(fi["LightGBM"], fi["LogReg"])[0]),
         "trip_level_shap_corr": {f: float(np.corrcoef(sh_lgb[f], sh_lr[f])[0, 1]) for f in F if sh_lgb[f].std() > 0 and sh_lr[f].std() > 0}}
tic(f"global importance saved; LGBM vs LR importance Spearman {agree['spearman_importance']:.3f}")

# ------------------------------------------------------------------ tract profiles, Moran's I, LISA
tr = gpd.read_file(ROOT / "data" / "external" / "tiger" / "tracts_norfolk.gpkg")
tr = tr[tr.GEOID.isin(smp.o_tract.unique())].sort_values("GEOID").reset_index(drop=True)
w = libpysal.weights.Queen.from_dataframe(tr, ids=tr.GEOID.tolist(), use_index=False)
islands = list(w.islands)
if islands:
    w = libpysal.weights.KNN.from_dataframe(tr, k=4, ids=tr.GEOID.tolist())
w.transform = "r"
order = tr.GEOID.tolist()
prof = {}
for mname, S in (("LightGBM", sh_lgb), ("LogReg", sh_lr)):
    tp = S.assign(o_tract=meta.o_tract.values).groupby("o_tract").mean().reindex(order)
    for g in dict.fromkeys(map(group_of, F)):
        tp[f"G_{g}"] = tp[[f for f in F if group_of(f) == g]].sum(1)
    prof[mname] = tp
    tp.to_csv(XAI / f"tract_shap_profile_{mname}.csv")
pt = pd.read_csv(TAB / "T09_test_by_tract.csv", dtype={"o_tract": str}).set_index("o_tract").reindex(order)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
pt["resid_lgbm"] = lg(pt.observed) - lg(pt.pred_lgbm)
pt["resid_lr"] = lg(pt.observed) - lg(pt.pred_lr)
pt["resid_pp_lgbm"] = 100 * (pt.observed - pt.pred_lgbm)
moran_rows, lisa = [], pd.DataFrame(index=order)
targets = {f"LightGBM: {c}": prof["LightGBM"][c] for c in prof["LightGBM"].columns if c.startswith("G_")}
targets.update({f"LightGBM: {f}": prof["LightGBM"][f] for f in gi[(gi.model == "LightGBM") & (gi.kind == "feature")].nsmallest(8, "rank").name})
targets.update({f"LogReg: {c}": prof["LogReg"][c] for c in prof["LogReg"].columns if c.startswith("G_")})
targets.update({"Residual (log-odds), LightGBM": pt.resid_lgbm, "Residual (log-odds), LogReg": pt.resid_lr,
                "Observed e-bike share": pt.observed})
for name, v in targets.items():
    v = v.fillna(v.mean()).values
    mi = esda.Moran(v, w, permutations=999)
    moran_rows.append(dict(variable=name, moran_I=mi.I, expected_I=mi.EI, z=mi.z_sim, p_sim=mi.p_sim))
    if name in ("Residual (log-odds), LightGBM", "LightGBM: G_Origin area", "LightGBM: G_Trip geometry",
                "Observed e-bike share", "LightGBM: G_Destination area"):
        ll = esda.Moran_Local(v, w, permutations=999, seed=SEED)
        lab = np.array(["Not significant"] * len(v), dtype=object)
        sig = ll.p_sim < 0.05
        for q, nm in ((1, "High-High"), (2, "Low-High"), (3, "Low-Low"), (4, "High-Low")):
            lab[sig & (ll.q == q)] = nm
        lisa[name] = lab
pd.DataFrame(moran_rows).to_csv(TAB / "T12_moran.csv", index=False)
lisa.to_csv(XAI / "lisa_clusters.csv")
tic("Moran's I and LISA saved")
print(pd.DataFrame(moran_rows).round(3).to_string(index=False))

# ------------------------------------------------------------------ zones: clusters of tract SHAP profiles
top = gi[(gi.model == "LightGBM") & (gi.kind == "feature")].nsmallest(10, "rank").name.tolist()
Xz = StandardScaler().fit_transform(prof["LightGBM"][top].fillna(0))
sil = {k: silhouette_score(Xz, KMeans(k, random_state=SEED, n_init=50).fit_predict(Xz)) for k in range(3, 7)}
kbest = max(sil, key=sil.get)
zl = KMeans(kbest, random_state=SEED, n_init=50).fit_predict(Xz)
# same procedure on the primary model's (spline LR) tract profiles, same k; agreement by adjusted Rand index
from sklearn.metrics import adjusted_rand_score  # noqa: E402
top_lr = gi[(gi.model == "LogReg") & (gi.kind == "feature")].nsmallest(10, "rank").name.tolist()
Xz_lr = StandardScaler().fit_transform(prof["LogReg"][top_lr].fillna(0))
zl_lr = KMeans(kbest, random_state=SEED, n_init=50).fit_predict(Xz_lr)
ari = float(adjusted_rand_score(zl, zl_lr))
zones = prof["LightGBM"][top].assign(zone=zl + 1, zone_lr=zl_lr + 1, observed=pt.observed.values, n=pt.n.values)
zones.to_csv(XAI / "tract_zones.csv")
zsum = zones.groupby("zone").agg(tracts=("n", "size"), trips=("n", "sum"), ebike_share=("observed", "mean"),
                                 **{f: (f, "mean") for f in top})
zsum.to_csv(TAB / "T13_zones.csv")
tic(f"zones: k={kbest} (silhouette {sil}); LGBM vs LR zone agreement ARI {ari:.3f}")
print(zsum.round(3).T.to_string())

# ------------------------------------------------------------------ ALE (LightGBM) and LR shape functions
ale_rows = []
contf = [f for f in gi[(gi.model == "LightGBM") & (gi.kind == "feature")].sort_values("rank").name if smp[f].nunique() > 2][:8]
X = smp[F].reset_index(drop=True)
for f in contf:
    x = X[f].values.astype(float)
    edges = np.arange(-0.5, 24, 1.0) if f == "hour" else np.unique(np.nanquantile(x, np.linspace(0, 1, 21)))
    k = np.clip(np.searchsorted(edges, x, side="left") - 1, 0, len(edges) - 2)
    lo, hi = X.copy(), X.copy()
    lo[f], hi[f] = edges[k], edges[k + 1]
    delta = bst.predict(hi, raw_score=True) - bst.predict(lo, raw_score=True)
    nb = len(edges) - 1
    for blk_name, msk in [("all", np.ones(len(X), bool))] + [(f"B{b}", smp.o_block.values == b) for b in sorted(smp.o_block.unique())]:
        okm = msk & ~np.isnan(x)
        s_ = np.bincount(k[okm], weights=delta[okm], minlength=nb)
        c_ = np.bincount(k[okm], minlength=nb).astype(float)
        a = np.concatenate([[0], np.cumsum(np.where(c_ > 0, s_ / np.maximum(c_, 1), 0))])
        a = a - (((a[1:] + a[:-1]) / 2) * c_).sum() / max(c_.sum(), 1)
        for e, v in zip(edges, a):
            ale_rows.append(dict(feature=f, subset=blk_name, x=e, ale=v))
    grid = np.quantile(x[~np.isnan(x)], np.linspace(0.01, 0.99, 60)) if f != "hour" else np.arange(24)
    G = pd.DataFrame(np.repeat(X.median(numeric_only=True).values[None, :], len(grid), 0), columns=F)
    G[f] = grid
    gz = pre.transform(G)
    ix = cols_of[f]
    shape = gz[:, ix] @ coef[ix] - (bg[:, ix] @ coef[ix]).mean()
    for e, v in zip(grid, shape):
        ale_rows.append(dict(feature=f, subset="LR shape", x=e, ale=v))
pd.DataFrame(ale_rows).to_csv(XAI / "ale_and_shapes.csv", index=False)
tic("ALE and LR shape functions saved")

# ------------------------------------------------------------------ interactions
import shap  # noqa: E402
sub = smp.sample(2000, random_state=SEED)
I = shap.TreeExplainer(bst).shap_interaction_values(sub[F])
np.savez_compressed(XAI / "shap_interactions.npz", values=I.astype("float32"), X=sub[F].values.astype("float32"),
                    features=np.array(F), block=sub.o_block.values)
M = np.abs(I).mean(0)
pairs = sorted(((2 * M[i, j], F[i], F[j]) for i in range(len(F)) for j in range(i + 1, len(F))), reverse=True)
pd.DataFrame([dict(f1=a, f2=b, mean_abs_interaction=v, share=v / M.sum()) for v, a, b in pairs]).to_csv(
    TAB / "T14_interactions.csv", index=False)
(OUT / "reports" / "07_xai.json").write_text(json.dumps({"agreement": agree, "k_zones": int(kbest), "silhouette": sil, "zone_ari_lgbm_vs_lr": ari,
                                                          "main_effect_share": float(np.trace(M) / M.sum()),
                                                          "w_islands": islands, "base_lgbm": float(base_lgb),
                                                          "base_lr": float(base_lr)}, indent=2))
tic(f"interactions saved; main-effect share {np.trace(M) / M.sum():.3f}")
