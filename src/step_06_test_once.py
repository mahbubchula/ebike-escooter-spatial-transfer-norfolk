"""Step 06: ONE-TIME evaluation on the 2026 test set (January to October).

Before the lock: the origin-only sensitivity models are fitted on final_train with the main model's settings
(no separate tuning; documented in PROTOCOL.md). Then the SHA-256 hash of every model file is written to
outputs/reports/06_TEST_LOCK.json; later runs stop if any model changed.

Reports for LightGBM and the spline logistic regression, per feature set (main, main_plus_history, origin_only):
overall metrics with day-block bootstrap 95% CIs (200 replicates), paired CI for PR-AUC(LightGBM) - PR-AUC(LogReg),
metrics by month, by spatial block, and observed vs predicted by origin tract.
Post-test spatial transfer (reported only, never used for choices): per block, a model trained on final_train
without trips that start or end in the block, evaluated on 2026 trips starting in the block, vs the locked model.

Run:  python src/step_06_test_once.py
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FS, OUT, SEED, TARGET, load, metrics  # noqa: E402

MOD, REP, TAB, PRED = OUT / "models", OUT / "reports", OUT / "tables", OUT / "predictions"
SEL = json.loads((REP / "03_selection_main.json").read_text())
rng = np.random.default_rng(SEED)


def lr_pipe(F, df):
    binary = [f for f in F if set(pd.unique(df[f].dropna())) <= {0, 1}]
    contin = [f for f in F if f not in binary]
    pre = ColumnTransformer([("cont", Pipeline([("imp", SimpleImputer(strategy="median")),
                                                ("spl", SplineTransformer(n_knots=5, degree=3)),
                                                ("sc", StandardScaler())]), contin),
                             ("bin", SimpleImputer(strategy="most_frequent"), binary)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=1.0, max_iter=3000))])


def lgb_params_for(F):
    p = dict(SEL["lgbm_params"])
    if "monotone_constraints" in p:
        p["monotone_constraints"] = [1 if f == "city_ebike_share_7d" else 0 for f in F]
    return p


# ------------------------------------------------------------------ origin-only models (before the lock)
if not (MOD / "lgbm_origin_only.txt").exists():
    Fo = FS["origin_only"]
    ft = load("final_train", Fo)
    lgb.train(lgb_params_for(Fo), lgb.Dataset(ft[Fo], ft[TARGET], params={"feature_pre_filter": False}),
              SEL["n_trees"]).save_model(str(MOD / "lgbm_origin_only.txt"))
    s = ft.sample(n=600_000, random_state=SEED)
    joblib.dump(lr_pipe(Fo, s).fit(s[Fo], s[TARGET]), MOD / "logreg_origin_only.joblib")
    print("origin-only models fitted on final_train")

SETS = ["main", "main_plus_history", "origin_only"]
files = sorted([MOD / f"lgbm_{s}.txt" for s in SETS] + [MOD / f"logreg_{s}.joblib" for s in SETS])
hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
lock = REP / "06_TEST_LOCK.json"
if lock.exists():
    if json.loads(lock.read_text())["model_sha256"] != hashes:
        sys.exit("STOP: models changed after the test set was evaluated.")
    print("Lock found and models unchanged: recomputing identical test results.")
else:
    lock.write_text(json.dumps({"first_evaluated": pd.Timestamp.now().isoformat(timespec="seconds"),
                                "model_sha256": hashes}, indent=2))
    print("Test lock written. First evaluation of the test set.")


def boot(y, pd_, day, reps=200):
    days = np.unique(day)
    idx = pd.Series(np.arange(len(day))).groupby(day).apply(np.array)
    res = {k: {m: [] for m in ("pr_auc", "roc_auc", "log_loss", "brier")} for k in pd_}
    diff = []
    for _ in range(reps):
        ii = np.concatenate(idx.loc[rng.choice(days, len(days), replace=True)].values)
        yb = y[ii]
        pr = {}
        for k, p in pd_.items():
            pb = p[ii]
            pr[k] = average_precision_score(yb, pb)
            res[k]["pr_auc"].append(pr[k])
            res[k]["roc_auc"].append(roc_auc_score(yb, pb))
            res[k]["log_loss"].append(log_loss(yb, np.clip(pb, 1e-6, 1 - 1e-6)))
            res[k]["brier"].append(brier_score_loss(yb, pb))
        diff.append(pr["LightGBM"] - pr["LogReg"])
    ci = {k: {m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in d.items()} for k, d in res.items()}
    return ci, (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))


thr = SEL["threshold_f1"]
THR = {s_: json.loads((REP / f"03_selection_{s_}.json").read_text())["threshold_f1"]
       if (REP / f"03_selection_{s_}.json").exists() else thr for s_ in SETS}
overall, groups, tracts = [], [], []
for s in SETS:
    F = FS[s]
    thr_s = THR[s]
    te = load("test", F)
    y, day = te[TARGET].values, te["day"].values
    p = {"LightGBM": lgb.Booster(model_file=str(MOD / f"lgbm_{s}.txt")).predict(te[F]),
         "LogReg": joblib.load(MOD / f"logreg_{s}.joblib").predict_proba(te[F])[:, 1]}
    ci, dci = boot(y, p, day)
    for k, pk in p.items():
        m = metrics(y, pk, thr_s)
        m.update(feature_set=s, model=k, primary=(k == SEL["primary_model"]),
                 **{f"{mm}_ci95": ci[k][mm] for mm in ci[k]})
        if k == "LightGBM":
            m["pr_auc_diff_lgbm_minus_lr_ci95"] = dci
        overall.append(m)
        print(f"[{s}] {k:8s} PR-AUC {m['pr_auc']:.4f} {tuple(round(v, 4) for v in ci[k]['pr_auc'])} AUC {m['roc_auc']:.4f} "
              f"logloss {m['log_loss']:.4f} int {m['cal_intercept']:+.3f} mean {m['mean_pred']:.4f} obs {m['observed_rate']:.4f}",
              flush=True)
    te["month"] = te.start_dt.dt.month
    for gcol in ("month", "o_block"):
        for gv, ix in te.groupby(gcol).indices.items():
            for k, pk in p.items():
                m = metrics(y[ix], pk[ix], thr_s)
                m.update(feature_set=s, model=k, group=gcol, value=int(gv))
                groups.append(m)
    if s == "main":
        tt = pd.DataFrame({"o_tract": te.o_tract, "y": y, "p_lgbm": p["LightGBM"], "p_lr": p["LogReg"]})
        tracts = tt.groupby("o_tract").agg(n=("y", "size"), observed=("y", "mean"), pred_lgbm=("p_lgbm", "mean"),
                                           pred_lr=("p_lr", "mean")).reset_index()
        pd.DataFrame({"start_dt": te.start_dt, "day": te.day, "o_tract": te.o_tract, "d_tract": te.d_tract,
                      "o_block": te.o_block, "y": y, "p_lgbm": p["LightGBM"].astype("float32"),
                      "p_logreg": p["LogReg"].astype("float32")}).to_parquet(PRED / "test_predictions_main.parquet", index=False)
pd.DataFrame(overall).to_csv(TAB / "T07_test_overall.csv", index=False)
pd.DataFrame(groups).to_csv(TAB / "T08_test_by_group.csv", index=False)
tracts.to_csv(TAB / "T09_test_by_tract.csv", index=False)

# ------------------------------------------------------------------ post-test spatial transfer (reported only)
F = FS["main"]
ft, te = load("final_train", F), load("test", F)
blk = pd.read_csv(TAB / "T03b_tract_blocks.csv", dtype={"GEOID": str}).set_index("GEOID")["block"]
ft["d_block"] = ft.d_tract.map(blk)
locked = {"LightGBM": lgb.Booster(model_file=str(MOD / "lgbm_main.txt")),
          "LogReg": joblib.load(MOD / "logreg_main.joblib")}
rows = []
for b in sorted(te.o_block.unique()):
    tb = ft[(ft.o_block != b) & (ft.d_block != b)]
    un = {"LightGBM": lgb.train(lgb_params_for(F), lgb.Dataset(tb[F], tb[TARGET], params={"feature_pre_filter": False}),
                                SEL["n_trees"])}
    s_ = tb.sample(n=min(600_000, len(tb)), random_state=SEED)
    un["LogReg"] = lr_pipe(F, s_).fit(s_[F], s_[TARGET])
    tm = ft.sample(n=len(tb), random_state=SEED + int(b))
    mt = {"LightGBM": lgb.train(lgb_params_for(F), lgb.Dataset(tm[F], tm[TARGET], params={"feature_pre_filter": False}),
                                SEL["n_trees"])}
    s_m = tm.sample(n=min(600_000, len(tm)), random_state=SEED)
    mt["LogReg"] = lr_pipe(F, s_m).fit(s_m[F], s_m[TARGET])
    vb = te[te.o_block == b]
    for k in ("LightGBM", "LogReg"):
        f_ = (lambda m: m.predict(vb[F])) if k == "LightGBM" else (lambda m: m.predict_proba(vb[F])[:, 1])
        for cond, mdl in (("seen", locked[k]), ("matched", mt[k]), ("unseen", un[k])):
            m = metrics(vb[TARGET].values, f_(mdl), thr)
            m.update(block=int(b), model=k, condition=cond)
            rows.append(m)
    print(f"post-test transfer block {b} done", flush=True)
pd.DataFrame(rows).to_csv(TAB / "T10_spatial_transfer_test.csv", index=False)
print(pd.DataFrame(rows).pivot_table(index=["model", "block"], columns="condition", values=["pr_auc", "roc_auc"]).round(4))
