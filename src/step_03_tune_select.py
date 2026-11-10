"""Step 03: tune LightGBM on validation, apply the pre-declared selection rule, refit on final_train.

LightGBM: Optuna TPE, 40 trials, objective validation log loss, early stopping on validation (lr 0.05).
Search includes an optional monotone-increasing constraint on the lagged citywide e-bike share.
Spline logistic regression: fitted on a 600k inner-train subsample (validation) and refitted on a 600k
final-train subsample.
Selection rule (PROTOCOL.md): lower validation log loss is primary; if |difference| < 0.001 the spline logistic
regression is primary. Threshold: maximises F1 on validation. The test split is NOT loaded.

Outputs: outputs/models/{lgbm,logreg}_{set}.*, outputs/reports/03_selection_{set}.json, tables/T05_tuning_{set}.csv

Run:  python src/step_03_tune_select.py --set main
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FS, OUT, SEED, TARGET, load, metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="main", choices=["main", "main_plus_history"])
ap.add_argument("--trials", type=int, default=40)
A = ap.parse_args()
F = FS[A.set]
tr, va = load("inner_train", F), load("valid", F)
y, yv = tr[TARGET].values, va[TARGET].values
dtr = lgb.Dataset(tr[F], y, free_raw_data=False, params={"feature_pre_filter": False})
dva = lgb.Dataset(va[F], yv, reference=dtr, free_raw_data=False)
MONO = [1 if f == "city_ebike_share_7d" else 0 for f in F]


def lgb_params(t):
    p = dict(objective="binary", metric="binary_logloss", verbosity=-1, seed=SEED, bagging_freq=1,
             deterministic=True, force_row_wise=True, learning_rate=0.05, feature_pre_filter=False,
             num_leaves=t.suggest_int("num_leaves", 7, 127, log=True),
             min_data_in_leaf=t.suggest_int("min_data_in_leaf", 200, 20000, log=True),
             feature_fraction=t.suggest_float("feature_fraction", 0.4, 1.0),
             bagging_fraction=t.suggest_float("bagging_fraction", 0.5, 1.0),
             lambda_l2=t.suggest_float("lambda_l2", 1e-2, 100, log=True),
             min_gain_to_split=t.suggest_float("min_gain_to_split", 0.0, 1.0))
    if t.suggest_categorical("monotone_supply", [True, False]):
        p["monotone_constraints"] = MONO
    return p


def objective(t):
    m = lgb.train(lgb_params(t), dtr, 4000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
    t.set_user_attr("best_iter", m.best_iteration)
    return m.best_score["valid_0"]["binary_logloss"]


t0 = time.time()
optuna.logging.set_verbosity(optuna.logging.WARNING)
st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
st.optimize(objective, n_trials=A.trials)
st.trials_dataframe().to_csv(OUT / "tables" / f"T05_tuning_{A.set}.csv", index=False)
best_t = optuna.trial.FixedTrial(st.best_params)
best = lgb_params(best_t)
n_trees = st.best_trial.user_attrs["best_iter"]
m_in = lgb.train(best, dtr, n_trees)
p_lgb = m_in.predict(va[F])
print(f"[{A.set}] tuning {(time.time() - t0) / 60:.1f} min; best {st.best_params}; trees {n_trees}")

binary = [f for f in F if set(pd.unique(tr[f].dropna())) <= {0, 1}]
contin = [f for f in F if f not in binary]


def make_lr():
    pre = ColumnTransformer([("cont", Pipeline([("imp", SimpleImputer(strategy="median")),
                                                ("spl", SplineTransformer(n_knots=5, degree=3)),
                                                ("sc", StandardScaler())]), contin),
                             ("bin", SimpleImputer(strategy="most_frequent"), binary)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=1.0, max_iter=3000))])


sub = tr.sample(n=600_000, random_state=SEED)
lr_in = make_lr().fit(sub[F], sub[TARGET])
p_lr = lr_in.predict_proba(va[F])[:, 1]
mv = {"LightGBM": metrics(yv, p_lgb), "LogReg": metrics(yv, p_lr)}
diff = mv["LightGBM"]["log_loss"] - mv["LogReg"]["log_loss"]
primary = "LightGBM" if diff < -0.001 else "LogReg"
pv = p_lgb if primary == "LightGBM" else p_lr
grid = np.linspace(0.02, 0.4, 191)
f1s = [f1_score(yv, (pv >= t).astype(int), zero_division=0) for t in grid]
thr = float(grid[int(np.argmax(f1s))])
for k, v in mv.items():
    print(f"  valid {k:8s} PR-AUC {v['pr_auc']:.4f} AUC {v['roc_auc']:.4f} logloss {v['log_loss']:.5f} "
          f"slope {v['cal_slope']:.3f} int {v['cal_intercept']:+.3f}")
print(f"  log-loss difference (LGBM - LR) {diff:+.5f} -> primary model: {primary}; F1 threshold {thr:.3f}")

# refit on final_train
ft = load("final_train", F)
m_fin = lgb.train(best, lgb.Dataset(ft[F], ft[TARGET], params={"feature_pre_filter": False}), n_trees)
m_fin.save_model(str(OUT / "models" / f"lgbm_{A.set}.txt"))
lr_fin = make_lr().fit(*(lambda s: (s[F], s[TARGET]))(ft.sample(n=600_000, random_state=SEED)))
joblib.dump(lr_fin, OUT / "models" / f"logreg_{A.set}.joblib")
(OUT / "reports" / f"03_selection_{A.set}.json").write_text(json.dumps({
    "feature_set": A.set, "features": F, "lgbm_params": best, "n_trees": n_trees, "search_trials": A.trials,
    "valid_metrics": mv, "logloss_diff_lgbm_minus_lr": diff, "primary_model": primary, "threshold_f1": thr,
    "final_train_rows": len(ft)}, indent=2, default=float))
print(f"[{A.set}] saved; total {(time.time() - t0) / 60:.1f} min")
