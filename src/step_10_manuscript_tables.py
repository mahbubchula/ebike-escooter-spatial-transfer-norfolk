"""Step 10: LaTeX tables for the manuscript, built only from saved results and the modelling data.

Writes outputs/manuscript_tables/tab_*.tex

Run:  python src/step_10_manuscript_tables.py
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, SPL, TARGET  # noqa: E402

MS = OUT / "manuscript_tables"
MS.mkdir(parents=True, exist_ok=True)
TAB, REP = OUT / "tables", OUT / "reports"


def write(name, body, caption, label, cols, note=None, size="\\small", wide=False):
    env = "tabularx" if "L" in cols else "tabular"
    width = "{\\linewidth}" if env == "tabularx" else ""
    tab = [f"\\begin{{{env}}}{width}{{{cols}}}", "\\toprule", *body, "\\bottomrule", f"\\end{{{env}}}"]
    if wide:
        tab = ["\\resizebox{\\textwidth}{!}{%", *tab, "}"]
    tex = ["\\begin{table}[!htbp]", "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}", *tab]
    if note:
        tex.append(f"\\par\\vspace{{2pt}}\\parbox{{\\linewidth}}{{\\footnotesize {note}}}")
    tex.append("\\end{table}")
    (MS / f"{name}.tex").write_text("\n".join(tex) + "\n")
    print("wrote", name)


def ci(s):
    return ast.literal_eval(s) if isinstance(s, str) else (np.nan, np.nan)


# ------------------------------------------------------------------ Table: sample by vehicle
cols = [TARGET, "trip_id", "start_dt", "od_dist_km", "hour", "is_weekend", "is_daylight", "temp_c", "precip_mm",
        "wind_kmh", "o_dist_cbd_km", "o_median_income", "o_pct_hh_no_car", "city_ebike_share_7d"]
d = pd.concat([pd.read_parquet(SPL / f"{s}.parquet", columns=cols) for s in ("final_train", "test")], ignore_index=True)
raw = pd.read_parquet(ROOT / "data" / "processed" / "norfolk_model_data.parquet",
                      columns=["trip_id", "dist_km", "duration_min", "same_tract"])
d = d.merge(raw, on="trip_id", how="left", validate="one_to_one")
d["o_median_income"] /= 1000
V = [("od_dist_km", "OD centroid distance (km)", 2), ("dist_km", "Ridden distance (km)$^{a}$", 2),
     ("duration_min", "Duration (min)$^{a}$", 1), ("hour", "Start hour", 1), ("temp_c", "Temperature (\\textdegree C)", 1),
     ("precip_mm", "Precipitation (mm/h)", 2), ("wind_kmh", "Wind speed (km/h)", 1),
     ("o_dist_cbd_km", "Origin distance to downtown (km)", 2), ("o_median_income", "Origin median income (US\\$000)", 1),
     ("o_pct_hh_no_car", "Origin car-free households (\\%)", 1),
     ("city_ebike_share_7d", "Lagged fleet e-bike share, 7 days (\\%)", 1)]
B = [("same_tract", "Same origin and destination tract (\\%)"), ("is_weekend", "Weekend (\\%)"),
     ("is_daylight", "Daylight (\\%)")]
body = ["Variable & E-scooter & E-bike & SMD \\\\", "\\midrule",
        f"Trips & {(d[TARGET] == 0).sum():,} & {(d[TARGET] == 1).sum():,} & \\\\"]
for v, lab_, dp in V:
    sc = 100 if v == "city_ebike_share_7d" else 1
    a, b = d.loc[d[TARGET] == 0, v] * sc, d.loc[d[TARGET] == 1, v] * sc
    smd = (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2)
    body.append(f"{lab_} & {a.mean():.{dp}f} ({a.std():.{dp}f}) & {b.mean():.{dp}f} ({b.std():.{dp}f}) & {smd:+.2f} \\\\")
for v, lab_ in B:
    a, b = d.loc[d[TARGET] == 0, v], d.loc[d[TARGET] == 1, v]
    smd = (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2)
    body.append(f"{lab_} & {100 * a.mean():.1f} & {100 * b.mean():.1f} & {smd:+.2f} \\\\")
write("tab_sample", body, "Trips by vehicle type, June 2021 to October 2026 (training and test periods). Mean (SD) for "
      "continuous variables, percentage for binary variables; SMD = standardized mean difference (e-bike minus e-scooter).",
      "tab:sample", "lrrr", note="$^{a}$ Recorded after the trip; descriptive only and never a model feature. The 2025 "
      "validation year is summarized in Table~\\ref{tab:split}.")
del d, raw

# ------------------------------------------------------------------ Table: split
t1 = pd.read_csv(TAB / "T01_split.csv")
use = {"inner_train": "Fit candidate models", "embargo": "Discarded (7-day gap)",
       "valid": "Tuning, model selection, threshold, spatial-transfer experiment",
       "final_train": "Refit of the selected configuration", "test": "Single evaluation (locked)"}
nm = {"inner_train": "Inner training", "embargo": "Embargo", "valid": "Validation", "final_train": "Final training",
      "test": "Test"}
body = ["Set & Period & Trips & E-bike (\\%) & Use \\\\", "\\midrule"]
for _, r in t1.iterrows():
    body.append(f"{nm[r.set]} & {r.start} to {r.end} & {int(r.trips):,} & {r.ebike_pct:.2f} & {use[r.set]} \\\\")
write("tab_split", body, "Temporal design.", "tab:split", "llrrL")

# ------------------------------------------------------------------ Table: performance
t4 = pd.read_csv(TAB / "T04_candidates_valid_main.csv").set_index("model")
t7 = pd.read_csv(TAB / "T07_test_overall.csv")
sel = {s: json.loads((REP / f"03_selection_{s}.json").read_text()) for s in ("main", "main_plus_history")}
body = ["& \\multicolumn{3}{c}{Validation 2025} & \\multicolumn{6}{c}{Test 2026} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-10}",
        "Model & PR-AUC & AUC & Log loss & PR-AUC [95\\% CI] & AUC & Log loss & Brier & Cal. slope & Mean pred. \\\\",
        "\\midrule", "\\multicolumn{10}{l}{\\textit{Untuned candidates, main feature set}} \\\\"]
for m, lab_ in (("Dummy", "Prior only"), ("LogReg (splines)", "Spline logistic regression"),
                ("LogReg + tract FE", "Logistic regression + tract fixed effects"), ("LightGBM", "LightGBM"),
                ("LightGBM (weighted)", "LightGBM, class-weighted"), ("XGBoost", "XGBoost"), ("CatBoost", "CatBoost")):
    r = t4.loc[m]
    body.append(f"{lab_} & {r.pr_auc:.3f} & {r.roc_auc:.3f} & {r.log_loss:.4f} & & & & & & \\\\")
body += ["\\midrule", "\\multicolumn{10}{l}{\\textit{Tuned and locked models}} \\\\"]
for fs, fsl in (("main", "main"), ("main_plus_history", "+ tract history"), ("origin_only", "origin only")):
    for m, ml in (("LogReg", "Spline LR"), ("LightGBM", "LightGBM")):
        r = t7[(t7.feature_set == fs) & (t7.model == m)].iloc[0]
        lo, hi = ci(r.pr_auc_ci95)
        if fs in sel:
            v = sel[fs]["valid_metrics"][m]
            val = f"{v['pr_auc']:.3f} & {v['roc_auc']:.3f} & {v['log_loss']:.4f}"
        else:
            val = "& &"
        star = "$^{\\dagger}$" if (fs == "main" and m == "LogReg") else ""
        body.append(f"{ml}, {fsl}{star} & {val} & {r.pr_auc:.3f} [{lo:.3f}, {hi:.3f}] & {r.roc_auc:.3f} & {r.log_loss:.4f} & "
                    f"{r.brier:.4f} & {r.cal_slope:.2f} & {100 * r.mean_pred:.1f}\\% \\\\")
obs = t7.observed_rate.iloc[0]
write("tab_performance", body, "Predictive performance. Validation results guided every modeling choice; the test set "
      "was evaluated once.", "tab:performance", "lrrrrrrrrr",
      note=f"$^{{\\dagger}}$ Primary model. Base rates: 3.77\\% (validation), {100 * obs:.2f}\\% (test). CI: day-block "
           "bootstrap (200 replicates). Origin-only models use the main model's settings without separate tuning.",
      size="\\scriptsize", wide=True)

# ------------------------------------------------------------------ Table: spatial transfer
tv = pd.read_csv(TAB / "T06_spatial_transfer_valid.csv")
tt = pd.read_csv(TAB / "T10_spatial_transfer_test.csv")
bl = pd.read_csv(TAB / "T03_spatial_blocks.csv").set_index("o_block")
body = ["& & & \\multicolumn{3}{c}{Spline LR, AUC (validation)} & \\multicolumn{3}{c}{LightGBM, AUC (validation)} & "
        "\\multicolumn{2}{c}{Penalty, test} \\\\",
        "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-11}",
        "Block & Tracts & Trips (\\%) & Seen & Matched & Unseen & Seen & Matched & Unseen & LR & LGBM \\\\", "\\midrule"]
for b in range(1, 7):
    cells = []
    for m in ("LogReg", "LightGBM"):
        for c in ("seen", "matched", "unseen"):
            cells.append(f"{tv[(tv.model == m) & (tv.block == b) & (tv.condition == c)].roc_auc.iloc[0]:.3f}")
    pens = []
    for m in ("LogReg", "LightGBM"):
        g = tt[(tt.model == m) & (tt.block == b)].set_index("condition").roc_auc
        pens.append(f"{g['matched'] - g['unseen']:+.3f}")
    body.append(f"B{b} & {int(bl.loc[b, 'tracts'])} & {bl.loc[b, 'trip_pct']:.1f} & " + " & ".join(cells) + " & " +
                " & ".join(pens) + " \\\\")
k = pd.read_csv(TAB / "T15_block_sensitivity.csv")
body.append("\\midrule")
body.append("\\multicolumn{11}{l}{\\textit{Trip-weighted AUC penalty (matched $-$ unseen), validation}} \\\\")
w6 = {}
for m in ("LogReg", "LightGBM"):
    p = tv[tv.model == m].pivot(index="block", columns="condition", values="roc_auc")
    n = tv[(tv.model == m) & (tv.condition == "unseen")].set_index("block").n
    w6[m] = np.average(p["matched"] - p["unseen"], weights=n.loc[p.index])
for K in (4, 6, 8):
    if K == 6:
        v = w6
    else:
        v = {m: np.average(k[(k.K == K) & (k.model == m)].auc_penalty, weights=k[(k.K == K) & (k.model == m)].n_valid)
             for m in ("LogReg", "LightGBM")}
    body.append(f"K = {K} blocks & & & \\multicolumn{{3}}{{c}}{{{v['LogReg']:.3f}}} & \\multicolumn{{3}}{{c}}{{{v['LightGBM']:.3f}}} & & \\\\")
write("tab_transfer", body, "Spatial transferability: AUC on trips starting in each block for models trained on all "
      "areas (seen), on a size-matched random subsample (matched) and without any trip starting or ending in the block "
      "(unseen).", "tab:transfer", "lrrrrrrrrrr",
      note="Penalty = matched minus unseen AUC. Blocks are k-means clusters of tract centroids (K = 6 in the main "
           "analysis); the bottom rows repeat the validation experiment with K = 4 and K = 8.", size="\\scriptsize", wide=True)

# ------------------------------------------------------------------ Table: explanations and spatial autocorrelation
gi = pd.read_csv(TAB / "T11_shap_importance.csv")
mo = pd.read_csv(TAB / "T12_moran.csv").set_index("variable")
body = ["& \\multicolumn{2}{c}{Mean $|$SHAP$|$ [95\\% CI]} & \\multicolumn{2}{c}{Moran's I of tract mean SHAP} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "Feature group & Spline LR & LightGBM & Spline LR & LightGBM \\\\", "\\midrule"]
for g in ("Trip geometry", "Fleet supply", "Destination area", "Origin area", "Weather", "Time"):
    cells = []
    for m in ("LogReg", "LightGBM"):
        r = gi[(gi.model == m) & (gi.kind == "group") & (gi.name == g)].iloc[0]
        cells.append(f"{r.mean_abs_shap:.3f} [{r.lo:.3f}, {r.hi:.3f}]")
    for m in ("LogReg", "LightGBM"):
        r = mo.loc[f"{m}: G_{g}"]
        cells.append(f"{r.moran_I:.2f}{'' if r.p_sim < 0.05 else ' (ns)'}")
    body.append(f"{g} & " + " & ".join(cells) + " \\\\")
body.append("\\midrule")
for nm_, lab_ in (("Residual (log-odds), LogReg", "Tract residual, spline LR"),
                  ("Residual (log-odds), LightGBM", "Tract residual, LightGBM"),
                  ("Observed e-bike share", "Observed tract e-bike share")):
    body.append(f"{lab_} & & & \\multicolumn{{2}}{{c}}{{{mo.loc[nm_, 'moran_I']:.2f} (p = {mo.loc[nm_, 'p_sim']:.3f})}} \\\\")
xj = json.loads((REP / "07_xai.json").read_text())
write("tab_xai", body, "Explanation importance and its spatial autocorrelation, 2026 test trips.", "tab:xai", "lcccc",
      note=f"Importance is design-weighted to all 2026 trips; CIs from a day-block bootstrap (300 replicates). Moran's I "
           f"uses queen contiguity of 80 tracts and 999 permutations; (ns) = p $\\geq$ 0.05. Rank correlation of feature-level "
           f"importance between the two models: {xj['agreement']['spearman_importance']:.2f}.", size="\\scriptsize", wide=True)
