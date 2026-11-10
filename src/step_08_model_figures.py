"""Step 08: model, transferability and explanation figures (studio theme), from saved results only.

Fig05_performance        validation candidates; test PR and calibration curves; monthly lift and calibration; CIs; tuning
Fig06_transferability    block maps of the unseen-area penalty; seen / size-matched / unseen by block (validation, test);
                         calibration in unseen areas
Fig07_explanations       beeswarm (primary model); group importance for both models; model agreement
Fig08_effects            ALE by spatial block (LightGBM) with spline shape functions (primary model)
Fig09_spatial_xai        tract maps of explanation groups; LISA clusters; Moran's I; zones
Fig10_interactions       interaction strengths; dependence of key pairs; zone profiles

Run:  python src/step_08_model_figures.py [--only fig05,fig06]
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fstudio as F  # noqa: E402
import plots as P  # noqa: E402
import plots_extra as PX  # noqa: E402

F.use_theme("studio")
mpl.rcParams.update({"font.weight": "bold"})
OUT = ROOT / "outputs"
FIG, TAB, XAI, REP = OUT / "figures", OUT / "tables", OUT / "xai", OUT / "reports"
ap = argparse.ArgumentParser()
ap.add_argument("--only", default="fig05,fig06,fig07,fig08,fig09,fig10")
ONLY = ap.parse_args().only.split(",")
FEATS = json.loads((ROOT / "configs" / "features.json").read_text())["main"]
SEL = json.loads((REP / "03_selection_main.json").read_text())
MC = {"LogReg": F.C6[3], "LightGBM": F.S4[2]}
MN = {"LogReg": "Spline LR (primary)", "LightGBM": "LightGBM"}
CC = {"seen": F.INK2, "matched": F.S4[3], "unseen": F.G1}
CN = {"seen": "Seen (all areas)", "matched": "Size-matched", "unseen": "Unseen area"}
GROUP_COL = {"Fleet supply": F.C6[0], "Trip geometry": F.C6[1], "Destination area": F.C6[2], "Origin area": F.C6[4],
             "Weather": F.C6[5], "Time": F.C6[3]}
LAB = {"city_ebike_share_7d": "Fleet e-bike share (7 d)", "city_trips_7d": "Fleet trips (7 d)", "od_dist_km": "OD distance",
       "o_dist_cbd_km": "Origin: distance to downtown", "hour": "Hour", "temp_c": "Temperature", "is_weekend": "Weekend",
       "is_holiday": "Holiday", "is_daylight": "Daylight", "precip_mm": "Rain (hour)", "precip_prev3h_mm": "Rain (prev. 3 h)",
       "snow_cm": "Snow", "wind_kmh": "Wind", "humidity_pct": "Humidity", "cloud_pct": "Cloud cover"}
CTX = {"pop_density_km2": "pop. density", "job_density_km2": "job density", "median_income": "median income",
       "median_age": "median age", "pct_age_18_29": "aged 18-29", "pct_hh_no_car": "car-free hh", "pct_commute_transit":
       "transit commute", "pct_commute_bicycle": "bike commute", "pct_commute_walk": "walk commute",
       "pct_jobs_retail_food": "retail/food jobs", "pct_degree": "degree", "pct_commute_wfh": "work from home"}
for k, v in CTX.items():
    LAB[f"o_{k}"], LAB[f"d_{k}"] = f"O: {v}", f"D: {v}"
captions = {}


def lab(f):
    return LAB.get(f, f)


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


def tracts():
    tr = gpd.read_file(ROOT / "data" / "external" / "tiger" / "tracts_norfolk.gpkg").to_crs(3857)
    tr = tr[tr["ALAND"] > 0.2 * (tr["ALAND"] + tr["AWATER"])].copy()
    blk = pd.read_csv(TAB / "T03b_tract_blocks.csv", dtype={"GEOID": str}).set_index("GEOID")["block"]
    tr["block"] = tr.GEOID.map(blk)
    return tr[tr.block.notna()]


def frame(ax, g):
    xmin, ymin, xmax, ymax = g.total_bounds
    ax.set_xlim(xmin - 500, xmax + 500)
    ax.set_ylim(ymin - 500, ymax + 500)
    ax.set_axis_off()
    ax.set_aspect("equal")


def hcb(fig, ax, cmap, norm, label, fmt=None, pos=(0.05, -0.04, 0.6, 0.03)):
    cax = ax.inset_axes(list(pos))
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      **({"format": fmt} if fmt else {}))
    cb.set_label(label, fontsize=6.6, labelpad=1)
    cb.ax.tick_params(labelsize=6.2, length=0, pad=1)
    cb.outline.set_visible(False)


def ci(s):
    return ast.literal_eval(s) if isinstance(s, str) else (np.nan, np.nan)


# =================================================================== Fig05 performance
if "fig05" in ONLY:
    fig = F.new_figure(180, 210)
    gs = fig.add_gridspec(3, 3)
    ax = fig.add_subplot(gs[0, 0])
    order = ["Dummy", "LogReg + tract FE", "LogReg (splines)", "XGBoost", "CatBoost", "LightGBM", "LightGBM (weighted)"]
    for k, (fs, mk, nm) in enumerate((("main", "o", "Main"), ("origin_only", "s", "Origin only"),
                                      ("main_plus_history", "D", "+ History"))):
        t = pd.read_csv(TAB / f"T04_candidates_valid_{fs}.csv").set_index("model").loc[order]
        ax.scatter(t.pr_auc, np.arange(len(order)) + (k - 1) * 0.22, marker=mk, s=26,
                   color=[F.MUTED if "Dummy" in m else (MC["LogReg"] if "LogReg" in m else MC["LightGBM"]) for m in order],
                   ec=F.BG, lw=0.5, label=nm, zorder=3)
    ax.axvline(0.0377, color=F.RULE, lw=0.8, ls="--")
    ax.set_yticks(range(len(order)), ["Dummy", "LR + tract FE", "Spline LR", "XGBoost", "CatBoost", "LightGBM", "LGBM weighted"],
                  fontsize=7)
    ax.set_xlabel("Validation PR-AUC (2025)")
    ax.legend(fontsize=6.5, loc="lower right", handletextpad=0.2)
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "a", "Candidates")

    pr = pd.read_parquet(OUT / "predictions" / "test_predictions_main.parquet")
    ax = fig.add_subplot(gs[0, 1])
    for k, col in (("p_logreg", "LogReg"), ("p_lgbm", "LightGBM")):
        p_, r_, _ = precision_recall_curve(pr.y, pr[k])
        ix = np.linspace(0, len(p_) - 1, 500).astype(int)
        ax.plot(r_[ix], p_[ix], color=MC[col], lw=1.7, label=f"{MN[col]} {average_precision_score(pr.y, pr[k]):.3f}")
    ax.axhline(pr.y.mean(), color=F.MUTED, lw=0.8, ls="--")
    ax.text(0.5, pr.y.mean() - 0.016, f"base rate {pr.y.mean():.3f}", ha="center", fontsize=6.6, color=F.INK2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 0.4)
    ax.legend(fontsize=6.5, loc="upper right", title="PR-AUC", title_fontsize=6.6)
    F.ygrid(ax, "both")
    F.panel(fig, ax, "b", "Precision-recall, 2026")

    ax = fig.add_subplot(gs[0, 2])
    for k, col in (("p_logreg", "LogReg"), ("p_lgbm", "LightGBM")):
        q = pd.qcut(pr[k], 10, labels=False, duplicates="drop")
        ax.plot(pr[k].groupby(q).mean(), pr.y.groupby(q).mean(), "o-", color=MC[col], ms=3.5, lw=1.4, mec=F.BG,
                label=MN[col])
    lim = 0.2
    ax.plot([0, lim], [0, lim], color=F.MUTED, lw=0.8, ls="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    F.pct_axis(ax)
    F.pct_axis(ax, "x")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed e-bike share")
    ax.legend(fontsize=6.5, loc="upper left")
    F.ygrid(ax, "both")
    F.panel(fig, ax, "c", "Calibration, 2026")

    g8 = pd.read_csv(TAB / "T08_test_by_group.csv")
    mo = g8[(g8.feature_set == "main") & (g8.group == "month")]
    ax = fig.add_subplot(gs[1, 0:2])
    for col in ("LogReg", "LightGBM"):
        d_ = mo[mo.model == col].sort_values("value")
        ax.plot(d_.value, d_.pr_auc / d_.observed_rate, "o-", color=MC[col], ms=4, lw=1.6, label=MN[col], mec=F.BG)
    ax.axhline(1, color=F.MUTED, lw=0.8, ls="--")
    d_ = mo[mo.model == "LogReg"].sort_values("value")
    ax2_vals = d_.observed_rate.values
    for x_, v_ in zip(d_.value, ax2_vals):
        ax.text(x_, 0.62, f"{100 * v_:.1f}%", ha="center", fontsize=6.4, color=F.INK2)
    ax.text(0.45, 0.62, "obs.", ha="right", fontsize=6.4, color=F.INK2)
    ax.set_xticks(range(1, 11), list("JFMAMJJASO"))
    ax.set_xlim(0.3, 10.5)
    ax.set_ylim(0.5, 3.0)
    ax.set_xlabel("2026 month (observed e-bike share along the bottom)")
    ax.set_ylabel("PR-AUC / base rate")
    ax.legend(fontsize=7, loc="upper right")
    F.ygrid(ax)
    F.panel(fig, ax, "d", "Lift over chance by month")

    ax = fig.add_subplot(gs[1, 2])
    for col in ("LogReg", "LightGBM"):
        d_ = mo[mo.model == col].sort_values("value")
        ax.plot(100 * d_.observed_rate, 100 * d_.mean_pred, "o", color=MC[col], ms=5, mec=F.BG, label=MN[col])
        for x_, y_, m_ in zip(100 * d_.observed_rate, 100 * d_.mean_pred, d_.value):
            if col == "LogReg":
                ax.text(x_ + 0.15, y_ - 0.35, "JFMAMJJASO"[int(m_) - 1], fontsize=6, color=F.INK2)
    ax.plot([2, 10], [2, 10], color=F.MUTED, lw=0.8, ls="--")
    ax.set_xlabel("Observed share (%)")
    ax.set_ylabel("Mean predicted (%)")
    ax.legend(fontsize=6.3, loc="upper left")
    F.ygrid(ax, "both")
    F.panel(fig, ax, "e", "Monthly level")

    ax = fig.add_subplot(gs[2, 0:2])
    t7 = pd.read_csv(TAB / "T07_test_overall.csv")
    rows = [(fs, m) for fs in ("main", "main_plus_history", "origin_only") for m in ("LogReg", "LightGBM")]
    yy = np.arange(len(rows))[::-1]
    for yi, (fs, m) in zip(yy, rows):
        r = t7[(t7.feature_set == fs) & (t7.model == m)].iloc[0]
        lo, hi = ci(r.pr_auc_ci95)
        ax.errorbar(r.pr_auc, yi, xerr=[[r.pr_auc - lo], [hi - r.pr_auc]], fmt="o" if m == "LogReg" else "D",
                    color=MC[m] if fs != "origin_only" else F.tint(MC[m], 0.4), ms=5.5, capsize=2.5, lw=1.2)
        ax.text(hi + 0.0015, yi, f"{r.pr_auc:.3f} (AUC {r.roc_auc:.3f})", va="center", fontsize=6.6)
    ax.axvline(t7.observed_rate.iloc[0], color=F.MUTED, lw=0.8, ls="--")
    ax.set_yticks(yy, [{"main": "Main", "main_plus_history": "+ Tract history", "origin_only": "Origin only"}[fs] +
                       f" · {'Spline LR' if m == 'LogReg' else 'LightGBM'}" for fs, m in rows], fontsize=7.2)
    ax.set_xlim(0.05, 0.135)
    ax.set_xlabel("Test PR-AUC, 2026 (95% CI)")
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "f", "Test PR-AUC")

    ax = fig.add_subplot(gs[2, 2])
    for fs, col, nm in (("main", F.S4[2], "Main"), ("main_plus_history", F.tint(F.S4[2], 0.45), "+ History")):
        t5 = pd.read_csv(TAB / f"T05_tuning_{fs}.csv").sort_values("number")
        ax.scatter(t5.number, t5.value, s=8, color=col, alpha=0.5, lw=0)
        ax.plot(t5.number, t5.value.cummin(), color=col, lw=1.6, drawstyle="steps-post", label=nm)
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("Validation log loss")
    ax.legend(fontsize=6.6, loc="upper right")
    F.ygrid(ax)
    F.panel(fig, ax, "g", "LightGBM tuning")
    F.save(fig, FIG / "Fig05_performance")
    captions["Fig05"] = (
        "**Figure 5. Predictive performance.** (a) Validation PR-AUC (2025) of untuned candidates for the main, origin-only "
        "and tract-history feature sets; the dashed line is the validation base rate (3.8%). Class weighting destroyed "
        "calibration and was rejected. (b) Precision-recall and (c) decile calibration curves of the locked main models on "
        "the 2026 test set. (d) PR-AUC divided by the monthly base rate (lift over chance; observed e-bike share printed "
        "along the bottom). (e) Mean predicted against observed monthly e-bike share. (f) Test PR-AUC with day-block "
        "bootstrap 95% CIs (200 replicates) for all locked models. (g) Optuna search for LightGBM (points = trials, line "
        f"= best so far). The spline logistic regression is the primary model under the pre-declared rule (validation log "
        f"loss within 0.001 of LightGBM); decision threshold {SEL['threshold_f1']:.3f} (maximum F1 on validation).")

# =================================================================== Fig06 transferability
if "fig06" in ONLY:
    tv = pd.read_csv(TAB / "T06_spatial_transfer_valid.csv")
    tt = pd.read_csv(TAB / "T10_spatial_transfer_test.csv")
    tr = tracts()
    fig = F.new_figure(180, 215)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.15, 1, 1])
    top = outer[0].subgridspec(1, 3, width_ratios=[1, 1, 1.05])
    for k, m in enumerate(("LogReg", "LightGBM")):
        ax = fig.add_subplot(top[k])
        v = tv[tv.model == m].pivot(index="block", columns="condition", values="roc_auc")
        pen = (v["matched"] - v["unseen"])
        g = tr.dissolve("block").reset_index()
        g["pen"] = g.block.map(pen)
        nrm = mpl.colors.TwoSlopeNorm(0, -0.03, 0.11)
        g.plot(ax=ax, column="pen", cmap=F.DIV, norm=nrm, edgecolor="white", lw=1.2)
        tr.boundary.plot(ax=ax, color="white", lw=0.25, alpha=0.6)
        for _, r in g.iterrows():
            c = r.geometry.representative_point()
            ax.text(c.x, c.y, f"B{int(r.block)}\n{r.pen:+.3f}", ha="center", va="center", fontsize=6.6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        frame(ax, g)
        hcb(fig, ax, F.DIV, nrm, "AUC penalty (size-matched − unseen)")
        F.panel(fig, ax, "ab"[k], f"{'Spline LR' if m == 'LogReg' else 'LightGBM'}: penalty")
    ax = fig.add_subplot(top[2])
    rows = []
    for m in ("LogReg", "LightGBM"):
        for b in range(1, 7):
            r = tv[(tv.model == m) & (tv.block == b) & (tv.condition == "unseen")].iloc[0]
            lo, hi = ci(r.diff_matched_roc_auc_ci)
            rows.append((m, b, lo, hi))
    yy = np.arange(6)[::-1]
    for j, m in enumerate(("LogReg", "LightGBM")):
        sub = [r for r in rows if r[0] == m]
        mid = [(r[2] + r[3]) / 2 for r in sub]
        ax.errorbar(mid, yy + (0.15 if j == 0 else -0.15), xerr=[[mid[i] - sub[i][2] for i in range(6)],
                                                              [sub[i][3] - mid[i] for i in range(6)]],
                    fmt="o" if j == 0 else "D", color=MC[m], ms=4.5, capsize=2, lw=1.1, label=MN[m])
    ax.axvline(0, color=F.INK2, lw=0.8)
    ax.set_yticks(yy, [f"B{b}" for b in range(1, 7)])
    ax.set_xlabel("AUC penalty, 95% CI")
    ax.set_ylim(-0.6, 6.4)
    ax.legend(fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "c", "Penalty with CIs")

    for row, (tab, phase) in enumerate(((tv, "Validation 2025"), (tt, "Test 2026"))):
        sub_gs = outer[row + 1].subgridspec(1, 2, width_ratios=[1.6, 1])
        ax = fig.add_subplot(sub_gs[0])
        x = np.arange(6)
        for j, m in enumerate(("LogReg", "LightGBM")):
            for c_i, cond in enumerate(("seen", "matched", "unseen")):
                v = tab[(tab.model == m) & (tab.condition == cond)].sort_values("block")
                ax.scatter(x + (j - 0.5) * 0.36 + (c_i - 1) * 0.09, v.roc_auc, s=22, marker="o" if j == 0 else "D",
                           color=CC[cond], ec=F.BG, lw=0.5, zorder=3)
            v_m = tab[(tab.model == m) & (tab.condition == "matched")].sort_values("block").roc_auc.values
            v_u = tab[(tab.model == m) & (tab.condition == "unseen")].sort_values("block").roc_auc.values
            ax.vlines(x + (j - 0.5) * 0.36, v_u, v_m, color=F.tint(MC[m], 0.3), lw=2.2, zorder=2)
        ax.set_xticks(x, [f"B{b}" for b in range(1, 7)])
        ax.set_ylabel("AUC")
        ax.set_ylim(0.49, 0.72)
        F.ygrid(ax)
        if row == 0:
            ax.legend(handles=[Line2D([], [], marker="o", ls="", color=CC[c], ms=5, label=CN[c]) for c in CC] +
                      [Line2D([], [], marker="o", ls="", color=F.INK2, mfc="none", ms=5, label="Spline LR (left)"),
                       Line2D([], [], marker="D", ls="", color=F.INK2, mfc="none", ms=5, label="LightGBM (right)")],
                      fontsize=6.3, ncol=3, loc="upper left", columnspacing=0.8)
        F.panel(fig, ax, "df"[row], f"{phase}: AUC by block")
        ax = fig.add_subplot(sub_gs[1])
        for j, m in enumerate(("LogReg", "LightGBM")):
            vm = tab[(tab.model == m) & (tab.condition == "matched")].sort_values("block")
            vu = tab[(tab.model == m) & (tab.condition == "unseen")].sort_values("block")
            yy = np.arange(6)[::-1] + (0.15 if j == 0 else -0.15)
            ax.hlines(yy, vm.cal_intercept, vu.cal_intercept, color=F.tint(MC[m], 0.4), lw=2)
            ax.scatter(vm.cal_intercept, yy, color=CC["matched"], s=18, zorder=3, marker="o" if j == 0 else "D")
            ax.scatter(vu.cal_intercept, yy, color=CC["unseen"], s=18, zorder=3, marker="o" if j == 0 else "D")
        ax.axvline(0, color=F.INK2, lw=0.8)
        ax.set_yticks(np.arange(6)[::-1], [f"B{b}" for b in range(1, 7)])
        ax.set_xlabel("Calibration intercept")
        F.clean_left(ax)
        F.ygrid(ax, "x")
        F.panel(fig, ax, "eg"[row], "Calibration")
    F.save(fig, FIG / "Fig06_transferability")
    captions["Fig06"] = (
        "**Figure 6. Spatial transferability.** For each of six spatial blocks, three models with identical settings are "
        "compared on trips starting in the block: trained on all areas (seen), on a random subsample of all areas of the "
        "same size as the unseen-area training set (size-matched), and on trips that neither start nor end in the block "
        "(unseen). (a, b) AUC penalty of not having seen the area (size-matched minus unseen) on 2025 validation trips, by "
        "block, for the spline logistic regression and LightGBM. (c) The same penalty with day-block bootstrap 95% CIs "
        "(200 replicates). (d, f) AUC by block and condition on 2025 validation and on the 2026 test set; bars span the "
        "size-matched to unseen difference. (e, g) Calibration intercept of size-matched (teal) and unseen (red) models; "
        "negative values mean e-bike use is over-predicted. Seen and size-matched models perform almost identically, so "
        "the penalty reflects spatial novelty rather than the loss of training data.")

# =================================================================== Fig07 explanations
if "fig07" in ONLY:
    sh = pd.read_parquet(XAI / "shap_test_sample.parquet")
    gi = pd.read_csv(TAB / "T11_shap_importance.csv")
    xj = json.loads((REP / "07_xai.json").read_text())
    fig = F.new_figure(180, 222)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.55, 1])
    row0 = outer[0].subgridspec(1, 2, width_ratios=[1.35, 1])
    row1 = outer[1].subgridspec(1, 2, width_ratios=[1, 1])
    ax = fig.add_subplot(row0[0])
    top = gi[(gi.model == "LogReg") & (gi.kind == "feature")].sort_values("rank").name.tolist()[:16]
    idx = np.random.default_rng(42).choice(len(sh), 12000, replace=False, p=sh.w / sh.w.sum())
    P.beeswarm(ax, sh.loc[idx, [f"lr_{f}" for f in top]].values, sh.loc[idx, [f"x_{f}" for f in top]].values,
               [lab(f) for f in top], order=list(range(len(top)))[::-1])
    ax.set_xlabel("SHAP value, spline LR (log-odds)")
    F.panel(fig, ax, "a", "Primary model, 2026")
    ax = fig.add_subplot(row0[1])
    gg = gi[gi.kind == "group"]
    order = gg[gg.model == "LogReg"].sort_values("mean_abs_shap").name.tolist()
    y = np.arange(len(order))
    for j, m in enumerate(("LogReg", "LightGBM")):
        d_ = gg[gg.model == m].set_index("name").loc[order]
        ax.barh(y + (0.19 if j == 0 else -0.19), d_.mean_abs_shap, height=0.36,
                color=[GROUP_COL[g] if j == 0 else F.tint(GROUP_COL[g], 0.5) for g in order],
                xerr=[d_.mean_abs_shap - d_.lo, d_.hi - d_.mean_abs_shap], error_kw=dict(lw=0.7, ecolor=F.INK2))
    ax.set_yticks(y, order, fontsize=7.4)
    ax.set_xlabel("Mean |SHAP| (95% CI)")
    ax.legend(handles=[Patch(color=F.INK2, label="Spline LR"), Patch(color=F.tint(F.INK2, 0.5), label="LightGBM")],
              fontsize=6.6, loc="lower right")
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "b", "Feature groups")
    ax = fig.add_subplot(row1[0])
    fi = gi[gi.kind == "feature"].pivot(index="name", columns="model", values="mean_abs_shap")
    ax.scatter(fi.LogReg, fi.LightGBM, s=26, color=[GROUP_COL[group_of(f)] for f in fi.index], ec=F.BG, lw=0.5, zorder=3)
    for f in fi.index:
        if fi.loc[f, "LightGBM"] > 0.07 or fi.loc[f, "LogReg"] > 0.15:
            ax.text(fi.loc[f, "LogReg"] * 1.08, fi.loc[f, "LightGBM"], lab(f), fontsize=6.2, va="center")
    lim = [0.002, 0.7]
    ax.plot(lim, lim, color=F.MUTED, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("Mean |SHAP|, spline LR")
    ax.set_ylabel("Mean |SHAP|, LightGBM")
    ax.text(0.03, 0.95, f"Spearman ρ = {xj['agreement']['spearman_importance']:.2f}", transform=ax.transAxes, fontsize=7,
            va="top")
    F.ygrid(ax, "both")
    F.panel(fig, ax, "c", "Importance agreement")
    ax = fig.add_subplot(row1[1])
    cr = pd.Series(xj["agreement"]["trip_level_shap_corr"])
    cr = cr.reindex(gi[(gi.model == "LogReg") & (gi.kind == "feature")].sort_values("rank").name).dropna().head(12)[::-1]
    yy = np.arange(len(cr))
    ax.hlines(yy, 0, cr.values, color=[GROUP_COL[group_of(f)] for f in cr.index], lw=2)
    ax.scatter(cr.values, yy, color=[GROUP_COL[group_of(f)] for f in cr.index], s=20, zorder=3)
    ax.axvline(0, color=F.INK2, lw=0.8)
    ax.set_yticks(yy, [lab(f) for f in cr.index], fontsize=6.6)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Trip-level SHAP correlation")
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "d", "Trip-level agreement")
    F.save(fig, FIG / "Fig07_explanations")
    captions["Fig07"] = (
        "**Figure 7. What separates e-bike from e-scooter trips.** Explanations on a stratified sample of 117,587 trips "
        "from 2026 (up to 1,500 per origin tract), design-weighted to all 2026 trips. (a) SHAP values of the primary "
        "spline logistic regression; because the model is additive in log-odds, each value is the feature's centred "
        "spline contribution (exact). (b) Mean absolute grouped SHAP value for both models with day-block bootstrap 95% "
        "CIs. (c) Mean absolute SHAP value per feature in both models (log scales; colour = group). (d) Correlation "
        "between the two models' trip-level SHAP values for the 12 most important features of the primary model. Group-"
        "level explanations agree; single correlated tract attributes and some time variables do not.")

# =================================================================== Fig08 effects
if "fig08" in ONLY:
    al = pd.read_csv(XAI / "ale_and_shapes.csv")
    feats = list(dict.fromkeys(al.feature))[:8]
    sh = pd.read_parquet(XAI / "shap_test_sample.parquet")
    fig = F.new_figure(180, 160)
    gs = fig.add_gridspec(2, 4)
    bcol = {f"B{i + 1}": c for i, c in enumerate(F.C6)}
    UNITS = {"city_ebike_share_7d": "Fleet e-bike share, 7 d", "o_dist_cbd_km": "Origin to downtown (km)",
             "od_dist_km": "OD distance (km)", "city_trips_7d": "Fleet trips, 7 d (000)", "temp_c": "Temperature (°C)",
             "hour": "Start hour"}
    for i, f in enumerate(feats):
        sub = gs[i // 4, i % 4].subgridspec(2, 1, height_ratios=[4, 0.8], hspace=0.05)
        ax = fig.add_subplot(sub[0])
        axn = fig.add_subplot(sub[1], sharex=ax)
        x_all = sh[f"x_{f}"].dropna()
        lo_x, hi_x = (np.nanpercentile(x_all, 1), np.nanpercentile(x_all, 99)) if f != "hour" else (-0.5, 23.5)
        for b, col in bcol.items():
            d_ = al[(al.feature == f) & (al.subset == b) & al.x.between(lo_x, hi_x)]
            ax.plot(d_.x, d_.ale, color=col, lw=1.0, alpha=0.85)
        d_ = al[(al.feature == f) & (al.subset == "all") & al.x.between(lo_x, hi_x)]
        PX.glow_line(ax, d_.x, d_.ale, F.INK, lw=1.6)
        d_ = al[(al.feature == f) & (al.subset == "LR shape") & al.x.between(lo_x, hi_x)]
        ax.plot(d_.x, d_.ale - np.average(d_.ale), color=MC["LogReg"], lw=1.6, ls=(0, (4, 1.5)))
        ax.axhline(0, color=F.RULE, lw=0.8, zorder=0)
        F.ygrid(ax)
        ax.tick_params(axis="x", labelbottom=False, length=0)
        if i % 4 == 0:
            ax.set_ylabel("Effect (log-odds)")
        bins = np.arange(-0.5, 24.5, 1) if f == "hour" else np.linspace(lo_x, hi_x, 30)
        axn.hist(x_all.clip(lo_x, hi_x), bins=bins, weights=sh.loc[x_all.index, "w"], color=F.tint(F.INK, 0.78), lw=0)
        axn.set_yticks([])
        axn.spines["left"].set_visible(False)
        axn.set_xlabel(UNITS.get(f, lab(f)), fontsize=7.4)
        if f == "city_ebike_share_7d":
            axn.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, 0))
        if f == "city_trips_7d":
            axn.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
        if f == "hour":
            axn.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlim(lo_x, hi_x)
        TT = {"city_ebike_share_7d": "Fleet e-bike share", "o_dist_cbd_km": "Distance to downtown",
              "od_dist_km": "OD distance", "city_trips_7d": "Fleet activity", "temp_c": "Temperature", "hour": "Hour"}
        F.panel(fig, ax, "abcdefgh"[i], TT.get(f, lab(f).replace("O: ", "Origin ").replace("D: ", "Dest. ")))
    h = [Line2D([], [], color=F.INK, lw=1.8, label="LightGBM ALE, all"), Line2D([], [], color=MC["LogReg"], lw=1.6,
         ls=(0, (4, 1.5)), label="Spline LR shape")] + [Line2D([], [], color=c, lw=1.2, label=b) for b, c in bcol.items()]
    fig.legend(handles=h, loc="outside lower center", ncol=8, fontsize=7)
    F.save(fig, FIG / "Fig08_effects")
    captions["Fig08"] = (
        "**Figure 8. Effects of the eight most important continuous features.** Black: accumulated local effects (ALE) "
        "of LightGBM on the 2026 explanation sample (20 quantile bins; hourly bins for hour). Coloured: ALE computed "
        "separately on trips starting in each spatial block (B1 to B6). Dashed purple: the spline shape function of the "
        "primary model (other features at their medians), centred. All curves are on the log-odds scale. Shape functions "
        "of tract attributes in the spline model are irregular because each attribute takes one value per tract and "
        "partly acts as a location identifier; group-level summaries are more reliable than single tract attributes. The strips "
        "show the design-weighted distribution of each feature; the x-axis spans the 1st to 99th percentile.")

# =================================================================== Fig09 spatial XAI
if "fig09" in ONLY:
    tr = tracts()
    prof = pd.read_csv(XAI / "tract_shap_profile_LogReg.csv", dtype={"o_tract": str}).set_index("o_tract")
    profg = pd.read_csv(XAI / "tract_shap_profile_LightGBM.csv", dtype={"o_tract": str}).set_index("o_tract")
    lisa = pd.read_csv(XAI / "lisa_clusters.csv", dtype={0: str}, index_col=0)
    lisa.index = lisa.index.astype(str)
    zones = pd.read_csv(XAI / "tract_zones.csv", dtype={"o_tract": str}).set_index("o_tract")
    mo = pd.read_csv(TAB / "T12_moran.csv")
    fig = F.new_figure(180, 225)
    outer9 = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.95])
    r0, r1, r2 = (outer9[i].subgridspec(1, 3) for i in range(3))

    class _G:
        def __getitem__(self, key):
            r, c = key
            rows = (r0, r1, r2)
            if isinstance(c, slice):
                return rows[r][c]
            return rows[r][c]
    gs = _G()
    maps = [("G_Trip geometry", "Trip geometry", prof), ("G_Destination area", "Destination area", prof),
            ("G_Fleet supply", "Fleet supply", prof), ("G_Origin area", "Origin area", prof)]
    for k, (col, ttl, src) in enumerate(maps):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        g = tr.merge(src[[col]], left_on="GEOID", right_index=True, how="left")
        lim = np.nanpercentile(np.abs(g[col]), 97)
        nrm = mpl.colors.TwoSlopeNorm(0, -lim, lim)
        g.plot(ax=ax, column=col, cmap=F.DIV, norm=nrm, edgecolor="white", lw=0.3, missing_kwds={"color": F.GRID})
        frame(ax, g)
        hcb(fig, ax, F.DIV, nrm, "Mean SHAP (log-odds)", pos=(0.02, 0.05, 0.38, 0.028))
        F.panel(fig, ax, "abcd"[k], ttl)
    LC = {"High-High": F.G1, "Low-Low": F.G0, "High-Low": F.tint(F.G1, 0.55), "Low-High": F.tint(F.G0, 0.55),
          "Not significant": F.tint(F.MUTED, 0.8)}
    for k, (col, ttl) in enumerate((("Residual (log-odds), LightGBM", "Residual clusters"),
                                    ("LightGBM: G_Origin area", "Origin-area clusters"))):
        ax = fig.add_subplot(gs[1, 1 + k])
        g = tr.merge(lisa[[col]], left_on="GEOID", right_index=True, how="left")
        g[col] = g[col].fillna("Not significant")
        g.plot(ax=ax, color=[LC[v] for v in g[col]], edgecolor="white", lw=0.3)
        frame(ax, g)
        if k == 0:
            ax.legend(handles=[Patch(color=c, label=l) for l, c in LC.items()], fontsize=5.8, loc="lower left",
                      ncol=1, frameon=False, handlelength=0.9, borderaxespad=0.1, labelspacing=0.25)
        F.panel(fig, ax, "ef"[k], ttl)
    ax = fig.add_subplot(gs[2, 0:2])
    sel = mo[mo.variable.str.startswith(("LightGBM: G_", "LogReg: G_", "Residual", "Observed"))].copy()
    sel["model"] = np.where(sel.variable.str.startswith("LogReg"), "LogReg",
                            np.where(sel.variable.str.startswith("LightGBM"), "LightGBM", "other"))
    sel["name"] = sel.variable.str.replace("LightGBM: G_", "").str.replace("LogReg: G_", "")
    names = ["Trip geometry", "Destination area", "Fleet supply", "Origin area", "Weather", "Time"]
    x = np.arange(len(names))
    for j, m in enumerate(("LogReg", "LightGBM")):
        v = sel[sel.model == m].set_index("name").reindex(names)
        bars = ax.bar(x + (j - 0.5) * 0.38, v.moran_I, width=0.36, color=MC[m], label=MN[m])
        for xi, (I_, p_) in zip(x + (j - 0.5) * 0.38, zip(v.moran_I, v.p_sim)):
            ax.text(xi, I_ + 0.02, "*" if p_ < 0.05 else "ns", ha="center", fontsize=6.4)
    oth = sel[sel.model == "other"].set_index("variable")
    for k, (nm, short) in enumerate((("Residual (log-odds), LogReg", "Residual LR"),
                                     ("Residual (log-odds), LightGBM", "Residual LGBM"),
                                     ("Observed e-bike share", "Observed share"))):
        ax.bar(len(names) + 0.3 + k * 0.5, oth.loc[nm, "moran_I"], width=0.42, color=[F.tint(F.G1, 0.3), F.G1, F.INK2][k])
        ax.text(len(names) + 0.3 + k * 0.5, oth.loc[nm, "moran_I"] + 0.02, "*", ha="center", fontsize=6.4)
    ax.set_xticks(list(x) + [len(names) + 0.3 + k * 0.5 for k in range(3)],
                  names + ["Resid. LR", "Resid. LGBM", "Observed"], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Moran's I (queen)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.8, loc="upper right")
    F.ygrid(ax)
    F.panel(fig, ax, "g", "Spatial autocorrelation")
    ax = fig.add_subplot(gs[2, 2])
    g = tr.merge(zones[["zone"]], left_on="GEOID", right_index=True, how="left")
    zc = {z: c for z, c in zip(range(1, 7), F.C6)}
    g.plot(ax=ax, color=[zc.get(int(z), F.GRID) if pd.notna(z) else F.GRID for z in g.zone], edgecolor="white", lw=0.3)
    frame(ax, g)
    ax.legend(handles=[Patch(color=zc[z], label=f"Zone {z}") for z in sorted(zones.zone.unique())], fontsize=6,
              loc="lower left", ncol=1, frameon=False, handlelength=0.9, borderaxespad=0.1, labelspacing=0.25)
    F.panel(fig, ax, "h", "Explanation zones")
    F.save(fig, FIG / "Fig09_spatial_xai")
    xj = json.loads((REP / "07_xai.json").read_text())
    captions["Fig09"] = (
        "**Figure 9. Where explanations and errors cluster.** (a-d) Tract mean SHAP value (log-odds) of four feature groups "
        "for the primary spline logistic regression on 2026 trips starting in each tract (red = pushes toward e-bike). "
        "(e, f) Local Moran (LISA) clusters (queen contiguity, 999 permutations, p < 0.05) of the LightGBM tract residual "
        "(observed minus predicted log-odds on all 2026 trips) and of the LightGBM origin-area contribution. (g) Global "
        "Moran's I of tract mean grouped SHAP values for both models, of tract residuals and of the observed e-bike share "
        "(* = p < 0.05; ns = not significant). (h) Zones from k-means clustering of standardized tract SHAP profiles "
        f"(LightGBM, ten most important features; k = {xj['k_zones']} chosen by silhouette, maximum "
        f"{max(xj['silhouette'].values()):.2f}); zones from the primary model's profiles agree moderately "
        f"(adjusted Rand index {xj['zone_ari_lgbm_vs_lr']:.2f}).")

# =================================================================== Fig10 interactions
if "fig10" in ONLY:
    z = np.load(XAI / "shap_interactions.npz", allow_pickle=True)
    I, X, fn, blk = z["values"], z["X"], list(z["features"]), z["block"]
    M = np.abs(I).mean(0)
    tot = M.sum(1) + M.sum(0) - np.diag(M)
    top = list(np.argsort(-tot)[:10])
    zt = pd.read_csv(TAB / "T13_zones.csv").set_index("zone")
    fig = F.new_figure(180, 190)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, 0:2])
    S = 2 * M[np.ix_(top, top)]
    np.fill_diagonal(S, np.nan)
    S[np.triu_indices(len(top), 1)] = np.nan
    im = ax.imshow(S, cmap=F.HEAT, vmin=0)
    for i in range(len(top)):
        for j in range(i):
            if S[i, j] >= np.nanpercentile(S, 80):
                ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color=F.text_on(im.cmap(im.norm(S[i, j]))))
    nm = [lab(fn[i]) for i in top]
    ax.set_xticks(range(len(top)), nm, rotation=60, ha="right", fontsize=6.8)
    ax.set_yticks(range(len(top)), nm, fontsize=6.8)
    ax.tick_params(length=0)
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    cax = ax.inset_axes([0.55, 0.88, 0.4, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Mean |interaction| (log-odds)", fontsize=6.6)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    F.panel(fig, ax, "a", "Pairwise interactions (LightGBM)")
    ax = fig.add_subplot(gs[0, 2])
    feats = [c for c in zt.columns if c not in ("tracts", "trips", "ebike_share")]
    H = zt[feats].T.values
    lim = np.abs(H).max()
    im2 = ax.imshow(H, cmap=F.DIV, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, F.minus(f"{H[i, j]:+.2f}"), ha="center", va="center", fontsize=5.6,
                    color=F.text_on(im2.cmap(im2.norm(H[i, j]))))
    ax.set_yticks(range(len(feats)), [lab(f) for f in feats], fontsize=6.4)
    ax.set_xticks(range(len(zt)), [f"Z{zz}\n{int(zt.loc[zz, 'tracts'])} tr\n{100 * zt.loc[zz, 'ebike_share']:.1f}%"
                                   for zz in zt.index], fontsize=6)
    ax.tick_params(length=0)
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    F.panel(fig, ax, "b", "Zone profiles")

    def dep(ax, f, by, title, letter, xlab, groups):
        j = fn.index(f)
        x = X[:, j]
        phi = I[:, j, :].sum(1)
        for name, msk, col in groups:
            ax.scatter(x[msk], phi[msk], s=3, color=col, alpha=0.25, lw=0, rasterized=True)
            xs, ys = x[msk], phi[msk]
            bins = np.unique(np.nanquantile(xs, np.linspace(0, 1, 13)))
            kk = np.clip(np.digitize(xs, bins) - 1, 0, len(bins) - 2)
            mx = [xs[kk == b].mean() for b in range(len(bins) - 1) if (kk == b).any()]
            my = [ys[kk == b].mean() for b in range(len(bins) - 1) if (kk == b).any()]
            PX.glow_line(ax, mx, my, col, lw=1.6, label=name)
        ax.axhline(0, color=F.RULE, lw=0.8, zorder=0)
        ax.set_xlabel(xlab)
        ax.set_ylabel(f"SHAP, {lab(f).lower()}", fontsize=7.6)
        ax.legend(fontsize=6.4, loc="best")
        F.ygrid(ax)
        F.panel(fig, ax, letter, title)

    cbd = X[:, fn.index("o_dist_cbd_km")]
    groups = [("Origin < 2 km", cbd < 2, F.G0), ("2-6 km", (cbd >= 2) & (cbd < 6), F.S4[3]), ("> 6 km", cbd >= 6, F.G1)]
    dep(fig.add_subplot(gs[1, 0]), "od_dist_km", "o_dist_cbd_km", "OD distance × centrality", "c", "OD distance (km)", groups)
    dep(fig.add_subplot(gs[1, 1]), "city_ebike_share_7d", "o_dist_cbd_km", "Fleet share × centrality", "d",
        "Fleet e-bike share, 7 d", groups)
    tmp = X[:, fn.index("temp_c")]
    dep(fig.add_subplot(gs[1, 2]), "hour", "temp_c", "Hour × temperature", "e", "Start hour",
        [("< 15 °C", tmp < 15, F.G0), ("15-25 °C", (tmp >= 15) & (tmp < 25), F.S4[2]), ("> 25 °C", tmp >= 25, F.G1)])
    F.save(fig, FIG / "Fig10_interactions")
    xj = json.loads((REP / "07_xai.json").read_text())
    captions["Fig10"] = (
        f"**Figure 10. Interactions and zone profiles.** (a) Mean absolute SHAP interaction values of LightGBM (both halves "
        f"summed) among the ten features with the largest total attribution, 2,000 trips from 2026; main effects account "
        f"for {100 * xj['main_effect_share']:.0f}% of total absolute attribution. (b) Mean SHAP value of the ten most "
        "important features in each explanation zone (columns: zone, number of tracts, observed 2026 e-bike share). (c-e) "
        "SHAP value of a feature against its value, split by a second variable (lines = binned means).")

cap = FIG / "figure_captions.md"
old = {}
if cap.exists():
    for block in cap.read_text().split("\n\n"):
        if block.startswith("**Figure"):
            old[f"Fig{block.split('Figure ')[1].split('.')[0].zfill(2)}"] = block.strip()
old.update(captions)
cap.write_text("# Figure captions (Norfolk)\n\n" + "\n\n".join(old[k] for k in sorted(old)) + "\n")
print("done:", ", ".join(captions))
