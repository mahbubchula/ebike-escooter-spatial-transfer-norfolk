"""Step 04: descriptive figures (studio theme), June 2021 to October 2026.

Fig01_fleet_time       weekly trips by vehicle; weekly e-bike share and lagged fleet share; year x month heat table;
                       hourly rose; weekday x hour heat map
Fig02_trip_profile     ridden distance and duration (split violins by year, descriptive only); share by OD distance;
                       speed ECDF; same-tract vs cross-tract share by year
Fig03_spatial          trip density; e-bike share by origin tract; spatial blocks; OD flow map; change between periods
Fig04_context          temperature, rain, wind, temperature x hour; neighbourhood decile heat table; tract scatter
Ridden distance, duration and speed are descriptive only; they are never model features.

Run:  python src/step_04_descriptive_figures.py [--only fig01,fig03]
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fstudio as F  # noqa: E402
import plots as P  # noqa: E402
import plots_extra as PX  # noqa: E402

F.use_theme("studio")
mpl.rcParams.update({"font.weight": "bold"})
OUT = ROOT / "outputs"
FIG = OUT / "figures"
ap = argparse.ArgumentParser()
ap.add_argument("--only", default="fig01,fig02,fig03,fig04")
ONLY = ap.parse_args().only.split(",")
SEED = 42
VEH = {1: ("E-bike", F.G1), 0: ("E-scooter", F.G0)}
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
YC = dict(zip(YEARS, [F.tint(F.G0, f) for f in (0.75, 0.6, 0.45, 0.3, 0.15, 0.0)]))
captions = {}

COLS = ["trip_id", "start_dt", "e_bike", "o_tract", "d_tract", "same_tract", "dist_km", "duration_min", "hour",
        "day_of_week", "is_weekend", "temp_c", "precip_mm", "wind_kmh", "is_daylight", "o_dist_cbd_km",
        "o_median_income", "o_pct_hh_no_car", "o_pct_age_18_29", "o_pop_density_km2", "o_job_density_km2",
        "o_pct_commute_walk", "o_pct_degree"]
d = pd.read_parquet(ROOT / "data" / "processed" / "norfolk_model_data.parquet", columns=COLS)
d = d[(d.start_dt >= "2021-06-01") & (d.start_dt < "2026-11-01") & d.o_tract.notna() & d.d_tract.notna()].copy()
d["year"] = d.start_dt.dt.year
d["month"] = d.start_dt.dt.month
d["day"] = d.start_dt.dt.normalize()
tf = pd.read_parquet(ROOT / "data" / "external" / "tract_features_norfolk.parquet").set_index("GEOID")


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


d["od_dist_km"] = hav(d.o_tract.map(tf.INTPTLAT.astype(float)).values, d.o_tract.map(tf.INTPTLON.astype(float)).values,
                      d.d_tract.map(tf.INTPTLAT.astype(float)).values, d.d_tract.map(tf.INTPTLON.astype(float)).values)
print(f"{len(d):,} trips; e-bike share {d.e_bike.mean():.4f}")


def heat(ax, M, xl, yl, cmap, vmin=None, vmax=None, annot=False, fmt="{:.0f}", scale=100, fs=6.4, origin="upper"):
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    if annot:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, fmt.format(scale * M[i, j]), ha="center", va="center", fontsize=fs,
                            color=F.text_on(im.cmap(im.norm(M[i, j]))))
    ax.set_xticks(range(len(xl)), xl)
    ax.set_yticks(range(len(yl)), yl)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def cbar(fig, ax, im, label, fmt=None, loc="right", shrink=0.85):
    cb = fig.colorbar(im, ax=ax, shrink=shrink, aspect=22, pad=0.02, location=loc, **({"format": fmt} if fmt else {}))
    cb.set_label(label, fontsize=7.4)
    cb.ax.tick_params(labelsize=6.8)
    cb.outline.set_visible(False)


# =================================================================== Fig01
if "fig01" in ONLY:
    fig = F.new_figure(180, 215)
    outer = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.25])
    wk = d.groupby([pd.Grouper(key="start_dt", freq="W-SUN"), "e_bike"]).size().unstack(fill_value=0)
    ax = fig.add_subplot(outer[0])
    ax.stackplot(wk.index, wk[0] / 1000, wk[1] / 1000, colors=[F.tint(F.G0, 0.25), F.G1], lw=0,
                 labels=["E-scooter", "E-bike"])
    ax.set_ylabel("Trips per week (000)")
    ax.set_xlim(wk.index.min(), wk.index.max())
    ax.legend(loc="upper left", fontsize=7.4, ncol=2)
    for yr in range(2022, 2027):
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color=F.RULE, lw=0.7, zorder=0)
    F.ygrid(ax)
    F.panel(fig, ax, "a", "Weekly trips by vehicle")

    ax = fig.add_subplot(outer[1], sharex=ax)
    daily = d.groupby("day")["e_bike"].agg(["sum", "size"])
    lag = daily["sum"].rolling(7).sum().shift(1) / daily["size"].rolling(7).sum().shift(1)
    sh = wk[1] / wk.sum(1)
    ax.fill_between(sh.index, 0, sh.values, color=F.tint(F.G1, 0.75), lw=0, step="mid")
    ax.plot(sh.index, sh.values, color=F.G1, lw=1.3, drawstyle="steps-mid", label="Weekly e-bike share")
    ax.plot(lag.index, lag.values, color=F.INK, lw=0.9, ls=(0, (3, 1.5)), label="Lagged 7-day share (fleet proxy)")
    for (a, b, lab, c) in (("2021-06-01", "2024-12-24", "Training", F.MUTED), ("2025-01-01", "2025-12-31", "Validation", F.S4[2]),
                           ("2026-01-01", "2026-10-31", "Test", F.S4[3])):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), ymin=0.94, ymax=1.0, color=c, alpha=0.8, lw=0)
        ax.text(pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2, 0.205, lab, ha="center", fontsize=6.8,
                color=F.INK)
    F.pct_axis(ax)
    ax.set_ylim(0, 0.215)
    ax.set_ylabel("E-bike share")
    ax.legend(loc="upper right", fontsize=7, bbox_to_anchor=(1, 0.9))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
    F.ygrid(ax)
    F.panel(fig, ax, "b", "E-bike share and fleet proxy")

    bot = outer[2].subgridspec(1, 3, width_ratios=[1.35, 0.9, 1.1])
    ax = fig.add_subplot(bot[0])
    ym = d.groupby(["year", "month"])["e_bike"].mean().unstack()
    im = heat(ax, ym.values, list("JFMAMJJASOND"), [str(y) for y in ym.index], F.SEQ, annot=True, fmt="{:.0f}", fs=6.2)
    cbar(fig, ax, im, "E-bike share", mpl.ticker.PercentFormatter(1, 0), loc="bottom", shrink=0.8)
    F.panel(fig, ax, "c", "Year × month (%)")

    ax = fig.add_subplot(bot[1], projection="polar")
    hs = d.groupby("hour")["e_bike"].mean()
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    nrm = mpl.colors.Normalize(hs.min(), hs.max())
    ax.bar(th, hs.values, width=2 * np.pi / 24 * 0.9, color=[F.SEQ(0.25 + 0.75 * nrm(v)) for v in hs.values],
           edgecolor=F.BG, lw=0.6)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(th[::3], [f"{h:02d}" for h in range(0, 24, 3)], fontsize=7)
    ax.set_yticks([0.04, 0.08])
    ax.set_yticklabels(["4%", "8%"], fontsize=6.4)
    ax.set_rlabel_position(100)
    ax.grid(color=F.GRID, lw=0.6)
    ax.spines["polar"].set_visible(False)
    F.panel(fig, ax, "d", "By hour")

    ax = fig.add_subplot(bot[2])
    wh = d.groupby(["day_of_week", "hour"])["e_bike"].mean().unstack()
    im = heat(ax, wh.values, [str(h) if h % 6 == 0 else "" for h in range(24)], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
              F.SEQ)
    ax.set_xlabel("Start hour")
    cbar(fig, ax, im, "E-bike share", mpl.ticker.PercentFormatter(1, 0), loc="bottom", shrink=0.8)
    F.panel(fig, ax, "e", "Weekday × hour")
    F.save(fig, FIG / "Fig01_fleet_time")
    captions["Fig01"] = (
        f"**Figure 1. Fleet use over time, Norfolk Lime, June 2021 to October 2026 ({len(d):,} trips).** (a) Weekly "
        "trips by vehicle (stacked). (b) Weekly e-bike share (filled) and the citywide e-bike share of the previous seven "
        "days (dashed), used as a strictly lagged proxy for the deployed fleet; bars at the top mark the training, "
        "validation and test periods. (c) E-bike share (%) by year and month. (d) E-bike share by start hour (radial "
        "length and colour). (e) E-bike share by day of week and start hour.")

# =================================================================== Fig02
if "fig02" in ONLY:
    s2 = d[d.duration_min.between(1, 120)].copy()
    s2["speed"] = s2["dist_km"] / (s2["duration_min"] / 60)
    s2 = s2[s2.speed.between(1, 35)]
    fig = F.new_figure(180, 205)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1])
    for j, (var, cap, lab, ttl) in enumerate((("dist_km", 8, "Ridden distance (km)", "Ridden distance"),
                                              ("duration_min", 40, "Duration (min)", "Trip duration"))):
        ax = fig.add_subplot(gs[0, j])
        L = {y: s2.loc[(s2.year == y) & (s2.e_bike == 0), var].clip(upper=cap).sample(4000, random_state=SEED).values for y in YEARS}
        R = {y: s2.loc[(s2.year == y) & (s2.e_bike == 1), var].clip(upper=cap).sample(4000, random_state=SEED).values for y in YEARS}
        PX.split_violin(ax, YEARS, L, R, colors=(F.G0, F.G1), width=0.44, bw=0.3)
        ax.set_ylabel(lab)
        ax.set_xticks(range(len(YEARS)), [str(y) for y in YEARS], fontsize=7.4)
        if j == 0:
            ax.legend(handles=[Patch(color=F.G0, alpha=0.5, label="E-scooter (left)"),
                               Patch(color=F.G1, alpha=0.5, label="E-bike (right)")], fontsize=7, loc="upper left", ncol=2)
        F.panel(fig, ax, "ab"[j], ttl)

    ax = fig.add_subplot(gs[1, 0])
    DB = [-0.01, 0.0, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 20]
    DL = ["Same\ntract", "<1", "1-1.5", "1.5-2", "2-2.5", "2.5-3", "3-4", "4-5", "5-7", ">7"]
    for grp, ls, mk in (("2021-2023", "--", "s"), ("2024-2026", "-", "o")):
        msk = d.year.between(2021, 2023) if grp == "2021-2023" else d.year.between(2024, 2026)
        g = d[msk].groupby(pd.cut(d.loc[msk, "od_dist_km"], DB), observed=False)["e_bike"].agg(["sum", "size"])
        p, lo, hi = F.wilson(g["sum"].values, g["size"].values)
        x = np.arange(len(g))
        ax.fill_between(x, lo, hi, color=F.G1, alpha=0.15, lw=0)
        ax.plot(x, p, color=F.G1, ls=ls, marker=mk, ms=3.5, lw=1.4, label=grp)
    ax.set_xticks(range(len(DL)), DL, fontsize=6.6, rotation=35)
    F.pct_axis(ax)
    ax.set_xlabel("Origin-destination centroid distance (km)")
    ax.set_ylabel("E-bike share")
    ax.legend(fontsize=7.2, loc="upper left")
    F.ygrid(ax)
    F.panel(fig, ax, "c", "Share by OD distance")

    ax = fig.add_subplot(gs[1, 1])
    for v in (0, 1):
        x = np.sort(s2.loc[s2.e_bike == v, "speed"].sample(60000, random_state=SEED).values)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=VEH[v][1], lw=1.8, label=VEH[v][0])
        ax.axvline(np.median(x), color=VEH[v][1], lw=0.8, ls=":")
    F.pct_axis(ax)
    ax.set_xlabel("Average ridden speed (km/h)")
    ax.set_ylabel("Cumulative share")
    ax.legend(fontsize=7.2, loc="lower right")
    F.ygrid(ax, "both")
    F.panel(fig, ax, "d", "Speed")

    ax = fig.add_subplot(gs[2, :])
    yy = np.arange(len(YEARS))
    for k, (lab, msk, col) in enumerate((("Same tract", d.same_tract == 1, F.S4[2]), ("Other tract", d.same_tract == 0, F.S4[3]),
                                         ("Weekday", d.is_weekend == 0, F.G0), ("Weekend", d.is_weekend == 1, F.C6[3]))):
        v = d[msk].groupby("year")["e_bike"].mean().reindex(YEARS)
        ax.plot(yy + (k - 1.5) * 0.12, v.values, marker="o", ms=5, lw=1.4, color=col, label=lab, mec=F.BG)
    ax.set_xticks(yy, [str(y) + ("*" if y in (2021, 2026) else "") for y in YEARS])
    F.pct_axis(ax)
    ax.set_ylabel("E-bike share")
    ax.legend(fontsize=7.2, ncol=4, loc="upper right")
    F.ygrid(ax)
    F.panel(fig, ax, "e", "Trip context by year")
    F.save(fig, FIG / "Fig02_trip_profile")
    captions["Fig02"] = (
        "**Figure 2. Trip profile.** (a, b) Ridden distance and duration by year for e-scooters (left half) and e-bikes "
        "(right half); 4,000 random trips per vehicle and year, trips of 1-120 min and 1-35 km/h, capped at 8 km and 40 min; "
        "horizontal bars are medians and thick ticks the interquartile range. Ridden distance, duration and speed are "
        "recorded after the trip and are never model features. (c) E-bike share by straight-line distance between origin "
        "and destination tract centroids (the pre-trip distance used in the models) for 2021-2023 and 2024-2026, with "
        "Wilson 95% bands. (d) Cumulative distribution of average ridden speed (dotted lines = medians). (e) E-bike share "
        "by year for trips within one tract or across tracts and on weekdays or weekends (* = partial year). Steps in (d) "
        "reflect the recording precision of distance (0.1 mile) and duration (whole minutes).")

# =================================================================== Fig03
if "fig03" in ONLY:
    tr = gpd.read_file(ROOT / "data" / "external" / "tiger" / "tracts_norfolk.gpkg").to_crs(3857)
    tr = tr[tr["ALAND"] > 0.2 * (tr["ALAND"] + tr["AWATER"])].copy()
    blk = pd.read_csv(OUT / "tables" / "T03b_tract_blocks.csv", dtype={"GEOID": str}).set_index("GEOID")["block"]
    a = d.groupby("o_tract").agg(n=("e_bike", "size"), p=("e_bike", "mean"))
    early = d[d.year <= 2023].groupby("o_tract")["e_bike"].agg(["mean", "size"])
    late = d[d.year >= 2024].groupby("o_tract")["e_bike"].agg(["mean", "size"])
    ok = (early["size"] >= 300) & (late["size"] >= 300)
    chg = (100 * (late["mean"] - early["mean"]))[ok]
    g = tr.merge(a, left_on="GEOID", right_index=True, how="left")
    g["dens"] = g["n"] / (g["ALAND"] / 1e6)
    g["chg"] = g.GEOID.map(chg)
    g["block"] = g.GEOID.map(blk)
    has = g[g.n.notna() & (g.n >= 300)]
    xmin, ymin, xmax, ymax = g[g.n.notna()].total_bounds
    base = F.tint(F.MUTED, 0.8)

    def frame(ax):
        ax.set_xlim(xmin - 600, xmax + 600)
        ax.set_ylim(ymin - 600, ymax + 600)
        ax.set_axis_off()
        ax.set_aspect("equal")

    def hcb(ax, cmap, norm, label, fmt=None, pos=(0.05, -0.05, 0.6, 0.03)):
        cax = ax.inset_axes(list(pos))
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                          **({"format": fmt} if fmt else {}))
        cb.set_label(label, fontsize=6.8, labelpad=1)
        cb.ax.tick_params(labelsize=6.3, length=0, pad=1)
        cb.outline.set_visible(False)

    fig = F.new_figure(180, 205)
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.3])
    gs = outer[0].subgridspec(1, 3)
    gb = outer[1].subgridspec(1, 2, width_ratios=[1.45, 1])
    ax = fig.add_subplot(gs[0])
    tr.plot(ax=ax, color=base, edgecolor="white", lw=0.3)
    nrm = mpl.colors.LogNorm(*np.nanpercentile(has["dens"], [3, 99]))
    has.plot(ax=ax, column="dens", cmap=F.HEAT, norm=nrm, edgecolor="white", lw=0.3)
    frame(ax)
    hcb(ax, F.HEAT, nrm, "Trips per km² (log)")
    F.panel(fig, ax, "a", "Trip origins")

    ax = fig.add_subplot(gs[1])
    tr.plot(ax=ax, color=base, edgecolor="white", lw=0.3)
    nrm = mpl.colors.Normalize(*np.nanpercentile(has["p"], [3, 97]))
    has.plot(ax=ax, column="p", cmap=F.SEQ, norm=nrm, edgecolor=F.RULE, lw=0.3)
    frame(ax)
    hcb(ax, F.SEQ, nrm, "E-bike share", mpl.ticker.PercentFormatter(1, 0))
    F.panel(fig, ax, "b", "E-bike share")

    ax = fig.add_subplot(gs[2])
    bcol = {i + 1: c for i, c in enumerate(F.C6)}
    g[g.block.notna()].plot(ax=ax, color=[bcol[int(b)] for b in g[g.block.notna()].block], edgecolor="white", lw=0.4, alpha=0.85)
    bt = pd.read_csv(OUT / "tables" / "T03_spatial_blocks.csv").set_index("o_block")
    for b, sub in g[g.block.notna()].groupby("block"):
        c = sub.dissolve().geometry.iloc[0].representative_point()
        ax.text(c.x, c.y, f"B{int(b)}  {bt.loc[int(b), 'trip_pct']:.0f}%", ha="center", va="center", fontsize=6.8,
                color=F.INK, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    frame(ax)
    F.panel(fig, ax, "c", "Spatial blocks")

    ax = fig.add_subplot(gb[0])
    tr.plot(ax=ax, color=F.tint(F.MUTED, 0.88), edgecolor="white", lw=0.4)
    cen = gpd.GeoSeries(gpd.points_from_xy(tf.INTPTLON.astype(float), tf.INTPTLAT.astype(float)), index=tf.index,
                        crs=4326).to_crs(3857)
    od = d[d.same_tract == 0].groupby(["o_tract", "d_tract"])["e_bike"].agg(n="size", p="mean").reset_index()
    od = od[od.n >= 500].sort_values("n")
    od = od.tail(250)
    segs = [[(cen[o].x, cen[o].y), (cen[t].x, cen[t].y)] for o, t in zip(od.o_tract, od.d_tract)]
    nrm = mpl.colors.Normalize(*np.nanpercentile(od["p"], [5, 95]))
    lc = LineCollection(segs, linewidths=0.3 + 4.5 * (od.n / od.n.max()) ** 0.7, colors=[F.SEQ(0.2 + 0.8 * nrm(v)) for v in od.p],
                        alpha=0.85, capstyle="round", zorder=3)
    ax.add_collection(lc)
    vol = d.groupby("o_tract").size()
    ax.scatter(cen.reindex(vol.index).x, cen.reindex(vol.index).y, s=3 + 60 * (vol / vol.max()) ** 0.7, color=F.G0,
               ec=F.BG, lw=0.4, zorder=4)
    frame(ax)
    hcb(ax, F.SEQ, nrm, "E-bike share of OD flow", mpl.ticker.PercentFormatter(1, 0), pos=(0.03, 0.04, 0.32, 0.025))
    F.panel(fig, ax, "d", "Top 250 OD flows")

    ax = fig.add_subplot(gb[1])
    tr.plot(ax=ax, color=base, edgecolor="white", lw=0.3)
    lim = np.nanpercentile(np.abs(g["chg"]), 95)
    nrm = mpl.colors.TwoSlopeNorm(0, -lim, lim)
    g[g.chg.notna()].plot(ax=ax, column="chg", cmap=F.DIV, norm=nrm, edgecolor=F.RULE, lw=0.3)
    frame(ax)
    hcb(ax, F.DIV, nrm, "Change (pp)")
    F.panel(fig, ax, "e", "2024-26 vs 2021-23")
    F.save(fig, FIG / "Fig03_spatial")
    captions["Fig03"] = (
        "**Figure 3. Spatial structure.** (a) Trip origins per km² of land by census tract (log scale). (b) E-bike share by "
        "origin tract (tracts with at least 300 trips; grey = fewer). (c) The six spatial blocks used for the spatial "
        "transferability tests, defined from tract centroids only (k-means), with each block's share of all trips. (d) "
        "The 250 largest flows between different tracts (at least 500 trips; line width ∝ trips, colour = e-bike share) and "
        "origin volumes (blue dots). (e) Change in e-bike share between 2021-2023 and 2024-2026 (percentage points; tracts "
        "with at least 300 trips in each period). Open-water tracts are not shown.")

# =================================================================== Fig04
if "fig04" in ONLY:
    fig = F.new_figure(180, 200)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15])
    ax = fig.add_subplot(gs[0, 0])
    TB = np.arange(-6, 38, 3)
    for v_lab, msk, col in (("Dry", d.precip_mm <= 0.1, F.G1), ("Rain", d.precip_mm > 0.1, F.G0)):
        s = d[msk]
        gg = s.groupby(pd.cut(s.temp_c, TB), observed=True)["e_bike"].agg(["sum", "size"])
        gg = gg[gg["size"] >= 300]
        x = np.array([iv.mid for iv in gg.index])
        pp, lo, hi = F.wilson(gg["sum"].values, gg["size"].values)
        ax.fill_between(x, lo, hi, color=col, alpha=0.16, lw=0)
        ax.plot(x, pp, color=col, lw=1.5, zorder=3)
        ax.scatter(x, pp, s=8 + 110 * (gg["size"] / d.shape[0] * 12) ** 0.7, color=col, ec=F.BG, lw=0.8, zorder=4,
                   label=v_lab)
    F.pct_axis(ax)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("E-bike share")
    ax.legend(fontsize=7.2, loc="upper left", markerscale=0.6)
    F.ygrid(ax)
    F.panel(fig, ax, "a", "Temperature and rain")

    ax = fig.add_subplot(gs[0, 1])
    g = d.groupby([pd.cut(d.temp_c, [-10, 0, 5, 10, 15, 20, 25, 30, 40]), "hour"], observed=False)["e_bike"].agg(["mean", "size"])
    M = np.where(g["size"].unstack().values >= 150, g["mean"].unstack().values, np.nan)
    im = heat(ax, M, [str(h) if h % 6 == 0 else "" for h in range(24)],
              ["<0", "0-5", "5-10", "10-15", "15-20", "20-25", "25-30", ">30"], F.SEQ, origin="lower")
    ax.set_xlabel("Start hour")
    ax.set_ylabel("Temperature (°C)")
    cbar(fig, ax, im, "E-bike share", mpl.ticker.PercentFormatter(1, 0))
    F.panel(fig, ax, "b", "Temperature × hour")

    ax = fig.add_subplot(gs[1, 0])
    WB = [0, 5, 10, 15, 20, 25, 30, 45]
    for yr_lab, msk, ls in (("2021-2023", d.year <= 2023, "--"), ("2024-2026", d.year >= 2024, "-")):
        for dl, dm, col in (("Day", d.is_daylight == 1, F.S4[2]), ("Night", d.is_daylight == 0, F.C6[3])):
            s = d[msk & dm]
            gg = s.groupby(pd.cut(s.wind_kmh, WB), observed=True)["e_bike"].agg(["sum", "size"])
            gg = gg[gg["size"] >= 300]
            p, lo, hi = F.wilson(gg["sum"].values, gg["size"].values)
            x = np.array([iv.mid for iv in gg.index])
            ax.fill_between(x, lo, hi, color=col, alpha=0.13, lw=0)
            ax.plot(x, p, color=col, ls=ls, marker="o", ms=2.6, lw=1.3)
    F.pct_axis(ax)
    ax.set_xlabel("Wind speed (km/h)")
    ax.set_ylabel("E-bike share")
    ax.legend(handles=[Line2D([], [], color=F.S4[2], lw=1.6, label="Daylight"), Line2D([], [], color=F.C6[3], lw=1.6, label="Dark"),
                       Line2D([], [], color=F.INK2, ls="--", label="2021-2023"), Line2D([], [], color=F.INK2, label="2024-2026")],
              fontsize=6.8, ncol=2, loc="upper left")
    F.ygrid(ax)
    F.panel(fig, ax, "c", "Wind and light")

    ax = fig.add_subplot(gs[1, 1])
    NB = [("o_median_income", "Income"), ("o_pct_degree", "Degree holders"), ("o_pct_hh_no_car", "Car-free households"),
          ("o_pct_age_18_29", "Aged 18-29"), ("o_pct_commute_walk", "Walk commuters"), ("o_job_density_km2", "Job density"),
          ("o_pop_density_km2", "Pop. density"), ("o_dist_cbd_km", "Distance to downtown")]
    S = []
    for var, _ in NB:
        q = pd.qcut(d[var].rank(method="first"), 5, labels=False)
        S.append(d.groupby(q)["e_bike"].mean().values)
    S = np.array(S)
    im = heat(ax, S, ["Q1", "Q2", "Q3", "Q4", "Q5"], [n for _, n in NB], F.SEQ, annot=True, fmt="{:.1f}", fs=6.4)
    ax.set_xlabel("Origin-tract quintile (Q1 = lowest)")
    cbar(fig, ax, im, "E-bike share", mpl.ticker.PercentFormatter(1, 0))
    F.panel(fig, ax, "d", "Neighborhood gradients (%)")

    ax = fig.add_subplot(gs[2, :])
    tt = d.groupby("o_tract").agg(p=("e_bike", "mean"), n=("e_bike", "size"), cbd=("o_dist_cbd_km", "first"),
                                  inc=("o_median_income", "first"), age=("o_pct_age_18_29", "first"))
    tt = tt[tt.n >= 300]
    blk = pd.read_csv(OUT / "tables" / "T03b_tract_blocks.csv", dtype={"GEOID": str}).set_index("GEOID")["block"]
    tt["block"] = tt.index.map(blk)
    for b, sub in tt.groupby("block"):
        ax.scatter(sub.cbd, sub.p, s=10 + 160 * (sub.n / tt.n.max()) ** 0.6, color=F.C6[int(b) - 1], ec=F.BG, lw=0.6,
                   alpha=0.9, label=f"B{int(b)}")
    o = np.argsort(tt.cbd.values)
    xs, ys, ws = tt.cbd.values[o], tt.p.values[o], tt.n.values[o]
    bins = np.quantile(xs, np.linspace(0, 1, 9))
    k = np.clip(np.digitize(xs, bins) - 1, 0, 7)
    PX.glow_line(ax, [xs[k == i].mean() for i in range(8)], [np.average(ys[k == i], weights=ws[k == i]) for i in range(8)],
                 F.INK, lw=1.6)
    F.pct_axis(ax)
    ax.set_xlabel("Origin tract distance to downtown (km)")
    ax.set_ylabel("Tract e-bike share")
    ax.legend(title="Block", fontsize=6.8, title_fontsize=7, ncol=6, loc="upper left", markerscale=0.7)
    F.ygrid(ax)
    F.panel(fig, ax, "e", "Tracts by distance to downtown")
    F.save(fig, FIG / "Fig04_context")
    captions["Fig04"] = (
        "**Figure 4. Weather and neighbourhood context.** (a) E-bike share by 3 °C temperature band in dry and rainy hours "
        "(marker area ∝ trips; Wilson 95% bands; bands with fewer than 300 trips omitted). (b) E-bike share by temperature "
        "and start hour (cells with fewer than 150 trips blank). (c) E-bike share by wind speed in daylight and darkness, "
        "2021-2023 (dashed) and 2024-2026 (solid). (d) E-bike share (%) by quintile of eight origin-tract characteristics "
        "(ACS 2023 5-year; LEHD LODES 2022). (e) Tracts with at least 300 trips: distance to downtown against e-bike share, "
        "coloured by spatial block (marker area ∝ trips); the line joins trip-weighted means of distance octiles.")

cap = FIG / "figure_captions.md"
old = {}
if cap.exists():
    for block in cap.read_text().split("\n\n"):
        if block.startswith("**Figure"):
            old[f"Fig{block.split('Figure ')[1].split('.')[0].zfill(2)}"] = block.strip()
old.update(captions)
cap.write_text("# Descriptive figure captions (Norfolk)\n\n" + "\n\n".join(old[k] for k in sorted(old)) + "\n")
print("done:", ", ".join(captions))
