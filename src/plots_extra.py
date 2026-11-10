"""plots_extra - additional advanced plot forms that follow the active figure-studio theme.

Complements plots.py with: calendar heatmap, polar stacked bars, radar, marimekko,
ECDF, split violins, clustermap with dendrogram, correlation network, bubble matrix,
bullet charts, Nightingale rose, gradient intervals, ROPE probability bars, parallel
coordinates, Q-Q envelope, REC curves, rank-probability heatmap, sunburst, 3D surface,
chord diagram and tornado chart. All colours come from fstudio tokens.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Patch, PathPatch, Rectangle, Wedge
from matplotlib.path import Path as MPath
from scipy.stats import gaussian_kde

import fstudio as F

B = "bold"


def _kde(d, xs, bw=None):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    return gaussian_kde(d, bw_method=bw)(xs)


# ---------------------------------------------------------------------------
# Time and composition
# ---------------------------------------------------------------------------
def calendar_heatmap(ax, dates, year, cmap=None, label_months=True):
    """Week x weekday grid of daily counts for one year (GitHub-style)."""
    cmap = cmap or F.SEQ
    s = pd.Series(1, index=pd.to_datetime(dates)).groupby(level=0).sum()
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    counts = s.reindex(days, fill_value=0)
    vmax = counts.max()
    first = days[0].weekday()
    for d, c in counts.items():
        week = (d.dayofyear - 1 + first) // 7
        wd = d.weekday()
        fc = F.tint(F.INK, 0.94) if c == 0 else cmap(0.18 + 0.82 * c / vmax)
        ax.add_patch(Rectangle((week, 6 - wd), 0.88, 0.88, fc=fc, lw=0))
    if label_months:
        for m in range(1, 13):
            d = pd.Timestamp(f"{year}-{m:02d}-01")
            ax.text((d.dayofyear - 1 + first) // 7, 7.25, d.strftime("%b"), fontsize=7, fontweight=B,
                    color=F.INK2, ha="left")
    ax.set_yticks([6 - i + 0.44 for i in range(7)], ["Mon", "", "Wed", "", "Fri", "", "Sun"], fontsize=6.8)
    ax.set_xlim(-0.5, 54)
    ax.set_ylim(-0.3, 8)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_aspect("equal")
    return vmax


def polar_stacked(ax, table: pd.DataFrame, colors: dict, inner=0.25):
    """Radial stacked bars: rows = angular categories (e.g. months), columns = stacked groups."""
    n = len(table)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n * 0.86
    base = np.full(n, inner * table.sum(axis=1).max())
    for c in table.columns:
        ax.bar(theta, table[c].values, width=width, bottom=base, color=colors[c], ec=F.BG, lw=0.6, label=c)
        base = base + table[c].values
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(theta, list(table.index), fontsize=7.4, fontweight=B)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(color=F.GRID, lw=0.6)
    return base


def marimekko(ax, ct: pd.DataFrame, colors, gap=0.012):
    """Mosaic: column widths = column share, segment heights = within-column share."""
    tot = ct.values.sum()
    x = 0.0
    for j, col in enumerate(ct.columns):
        w = ct[col].sum() / tot
        y = 0.0
        for i, row in enumerate(ct.index):
            h = ct.loc[row, col] / ct[col].sum()
            ax.add_patch(Rectangle((x, y), w - gap, h - gap / 2, fc=colors[i], ec=F.BG, lw=0.8))
            if h > 0.07 and w > 0.08:
                ax.text(x + (w - gap) / 2, y + h / 2, f"{h:.0%}", ha="center", va="center", fontsize=7,
                        fontweight=B, color=F.text_on(colors[i]))
            y += h
        ax.text(x + (w - gap) / 2, -0.04, f"{col}\n(n = {int(ct[col].sum())})", ha="center", va="top",
                fontsize=7.2, fontweight=B)
        x += w
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    F.pct_axis(ax)
    ax.spines["bottom"].set_visible(False)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
def ecdf(ax, data: dict, colors: dict, lw=1.8, show_median=True):
    for k, v in data.items():
        v = np.sort(np.asarray(v, float)[~np.isnan(v)])
        y = np.arange(1, len(v) + 1) / len(v)
        ax.step(v, y, where="post", color=colors[k], lw=lw, label=f"{k} (n = {len(v)})")
        if show_median:
            ax.plot([np.median(v)] * 2, [0, 0.5], color=colors[k], lw=0.8, ls=(0, (2, 2)))
    ax.axhline(0.5, color=F.RULE, lw=0.8)
    ax.set_ylim(0, 1.01)
    F.pct_axis(ax)
    F.ygrid(ax, "both")


def split_violin(ax, groups, left: dict, right: dict, colors=(None, None), width=0.42, bw=0.35):
    """Two half-violins per group (left/right), with medians and quartile ticks."""
    cl, cr = colors[0] or F.G0, colors[1] or F.G1
    for i, g in enumerate(groups):
        for side, data, col in [(-1, left[g], cl), (1, right[g], cr)]:
            d = np.asarray(data, float)
            d = d[~np.isnan(d)]
            if len(d) < 5:
                continue
            ys = np.linspace(d.min(), d.max(), 200)
            k = _kde(d, ys, bw)
            k = k / k.max() * width
            ax.fill_betweenx(ys, i, i + side * k, color=col, alpha=0.45, lw=0)
            ax.plot(i + side * k, ys, color=col, lw=1.0)
            q1, m, q3 = np.percentile(d, [25, 50, 75])
            ax.plot([i, i + side * width * 0.55], [m, m], color=F.INK, lw=1.6)
            ax.plot([i + side * 0.04] * 2, [q1, q3], color=col, lw=2.6, solid_capstyle="butt")
    ax.set_xticks(range(len(groups)), groups)
    F.ygrid(ax)


# ---------------------------------------------------------------------------
# Matrices and networks
# ---------------------------------------------------------------------------
def clustermap(fig, spec, C: pd.DataFrame, group_of: dict, group_colors: dict, cmap=None):
    """Correlation matrix ordered by average-linkage clustering, with a top dendrogram
    and a family colour strip. Returns (ax_matrix, ax_dendrogram)."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    cmap = cmap or F.DIV
    D = 1 - C.abs().values
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), "average")
    gs = spec.subgridspec(3, 1, height_ratios=[0.9, 0.12, 5], hspace=0.02)
    axd = fig.add_subplot(gs[0])
    dn = dendrogram(Z, ax=axd, no_labels=True, color_threshold=0, above_threshold_color=F.INK2)
    for coll in axd.collections:
        coll.set_linewidth(0.7)
    axd.axis("off")
    order = dn["leaves"]
    lab = [C.index[i] for i in order]
    axs = fig.add_subplot(gs[1])
    for j, l in enumerate(lab):
        axs.add_patch(Rectangle((j, 0), 1, 1, fc=group_colors[group_of[l]], lw=0))
    axs.set_xlim(0, len(lab))
    axs.set_ylim(0, 1)
    axs.axis("off")
    ax = fig.add_subplot(gs[2])
    M = C.values[np.ix_(order, order)]
    ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="nearest",
              extent=(0, len(lab), len(lab), 0))
    ax.set_xticks(np.arange(len(lab)) + 0.5, lab, rotation=90, fontsize=5.4)
    ax.set_yticks(np.arange(len(lab)) + 0.5, lab, fontsize=5.4)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(group_colors[group_of[t.get_text()]])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    axd.set_xlim(0, 10 * len(lab))
    return ax, axd


def network(ax, nodes: list, R: pd.DataFrame, colors: dict, threshold=0.15, size=None, outcome=None):
    """Circular correlation network: edge width/alpha ~ |r|, colour = sign; outcome node in the centre."""
    ring = [n for n in nodes if n != outcome]
    ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(ring), endpoint=False)
    pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(ring, ang)}
    if outcome:
        pos[outcome] = (0.0, 0.0)
    edges = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            r = R.loc[a, b]
            if abs(r) >= threshold:
                edges.append((abs(r), a, b, r))
    for w, a, b, r in sorted(edges):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        col = F.G1 if r > 0 else F.G0
        ax.plot([x0, x1], [y0, y1], color=col, lw=0.6 + 9 * w ** 1.5, alpha=min(0.25 + w, 0.95),
                solid_capstyle="round", zorder=1)
    size = size or {}
    for n, (x, y) in pos.items():
        s = size.get(n, 1.0)
        ax.scatter(x, y, s=520 * s, color=colors[n], ec=F.BG, lw=1.6, zorder=3)
        ax.text(x, y, n, ha="center", va="center", fontsize=7.4, fontweight=B, color=F.text_on(colors[n]),
                zorder=4)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")


def bubble_matrix(ax, V: pd.DataFrame, C: pd.DataFrame | None = None, vmax=1.0, cmap=None, ref=None,
                  lower=True, fmt="{:.2f}"):
    """Lower-triangle matrix of circles: area ~ value, colour ~ C (default value)."""
    cmap = cmap or F.SEQ
    labs = list(V.index)
    n = len(labs)
    C = V if C is None else C
    for i in range(n):
        for j in range(n):
            if (lower and j >= i) or (not lower and i == j):
                continue
            v = abs(V.iloc[i, j])
            c = abs(C.iloc[i, j])
            ax.scatter(j, i, s=(v / vmax) * 330, color=cmap(min(c / vmax, 1) * 0.85 + 0.1), ec=F.BG, lw=0.6,
                       zorder=3)
            if ref is not None and c >= ref:
                ax.scatter(j, i, s=(v / vmax) * 330, fc="none", ec=F.INK, lw=1.4, zorder=4)
    ax.set_xticks(range(n - 1), labs[:-1], rotation=90)
    ax.set_yticks(range(1, n), labs[1:])
    ax.set_xlim(-0.6, n - 1.4)
    ax.set_ylim(n - 0.4, 0.4)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for k in range(n):
        ax.axhline(k, color=F.GRID, lw=0.5, zorder=0)
        ax.axvline(k, color=F.GRID, lw=0.5, zorder=0)


def bullet(ax, labels, values, targets, bands=(0.6, 0.7, 0.8, 1.0), color=None, marker_values=None,
           marker_label=None, xlim=(0.4, 1.0)):
    """Bullet charts: qualitative bands, a bar for the value and a target tick."""
    color = color or F.INK
    n = len(labels)
    shades = [F.tint(F.INK, f) for f in (0.78, 0.86, 0.93)]
    for i in range(n):
        y = n - 1 - i
        lo = xlim[0]
        for b_, sh in zip(bands[1:], shades):
            ax.barh(y, b_ - lo, left=lo, height=0.78, color=sh, lw=0)
            lo = b_
        ax.barh(y, values[i] - xlim[0], left=xlim[0], height=0.3, color=color[i] if isinstance(color, list) else color)
        ax.plot([targets[i]] * 2, [y - 0.32, y + 0.32], color=F.G1, lw=2)
        if marker_values is not None:
            ax.scatter(marker_values[i], y, marker="D", s=26, color=F.BG, ec=F.INK, lw=1.0, zorder=4)
    ax.set_yticks(range(n), labels[::-1])
    ax.set_xlim(*xlim)
    F.clean_left(ax)


# ---------------------------------------------------------------------------
# Estimates and comparison
# ---------------------------------------------------------------------------
def rose(ax, labels, values, colors, lo=None, hi=None, start=90):
    """Nightingale rose (polar bars, radius ~ sqrt(value) so area ~ value)."""
    n = len(labels)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.sqrt(np.asarray(values, float))
    ax.bar(theta, r, width=2 * np.pi / n * 0.92, color=colors, ec=F.BG, lw=1.0, alpha=0.9)
    if lo is not None:
        for t, a, b in zip(theta, np.sqrt(lo), np.sqrt(hi)):
            ax.plot([t, t], [a, b], color=F.INK, lw=1.2)
    for t, rr, l, v in zip(theta, r, labels, values):
        ax.text(t, max(rr, np.sqrt(hi[list(labels).index(l)]) if hi is not None else rr) + 0.2 * r.max(),
                f"{l}\n{v:.1f}%", ha="center", va="center",
                fontsize=7, fontweight=B)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_ylim(0, r.max() * 1.45)


def gradient_interval(ax, y, draws, color, levels=(50, 80, 95), height=0.34, median=True):
    """Horizontal gradient interval: nested quantile bars (darker = narrower)."""
    d = np.asarray(draws, float)
    for k, lv in enumerate(sorted(levels, reverse=True)):
        a, b = np.percentile(d, [(100 - lv) / 2, 100 - (100 - lv) / 2])
        ax.barh(y, b - a, left=a, height=height * (0.55 + 0.45 * (k + 1) / len(levels)),
                color=F.tint(color, 0.75 - 0.3 * k), lw=0, zorder=2)
    if median:
        ax.scatter(np.median(d), y, s=34, color=color, ec=F.BG, lw=1, zorder=4)


def rope_bars(ax, labels, p_better, p_rope, p_worse, names=("Better", "Practically equivalent", "Worse")):
    """Stacked horizontal probability bars (Bayesian comparison vs a reference)."""
    cols = [F.G0, F.tint(F.INK, 0.8), F.G1]
    n = len(labels)
    y = np.arange(n)[::-1]
    left = np.zeros(n)
    for vals, c, nm in zip([p_better, p_rope, p_worse], cols, names):
        vals = np.nan_to_num(np.asarray(vals, float))
        ax.barh(y, vals, left=left, color=c, height=0.66, ec=F.BG, lw=0.8, label=nm)
        for yy, l, v in zip(y, left, vals):
            if v > 0.12:
                ax.text(l + v / 2, yy, f"{v:.0%}", ha="center", va="center", fontsize=7, fontweight=B,
                        color=F.text_on(c))
        left += vals
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    F.pct_axis(ax, "x")
    F.clean_left(ax)


def parallel_coords(ax, df: pd.DataFrame, colors: dict, higher_better: dict, label_fmt="{:.3f}"):
    """Parallel coordinates with each axis scaled to its range; up = better on every axis."""
    cols = list(df.columns)
    Z = df.copy().astype(float)
    for c in cols:
        lo, hi = df[c].min(), df[c].max()
        z = (df[c] - lo) / (hi - lo) if hi > lo else df[c] * 0 + 0.5
        Z[c] = z if higher_better[c] else 1 - z
    x = np.arange(len(cols))
    for k in x:
        ax.axvline(k, color=F.RULE, lw=1.0, zorder=0)
        ax.text(k, 1.07, cols[k], ha="center", fontsize=7.6, fontweight=B)
        lo_v = df[cols[k]].min() if higher_better[cols[k]] else df[cols[k]].max()
        hi_v = df[cols[k]].max() if higher_better[cols[k]] else df[cols[k]].min()
        ax.text(k, -0.07, label_fmt.format(lo_v), ha="center", fontsize=6.4, color=F.INK2, fontweight=B)
        ax.text(k, 1.015, label_fmt.format(hi_v), ha="center", fontsize=6.4, color=F.INK2, fontweight=B)
    for m, row in Z.iterrows():
        ax.plot(x, row.values, color=colors[m], lw=2.0, marker="o", ms=4.5, mec=F.BG, alpha=0.9)
    ax.set_xlim(-0.25, len(cols) - 0.75)
    ax.set_ylim(-0.12, 1.12)
    ax.axis("off")


def qq_envelope(ax, resid, n_sim=500, seed=0, color=None):
    """Normal Q-Q plot with a 95% parametric envelope."""
    from scipy import stats
    color = color or F.G0
    r = np.sort((resid - resid.mean()) / resid.std(ddof=1))
    n = len(r)
    q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    rng = np.random.default_rng(seed)
    sims = np.sort(rng.standard_normal((n_sim, n)), axis=1)
    lo, hi = np.percentile(sims, [2.5, 97.5], axis=0)
    ax.fill_between(q, lo, hi, color=F.tint(color, 0.8), lw=0)
    ax.plot(q, q, color=F.INK2, lw=0.9, ls=(0, (3, 2)))
    out = (r < lo) | (r > hi)
    ax.scatter(q[~out], r[~out], s=9, color=color, lw=0, alpha=0.8)
    ax.scatter(q[out], r[out], s=12, color=F.G1, lw=0)
    F.ygrid(ax, "both")
    return out.mean()


def rec_curves(ax, y, preds: dict, colors: dict, eps_max=1.5):
    """Regression error characteristic: share of respondents with |error| <= tolerance."""
    eps = np.linspace(0, eps_max, 300)
    for k, p in preds.items():
        e = np.abs(np.asarray(p) - y)
        acc = [(e <= t).mean() for t in eps]
        aoc = np.trapezoid(1 - np.array(acc), eps) / eps_max
        ax.plot(eps, acc, color=colors[k], lw=1.8, label=f"{k}  AOC {aoc:.3f}")
    ax.set_xlim(0, eps_max)
    ax.set_ylim(0, 1.01)
    F.pct_axis(ax)
    F.ygrid(ax, "both")


def rank_heatmap(ax, ranks: pd.DataFrame, order, colors=None, max_rank=10, min_text=0.1, fontsize=6.4):
    """P(rank = k) per variable from bootstrap rank draws."""
    k = np.arange(1, max_rank + 1)
    M = np.array([[(ranks[v].round() == kk).mean() for kk in k] for v in order])
    ax.imshow(M, cmap=F.HEAT, aspect="auto", vmin=0, vmax=max(0.5, M.max()),
              extent=(0.5, max_rank + 0.5, len(order) - 0.5, -0.5))
    for i in range(len(order)):
        for j in range(max_rank):
            if M[i, j] >= min_text:
                fc = F.HEAT(min(M[i, j] / max(0.5, M.max()), 1))
                ax.text(j + 1, i, f"{M[i, j]:.0%}".replace("%", ""), ha="center", va="center", fontsize=fontsize,
                        fontweight=B, color=F.text_on(fc))
    ax.set_yticks(range(len(order)), order)
    ax.set_xticks(k)
    if colors:
        for t, v in zip(ax.get_yticklabels(), order):
            t.set_color(colors[v])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return M


def sunburst(ax, inner: dict, outer: dict, inner_colors: dict, min_label=0.035, inner_labels=None):
    """Two-ring sunburst: inner = groups, outer = members (dict group -> {member: value})."""
    total = sum(inner.values())
    start = 90.0
    for g, gv in inner.items():
        span = 360 * gv / total
        ax.add_patch(Wedge((0, 0), 0.62, start - span, start, width=0.3, fc=inner_colors[g], ec=F.BG, lw=1.4))
        mid = np.deg2rad(start - span / 2)
        if gv / total > min_label:
            ax.text(0.47 * np.cos(mid), 0.47 * np.sin(mid), f"{(inner_labels or {}).get(g, g)}\n{gv / total:.0%}", ha="center", va="center",
                    fontsize=5.8, fontweight=B, color=F.text_on(inner_colors[g]), linespacing=1.0)
        s2 = start
        for m, mv in sorted(outer[g].items(), key=lambda kv: -kv[1]):
            sp = 360 * mv / total
            ax.add_patch(Wedge((0, 0), 1.0, s2 - sp, s2, width=0.36, fc=F.tint(inner_colors[g], 0.35), ec=F.BG,
                               lw=1.0))
            if mv / total > min_label:
                a = np.deg2rad(s2 - sp / 2)
                ax.text(0.82 * np.cos(a), 0.82 * np.sin(a), m, ha="center", va="center", fontsize=6.6,
                        fontweight=B, rotation=0)
            s2 -= sp
        start -= span
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")


def surface3d(ax, x1, x2, Z, counts=None, min_n=15, cmap=None, elev=26, azim=-128):
    """3D surface of a 2D effect; unsupported cells drawn transparent grey."""
    cmap = cmap or F.DIV
    X, Y = np.meshgrid(x1, x2, indexing="ij")
    vmax = np.abs(Z).max()
    fc = cmap((Z / vmax + 1) / 2)
    if counts is not None:
        sup = np.ones_like(Z, bool)
        sup[:-1, :-1] = counts >= min_n
        fc[~sup] = (0.85, 0.85, 0.85, 0.35)
    ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1, lw=0.4, edgecolor=F.BG, shade=False,
                    antialiased=True)
    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_edgecolor(F.RULE)
        a._axinfo["grid"]["color"] = F.GRID
    ax.tick_params(labelsize=6.6, pad=0)


def chord(ax, nodes: list, colors: dict, links: list, label_r=1.12):
    """Chord diagram: nodes on a circle, ribbons between pairs with width ~ weight.
    links: (a, b, weight, color)."""
    n = len(nodes)
    weight = {k: 0.0 for k in nodes}
    for a, b, w, _ in links:
        weight[a] += w
        weight[b] += w
    tot = sum(weight.values())
    gap = 2 * np.pi * 0.02
    span = {k: (2 * np.pi - gap * n) * weight[k] / tot for k in nodes}
    start, arcs = np.pi / 2, {}
    for k in nodes:
        arcs[k] = [start, start - span[k]]
        a0, a1 = np.rad2deg(start - span[k]), np.rad2deg(start)
        ax.add_patch(Wedge((0, 0), 1.0, a0, a1, width=0.07, fc=colors[k], ec=F.BG, lw=0.6))
        mid = start - span[k] / 2
        ax.text(label_r * np.cos(mid), label_r * np.sin(mid), k, ha="center", va="center", fontsize=7.2,
                fontweight=B, color=colors[k])
        start -= span[k] + gap
    cursor = {k: arcs[k][0] for k in nodes}
    for a, b, w, col in sorted(links, key=lambda t: t[2]):
        wa = span[a] * w / weight[a]
        wb = span[b] * w / weight[b]
        a0, a1 = cursor[a], cursor[a] - wa
        b0, b1 = cursor[b], cursor[b] - wb
        cursor[a] -= wa
        cursor[b] -= wb
        r = 0.93
        p = lambda t: (r * np.cos(t), r * np.sin(t))  # noqa: E731
        verts = [p(a0)] + [p(t) for t in np.linspace(a0, a1, 8)] + [(0, 0), p(b0)] + \
                [p(t) for t in np.linspace(b0, b1, 8)] + [(0, 0), p(a0)]
        codes = [MPath.MOVETO] + [MPath.LINETO] * 8 + [MPath.CURVE3, MPath.CURVE3] + [MPath.LINETO] * 8 + \
                [MPath.CURVE3, MPath.CURVE3]
        ax.add_patch(PathPatch(MPath(verts, codes), fc=col, ec="none", alpha=0.55, zorder=1))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")


def tornado(ax, labels, deltas: dict, colors: dict, height=0.36):
    """Tornado chart of signed changes for several series (e.g. pipelines)."""
    keys = list(deltas)
    n = len(labels)
    y = np.arange(n)[::-1]
    for k, key in enumerate(keys):
        off = (k - (len(keys) - 1) / 2) * height
        v = np.asarray(deltas[key], float)
        ax.barh(y + off, v, height=height * 0.92, color=colors[key], lw=0, label=key)
        for yy, vv in zip(y + off, v):
            ax.text(vv + (0.0008 if vv >= 0 else -0.0008), yy, F.minus(f"{vv:+.3f}"), va="center",
                    ha="left" if vv >= 0 else "right", fontsize=6.4, fontweight=B, color=colors[key])
    ax.axvline(0, color=F.INK, lw=1.0)
    ax.set_yticks(y, labels)
    F.clean_left(ax)
    F.ygrid(ax, "x")


def radar(ax, labels, series: dict, colors: dict, rmin=None, rmax=None, fill=True):
    """Radar / spider chart on a polar axis."""
    n = len(labels)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    th_c = np.r_[th, th[:1]]
    allv = np.concatenate([np.asarray(v, float) for v in series.values()])
    rmin = rmin if rmin is not None else np.floor(allv.min() * 2) / 2
    rmax = rmax if rmax is not None else np.ceil(allv.max() * 2) / 2
    for k, v in series.items():
        v = np.r_[v, v[:1]]
        ax.plot(th_c, v, color=colors[k], lw=1.8, label=k)
        ax.scatter(th, v[:-1], s=16, color=colors[k], ec=F.BG, lw=0.6, zorder=3)
        if fill:
            ax.fill(th_c, v, color=colors[k], alpha=0.12)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(th, labels, fontsize=7.4, fontweight=B)
    ax.set_ylim(rmin, rmax)
    ax.set_yticks(np.arange(rmin, rmax + 0.01, 0.5))
    ax.tick_params(axis="y", labelsize=6.2)
    ax.spines["polar"].set_color(F.RULE)
    ax.grid(color=F.GRID, lw=0.7)


# ---------------------------------------------------------------------------
# Polished line rendering: glow + halo + sigmoid connectors
# ---------------------------------------------------------------------------
def _sigmoid_xy(x0, y0, x1, y1, n=40, k=10.0):
    """Smooth S-shaped connector between two points (logistic easing)."""
    t = np.linspace(0, 1, n)
    s = 1 / (1 + np.exp(-k * (t - 0.5)))
    s = (s - s[0]) / (s[-1] - s[0])
    return x0 + (x1 - x0) * t, y0 + (y1 - y0) * s


def sigmoid_path(xs, ys, n=40, k=10.0):
    """Concatenate sigmoid connectors through all points."""
    X, Y = [xs[0]], [ys[0]]
    for i in range(len(xs) - 1):
        a, b = _sigmoid_xy(xs[i], ys[i], xs[i + 1], ys[i + 1], n, k)
        X.extend(a[1:])
        Y.extend(b[1:])
    return np.array(X), np.array(Y)


def glow_line(ax, x, y, color, lw=2.2, glow=True, halo=True, zorder=3, alpha=1.0, ls="-", label=None):
    """Line with a soft coloured glow underneath and a background-coloured halo."""
    import matplotlib.patheffects as pe
    if glow:
        for w, a in [(lw * 4.2, 0.05), (lw * 2.8, 0.09), (lw * 1.8, 0.14)]:
            ax.plot(x, y, color=color, lw=w, alpha=a * alpha, solid_capstyle="round", zorder=zorder - 0.2)
    effects = [pe.Stroke(linewidth=lw + 2.4, foreground=F.BG), pe.Normal()] if halo else None
    return ax.plot(x, y, color=color, lw=lw, alpha=alpha, solid_capstyle="round", solid_joinstyle="round",
                   ls=ls, path_effects=effects, zorder=zorder, label=label)[0]


def ring_markers(ax, x, y, color, s=58, zorder=5):
    """Data markers: coloured disc with white ring and soft outer ring."""
    ax.scatter(x, y, s=s * 2.0, color=color, alpha=0.18, lw=0, zorder=zorder - 0.1)
    ax.scatter(x, y, s=s, color=color, ec=F.BG, lw=1.4, zorder=zorder)


def bump_smooth(ax, ranks, labels, periods, colors, lw=2.6, label_side="right"):
    """Bump chart with sigmoid connectors, glow and ringed markers. ranks: (n_entities, n_periods)."""
    ranks = np.asarray(ranks, float)
    n, T = ranks.shape
    x = np.arange(T)
    order = np.argsort(-ranks[:, -1])           # draw best-ranked last (on top)
    for i in order:
        X, Y = sigmoid_path(x, ranks[i])
        glow_line(ax, X, Y, colors[i], lw=lw, zorder=3 + (n - ranks[i, -1]) * 0.01)
        ring_markers(ax, x, ranks[i], colors[i], s=56, zorder=6)
        for xi, r in zip(x, ranks[i]):
            ax.text(xi, r, f"{int(r)}", ha="center", va="center", fontsize=6.0, fontweight=B,
                    color=F.text_on(colors[i]), zorder=7)
        ax.text(T - 1 + 0.22, ranks[i, -1], labels[i], va="center", fontsize=7.8 * F.theme()["scale"],
                color=colors[i], fontweight=B)
    ax.set_xticks(x, periods)
    ax.set_yticks(range(1, n + 1))
    ax.set_ylim(n + 0.6, 0.4)
    ax.set_xlim(-0.3, T - 1 + 1.25)
    ax.set_ylabel("Rank")
    for t in range(T):
        ax.axvline(t, color=F.GRID, lw=0.8, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def slope_smooth(ax, labels, before, after, names, colors, fmt="{:.3f}", lw=2.4):
    """Slopegraph with sigmoid connectors, glow, ringed markers and spread direct labels."""
    n = len(labels)
    for i in range(n):
        X, Y = _sigmoid_xy(0, before[i], 1, after[i], n=60, k=8)
        glow_line(ax, X, Y, colors[i], lw=lw)
        ring_markers(ax, [0, 1], [before[i], after[i]], colors[i], s=50)
    rng_ = max(max(before), max(after)) - min(min(before), min(after))
    lims = [min(min(before), min(after)), max(max(before), max(after))]
    for side, vals, ha, x in [(0, before, "right", -0.07), (1, after, "left", 1.07)]:
        ys = F.spread(vals, rng_ * 0.085)
        lims = [min(lims[0], min(ys)), max(lims[1], max(ys))]
        for i in range(n):
            txt = f"{labels[i]}  {fmt.format(vals[i])}" if side else fmt.format(vals[i])
            ax.text(x, ys[i], txt, ha=ha, va="center", fontsize=7.6 * F.theme()["scale"], color=colors[i],
                    fontweight=B)
    ax.set_xticks([0, 1], names)
    ax.set_xlim(-0.42, 1.75)
    ax.set_ylim(lims[0] - rng_ * 0.06, lims[1] + rng_ * 0.06)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    for x in (0, 1):
        ax.axvline(x, color=F.RULE, lw=1.0, zorder=0)


def parallel_smooth(ax, df: pd.DataFrame, colors: dict, higher_better: dict, label_fmt="{:.2f}", lw=2.2):
    """Parallel coordinates with sigmoid connectors between axes, glow and ringed markers."""
    cols = list(df.columns)
    Z = df.copy().astype(float)
    for c in cols:
        lo, hi = df[c].min(), df[c].max()
        z = (df[c] - lo) / (hi - lo) if hi > lo else df[c] * 0 + 0.5
        Z[c] = z if higher_better[c] else 1 - z
    x = np.arange(len(cols))
    for k in x:
        ax.plot([k, k], [0, 1], color=F.RULE, lw=1.4, solid_capstyle="round", zorder=0)
        ax.text(k, 1.075, cols[k], ha="center", fontsize=7.6, fontweight=B)
        lo_v = df[cols[k]].min() if higher_better[cols[k]] else df[cols[k]].max()
        hi_v = df[cols[k]].max() if higher_better[cols[k]] else df[cols[k]].min()
        ax.text(k, -0.075, label_fmt.format(lo_v), ha="center", fontsize=6.4, color=F.INK2, fontweight=B)
        ax.text(k, 1.02, label_fmt.format(hi_v), ha="center", fontsize=6.4, color=F.INK2, fontweight=B)
    for m, row in Z.iterrows():
        X, Y = sigmoid_path(x, row.values, n=30, k=7)
        glow_line(ax, X, Y, colors[m], lw=lw)
        ring_markers(ax, x, row.values, colors[m], s=30)
    ax.set_xlim(-0.25, len(cols) - 0.75)
    ax.set_ylim(-0.13, 1.13)
    ax.axis("off")


def calendar_span(ax, dates, start, end, cmap=None, gaps=()):
    """Week x weekday grid of daily counts over an arbitrary period (may cross a year boundary)."""
    cmap = cmap or F.SEQ
    s = pd.Series(1, index=pd.to_datetime(dates)).groupby(level=0).sum()
    days = pd.date_range(start, end)
    counts = s.reindex(days, fill_value=0)
    vmax = counts.max()
    first = days[0].weekday()
    for i, (d, c) in enumerate(counts.items()):
        week = (i + first) // 7
        fc = F.tint(F.INK, 0.94) if c == 0 else cmap(0.18 + 0.82 * c / vmax)
        ax.add_patch(Rectangle((week, 6 - d.weekday()), 0.88, 0.88, fc=fc, lw=0))
    for m in pd.date_range(days[0].replace(day=1), end, freq="MS"):
        if m < days[0]:
            continue
        w = ((m - days[0]).days + first) // 7
        ax.text(w, 7.25, m.strftime("%b %Y") if m.month in (1, 8) else m.strftime("%b"), fontsize=7, fontweight=B,
                color=F.INK2, ha="left")
    ax.set_yticks([6 - i + 0.44 for i in range(7)], ["Mon", "", "Wed", "", "Fri", "", "Sun"], fontsize=6.8)
    ax.set_xlim(-0.5, (len(days) + first) // 7 + 1)
    ax.set_ylim(-0.3, 8)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_aspect("equal")
    return vmax, first
