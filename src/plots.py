"""plots - advanced figure types for fstudio. Every function draws on a given axes
(or subplotspec) and reads colours from the ACTIVE theme at call time.

    import fstudio as F, plots as P
    F.use_theme("earth")
    fig = F.new_figure(180, 90); ax = fig.add_subplot(121)
    P.raincloud(ax, [a, b], ["Control", "Treated"])
"""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Patch, Rectangle
from matplotlib.path import Path as MPath
from scipy.stats import gaussian_kde

import fstudio as F

_rng = np.random.default_rng(7)


def _kde(d, xs, bw=0.35):
    return gaussian_kde(np.asarray(d, float), bw_method=bw)(xs)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
def raincloud(ax, data, labels, colors=None, bw=0.35, show_stats=True, unit=""):
    """Horizontal raincloud: half-density + box + jittered points per group."""
    colors = colors or ([F.G0, F.G1] if len(data) == 2 else F.series(len(data)))
    lo_all = min(np.min(d) for d in data)
    hi_all = max(np.max(d) for d in data)
    pad = 0.06 * (hi_all - lo_all)
    for g, (d, col) in enumerate(zip(data, colors)):
        d = np.asarray(d, float)
        d = d[~np.isnan(d)]
        xs = np.linspace(d.min() - pad, d.max() + pad, 300)
        ys = _kde(d, xs, bw)
        ys = ys / ys.max() * 0.52
        ax.fill_between(xs, g + 0.06, g + 0.06 + ys, color=col, alpha=0.30, lw=0, zorder=2)
        ax.plot(xs, g + 0.06 + ys, color=col, lw=1.1, zorder=3)
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        lo, hi = np.percentile(d, [2.5, 97.5])
        ax.plot([lo, hi], [g - 0.04] * 2, color=F.INK2, lw=0.8, zorder=3)
        ax.add_patch(Rectangle((q1, g - 0.09), q3 - q1, 0.10, fc=F.BG, ec=F.INK, lw=0.9, zorder=4))
        ax.plot([med, med], [g - 0.09, g + 0.01], color=col, lw=2.2, zorder=5, solid_capstyle="butt")
        ax.scatter(d + _rng.normal(0, 0.004 * (hi_all - lo_all), len(d)),
                   g + _rng.uniform(-0.36, -0.15, len(d)), s=2.2, color=col, alpha=0.45, lw=0,
                   zorder=2, rasterized=True)
    if show_stats and len(data) == 2:
        d0, d1 = (np.asarray(x, float) for x in data)
        smd = (d1.mean() - d0.mean()) / np.sqrt((d0.var(ddof=1) + d1.var(ddof=1)) / 2)
        ax.text(0.98, 0.98, F.minus(f"Δ median {np.median(d1) - np.median(d0):+.2f} {unit}\n"
                                    f"SMD {smd:+.2f}"), transform=ax.transAxes, ha="right",
                va="top", fontsize=8 * F.theme()["scale"], color=F.INK2)
    ax.set_yticks(range(len(data)), labels)
    for t, c in zip(ax.get_yticklabels(), colors):
        t.set_color(c)
        t.set_fontweight("bold")
    F.clean_left(ax)
    ax.set_ylim(-0.45, len(data) - 1 + (1.12 if show_stats and len(data) == 2 else 0.72))
    F.ygrid(ax, "x")


def ridgeline(ax, data, labels, colors=None, overlap=1.8, bw=0.3, quantiles=True):
    """Stacked density ridges (many groups/periods). Colours from SEQ ramp if not given."""
    n = len(data)
    colors = colors or [F.SEQ(0.25 + 0.65 * i / max(n - 1, 1)) for i in range(n)]
    lo = min(np.min(d) for d in data)
    hi = max(np.max(d) for d in data)
    xs = np.linspace(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo), 400)
    dens = [_kde(d, xs, bw) for d in data]
    top = max(d.max() for d in dens)
    for i in range(n - 1, -1, -1):
        y0 = i
        y = dens[i] / top * overlap
        ax.fill_between(xs, y0, y0 + y, color=colors[i], alpha=0.85, lw=0, zorder=n - i)
        ax.plot(xs, y0 + y, color=F.BG, lw=1.2, zorder=n - i + 0.1)
        ax.plot(xs, y0 + y, color=F.INK, lw=0.5, zorder=n - i + 0.2)
        if quantiles:
            med = np.median(data[i])
            ax.plot([med, med], [y0, y0 + np.interp(med, xs, y)], color=F.INK, lw=0.9,
                    zorder=n - i + 0.3)
    ax.set_yticks(range(n), labels)
    F.clean_left(ax)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.2, n - 1 + overlap + 0.1)
    F.ygrid(ax, "x")


def beeswarm(ax, values, feature_values, labels, order=None, width=0.38, cbar=True):
    """SHAP-style beeswarm: rows = features, colour = within-feature percentile (DIV map).
    values / feature_values: 2-D arrays (n_obs, n_features)."""
    values, feature_values = np.asarray(values, float), np.asarray(feature_values, float)
    p = values.shape[1]
    order = order if order is not None else np.argsort(np.abs(values).mean(0))
    for row, j in enumerate(order):
        v = values[:, j]
        from scipy.stats import rankdata
        fv = feature_values[:, j]
        rank = np.full(len(v), 0.5)
        ok = ~np.isnan(fv)
        rank[ok] = (rankdata(fv[ok], method="average") - 0.5) / ok.sum()
        edges = np.linspace(v.min(), v.max() + 1e-9, 71)
        b = np.clip(np.digitize(v, edges) - 1, 0, 69)
        off = np.zeros(len(v))
        cmax = np.bincount(b).max()
        for k in np.unique(b):
            idx = np.where(b == k)[0]
            idx = idx[_rng.permutation(len(idx))]
            s = width * np.sqrt(len(idx) / cmax)
            off[idx] = np.linspace(-s, s, len(idx)) if len(idx) > 1 else 0
        ax.scatter(v, row + off, c=rank, cmap=F.DIV, vmin=0, vmax=1, s=4.5, lw=0,
                   rasterized=True, zorder=2)
    ax.axvline(0, color=F.INK2, lw=0.8, zorder=1)
    ax.set_yticks(range(p), [labels[j] for j in order])
    F.clean_left(ax)
    F.ygrid(ax, "x")
    if cbar:
        import matplotlib as mpl
        cax = ax.inset_axes([0.70, 0.05, 0.26, 0.03])
        cb = ax.figure.colorbar(mpl.cm.ScalarMappable(cmap=F.DIV), cax=cax, orientation="horizontal")
        cb.set_ticks([0, 1], labels=["Low", "High"])
        cb.ax.tick_params(labelsize=7.5, length=0, pad=1)
        cb.outline.set_visible(False)
        cax.set_title("Feature value", fontsize=7.5, pad=2, fontweight="bold", color=F.INK2)


# ---------------------------------------------------------------------------
# Estimates and uncertainty
# ---------------------------------------------------------------------------
def half_eye(ax, draws, labels, colors=None, log_scale=False, ref=None, ticks=None, annotate=True,
             text_x=None):
    """Posterior / bootstrap distributions: density + 66% and 95% intervals + median.
    With log_scale=True the draws are log-odds and the axis shows odds ratios."""
    n = len(draws)
    colors = colors or (F.series(n) if n <= 6 else [F.G0] * n)
    tr = np.exp if log_scale else (lambda z: z)
    for row, (d, col) in enumerate(zip(draws[::-1], colors[::-1])):
        d = np.asarray(d, float)
        g = np.linspace(d.min(), d.max(), 400)
        k = _kde(d, g, "scott")
        k = k / k.max() * 0.78
        ax.fill_between(tr(g), row, row + k, color=col, alpha=0.32, lw=0, zorder=2)
        ax.plot(tr(g), row + k, color=col, lw=1.0, zorder=3)
        q = tr(np.percentile(d, [2.5, 17, 50, 83, 97.5]))
        ax.plot([q[0], q[4]], [row] * 2, color=F.INK, lw=1.0, solid_capstyle="butt", zorder=4)
        ax.plot([q[1], q[3]], [row] * 2, color=F.INK, lw=3.4, solid_capstyle="butt", zorder=4)
        ax.scatter(q[2], row, s=26, fc=F.BG, ec=F.INK, lw=1.1, zorder=5)
        if annotate and text_x is not None:
            ax.text(text_x, row + 0.18, f"{q[2]:.2f} [{q[0]:.2f}, {q[4]:.2f}]", fontsize=8,
                    fontweight="bold", va="center", zorder=8)
    if ref is not None:
        ax.axvline(ref, color=F.INK2, lw=0.8, zorder=1)
    if log_scale:
        F.plain_log(ax, ticks or [0.5, 0.75, 1, 1.5, 2, 3])
    ax.set_yticks(range(n), labels[::-1])
    F.clean_left(ax)
    ax.set_ylim(-0.4, n)
    F.ygrid(ax, "x")


def forest(ax, labels, est, lo, hi, groups=None, ref=1.0, log_scale=True, ticks=None,
           color=None, values_col=True):
    """Forest plot with alternating row bands, optional group headers and a value column."""
    color = color or F.G0
    n = len(labels)
    y = np.arange(n)[::-1]
    for i, yy in enumerate(y):
        if i % 2 == 0:
            ax.axhspan(yy - 0.5, yy + 0.5, color=F.tint(F.INK, 0.96), lw=0, zorder=0)
    sizes = 28 if groups is None else [40 if g == "overall" else 26 for g in groups]
    ax.hlines(y, lo, hi, color=color, lw=1.5, zorder=2)
    mk = ["D" if (groups is not None and groups[i] == "overall") else "s" for i in range(n)]
    for i in range(n):
        ax.scatter(est[i], y[i], marker=mk[i], s=sizes if np.isscalar(sizes) else sizes[i],
                   color=color, ec=F.BG, lw=0.8, zorder=3)
    if ref is not None:
        ax.axvline(ref, color=F.INK2, lw=0.8, zorder=1)
    if log_scale:
        F.plain_log(ax, ticks or [0.5, 1, 2, 4])
    ax.set_yticks(y, labels)
    F.clean_left(ax)
    ax.set_ylim(-0.6, n - 0.4)
    if values_col:
        for i in range(n):
            ax.text(1.02, y[i], f"{est[i]:.2f} [{lo[i]:.2f}, {hi[i]:.2f}]",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=7.8, color=F.INK)
    F.ygrid(ax, "x")


def fan_chart(ax, x, draws, color=None, levels=(50, 80, 95), spaghetti=0, label=None):
    """Nested uncertainty bands (fan) + median, optional thin spaghetti draws.
    draws: (n_draws, len(x))."""
    color = color or F.G0
    draws = np.asarray(draws, float)
    for k, lev in enumerate(sorted(levels, reverse=True)):
        lo, hi = np.percentile(draws, [50 - lev / 2, 50 + lev / 2], axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.14 + 0.12 * k, lw=0, zorder=1 + k)
    if spaghetti:
        for row in draws[_rng.choice(len(draws), spaghetti, replace=False)]:
            ax.plot(x, row, color=color, lw=0.35, alpha=0.12, zorder=4)
    ax.plot(x, np.median(draws, 0), color=color, lw=2.0, zorder=6, label=label)
    F.ygrid(ax)


def lollipop(ax, labels, values, lo=None, hi=None, color=None, sort=True, fmt="{:.2f}", ref=0):
    """Ranked lollipop with optional interval and value labels."""
    color = color or F.G0
    v = np.asarray(values, float)
    o = np.argsort(v) if sort else np.arange(len(v))[::-1]
    y = np.arange(len(v))
    ax.hlines(y, ref, v[o], color=F.RULE, lw=1.4, zorder=1)
    if lo is not None:
        ax.hlines(y, np.asarray(lo)[o], np.asarray(hi)[o], color=color, lw=2.6, alpha=0.35, zorder=2)
    ax.scatter(v[o], y, s=46, color=color, ec=F.BG, lw=1, zorder=3)
    for yy, val in zip(y, v[o]):
        ax.text(val, yy + 0.32, fmt.format(val), ha="center", fontsize=7.6, color=F.INK2)
    ax.set_yticks(y, [labels[i] for i in o])
    F.clean_left(ax)
    F.ygrid(ax, "x")


def dose_response(ax, x, k, n, color=None, labels=None, ref=None, size_scale=0.33):
    """Proportion by ordered category with Wilson 95% CI; marker area ∝ n."""
    color = color or F.G1
    p, lo, hi = F.wilson(k, n)
    if ref is not None:
        ax.axhline(ref, color=F.MUTED, lw=0.8, zorder=1)
    ax.fill_between(x, 0, p, color=color, alpha=0.10, lw=0, zorder=1)
    ax.vlines(x, lo, hi, color=color, lw=1.4, zorder=2)
    ax.plot(x, p, color=color, lw=1.4, zorder=2)
    ax.scatter(x, p, s=np.asarray(n) * size_scale + 12, color=color, ec=F.BG, lw=1, zorder=3)
    if labels is not None:
        ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    F.pct_axis(ax)
    F.ygrid(ax)


# ---------------------------------------------------------------------------
# Comparisons and change
# ---------------------------------------------------------------------------
def dumbbell(ax, labels, a, b, a_ci=None, b_ci=None, names=("Group 0", "Group 1"), ratio=True,
             pct=True):
    """Two-group comparison per row with optional CIs and ratio label."""
    c0, c1 = F.G0, F.G1
    n = len(labels)
    for row in range(n):
        y = n - 1 - row
        ax.plot([a[row], b[row]], [y, y], color=F.tint(F.INK, 0.88), lw=7, solid_capstyle="butt",
                zorder=1)
        for val, ci, col, dy in [(a[row], a_ci, c0, -0.22), (b[row], b_ci, c1, 0.22)]:
            if ci is not None:
                ax.plot(ci[row], [y + dy] * 2, color=col, lw=1.4, zorder=2)
            ax.plot([val, val], [y, y + dy], color=col, lw=0.8, zorder=2)
            ax.scatter(val, y + dy, s=46, color=col, ec=F.BG, lw=1, zorder=3)
        if ratio and a[row]:
            ax.text(1.0, y, f"×{b[row] / a[row]:.1f}", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=9 * F.theme()["scale"], fontweight="bold")
    ax.set_yticks(range(n), labels[::-1])
    F.clean_left(ax)
    ax.set_ylim(-0.7, n - 0.3)
    if pct:
        F.pct_axis(ax, "x")
    F.ygrid(ax, "x")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, mec=F.BG, ms=6, label=l)
                       for c, l in zip([c1, c0], names[::-1])], loc="lower right", ncol=2,
              fontsize=7.8)


def slopegraph(ax, labels, before, after, names=("Before", "After"), colors=None, fmt="{:.2f}",
               highlight=None):
    """Before -> after per entity with direct labels (spread to avoid overlaps)."""
    n = len(labels)
    colors = colors or (F.series(n) if n <= 6 else [F.MUTED] * n)
    if highlight is not None:
        colors = [c if lab in highlight else F.tint(F.INK, 0.75) for c, lab in zip(colors, labels)]
    for i in range(n):
        ls, mk = F.line_style(i)
        ax.plot([0, 1], [before[i], after[i]], color=colors[i], lw=1.6, ls=ls, marker=mk, ms=5,
                mec=F.BG, mew=0.9)
    rng = max(max(before), max(after)) - min(min(before), min(after))
    for side, vals, ha, x in [(0, before, "right", -0.05), (1, after, "left", 1.05)]:
        ys = F.spread(vals, rng * 0.06)
        for i in range(n):
            txt = f"{labels[i]}  {fmt.format(vals[i])}" if side else f"{fmt.format(vals[i])}"
            ax.text(x, ys[i], txt, ha=ha, va="center", fontsize=8 * F.theme()["scale"],
                    color=colors[i], fontweight="bold" if side else "normal")
    ax.set_xticks([0, 1], names)
    ax.set_xlim(-0.35, 1.65)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    for x in (0, 1):
        ax.axvline(x, color=F.RULE, lw=0.8, zorder=0)


def bump_chart(ax, ranks, labels, periods, colors=None, highlight=None):
    """Rank trajectories across periods. ranks: (n_entities, n_periods), 1 = top."""
    ranks = np.asarray(ranks, float)
    n, T = ranks.shape
    colors = colors or (F.series(n) if n <= 6 else [F.MUTED] * n)
    if highlight is not None:
        colors = [c if lab in highlight else F.tint(F.INK, 0.78) for c, lab in zip(colors, labels)]
    x = np.arange(T)
    for i in range(n):
        xs = np.linspace(0, T - 1, 200)
        ys = np.interp(xs, x, ranks[i])
        ax.plot(xs, ys, color=colors[i], lw=2.4, solid_capstyle="round", zorder=2)
        ax.scatter(x, ranks[i], s=60, color=colors[i], ec=F.BG, lw=1.2, zorder=3)
        ax.text(T - 1 + 0.12, ranks[i, -1], labels[i], va="center", fontsize=8 * F.theme()["scale"],
                color=colors[i], fontweight="bold")
    ax.set_xticks(x, periods)
    ax.set_yticks(range(1, n + 1))
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_xlim(-0.2, T - 1 + 0.9)
    ax.set_ylabel("Rank")
    F.ygrid(ax)


def diverging_bars(ax, items, neg, pos, neg_labels, pos_labels, neg_colors=None, pos_colors=None,
                   total_label=True):
    """Diverging stacked bars (Likert / frequency). neg, pos: arrays (n_items, k) of shares
    (0-1); neg[:, 0] sits next to zero."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    neg_colors = neg_colors or [F.tint(F.G0, 0.70), F.tint(F.INK, 0.90), F.tint(F.INK, 0.62)][:neg.shape[1]]
    pos_colors = pos_colors or F.ORD[:pos.shape[1]]
    n = len(items)
    for r in range(n):
        y = n - 1 - r
        left = 0.0
        for j in range(neg.shape[1]):
            ax.barh(y, -neg[r, j], left=-left, height=0.66, color=neg_colors[j], ec=F.BG, lw=0.8)
            left += neg[r, j]
        right = 0.0
        for j in range(pos.shape[1]):
            ax.barh(y, pos[r, j], left=right, height=0.66, color=pos_colors[j], ec=F.BG, lw=0.8)
            right += pos[r, j]
        if total_label:
            ax.text(right + 0.012, y, f"{right:.0%}", va="center", fontsize=8.5 * F.theme()["scale"],
                    fontweight="bold", color=F.G1)
    ax.axvline(0, color=F.INK, lw=0.9, zorder=3)
    ax.set_yticks(range(n), items[::-1])
    F.clean_left(ax)
    from matplotlib import ticker
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{abs(v):.0%}"))
    F.ygrid(ax, "x")
    h = [Patch(fc=c, label=l) for c, l in zip(neg_colors, neg_labels)] + \
        [Patch(fc=c, label=l) for c, l in zip(pos_colors, pos_labels)]
    ax.legend(handles=h, ncol=len(h), loc="upper center", bbox_to_anchor=(0.5, -0.22),
              fontsize=7.8, handlelength=1, columnspacing=1)


# ---------------------------------------------------------------------------
# Composition and flows
# ---------------------------------------------------------------------------
def waffle(ax, counts, labels, colors=None, ncol=25, gap=0.24, title=None):
    """Waffle chart (1 square = 1 unit). Keep it in its own gridspec row."""
    colors = colors or F.series(len(counts))
    cells = []
    for i, c in enumerate(counts):
        cells += [i] * int(c)
    for i, k in enumerate(cells):
        r, c = divmod(i, ncol)
        ax.add_patch(Rectangle((c, -r), 1 - gap, 1 - gap, lw=0, fc=colors[k]))
    nrow = -(-len(cells) // ncol)
    ax.set_xlim(-0.3, ncol)
    ax.set_ylim(-nrow + 1 - 0.4 + gap, 1.0)
    ax.set_aspect("equal", adjustable="box", anchor="NW")
    ax.axis("off")
    tot = sum(counts)
    ax.legend(handles=[Patch(fc=c, label=f"{l}  {n} ({n / tot:.0%})")
                       for c, l, n in zip(colors, labels, counts)],
              loc="upper left", bbox_to_anchor=(0, -0.02), fontsize=8, handlelength=1,
              handleheight=1, title=title, title_fontsize=7.8, alignment="left")


def _ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha):
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b),
             (x0, y0a)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4,
             MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="none", alpha=alpha, zorder=1))


def alluvial(ax, stages, stage_names, order=None, colors=None, gap=0.02, width=0.06):
    """Alluvial / Sankey flows between categorical stages.
    stages: list of equal-length arrays (one per stage) of category labels.
    Flow colour follows the category in the first stage."""
    stages = [np.asarray(s) for s in stages]
    n = len(stages[0])
    cats = [order[i] if order else sorted(set(s), key=lambda c: (str(c))) for i, s in enumerate(stages)]
    first = cats[0]
    colors = colors or dict(zip(first, F.series(len(first))))
    pos = []
    for s, cs in zip(stages, cats):
        y, d = 0.0, {}
        tot_gap = gap * (len(cs) - 1)
        for c in cs:
            h = (s == c).sum() / n * (1 - tot_gap)
            d[c] = [y, y + h]
            y += h + gap
        pos.append(d)
    xs = np.linspace(0, 1, len(stages))
    for k in range(len(stages) - 1):
        off_a = {c: pos[k][c][0] for c in cats[k]}
        off_b = {c: pos[k + 1][c][0] for c in cats[k + 1]}
        for ca in cats[k]:
            for cb in cats[k + 1]:
                # later stages are split by first-stage category so colour follows it through
                for c0 in (cats[0] if k else [ca]):
                    m = (stages[k] == ca) & (stages[k + 1] == cb) & (stages[0] == c0)
                    if not m.any():
                        continue
                    h = m.sum() / n * (1 - gap * (len(cats[k]) - 1))
                    hb = m.sum() / n * (1 - gap * (len(cats[k + 1]) - 1))
                    _ribbon(ax, xs[k] + width / 2, xs[k + 1] - width / 2, off_a[ca], off_a[ca] + h,
                            off_b[cb], off_b[cb] + hb, colors.get(c0, F.MUTED), 0.42)
                    off_a[ca] += h
                    off_b[cb] += hb
    for k, (x, d) in enumerate(zip(xs, pos)):
        for c, (y0, y1) in d.items():
            ax.add_patch(Rectangle((x - width / 2, y0), width, y1 - y0, fc=F.INK, ec=F.BG, lw=0.6,
                                   zorder=3))
            ha, dx = ("right", -width) if k == 0 else ("left", width)
            if 0 < k < len(xs) - 1:
                ha, dx = "center", 0
            ax.text(x + dx * 0.7, (y0 + y1) / 2, f"{c}\n{(stages[k] == c).sum()}", ha=ha,
                    va="center", fontsize=7.6 * F.theme()["scale"], color=F.INK, zorder=4,
                    bbox=dict(fc=F.BG, ec="none", pad=0.6, alpha=0.8) if 0 < k < len(xs) - 1 else None)
        ax.text(x, 1.06, stage_names[k], ha="center", fontweight="bold",
                fontsize=9 * F.theme()["scale"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(1.12, -0.04)
    ax.axis("off")


def upset(fig, spec, sets, top=8, color=None):
    """UpSet plot (intersections of up to ~6 sets). sets: dict name -> set of ids.
    Returns (ax_bars, ax_matrix, ax_sizes)."""
    color = color or F.G0
    names = list(sets)
    universe = set().union(*sets.values())
    combos = {}
    for e in universe:
        key = tuple(int(e in sets[nm]) for nm in names)
        combos[key] = combos.get(key, 0) + 1
    items = sorted(combos.items(), key=lambda kv: -kv[1])[:top]
    gs = spec.subgridspec(2, 2, height_ratios=[2.2, 1], width_ratios=[1, 3.2], hspace=0.05, wspace=0.04)
    axb = fig.add_subplot(gs[0, 1])
    axm = fig.add_subplot(gs[1, 1], sharex=axb)
    axs = fig.add_subplot(gs[1, 0], sharey=axm)
    x = np.arange(len(items))
    axb.bar(x, [v for _, v in items], color=color, width=0.62, ec=F.BG, lw=0.8)
    for xi, (_, v) in zip(x, items):
        axb.text(xi, v, f"{v}", ha="center", va="bottom", fontsize=7.6, color=F.INK2)
    axb.set_ylabel("Size")
    axb.tick_params(axis="x", labelbottom=False, length=0)
    F.ygrid(axb)
    for j in range(len(names)):
        if j % 2 == 0:
            axm.axhspan(j - 0.5, j + 0.5, color=F.tint(F.INK, 0.95), lw=0)
    for xi, (key, _) in zip(x, items):
        on = [j for j, b in enumerate(key) if b]
        axm.scatter([xi] * len(names), range(len(names)), s=36, color=F.tint(F.INK, 0.82), zorder=2)
        axm.scatter([xi] * len(on), on, s=40, color=F.INK, zorder=3)
        if len(on) > 1:
            axm.plot([xi, xi], [min(on), max(on)], color=F.INK, lw=1.6, zorder=2)
    axm.set_yticks(range(len(names)), names)
    axm.set_xticks([])
    axm.tick_params(axis="y", length=0)
    for s_ in ("left", "bottom"):
        axm.spines[s_].set_visible(False)
    sizes = [len(sets[nm]) for nm in names]
    axs.barh(range(len(names)), sizes, color=F.MUTED, height=0.55)
    axs.invert_xaxis()
    axs.set_xlabel("Set size")
    axs.tick_params(axis="y", labelleft=False, length=0)
    axs.spines["left"].set_visible(False)
    axs.spines["right"].set_visible(True)
    axm.set_ylim(-0.6, len(names) - 0.4)
    return axb, axm, axs


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------
def heat_table(ax, data, row_labels, col_labels, higher_better=None, fmt="{:.3f}", ref_rows=(),
               row_colors=None):
    """Metric table as a heatmap, shaded by within-column rank (darker = better)."""
    data = np.asarray(data, float)
    nr, nc = data.shape
    hb = higher_better or [True] * nc
    for j in range(nc):
        col = data[[i for i in range(nr) if i not in ref_rows], j]
        lo, hi = col.min(), col.max()
        for i in range(nr):
            v = data[i, j]
            if i in ref_rows:
                fc = F.tint(F.INK, 0.94)
            else:
                s = (v - lo) / (hi - lo) if hi > lo else 1.0
                s = s if hb[j] else 1 - s
                fc = F.HEAT(0.18 + 0.72 * s)
            ax.add_patch(Rectangle((j, -i), 0.96, 0.94, fc=fc, lw=0))
            ax.text(j + 0.48, -i + 0.47, fmt.format(v), ha="center", va="center",
                    fontsize=9 * F.theme()["scale"], fontweight="bold", color=F.text_on(fc))
    ax.set_xlim(-0.02, nc)
    ax.set_ylim(-nr + 0.94, 1)
    ax.set_xticks(np.arange(nc) + 0.48, [f"{c} {'↑' if h else '↓'}" for c, h in zip(col_labels, hb)],
                  fontweight="bold")
    ax.xaxis.tick_top()
    ax.set_yticks(-np.arange(nr) + 0.47, row_labels)
    if row_colors:
        for t, c in zip(ax.get_yticklabels(), row_colors):
            t.set_color(c)
            t.set_fontweight("bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def corr_heatmap(ax, X, labels, method="spearman", cluster=True, triangle=True, annotate=True,
                 cbar=True):
    """Correlation matrix (lower triangle, hierarchical ordering, diverging map)."""
    import pandas as pd
    df = pd.DataFrame(np.asarray(X, float), columns=labels)
    C = df.corr(method=method).to_numpy()
    order = np.arange(len(labels))
    if cluster:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
        D = 1 - np.abs(C)
        np.fill_diagonal(D, 0)
        order = leaves_list(linkage(squareform(D, checks=False), "average"))
    C = C[np.ix_(order, order)]
    lab = [labels[i] for i in order]
    n = len(lab)
    for i in range(n):
        for j in range(n):
            if i == j or (triangle and j > i):
                continue
            v = C[i, j]
            fc = F.DIV((v + 1) / 2)
            ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fc=fc, lw=0))
            if annotate and i != j:
                ax.text(j, i, F.minus(f"{v:.2f}").replace("0.", "."), ha="center", va="center",
                        fontsize=7.2 * F.theme()["scale"], color=F.text_on(fc))
    if triangle:
        ax.set_xlim(-0.6, n - 1.4)
        ax.set_ylim(n - 0.4, 0.4)
        ax.set_xticks(range(n - 1), lab[:-1])
        ax.set_yticks(range(1, n), lab[1:])
    else:
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(n - 0.4, -0.6)
        ax.set_xticks(range(n), lab)
        ax.set_yticks(range(n), lab)
    F.rotate(ax, 40)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if cbar:
        import matplotlib as mpl
        cax = ax.inset_axes([0.62, 0.92, 0.34, 0.035])
        cb = ax.figure.colorbar(mpl.cm.ScalarMappable(cmap=F.DIV, norm=mpl.colors.Normalize(-1, 1)),
                                cax=cax, orientation="horizontal")
        cb.set_ticks([-1, 0, 1], labels=[F.minus("-1"), "0", "1"])
        cb.ax.tick_params(labelsize=7, length=0, pad=1)
        cb.outline.set_visible(False)
        cax.set_title(f"{method.capitalize()} ρ" if method == "spearman" else "r", fontsize=7.5,
                      pad=2, color=F.INK2, fontweight="bold")


def joint_plot(fig, spec, x, y, groups=None, group_labels=None, kind="scatter", bins=40):
    """Scatter/hexbin with marginal densities. Returns (ax, ax_top, ax_right)."""
    gs = spec.subgridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)
    x, y = np.asarray(x, float), np.asarray(y, float)
    if groups is None:
        groups = np.zeros(len(x), int)
        group_labels = group_labels or [""]
    ug = list(dict.fromkeys(groups))
    cols = [F.G0, F.G1] if len(ug) == 2 else F.series(len(ug))
    if kind == "hexbin":
        ax.hexbin(x, y, gridsize=bins // 2, cmap=F.SEQ, mincnt=1, linewidths=0.2, edgecolors=F.BG)
    for g, col in zip(ug, cols):
        m = groups == g
        if kind == "scatter":
            ax.scatter(x[m], y[m], s=9, color=col, alpha=0.55, lw=0, rasterized=True)
        xs = np.linspace(x.min(), x.max(), 200)
        ys = np.linspace(y.min(), y.max(), 200)
        axt.fill_between(xs, 0, _kde(x[m], xs, "scott"), color=col, alpha=0.3, lw=0)
        axt.plot(xs, _kde(x[m], xs, "scott"), color=col, lw=1.1)
        axr.fill_betweenx(ys, 0, _kde(y[m], ys, "scott"), color=col, alpha=0.3, lw=0)
        axr.plot(_kde(y[m], ys, "scott"), ys, color=col, lw=1.1)
    for a in (axt, axr):
        a.axis("off")
    F.ygrid(ax, "both")
    if group_labels and len(ug) > 1:
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, ms=5, label=l)
                           for c, l in zip(cols, group_labels)], loc="upper left", fontsize=7.8)
    return ax, axt, axr


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------
def roc_panel(ax, y, probs, inset=False):
    """ROC curves for several models (dict name -> probabilities); optional zoom inset."""
    from sklearn.metrics import roc_auc_score, roc_curve
    ax.plot([0, 1], [0, 1], color=F.RULE, lw=0.9)
    cols = F.series(len(probs))
    ins = ax.inset_axes([0.47, 0.08, 0.5, 0.46]) if inset else None
    for i, (nm, p) in enumerate(probs.items()):
        f, t, _ = roc_curve(y, p)
        ls, _m = F.line_style(i)
        ax.plot(f, t, color=cols[i], lw=1.4, ls=ls, label=f"{nm} {roc_auc_score(y, p):.3f}")
        if ins is not None:
            ins.plot(f, t, color=cols[i], lw=1.2, ls=ls)
    if ins is not None:
        ins.set_xlim(0.05, 0.3)
        ins.set_ylim(0.45, 0.85)
        ins.tick_params(labelsize=6.5, length=2)
        for s in ins.spines.values():
            s.set_visible(True)
            s.set_color(F.MUTED)
            s.set_linewidth(0.6)
        ax.indicate_inset_zoom(ins, edgecolor=F.MUTED, lw=0.6, alpha=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    if inset:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.4,
                  title="AUC", title_fontsize=7.4, columnspacing=1)
    else:
        ax.legend(loc="lower right", fontsize=7.4, title="AUC", title_fontsize=7.4)
    F.ygrid(ax, "both")


def calibration_panel(ax, y, probs, bins=10):
    """Decile calibration curves (dict name -> probabilities)."""
    import pandas as pd
    ax.plot([0, 1], [0, 1], color=F.RULE, lw=0.9)
    for i, (nm, p) in enumerate(probs.items()):
        q = pd.qcut(p, bins, labels=False, duplicates="drop")
        ls, mk = F.line_style(i)
        ax.plot(pd.Series(p).groupby(q).mean(), pd.Series(y).groupby(q).mean(),
                color=F.series(len(probs))[i], lw=1.1, ls=ls, marker=mk, ms=3.6, mec=F.BG, mew=0.6,
                label=nm)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed proportion")
    F.ygrid(ax, "both")


def decision_curve(ax, y, probs, thresholds=None):
    """Net benefit vs threshold with treat-all / treat-none references."""
    y = np.asarray(y)
    ts = thresholds if thresholds is not None else np.linspace(0.02, 0.85, 120)
    prev = y.mean()
    ax.plot(ts, prev - (1 - prev) * ts / (1 - ts), color=F.INK2, lw=1.1, ls=(0, (4, 2)))
    ax.axhline(0, color=F.INK, lw=0.9)
    for i, (nm, p) in enumerate(probs.items()):
        nb = [((p >= t) & (y == 1)).mean() - ((p >= t) & (y == 0)).mean() * t / (1 - t) for t in ts]
        ls, _m = F.line_style(i)
        ax.plot(ts, nb, color=F.series(len(probs))[i], lw=1.4, ls=ls, label=nm)
    ax.text(0.04, prev * 0.88, "Treat all", fontsize=7.8, color=F.INK2, rotation=-35)
    ax.text(ts[-1], 0.006, "Treat none", fontsize=7.8, color=F.INK, ha="right")
    ax.set_ylim(-0.04, prev * 1.08)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    F.ygrid(ax)


def waterfall(ax, labels, contributions, base, sd=None, raise_color=None, lower_color=None):
    """Additive explanation waterfall from base to prediction (largest at top)."""
    rc, lc = raise_color or F.G1, lower_color or F.G0
    c = np.asarray(contributions, float)
    o = np.argsort(np.abs(c))
    run = base
    for row, i in enumerate(o):
        col = rc if c[i] > 0 else lc
        ax.barh(row, c[i], left=run, height=0.62, color=col, lw=0, zorder=2)
        end = run + c[i]
        e = 0 if sd is None else sd[i]
        if sd is not None:
            ax.errorbar(end, row, xerr=e, color=F.INK, lw=0.8, capsize=2, zorder=3)
        ax.text(end + (0.06 + e if c[i] > 0 else -0.06 - e) * max(1, np.abs(c).max()), row,
                F.minus(f"{c[i]:+.2f}"), va="center", ha="left" if c[i] > 0 else "right",
                fontsize=7.8, bbox=dict(fc=F.BG, ec="none", pad=0.4, alpha=0.85), zorder=4)
        if row < len(o) - 1:
            ax.plot([end, end], [row + 0.31, row + 0.69], color=F.MUTED, lw=0.6, zorder=1)
        run = end
    ax.set_yticks(range(len(o)), [labels[i] for i in o])
    F.clean_left(ax)
    ax.axvline(base, color=F.INK2, lw=0.8, ls=(0, (3, 2)))
    ax.axvline(run, color=F.INK, lw=1.0)
    ax.text(run, len(o) - 0.35, F.minus(f"f(x) = {run:.2f}"), ha="center", fontsize=7.8,
            fontweight="bold", bbox=dict(fc=F.BG, ec="none", pad=0.6))
    ax.set_ylim(-0.7, len(o) + 0.1)
    ends = base + np.concatenate([[0], np.cumsum(c[o])])
    span = ends.max() - ends.min()
    ax.set_xlim(ends.min() - 0.32 * span, ends.max() + 0.28 * span)
    F.ygrid(ax, "x")


def effect_panel(fig, spec, x, y, n=None, band=None, color=None, labels=None, xlabel=""):
    """Effect curve (ALE/PDP) with optional band and a support strip below. Returns (ax, ax_n)."""
    color = color or F.G0
    gs = spec.subgridspec(2, 1, height_ratios=[4, 0.85], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axn = fig.add_subplot(gs[1], sharex=ax)
    x, y = np.asarray(x, float), np.asarray(y, float)
    if band is not None:
        ax.fill_between(x, y - band, y + band, color=color, alpha=0.18, lw=0)
    ax.axhline(0, color=F.RULE, lw=0.8)
    ax.plot(x, y, color=color, lw=1.8)
    if n is not None:
        ax.scatter(x, y, s=np.asarray(n) * 0.28 + 10, color=color, ec=F.BG, lw=1, zorder=3)
        w = 0.6 if labels is not None else 0.7 * np.min(np.diff(x))
        axn.bar(x, n, width=w, color=F.tint(F.INK, 0.8), lw=0)
    axn.set_yticks([])
    axn.spines["left"].set_visible(False)
    if labels is not None:
        axn.set_xticks(x, labels)
    axn.set_xlabel(xlabel)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    F.ygrid(ax)
    return ax, axn
