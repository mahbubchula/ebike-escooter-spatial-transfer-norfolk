"""fstudio - multi-theme figure design system (Mahbub's figure studio).

    import sys; sys.path.insert(0, "<skill dir>/scripts")
    import fstudio as F
    F.use_theme("classic")                  # classic | nature | earth | poster | midnight | mono
    fig = F.new_figure(180, 120)            # exact print size in mm
    ax = fig.add_subplot(...)
    F.panel(fig, ax, "a", "Short title")
    F.save(fig, "outputs/figures/Fig01_name")   # png/ tiff/ pdf/ svg/ + fit check

Tokens (F.INK, F.G0, F.S4, F.C6, ...) always refer to the ACTIVE theme, so read them
after use_theme(). Every palette was validated for colour-vision deficiency
(see references/themes.md).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_hex

# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------
THEMES = {
    "classic": dict(
        label="Classic journal (Times, blue/crimson)",
        use="Default for papers and theses; the original house style.",
        font=["Times New Roman", "Times", "DejaVu Serif"], serif=True, letter="({})",
        bg="#FFFFFF", ink="#141414", ink2="#474747", muted="#8C8C8C", rule="#CFCFCF", grid="#ECECEC",
        G0="#3A6EA5", G1="#D1495B",
        S4=["#2A6FBB", "#D46A28", "#2A9D8F", "#A63D8F"],
        C6=["#C8414B", "#2F6DB5", "#CF7A22", "#8E5BB0", "#2E9C82", "#9C6B1E"],
        ORD=["#E99484", "#D8665D", "#B8364A", "#7E1A33"],
        SEQ=["#FBF4F1", "#F3C6BC", "#E28A80", "#C23B4E", "#7E1A33", "#4A0F1F"],
        DIV=["#2F6DB5", "#E3E3E3", "#C8414B"],
        HEAT=["#F4F7FB", "#9FBDE0", "#2A6FBB", "#16406F"],
        fill="#9FBDE0", scale=1.0),
    "nature": dict(
        label="Nature-style sans (Arial, Okabe–Ito)",
        use="Journals that require sans-serif lettering (Nature, Cell, Science, PLOS).",
        font=["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], serif=False, letter="{}",
        bg="#FFFFFF", ink="#111111", ink2="#444444", muted="#8A8A8A", rule="#CCCCCC", grid="#EDEDED",
        G0="#0072B2", G1="#D55E00",
        S4=["#0072B2", "#E69F00", "#009E73", "#B03A7A"],
        C6=["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7"],
        ORD=["#EDA06E", "#E27A3A", "#D55E00", "#8F3F00"],
        SEQ=["#F2F7FB", "#C5DCEC", "#7FB2D6", "#3C8DC4", "#0072B2", "#003E66"],
        DIV=["#0072B2", "#E8E8E8", "#D55E00"],
        HEAT=["#F2F7FB", "#A6C8E3", "#3C8DC4", "#00507D"],
        fill="#A6C8E3", scale=0.95),
    "earth": dict(
        label="Earth tones (Georgia, terracotta/ocean)",
        use="Environment, agriculture, social science, transport; warm and calm.",
        font=["Georgia", "Charter", "DejaVu Serif"], serif=True, letter="({})",
        bg="#FBF8F3", ink="#2B2118", ink2="#5A4E44", muted="#968A7E", rule="#D8CFC4", grid="#EEE7DD",
        G0="#24719E", G1="#C4502F",
        S4=["#24719E", "#C4502F", "#C9A227", "#7B2D6B"],
        C6=["#C4502F", "#24719E", "#C9A227", "#3C8D5A", "#7B2D6B", "#A65E2E"],
        ORD=["#E39C7E", "#D77A57", "#C4502F", "#7A2E18"],
        SEQ=["#FBF3EC", "#F1D2BC", "#E3A07D", "#C4502F", "#8E3216", "#561B0B"],
        DIV=["#24719E", "#ECE6DC", "#C4502F"],
        HEAT=["#F4F1EC", "#B9CCD8", "#4F8DB3", "#164A6A"],
        fill="#B9CCD8", scale=1.0),
    "poster": dict(
        label="Poster bold (Avenir, indigo/coral)",
        use="Conference posters, infographics, teaching slides on white.",
        font=["Avenir Next", "Avenir", "Helvetica Neue", "Arial", "DejaVu Sans"], serif=False,
        letter="{}", upper=True,
        bg="#FFFFFF", ink="#1B1B2F", ink2="#4A4A63", muted="#9090A6", rule="#D3D3E0", grid="#EEEEF5",
        G0="#3A4CC0", G1="#F2545B",
        S4=["#3A4CC0", "#F28C28", "#12A38A", "#C2185B"],
        C6=["#F2545B", "#3A4CC0", "#F28C28", "#12A38A", "#9B4DCA", "#1B9AAA"],
        ORD=["#F59290", "#F06A71", "#E03E47", "#9E1A22"],
        SEQ=["#FFF4F3", "#FCC9C6", "#F7908E", "#F2545B", "#C0262D", "#7A1218"],
        DIV=["#3A4CC0", "#EDEDF3", "#F2545B"],
        HEAT=["#F3F4FC", "#B5BCEB", "#5E6FD6", "#25308A"],
        fill="#B5BCEB", scale=1.18),
    "midnight": dict(
        label="Midnight dark (Helvetica Neue, for slides)",
        use="Dark-background presentations, dashboards, talks.",
        font=["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], serif=False, letter="{}",
        bg="#0F1726", ink="#F2F4F8", ink2="#B8C0CC", muted="#7C8798", rule="#3A475C", grid="#1E2A3D",
        G0="#4A90E2", G1="#E5534B",
        S4=["#4A90E2", "#C9781F", "#2BA88A", "#B5479B"],
        C6=["#E5534B", "#4A90E2", "#C9781F", "#2BA88A", "#A974D9", "#7E9E2E"],
        ORD=["#8A3433", "#B23F3D", "#E5534B", "#FF9A8F"],
        SEQ=["#141E33", "#1F3A63", "#2F5F9E", "#4A90E2", "#8CBDF2", "#DDEBFC"],
        DIV=["#4A90E2", "#2A3345", "#E5534B"],
        HEAT=["#141E33", "#24406B", "#3B6FB0", "#8CBDF2"],
        fill="#2F5F9E", scale=1.05),
    "mono": dict(
        label="Monochrome + accent (Palatino, print-safe)",
        use="Greyscale print, theses printed in B/W; one red accent for the key result.",
        font=["Palatino", "Palatino Linotype", "Book Antiqua", "DejaVu Serif"], serif=True,
        letter="({})", mono=True,
        bg="#FFFFFF", ink="#111111", ink2="#4D4D4D", muted="#8F8F8F", rule="#CCCCCC", grid="#EEEEEE",
        G0="#8A8A8A", G1="#C8102E",
        S4=["#1F1F1F", "#6B6B6B", "#A3A3A3", "#C8102E"],
        C6=["#1F1F1F", "#505050", "#808080", "#ADADAD", "#C8102E", "#7A0A1C"],
        ORD=["#ADADAD", "#808080", "#505050", "#1F1F1F"],
        SEQ=["#F7F7F7", "#D4D4D4", "#A3A3A3", "#6B6B6B", "#333333", "#111111"],
        DIV=["#505050", "#EDEDED", "#C8102E"],
        HEAT=["#F7F7F7", "#BDBDBD", "#6B6B6B", "#1F1F1F"],
        fill="#BDBDBD", scale=1.0),
    "studio": dict(
        label="Studio (classic colours, white, Times bold, badge letters)",
        use="Main AITP set: classic blue/crimson palette on white with open frames and refined detail.",
        font=["Times New Roman", "Times", "DejaVu Serif"], serif=True, letter="{}", badge=True,
        bg="#FFFFFF", ink="#141414", ink2="#444444", muted="#8C8C8C", rule="#D4D4D4", grid="#EBEBEB",
        G0="#2F6DB5", G1="#C8414B",
        S4=["#2F6DB5", "#C8414B", "#CF7A22", "#2E9C82"],
        C6=["#C8414B", "#2F6DB5", "#CF7A22", "#8E5BB0", "#2E9C82", "#9C6B1E"],
        ORD=["#E99484", "#D8665D", "#B8364A", "#7E1A33"],
        SEQ=["#FFFFFF", "#FBE3DF", "#F1A99F", "#D8665D", "#B23446", "#6E1428"],
        DIV=["#1D4E89", "#5B8FCB", "#F4F4F4", "#E07B73", "#A8202F"],
        HEAT=["#FFFFFF", "#DCE8F5", "#8DB4DE", "#2F6DB5", "#173C6B"],
        fill="#DCE8F5", scale=1.0),
    "atlas": dict(
        label="Atlas editorial (Times bold, petrol/vermilion/saffron on warm panels)",
        use="Distinctive journal set: tinted plotting panels, white grid, badge panel letters.",
        font=["Times New Roman", "Times", "DejaVu Serif"], serif=True, letter="{}", badge=True,
        bg="#FFFFFF", panel="#F6F3EE", ink="#1C1B22", ink2="#4A4752", muted="#8E8A96", rule="#D9D3CA",
        grid="#FFFFFF",
        G0="#1F6F8B", G1="#E4572E",
        S4=["#1F6F8B", "#E4572E", "#F3A712", "#7D5BA6"],
        C6=["#1F6F8B", "#E4572E", "#F3A712", "#29A19C", "#7D5BA6", "#C2366B"],
        ORD=["#FCE0A8", "#F6B656", "#E4572E", "#8C2D3F"],
        SEQ=["#FFF8EC", "#FCE0A8", "#F6B656", "#E4572E", "#A23B72", "#4B1D52"],
        DIV=["#164E63", "#4FA3B5", "#F6F3EE", "#F29A6E", "#B8361A"],
        HEAT=["#F4F8F7", "#B9DDD8", "#5DB3AC", "#1F6F8B", "#16324F"],
        fill="#B9DDD8", scale=1.0),
    "atlas_white": dict(
        label="Atlas colours on white (Times bold, petrol/vermilion/saffron, Nature-style white panels)",
        use="Atlas palette with plain white plotting area, light grey grid and normal axis lines.",
        font=["Times New Roman", "Times", "DejaVu Serif"], serif=True, letter="{}", badge=True,
        bg="#FFFFFF", ink="#1C1B22", ink2="#4A4752", muted="#8E8A96", rule="#D4D4D4", grid="#ECECEC",
        G0="#1F6F8B", G1="#E4572E",
        S4=["#1F6F8B", "#E4572E", "#F3A712", "#7D5BA6"],
        C6=["#1F6F8B", "#E4572E", "#F3A712", "#29A19C", "#7D5BA6", "#C2366B"],
        ORD=["#FCE0A8", "#F6B656", "#E4572E", "#8C2D3F"],
        SEQ=["#FFF8EC", "#FCE0A8", "#F6B656", "#E4572E", "#A23B72", "#4B1D52"],
        DIV=["#164E63", "#4FA3B5", "#F4F2EE", "#F29A6E", "#B8361A"],
        HEAT=["#F4F8F7", "#B9DDD8", "#5DB3AC", "#1F6F8B", "#16324F"],
        fill="#B9DDD8", scale=1.0),
}
MONO_STYLES = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D"), ((0, (5, 1, 1, 1)), "v"), ("-", "P")]

_ACTIVE = {}


def tint(color, f=0.7, toward=None):
    """Blend a colour toward the background (f = share of background)."""
    a = np.array(to_rgb(color))
    b = np.array(to_rgb(toward or _ACTIVE.get("bg", "#FFFFFF")))
    return to_hex(a * (1 - f) + b * f)


def use_theme(name="classic"):
    """Activate a theme: sets rcParams and the module tokens (F.INK, F.G0, ...)."""
    t = dict(THEMES[name])
    _ACTIVE.clear()
    _ACTIVE.update(t, name=name)
    s = t["scale"]
    g = globals()
    for k in ["bg", "ink", "ink2", "muted", "rule", "grid", "fill"]:
        g[k.upper()] = t[k]
    for k in ["G0", "G1", "S4", "C6", "ORD"]:
        g[k] = t[k]
    g["G0_LIGHT"], g["G1_LIGHT"] = tint(t["G0"], 0.68), tint(t["G1"], 0.68)
    g["SEQ"] = LinearSegmentedColormap.from_list(f"{name}_seq", t["SEQ"])
    g["DIV"] = LinearSegmentedColormap.from_list(f"{name}_div", t["DIV"])
    g["HEAT"] = LinearSegmentedColormap.from_list(f"{name}_heat", t["HEAT"])
    g["THEME"] = name
    fam = "serif" if t["serif"] else "sans-serif"
    f0 = t["font"][0]
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update({
        # A list of real family names enables per-glyph fallback (e.g. arrows, Greek).
        "font.family": t["font"] + (["DejaVu Serif"] if t["serif"] else ["DejaVu Sans"]),
        f"font.{fam}": t["font"], "font.size": 9 * s,
        "mathtext.fontset": "custom", "mathtext.rm": f0, "mathtext.it": f"{f0}:italic",
        "mathtext.bf": f"{f0}:bold", "mathtext.cal": f"{f0}:italic", "mathtext.sf": f0,
        "axes.labelsize": 10 * s, "axes.labelweight": "bold", "axes.labelcolor": t["ink"],
        "axes.titlesize": 10 * s, "axes.titleweight": "bold", "axes.facecolor": t["bg"],
        "axes.edgecolor": t["ink2"], "axes.linewidth": 0.8, "axes.labelpad": 3,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=t["S4"]),
        "xtick.labelsize": 9 * s, "ytick.labelsize": 9 * s, "xtick.color": t["ink2"],
        "ytick.color": t["ink2"], "xtick.labelcolor": t["ink"], "ytick.labelcolor": t["ink"],
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
        "grid.color": t["grid"], "grid.linewidth": 0.6,
        "legend.fontsize": 8.5 * s, "legend.frameon": False, "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3, "legend.labelspacing": 0.35, "legend.labelcolor": t["ink"],
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "patch.edgecolor": t["bg"],
        "text.color": t["ink"], "axes.unicode_minus": True,
        "figure.facecolor": t["bg"], "savefig.facecolor": t["bg"], "savefig.edgecolor": t["bg"],
        "figure.dpi": 150, "savefig.dpi": 600,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.constrained_layout.h_pad": 0.04, "figure.constrained_layout.w_pad": 0.04,
        "figure.constrained_layout.hspace": 0.05, "figure.constrained_layout.wspace": 0.04,
    })
    if "panel" in t:   # tinted plotting panel, white grid, no frame
        g["PANEL"] = t["panel"]
        mpl.rcParams.update({"axes.facecolor": t["panel"], "axes.spines.left": False,
                             "axes.spines.bottom": False, "xtick.major.size": 0, "ytick.major.size": 0,
                             "xtick.major.pad": 3, "ytick.major.pad": 3, "grid.linewidth": 0.9})
    else:
        g["PANEL"] = t["bg"]
    return t


def theme():
    return dict(_ACTIVE)


def series(n):
    """Colours for n entities: <=4 overlapping (S4), <=6 adjacent (C6). Never invent hues."""
    if n <= 4:
        return S4[:n]
    if n <= 6:
        return C6[:n]
    raise ValueError("More than 6 categories: fold the tail into 'Other' or use small multiples.")


def line_style(i):
    """(linestyle, marker) for series i - mandatory secondary encoding in the mono theme."""
    return MONO_STYLES[i % len(MONO_STYLES)] if _ACTIVE.get("mono") else ("-", "o")


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def mm(v):
    return v / 25.4


def new_figure(w_mm=180, h_mm=120):
    """Exact-size figure: 180 mm double column, 88 mm single column (height <= 225 mm)."""
    fig = plt.figure(figsize=(mm(w_mm), mm(h_mm)), layout="constrained")
    fig._panels = []
    return fig


def new_slide(aspect="16:9"):
    """Slide-sized figure (13.33 x 7.5 in for 16:9, 10 x 7.5 in for 4:3)."""
    w, h = (338.7, 190.5) if aspect == "16:9" else (254, 190.5)
    return new_figure(w, h)


def panel(fig, ax, letter, title=""):
    """Register a panel letter + short bold title, drawn at the panel's outer edge on save."""
    ax.set_title(" ", loc="left", fontsize=11.5 * _ACTIVE["scale"], fontweight="bold",
                 pad=7, color=(0, 0, 0, 0))
    fig._panels.append((ax, letter, title))


def ygrid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, lw=0.6, zorder=0)


def clean_left(ax):
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def pct_axis(ax, axis="y", decimals=0):
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(
        ticker.PercentFormatter(1.0, decimals=decimals))


def plain_log(ax, ticks, axis="x"):
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    a.set_major_locator(ticker.FixedLocator(ticks))
    a.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_locator(ticker.NullLocator())


def rotate(ax, deg=30, axis="x"):
    for t in (ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()):
        t.set_rotation(deg)
        t.set_ha("right")
        t.set_rotation_mode("anchor")


def minus(text):
    return text.replace("-", "−")


def spread(ys, gap):
    """Nudge line-end label positions apart so they never overlap."""
    ys = np.asarray(ys, float)
    o = np.argsort(ys)
    out = ys[o].copy()
    for _ in range(60):
        for i in range(1, len(out)):
            if out[i] - out[i - 1] < gap:
                mid = (out[i] + out[i - 1]) / 2
                out[i - 1], out[i] = mid - gap / 2, mid + gap / 2
    res = np.empty_like(out)
    res[o] = out
    return res


def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def text_on(color):
    """INK or white, whichever reads better on a filled cell of this colour."""
    r, g, b = to_rgb(color)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#FFFFFF" if lum < 0.52 else "#141414"


# ---------------------------------------------------------------------------
# Export + fit check
# ---------------------------------------------------------------------------
def fit_check(fig, renderer=None):
    """Texts that fall outside the canvas (clipped). Empty list = OK."""
    r = renderer or fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    bad = []
    for ax in fig.axes:
        if not ax.axison:
            continue
        x0, x1 = sorted(ax.get_xlim())
        texts = list(ax.texts) + [ax.xaxis.label, ax.yaxis.label]
        texts += [t for t, v in zip(ax.get_xticklabels(), ax.get_xticks()) if x0 <= v <= x1]
        for t in texts:
            if not t.get_visible() or not t.get_text().strip():
                continue
            e = t.get_window_extent(r)
            if e.x1 > W + 1 or e.x0 < -1 or e.y0 < -1 or e.y1 > H + 1:
                bad.append(t.get_text()[:40])
    return bad


def _letter(letter):
    t = _ACTIVE
    s = letter.upper() if t.get("upper") else letter
    return t["letter"].format(s)


def save(fig, path, formats=("png", "tiff", "pdf", "svg"), by_format=True):
    """Freeze layout, place panel letters/titles, check fit, export.

    path = "<out_dir>/Fig01_name". With by_format=True (default) files go to
    <out_dir>/png/, <out_dir>/tiff/, <out_dir>/pdf/ and <out_dir>/svg/."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fig.set_layout_engine("none")
    W, H = fig.bbox.width, fig.bbox.height
    s = _ACTIVE["scale"]
    for ax, letter, title in getattr(fig, "_panels", []):
        yb = ax.yaxis.get_tightbbox(r)
        use_y = ax.axison and yb is not None and yb.width > 0
        x0 = min(ax.bbox.x0, yb.x0) if use_y else ax.bbox.x0
        ty = ax._left_title.get_window_extent(r).y0
        if _ACTIVE.get("badge"):   # white letter on a filled rounded badge
            lt = fig.text(x0 / W + 0.004, ty / H, _letter(letter), fontsize=10 * s, fontweight="bold",
                          va="bottom", ha="left", color="#FFFFFF",
                          bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.25", fc=INK, ec="none"))
        else:
            lt = fig.text(x0 / W, ty / H, _letter(letter), fontsize=11.5 * s, fontweight="bold",
                          va="bottom", ha="left", color=INK)
        if title:
            x1 = lt.get_window_extent(r).x1 + (7 if _ACTIVE.get("badge") else 4) * fig.dpi / 72
            fig.text(x1 / W, ty / H, title, fontsize=10.5 * s, fontweight="bold",
                     va="bottom", ha="left", color=INK)
    bad = fit_check(fig, r)
    if bad:
        print(f"  [fit] {Path(path).name}: clipped text -> {bad} "
              "(shorten these labels or widen that column)")
    stem = Path(path)
    for fmt in formats:
        out = (stem.parent / fmt / f"{stem.name}.{fmt}") if by_format else stem.with_suffix("." + fmt)
        out.parent.mkdir(parents=True, exist_ok=True)
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if fmt == "tiff" else {}
        fig.savefig(out, **kw)
    plt.close(fig)
    print("  wrote", stem.name, "(fit OK)" if not bad else "")
    return bad


use_theme("classic")
