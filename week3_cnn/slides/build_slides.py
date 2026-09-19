"""Build the Week 3 parts 7-8 slide deck from the measured results.

Every number on a slide comes from week3_cnn/results/*.json (the six ladder
runs) or from the trackio database those runs wrote, so re-running the ladder
and re-running this script keeps the deck honest.

    pip install python-pptx
    python slides/build_slides.py            # -> slides/week3_parts7_8.pptx

Figures are rendered to a temporary directory and embedded in the .pptx; no
PNGs are left in the repository.
"""

import json
from collections import namedtuple
import os
import sqlite3
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
WEEK3 = os.path.dirname(HERE)
RESULTS = os.path.join(WEEK3, "results")
DATA_ROOT = os.path.join(WEEK3, "data")
TRACKIO_DB = os.path.expanduser("~/.cache/huggingface/trackio/week3-cnn.db")
OUT = os.path.join(HERE, "week3_parts7_8.pptx")

RUNGS = [
    ("part7_0_config_d", "0", "part6 config D (18k)"),
    ("part7_1_full_data", "1", "+ full data (45k)"),
    ("part7_2a_batchnorm", "2a", "+ BatchNorm"),
    ("part7_2b_onecycle", "2b", "+ OneCycle LR"),
    ("part8_3b_deeper", "3b", "2 convs per stage"),
    ("part8_3c_wider", "3c", "widths x2"),
]
PARAMS = {  # from build_model(), printed by each run
    "0": 188_810,
    "1": 188_810,
    "2a": 188_970,
    "2b": 188_970,
    "3b": 272_234,
    "3c": 1_672_010,
}

# ------------------------------------------------------------------ palette

INK = RGBColor(0x1C, 0x22, 0x2B)  # body text
MUTED = RGBColor(0x6B, 0x75, 0x82)  # captions, kickers
ACCENT = RGBColor(0xC2, 0x55, 0x1E)  # the one warm colour: emphasis
BLUE = RGBColor(0x1F, 0x4E, 0x79)  # headings
PAPER = RGBColor(0xFA, 0xF9, 0xF7)
RULE = RGBColor(0xD8, 0xD3, 0xCB)

C_TRAIN = "#c2551e"
C_VAL = "#1f4e79"
C_TEST = "#6b7582"
C_GRID = "#d8d3cb"
MUTED_HEX = "#6b7582"

SANS = "Helvetica Neue"
MONO = "Menlo"

W, H = Inches(13.333), Inches(7.5)
M = Inches(0.85)  # side margin


# ------------------------------------------------------------------- data


def load_all():
    runs = {}
    for fname, key, _ in RUNGS:
        with open(os.path.join(RESULTS, fname + ".json")) as f:
            runs[key] = json.load(f)
    return runs


def best(run):
    """(train, val, test, gap) at the best-validation epoch."""
    i = run["best_epoch"] - 1
    tr, va = run["train"][i], run["val"][i]
    return tr, va, run.get("test"), tr - va


def trackio_series(run_name, key):
    """Pull one logged metric out of the trackio database, in step order."""
    if not os.path.exists(TRACKIO_DB):
        return [], []
    con = sqlite3.connect(f"file:{TRACKIO_DB}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT json_extract(metrics,'$.epoch'), json_extract(metrics,?) "
        "FROM metrics WHERE run_name = ? AND json_extract(metrics,?) IS NOT NULL "
        "ORDER BY id",
        (f"$.{key}", run_name, f"$.{key}"),
    ).fetchall()
    con.close()
    return [r[0] for r in rows], [r[1] for r in rows]


# ---------------------------------------------------------------- figures


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=11, color="#6b7582")
    ax.set_ylabel(ylabel, fontsize=11, color="#6b7582")
    ax.grid(True, color=C_GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors="#6b7582", labelsize=10)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="#faf9f7")
    plt.close(fig)
    return path


def fig_staircase(runs, path, highlight=None):
    fig, ax = plt.subplots(figsize=(10.4, 4.6), facecolor="#faf9f7")
    ax.set_facecolor("#faf9f7")
    labels, vals, tests = [], [], []
    for _, key, label in RUNGS:
        labels.append(f"{key}\n{label}")
        tr, va, te, _ = best(runs[key])
        vals.append(va)
        tests.append(te)
    x = range(len(labels))
    bars = ax.bar([i - 0.19 for i in x], vals, 0.38, label="validation", color=C_VAL)
    ax.bar([i + 0.19 for i in x], tests, 0.38, label="test", color=C_TEST)
    if highlight is not None:
        for i, b in enumerate(bars):
            if i != highlight:
                b.set_alpha(0.28)
    for i, v in enumerate(vals):
        ax.text(i - 0.19, v + 0.008, f"{v:.3f}", ha="center", fontsize=10, color=C_VAL)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0.65, 0.97)
    style(ax, "", "accuracy")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    return save(fig, path)


def fig_curves(runs, path, keys, title=None, show_train=True):
    """Train (dashed) and validation (solid) accuracy for one or two rungs."""
    fig, ax = plt.subplots(figsize=(9.2, 4.5), facecolor="#faf9f7")
    ax.set_facecolor("#faf9f7")
    shades = [(C_VAL, C_TRAIN), ("#8aa4bd", "#e0a184")]
    for n, key in enumerate(keys):
        r = runs[key]
        label = dict((k, l) for _, k, l in RUNGS)[key]
        epochs = range(1, len(r["val"]) + 1)
        cv, ct = shades[len(keys) - 1 - n] if len(keys) > 1 else shades[0]
        newest = n == len(keys) - 1
        if show_train:
            ax.plot(epochs, r["train"], "--", color=cv, lw=1.5,
                    alpha=0.55 if newest else 0.35,
                    label=f"{key} train" if newest else None)
        ax.plot(epochs, r["val"], "-", color=cv, lw=3.0 if newest else 2.0,
                label=f"rung {key}  ·  {label}  ·  val")
    # Final validation numbers, printed where the lines end.
    for n, key in enumerate(keys):
        cv, _ = shades[len(keys) - 1 - n] if len(keys) > 1 else shades[0]
        _, va, _, _ = best(runs[key])
        ax.annotate(f"{va:.3f}", xy=(30, runs[key]["val"][-1]), xytext=(6, -4),
                    textcoords="offset points", fontsize=12, color=cv, weight="bold")
    ax.set_xlim(0, 33)
    style(ax, "epoch", "accuracy")
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels_[::-1], frameon=False, fontsize=10.5,
              loc="lower right")
    if title:
        ax.set_title(title, fontsize=12, color="#1c222b", loc="left")
    return save(fig, path)


def fig_gap_fill(runs, path, key):
    r = runs[key]
    epochs = list(range(1, len(r["val"]) + 1))
    fig, ax = plt.subplots(figsize=(9.2, 4.5), facecolor="#faf9f7")
    ax.set_facecolor("#faf9f7")
    ax.fill_between(epochs, r["val"], r["train"], color=C_TRAIN, alpha=0.16)
    ax.plot(epochs, r["train"], "--", color=C_TRAIN, lw=2.2, label="train")
    ax.plot(epochs, r["val"], "-", color=C_VAL, lw=2.6, label="validation")
    tr, va, _, gap = best(runs[key])
    ax.annotate(
        f"gap {gap:.3f}",
        xy=(epochs[-1], (tr + va) / 2),
        xytext=(-92, 0),
        textcoords="offset points",
        fontsize=12,
        color=C_TRAIN,
        va="center",
    )
    style(ax, "epoch", "accuracy")
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    return save(fig, path)


def fig_lr(runs, path):
    fig, ax = plt.subplots(figsize=(9.2, 4.2), facecolor="#faf9f7")
    ax.set_facecolor("#faf9f7")
    epochs = range(1, 31)
    ax.plot(epochs, runs["2b"]["lr"], "-", color=C_VAL, lw=2.6, label="2b: OneCycle")
    ax.plot(epochs, [1e-3] * 30, "--", color=C_TEST, lw=2.0, label="2a: constant 1e-3")
    ax.axhline(3e-3, color=C_TRAIN, lw=1.0, ls=":")
    ax.text(30, 3.05e-3, "peak 3e-3", ha="right", fontsize=10, color=C_TRAIN)
    style(ax, "epoch", "learning rate")
    ax.legend(frameon=False, fontsize=11, loc="lower left")
    return save(fig, path)


def fig_params(runs, path):
    fig, ax = plt.subplots(figsize=(9.6, 4.5), facecolor="#faf9f7")
    ax.set_facecolor("#faf9f7")
    xs = [PARAMS[k] for _, k, _ in RUNGS]
    ys = [best(runs[k])[1] for _, k, _ in RUNGS]
    ax.plot(xs, ys, "-", color=C_GRID, lw=2.0, zorder=1)
    for (_, key, _), p, va in zip(RUNGS, xs, ys):
        color = C_TRAIN if key in ("3b", "3c") else C_VAL
        ax.scatter(p, va, s=130, color=color, zorder=3)
        dx = -14 if key == "3c" else 14
        ha = "right" if key == "3c" else "left"
        ax.annotate(f"{key}", (p, va), xytext=(dx, -5), textcoords="offset points",
                    fontsize=13, color=color, ha=ha, weight="bold")
    ax.annotate("free\n(same 189k parameters)", xy=(1.9e5, 0.762), xytext=(24, -6),
                textcoords="offset points", fontsize=11, color=C_VAL)
    ax.annotate("paid for\nwith parameters", xy=(6e5, 0.893), fontsize=11,
                color=C_TRAIN, ha="center")
    ax.set_xscale("log")
    ax.set_ylim(0.715, 0.935)
    ax.set_xlim(1.4e5, 2.6e6)
    style(ax, "parameters (log scale)", "validation accuracy")
    return save(fig, path)


def flat_epoch(series, tol=0.15):
    """First epoch from which the series stays within `tol` of its eventual best.

    For validation loss this is the point where further training stops buying
    anything measurable on held-out data.
    """
    lo = min(series)
    for i, v in enumerate(series):
        if v <= lo * (1 + tol):
            return i + 1
    return len(series)


def fig_valloss(runs, path, key="3c"):
    """Two panels: the loss gap calls it nine epochs before the accuracy does."""
    r = runs[key]
    epochs = list(range(1, len(r["val"]) + 1))
    flat = flat_epoch(r["val_loss"])
    peak = r["best_epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), facecolor="#faf9f7")

    ax = axes[0]
    ax.set_facecolor("#faf9f7")
    ax.fill_between(epochs, r["train_loss"], r["val_loss"], color=C_TRAIN, alpha=0.14)
    ax.plot(epochs, r["val_loss"], "-", color=C_TRAIN, lw=2.6, label="validation")
    ax.plot(epochs, r["train_loss"], "--", color=C_TRAIN, lw=1.8, alpha=0.7,
            label="train")
    ax.axvline(flat, color="#6b7582", lw=1.2, ls=":")
    ax.annotate(
        f"epoch {flat}:\nvalidation loss\nstops improving",
        xy=(flat, 0.75), xytext=(flat + 1.2, 0.80), fontsize=10.5, color="#3d4752",
    )
    style(ax, "epoch", "cross-entropy")
    ax.set_title("loss", fontsize=13, color="#1c222b", loc="left")
    ax.legend(frameon=False, fontsize=10.5, loc="upper right")

    ax = axes[1]
    ax.set_facecolor("#faf9f7")
    ax.fill_between(epochs, r["val"], r["train"], color=C_VAL, alpha=0.14)
    ax.plot(epochs, r["val"], "-", color=C_VAL, lw=2.6, label="validation")
    ax.plot(epochs, r["train"], "--", color=C_VAL, lw=1.8, alpha=0.7, label="train")
    ax.axvline(flat, color="#6b7582", lw=1.2, ls=":")
    ax.axvline(peak, color=C_VAL, lw=1.2, ls=":")
    ax.annotate(
        f"epoch {peak}:\nvalidation accuracy\nfinally peaks",
        xy=(peak, 0.66), xytext=(peak - 0.8, 0.62), fontsize=10.5, color=C_VAL,
        ha="right",
    )
    ax.set_ylim(0.55, 1.01)
    style(ax, "epoch", "accuracy")
    ax.set_title("accuracy — same run, same epochs", fontsize=13, color="#1c222b",
                 loc="left")
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    return save(fig, path)


def smooth(y, w=21):
    """Rolling mean -- the raw per-20-batch signal is far too noisy to read."""
    out = []
    for i in range(len(y)):
        lo, hi = max(0, i - w // 2), min(len(y), i + w // 2 + 1)
        out.append(sum(y[lo:hi]) / (hi - lo))
    return out


def fig_gradnorm(path):
    fig, ax = plt.subplots(figsize=(9.6, 4.3), facecolor="#faf9f7")
    ax.set_facecolor("#faf9f7")
    for run_name, label, color in [
        ("ladder/1_full_data", "1: no BatchNorm, constant lr", C_TEST),
        ("ladder/2a_batchnorm", "2a: BatchNorm, constant lr", C_VAL),
        ("ladder/2b_onecycle", "2b: BatchNorm + OneCycle", C_TRAIN),
    ]:
        x, y = trackio_series(run_name, "grad_norm")
        if x:
            ax.plot(x, y, lw=0.7, color=color, alpha=0.18)
            ax.plot(x, smooth(y), lw=2.6, color=color, label=label)
    ax.axhspan(1.0, 8.0, color="#6b7582", alpha=0.06, zorder=0)
    ax.text(0.4, 8.6, "a healthy band — this is what NOT broken looks like",
            fontsize=10.5, color="#6b7582")
    ax.set_ylim(0, 11)
    style(ax, "epoch", "gradient norm")
    ax.legend(frameon=False, fontsize=10.5, loc="lower right")
    return save(fig, path)


def fig_actstd(path):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), facecolor="#faf9f7",
                             sharey=True)
    for ax, (run_name, title) in zip(
        axes,
        [("ladder/1_full_data", "rung 1: no BatchNorm"),
         ("ladder/2a_batchnorm", "rung 2a: BatchNorm")],
    ):
        ax.set_facecolor("#faf9f7")
        for i, color in zip((1, 2, 3), (C_VAL, C_TRAIN, C_TEST)):
            x, y = trackio_series(run_name, f"act_std/conv{i}")
            if x:
                ax.plot(x, y, lw=2.2, color=color, label=f"conv{i}")
        style(ax, "epoch", "pre-activation std")
        ax.set_title(title, fontsize=12, color="#1c222b", loc="left")
    axes[0].legend(frameon=False, fontsize=10, loc="upper left")
    return save(fig, path)


def fig_weightnorm(runs, path):
    fig, ax = plt.subplots(figsize=(9.2, 4.3), facecolor="#faf9f7")
    ax.set_facecolor("#faf9f7")
    for key, color in [("1", C_TEST), ("2a", C_VAL), ("2b", C_TRAIN)]:
        r = runs[key]
        ax.plot(range(1, len(r["weight_norm"]) + 1), r["weight_norm"], lw=2.2,
                color=color, label=f"rung {key}")
    style(ax, "epoch", "||w||")
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    return save(fig, path)


# ------------------------------------------------------------------- maths
#
# python-pptx cannot write PowerPoint equations, so every formula is typeset
# with matplotlib's mathtext (a LaTeX subset) and embedded as a transparent
# image. Rendering at MATH_DPI and placing at native size keeps one consistent
# type size across every equation in the deck.

MATH_DPI = 300


def math_png(expr, path, fontsize=30, color="#1c222b"):
    """Typeset one LaTeX expression to a tightly-cropped transparent PNG."""
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, expr, fontsize=fontsize, color=color)
    fig.savefig(path, dpi=MATH_DPI, transparent=True, bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    return path


_math_seq = [0]


def math_block(slide, expr, x, y, tmp, fontsize=30, color="#1c222b", center=False):
    """Render `expr` and place it at native size; returns (width, height) in EMU."""
    _math_seq[0] += 1
    path = os.path.join(tmp, f"eq{_math_seq[0]}.png")
    math_png(expr, path, fontsize=fontsize, color=color)
    px_h, px_w = plt.imread(path).shape[:2]
    w = Emu(int(px_w / MATH_DPI * 914400))
    h = Emu(int(px_h / MATH_DPI * 914400))
    left = int((W - w) / 2) if center else x
    slide.shapes.add_picture(path, left, y, width=w, height=h)
    return w, h


# ------------------------------------------------------------ slide kit


def blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = PAPER
    return s


def text(slide, x, y, w, h, runs_spec, align=PP_ALIGN.LEFT, spacing=1.0):
    """runs_spec: list of (string, size, bold, colour, font) paragraphs."""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, (s, size, bold, color, font) in enumerate(runs_spec):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(6)
        r = p.add_run()
        r.text = s
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = font
    return box


def rule(slide, y, x=M, w=None):
    from pptx.enum.shapes import MSO_SHAPE

    w = w or (W - 2 * M)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, Pt(1.2))
    line.fill.solid()
    line.fill.fore_color.rgb = RULE
    line.line.fill.background()
    line.shadow.inherit = False
    return line


def header(slide, kicker, title):
    if kicker:
        text(slide, M, Inches(0.5), W - 2 * M, Inches(0.35),
             [(kicker.upper(), 12, True, ACCENT, SANS)])
    text(slide, M, Inches(0.85), W - 2 * M, Inches(0.8),
         [(title, 30, True, BLUE, SANS)])
    rule(slide, Inches(1.62))


def bullets(slide, items, top=Inches(2.0), size=17, width=None, left=M):
    """items: list of str, or (str, 'mono'|'note'|'lead') tuples."""
    specs = []
    for it in items:
        kind = "body"
        if isinstance(it, tuple):
            it, kind = it
        if kind == "mono":
            specs.append((it, 15, False, INK, MONO))
        elif kind == "note":
            specs.append((it, 15, False, MUTED, SANS))
        elif kind == "lead":
            specs.append((it, size + 2, True, INK, SANS))
        else:
            specs.append((it, size, False, INK, SANS))
    return text(slide, left, top, width or (W - 2 * M), H - top - Inches(0.5),
                specs, spacing=1.15)


def picture(slide, path, top=Inches(1.95), height=Inches(4.3)):
    pic = slide.shapes.add_picture(path, Emu(0), top, height=height)
    pic.left = int((W - pic.width) / 2)
    return pic


def caption(slide, s, y=Inches(6.45)):
    text(slide, M, y, W - 2 * M, Inches(0.6), [(s, 14, False, MUTED, SANS)],
         align=PP_ALIGN.CENTER)


def content_slide(prs, kicker, title, items, top=Inches(2.0)):
    s = blank(prs)
    header(s, kicker, title)
    bullets(s, items, top=top)
    return s


def flow_slide(prs, kicker, title, blocks, tmp, top=Inches(1.95)):
    """A slide that interleaves prose and typeset equations down the page.

    blocks: ("text"|"lead"|"note", str) | ("eq", expr) | ("eq", expr, fontsize)
            | ("gap", inches)
    """
    s = blank(prs)
    header(s, kicker, title)
    y = top
    for blk in blocks:
        kind = blk[0]
        if kind == "gap":
            y = Emu(int(y) + Inches(blk[1]))
            continue
        if kind == "eq":
            size = blk[2] if len(blk) > 2 else 28
            _, h = math_block(s, blk[1], M + Inches(0.4), y, tmp, fontsize=size)
            y = Emu(int(y) + int(h) + Inches(0.26))
            continue
        size, bold, color = {
            "lead": (18, True, INK),
            "text": (17, False, INK),
            "note": (15, False, MUTED),
        }[kind]
        box = text(s, M, y, W - 2 * M, Inches(0.5),
                   [(blk[1], size, bold, color, SANS)], spacing=1.15)
        # Estimate consumed height: the textbox autosizes, so advance by lines.
        chars_per_line = int((W - 2 * M) / Emu(1) / 914400 * 72 / (size * 0.50))
        lines = max(1, -(-len(blk[1]) // max(1, chars_per_line)))
        y = Emu(int(y) + Inches(lines * size * 1.15 / 72 + 0.16))
    return s


def defs_slide(prs, kicker, title, rows, tmp, note=None):
    """Symbol on the left, plain-English meaning on the right."""
    s = blank(prs)
    header(s, kicker, title)
    y = Inches(1.88)
    step = Inches(0.54)
    for symbol, meaning in rows:
        _, h = math_block(s, symbol, M, Emu(int(y) + Inches(0.02)), tmp, fontsize=21)
        text(s, M + Inches(1.5), y, W - 2 * M - Inches(1.5), Inches(0.5),
             [(meaning, 16, False, INK, SANS)], spacing=1.05)
        y = Emu(int(y) + step)
    if note:
        text(s, M, max(Emu(int(y) + Inches(0.12)), H - Inches(0.82)),
             W - 2 * M, Inches(0.6), [(note, 14.5, False, MUTED, SANS)],
             spacing=1.15)
    return s


def code_slide(prs, kicker, title, source, lines, note=None, size=14.5):
    """Real code, lifted from the repository. `lines` is a list of source lines;
    wrapping one in a 1-tuple marks it as the change this rung introduces."""
    s = blank(prs)
    header(s, kicker, title)
    text(s, M, Inches(1.90), W - 2 * M, Inches(0.32),
         [(source, 13.5, True, MUTED, MONO)])
    specs = []
    for ln in lines:
        if isinstance(ln, tuple):
            specs.append(("  " + ln[0], size, True, ACCENT, MONO))
        else:
            specs.append(("  " + ln, size, False, INK, MONO))
    text(s, M, Inches(2.32), W - 2 * M, H - Inches(3.0), specs, spacing=1.08)
    if note:
        text(s, M, H - Inches(1.02), W - 2 * M, Inches(0.7),
             [(note, 14.5, False, MUTED, SANS)], spacing=1.15)
    return s


def fig_receptive(path):
    """The receptive field of one last-stage unit, drawn on a real CIFAR image."""
    from torchvision import datasets

    ds = datasets.CIFAR10(root=DATA_ROOT, train=False, download=False)
    img, _ = ds[12]  # a horse: the object fills the frame

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), facecolor="#faf9f7")
    for ax, (rf, chain, title) in zip(
        axes,
        [(22, [3, 4, 8, 10, 18, 22], "1 conv per stage  (part 6, rung 2b)"),
         (36, [3, 5, 6, 10, 14, 16, 24, 32, 36], "2 convs per stage  (rung 3b)")],
    ):
        ax.imshow(img, extent=(0, 32, 32, 0))
        ax.set_xlim(-6, 38)
        ax.set_ylim(38, -6)
        # The image boundary.
        ax.add_patch(plt.Rectangle((0, 0), 32, 32, fill=False, ec="#1c222b", lw=2.0))
        # Every intermediate receptive field, growing outward from the centre.
        for r in chain[:-1]:
            ax.add_patch(plt.Rectangle((16 - r / 2, 16 - r / 2), r, r, fill=False,
                                       ec=C_VAL, lw=0.9, alpha=0.45))
        ax.add_patch(plt.Rectangle((16 - rf / 2, 16 - rf / 2), rf, rf, fill=False,
                                   ec=C_TRAIN, lw=3.0))
        ax.text(16, 16 - rf / 2 - 1.6, f"{rf} px", ha="center", fontsize=14,
                color=C_TRAIN, weight="bold")
        ax.set_title(title, fontsize=12.5, color="#1c222b", loc="left")
        ax.axis("off")
    axes[0].text(16, 36.5, "image is 32 px", ha="center", fontsize=11,
                 color="#6b7582")
    axes[1].text(16, 36.5, "reaches past the frame", ha="center", fontsize=11,
                 color=C_TRAIN)
    return save(fig, path)


Row = namedtuple("Row", "name shape rf kind params")


def walk_arch(convs, widths=(32, 64, 64)):
    """Walk a real model, recording each operator's output shape and the
    receptive field a single unit of that output responds to.

    Built by running the actual network, so these numbers cannot drift away
    from part8_bigger_cnn.build_model().
    """
    import sys

    import torch
    import torch.nn as nn

    if WEEK3 not in sys.path:  # so this runs from the repo root too
        sys.path.insert(0, WEEK3)
    from part8_bigger_cnn import build_model

    model = build_model(widths=widths, convs=convs)
    x = torch.zeros(1, 3, 32, 32)
    r, j = 1, 1
    pending = 0  # BatchNorm parameters fold into the Conv row above them
    rows = [Row("input image", "3 x 32 x 32", 1, "conv", 0)]
    for mod in model:
        x = mod(x)
        shape = " x ".join(str(d) for d in x.shape[1:])
        n = sum(p.numel() for p in mod.parameters())
        if isinstance(mod, nn.BatchNorm2d):
            rows[-1] = rows[-1]._replace(params=rows[-1].params + n)
        elif isinstance(mod, nn.Conv2d):
            r += (mod.kernel_size[0] - 1) * j
            rows.append(Row(f"Conv 3x3 -> {mod.out_channels}  +BN +ReLU",
                            shape, r, "conv", n))
        elif isinstance(mod, nn.MaxPool2d):
            r += (mod.kernel_size - 1) * j
            j *= mod.stride
            rows.append(Row("MaxPool 2x2", shape, r, "conv", 0))
        elif isinstance(mod, nn.Flatten):
            # Past this point there is no receptive field to speak of: the head
            # is dense, so every unit reads all spatial positions at once.
            rows.append(Row("Flatten", shape, None, "head", 0))
        elif isinstance(mod, nn.Linear):
            rows.append(Row(f"Linear -> {mod.out_features}", shape, None,
                            "head", n))
    return rows


def fig_shapes(path):
    """Shape and receptive field, operator by operator, 1 conv vs 2 convs."""
    left, right = walk_arch(1), walk_arch(2)
    n = max(len(left), len(right))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8), facecolor="#faf9f7")
    for ax, rows, title in zip(
        axes,
        [left, right],
        ["1 conv per stage   (rung 2b)", "2 convs per stage   (rung 3b)"],
    ):
        ax.set_facecolor("#faf9f7")
        ax.set_xlim(0, 10.6)
        ax.set_ylim(n + 2.4, -1.7)
        ax.axis("off")
        ax.text(0, -1.30, title, fontsize=13.5, color="#1c222b", weight="bold")
        for x, head in [(0, "operator"), (5.3, "output shape"),
                        (8.3, "one unit sees")]:
            ax.text(x, -0.50, head, fontsize=10.5, color="#6b7582")
        ax.plot([0, 10.6], [-0.18, -0.18], color=C_GRID, lw=1.2)
        for i, row in enumerate(rows):
            name, shape, rf, kind = row.name, row.shape, row.rf, row.kind
            full = kind == "conv" and rf >= 32
            if full:
                ax.add_patch(plt.Rectangle((-0.25, i + 0.05), 10.8, 0.9,
                                           color=C_TRAIN, alpha=0.12, zorder=0))
            ink = MUTED_HEX if kind == "head" else "#1c222b"
            ax.text(0, i + 0.62, name, fontsize=11.5, family="monospace",
                    color=ink)
            ax.text(5.3, i + 0.62, shape, fontsize=11.5, family="monospace",
                    color=MUTED_HEX)
            if kind == "head":
                # A dense layer reads every position, so it always spans the
                # image. Saying "22 px" here would be simply wrong.
                ax.text(8.3, i + 0.62, "all 16 pos.", fontsize=11,
                        family="monospace", color=MUTED_HEX, style="italic")
            else:
                ax.text(8.3, i + 0.62, f"{rf} px", fontsize=12.5,
                        family="monospace",
                        color=C_TRAIN if full else C_VAL,
                        weight="bold" if full else "normal")
        # Separate the convolutional stack from the dense head.
        h0 = next(i for i, r in enumerate(rows) if r.kind == "head")
        ax.plot([-0.25, 10.55], [h0 + 0.02, h0 + 0.02], color="#6b7582", lw=1.0,
                ls=(0, (4, 3)))
        # The verdict for this architecture.
        conv_max = max(r.rf for r in rows if r.kind == "conv")
        if conv_max >= 32:
            msg = ("A single CONV unit spans the image,\n"
                   "two layers before the head.")
            col = C_TRAIN
        else:
            msg = (f"No conv unit ever exceeds {conv_max} px.\n"
                   "Only the dense head spans the image.")
            col = C_VAL
        ax.text(0, n + 0.9, msg, fontsize=12.5, color=col, weight="bold",
                va="top", linespacing=1.5)
    return save(fig, path)


def fig_widths(path):
    """Same depth, same spatial sizes -- only the channel count moves."""
    left, right = walk_arch(2, (32, 64, 64)), walk_arch(2, (64, 128, 256))
    n = max(len(left), len(right))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8), facecolor="#faf9f7")
    for ax, rows, title, accent in zip(
        axes, [left, right],
        ["rung 3b   widths 32-64-64", "rung 3c   widths 64-128-256"],
        [C_VAL, C_TRAIN],
    ):
        ax.set_facecolor("#faf9f7")
        ax.set_xlim(0, 10.6)
        ax.set_ylim(n + 2.4, -1.7)
        ax.axis("off")
        ax.text(0, -1.30, title, fontsize=13.5, color=accent, weight="bold")
        for x, head in [(0, "operator"), (5.0, "output shape"),
                        (8.5, "parameters")]:
            ax.text(x, -0.50, head, fontsize=10.5, color=MUTED_HEX)
        ax.plot([0, 10.6], [-0.18, -0.18], color=C_GRID, lw=1.2)
        for i, row in enumerate(rows):
            ax.text(0, i + 0.62, row.name, fontsize=11.5, family="monospace",
                    color=MUTED_HEX if row.kind == "head" else "#1c222b")
            # The channel count is the first number of the shape: the features.
            ax.text(5.0, i + 0.62, row.shape, fontsize=11.5, family="monospace",
                    color=accent if row.kind == "conv" and "x" in row.shape
                    else MUTED_HEX)
            if row.params:
                ax.text(10.4, i + 0.62, f"{row.params:,}", fontsize=11.5,
                        family="monospace", color="#1c222b", ha="right")
        h0 = next(i for i, r in enumerate(rows) if r.kind == "head")
        ax.plot([-0.25, 10.55], [h0 + 0.02, h0 + 0.02], color=MUTED_HEX, lw=1.0,
                ls=(0, (4, 3)))
        total = sum(r.params for r in rows)
        ax.text(0, n + 0.9, f"total  {total:,} parameters", fontsize=13,
                color=accent, weight="bold", va="top")
    return save(fig, path)


def figure_slide(prs, kicker, title, img, cap=None, height=Inches(4.3)):
    s = blank(prs)
    header(s, kicker, title)
    picture(s, img, height=height)
    if cap:
        caption(s, cap)
    return s


def section_slide(prs, kicker, title, sub=None):
    s = blank(prs)
    text(s, M, Inches(2.6), W - 2 * M, Inches(0.4),
         [(kicker.upper(), 14, True, ACCENT, SANS)])
    text(s, M, Inches(3.0), W - 2 * M, Inches(1.2), [(title, 40, True, BLUE, SANS)])
    rule(s, Inches(4.3), w=Inches(3.0))
    if sub:
        text(s, M, Inches(4.6), Inches(9.5), Inches(1.2),
             [(sub, 18, False, MUTED, SANS)], spacing=1.2)
    return s


def table_slide(prs, kicker, title, headers, rows, note=None, widths=None):
    s = blank(prs)
    header(s, kicker, title)
    ncol = len(headers)
    widths = widths or [1.0] * ncol
    total = W - 2 * M
    scale = total / sum(widths)
    xs, x = [], M
    for w in widths:
        xs.append((x, int(w * scale)))
        x += int(w * scale)
    y = Inches(2.05)
    for (x0, w), h in zip(xs, headers):
        text(s, x0, y, w, Inches(0.4), [(h, 13, True, MUTED, SANS)])
    rule(s, Inches(2.5))
    for i, row in enumerate(rows):
        ry = Inches(2.62) + Inches(0.46) * i
        for j, ((x0, w), cell) in enumerate(zip(xs, row)):
            bold = isinstance(cell, tuple)
            val = cell[0] if bold else cell
            color = ACCENT if bold else INK
            font = MONO if j > 0 else SANS
            text(s, x0, ry, w, Inches(0.4), [(val, 15, bold, color, font)])
    if note:
        text(s, M, H - Inches(1.15), W - 2 * M, Inches(0.7),
             [(note, 14, False, MUTED, SANS)], spacing=1.15)
    return s


# ------------------------------------------------------------------ deck


def build():
    runs = load_all()
    tmp = tempfile.mkdtemp(prefix="week3slides_")
    f = lambda n: os.path.join(tmp, n + ".png")

    F = {
        "stair": fig_staircase(runs, f("stair")),
        "rung0": fig_gap_fill(runs, f("rung0"), "0"),
        "data": fig_curves(runs, f("data"), ["0", "1"]),
        "bn": fig_curves(runs, f("bn"), ["1", "2a"]),
        "lr": fig_lr(runs, f("lr")),
        "cycle": fig_curves(runs, f("cycle"), ["2a", "2b"]),
        "deep": fig_curves(runs, f("deep"), ["2b", "3b"]),
        "wide": fig_curves(runs, f("wide"), ["3b", "3c"]),
        "params": fig_params(runs, f("params")),
        "rf": fig_receptive(f("rf")),
        "shapes": fig_shapes(f("shapes")),
        "widths": fig_widths(f("widths")),
        "gap": fig_gap_fill(runs, f("gap"), "3c"),
        "valloss": fig_valloss(runs, f("valloss")),
        "grad": fig_gradnorm(f("grad")),
        "act": fig_actstd(f("act")),
        "wnorm": fig_weightnorm(runs, f("wnorm")),
    }

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    b = lambda k: best(runs[k])
    pct = lambda v: f"{v:.3f}"
    delta = lambda a, z: f"+{(b(z)[1] - b(a)[1]) * 100:.1f}"

    # ---------------------------------------------------------- 1. title
    s = blank(prs)
    text(s, M, Inches(2.3), W - 2 * M, Inches(0.4),
         [("WEEK 3  ·  PARTS 7 AND 8", 15, True, ACCENT, SANS)])
    text(s, M, Inches(2.75), W - 2 * M, Inches(1.4),
         [("From 73% to 91%", 54, True, BLUE, SANS)])
    text(s, M, Inches(4.1), Inches(9.6), Inches(1.6),
         [("Six rungs on CIFAR-10, one change at a time.", 21, False, INK, SANS),
          ("Every number measured on the same 5,000 validation images.",
           17, False, MUTED, SANS)], spacing=1.25)
    rule(s, Inches(5.9), w=Inches(3.0))

    # ------------------------------------------------- 2. where part 6 left us
    tr0, va0, te0, gap0 = b("0")
    figure_slide(
        prs, "the starting point",
        "Part 6 ended here: 73%, and stuck",
        F["rung0"],
        f"Rung 0 — part 6's config D. train {pct(tr0)}, val {pct(va0)}, "
        f"test {pct(te0)}, gap {gap0:.3f}.",
    )

    content_slide(
        prs, "diagnosis", "Read the training accuracy first",
        [("A 3-point gap is not an overfitting problem.", "lead"),
         "",
         f"train {pct(tr0)}  —  the model cannot fit the images it was trained on.",
         f"val   {pct(va0)}  —  and it is only 3 points behind.",
         "",
         ("Overfitting  =  train high, val far below.  Fix: regularise, add data.",
          "note"),
         ("Underfitting =  train itself is low.  Fix: more data, better "
          "optimisation, more capacity.", "note"),
         "",
         "Part 6 regularised an overfitting model until it underfit. "
         "Parts 7 and 8 raise the ceiling back up.",
         ],
    )

    # -------------------------------------------------------- 3. the method
    content_slide(
        prs, "method", "Three rules, and they matter more than the rungs",
        [("ONE CHANGE PER RUNG", "lead"),
         "Each rung differs from the one above it in exactly one respect, "
         "so each gain has exactly one cause.",
         "",
         ("ONE FIXED YARDSTICK", "lead"),
         "Every run — every rung, every sweep, every control — is scored on the "
         "same 5,000 validation images. The test set is opened once, at the end.",
         "",
         ("MEASURE THE CONFOUND", "lead"),
         "When a change drags a second change along with it, run the control "
         "that separates them.",
         ],
    )

    # ------------------------------------------------------ 4. the ladder
    figure_slide(
        prs, "the whole story on one slide", "Six rungs, 73% → 91%",
        F["stair"],
        "Validation and test move together throughout — the validation split "
        "is doing its job.",
    )

    table_slide(
        prs, "the whole story on one slide", "The same six rungs, as numbers",
        ["rung", "params", "min", "train", "val", "test", "gap"],
        [[f"{k}: {lab}", f"{PARAMS[k]:,}", f"{runs[k]['minutes']:.1f}",
          pct(b(k)[0]), (pct(b(k)[1]),), pct(b(k)[2]), f"{b(k)[3]:.3f}"]
         for _, k, lab in RUNGS],
        note="M1 Pro, 30 epochs each, best-validation epoch. "
             "The rest of the deck unpacks one rung at a time: "
             "the idea, the maths, why it should help, then what it actually did.",
        widths=[2.6, 1.0, 0.7, 0.8, 0.8, 0.8, 0.8],
    )

    # ============================================== ACT II — part 7
    section_slide(
        prs, "part 7", "Make it fit",
        "The model is not the bottleneck yet. Three rungs that cost almost no "
        "parameters: the data we already own, and how we train.",
    )

    # ------------------------------------------------ rung 1: full data
    content_slide(
        prs, "rung 1 · full data · the idea", "The cheapest capacity is data you already have",
        [("Part 6 trained on 18,000 of the 45,000 images available.", "lead"),
         "",
         "That was a habit from wanting fast iterations, not a decision about "
         "the model. CIFAR-10 ships 50,000 training images; we hold 5,000 back "
         "as validation and had been ignoring 27,000 of the rest.",
         "",
         "Augmentation re-uses the same 18,000 pictures with random crops and "
         "flips. New images carry information augmentation cannot manufacture.",
         "",
         ("Rung 1: n_train 18,000 → 45,000. Nothing else moves.", "note"),
         ],
    )

    flow_slide(
        prs, "rung 1 · full data · the maths", "More data drags a confound along",
        [("lead", "An epoch is a pass over the training set, so the number of "
                  "gradient updates depends on how big that set is:"),
         ("eq", r"$\mathrm{updates} \;=\; "
                r"\left\lceil \frac{N_{\mathrm{train}}}{\mathrm{batch}} \right\rceil "
                r"\times \mathrm{epochs}$", 30),
         ("eq", r"$\mathrm{rung\ 0:}\quad \left\lceil \frac{18000}{128}"
                r"\right\rceil \times 30 \;=\; 141 \times 30 \;=\; 4\,230$", 24),
         ("eq", r"$\mathrm{rung\ 1:}\quad \left\lceil \frac{45000}{128}"
                r"\right\rceil \times 30 \;=\; 352 \times 30 \;=\; 10\,560$", 24),
         ("gap", 0.1),
         ("text", "So 'more data' is really two changes at once: 2.5x the "
                  "images AND 2.5x the gradient updates. Either could explain "
                  "a gain."),
         ("note", "The control: train on the same 18,000 images for 75 epochs "
                  "— matching the update count with no new pictures. If the "
                  "updates were doing the work, it should catch up."),
         ],
        tmp,
    )

    content_slide(
        prs, "rung 1 · full data · why it should help",
        "Two sources of error, and data only shrinks one",
        [("What the network gets wrong splits in two:", "lead"),
         "",
         "· what the architecture cannot represent at all — a ceiling that more "
         "data will not move;",
         "· what it cannot pin down from the sample it saw — 18,000 examples "
         "leave many hypotheses consistent with the data.",
         "",
         "More distinct images shrink the second term. The signature of that "
         "working is specific, and worth watching for:",
         "",
         (">> accuracy goes up AND the train/val gap gets SMALLER.", "lead"),
         ("If a change raised accuracy by widening the gap, it bought capacity, "
          "not generalisation.", "note"),
         ],
    )

    tr1, va1, te1, gap1 = b("1")
    figure_slide(
        prs, "rung 1 · full data · the evidence",
        f"val {pct(va0)} → {pct(va1)}  ({delta('0', '1')} points), and the gap narrows",
        F["data"],
        f"gap {gap0:.3f} → {gap1:.3f} — exactly the data-limited signature. "
        f"The equal-steps control on 18k images landed well short, so it is the "
        f"images, not the updates.",
    )

    code_slide(
        prs, "rung 1 · full data · the code", "One argument, threaded through",
        "part7_improving_cnn.py  ·  make_splits()",
        ["N_POOL, N_VAL = 50000, 5000",
         "N_FULL = N_POOL - N_VAL      # 45,000",
         "",
         "order     = torch.randperm(N_POOL, generator=g.manual_seed(0))",
         ("train_idx = order[:n_train].tolist()      # <- the only knob",),
         "val_idx   = order[N_FULL:].tolist()        # never moves",
         "",
         "return (Subset(augmented, train_idx),      # training, augmented",
         "        Subset(clean,     val_idx),        # validation, clean",
         "        Subset(clean,     train_idx[:2000]))   # probe",
         ],
        note="The validation slice is taken from the END of a fixed shuffle, so "
             "it is identical whatever n_train is — that is what makes rung 0 "
             "and rung 1 comparable. And because train_idx is a prefix, the "
             "18,000 images are a subset of the 45,000: rung 1 adds data, it "
             "does not swap it.",
    )

    # ------------------------------------------------ rung 2a: batchnorm
    content_slide(
        prs, "rung 2a · batchnorm · the idea",
        "Every layer is aiming at a moving target",
        [("Layer 3 learns a mapping from what layer 2 sends it. "
          "Then layer 2's weights change, and the scale of that input shifts "
          "underneath layer 3.", "lead"),
         "",
         "The whole stack is doing this to itself, simultaneously, every step. "
         "The deeper the network, the worse it gets: a small change early is "
         "amplified by everything above it.",
         "",
         "BatchNorm pins the distribution. After every conv, normalise each "
         "channel across the batch, then let the network learn the scale and "
         "offset it actually wants.",
         "",
         ("Rung 2a: Conv → ReLU → Pool  becomes  Conv → BatchNorm → ReLU → Pool.",
          "note"),
         ],
    )

    defs_slide(
        prs, "rung 2a · batchnorm · the maths  (1 of 3)",
        "Every symbol, before any formula",
        [(r"$B$", "one mini-batch — the examples used for a single gradient step."),
         (r"$m$", "how many examples are in it. In our runs, m = 128."),
         (r"$x_i$", "one scalar value being normalised, at position i."),
         (r"$\mu_B$", "the empirical MEAN — empirical because it is measured "
                      "on this batch, not known in advance."),
         (r"$\sigma_B^2$", "the empirical VARIANCE of those same values."),
         (r"$\epsilon$", "a tiny constant so we never divide by zero "
                         "(PyTorch: 1e-5)."),
         (r"$\hat{x}_i$", "the normalised value: mean 0, variance 1 across "
                          "the batch."),
         (r"$\gamma,\ \beta$", "LEARNED scale and shift — the network's chance "
                               "to undo the normalisation if it wants to."),
         (r"$k$", "which dimension gets its own statistics. The one that "
                  "trips people up — see 3 of 3."),
         ],
        tmp,
        note="Only γ and β are learned. μ, σ² and x̂ are recomputed from "
             "scratch on every batch.",
    )

    s = blank(prs)
    header(s, "rung 2a · batchnorm · the maths  (2 of 3)", "The three equations")
    text(s, M, Inches(1.95), W - 2 * M, Inches(0.4),
         [("1.  MEASURE the batch, separately for each dimension k",
           15, True, ACCENT, SANS)])
    math_block(s, r"$\mu_B^{(k)} = \frac{1}{m}\sum_{i=1}^{m} x_i^{(k)}$",
               M + Inches(0.5), Inches(2.42), tmp, fontsize=27)
    math_block(s, r"$\left(\sigma_B^{(k)}\right)^2 = \frac{1}{m}\sum_{i=1}^{m}"
                  r"\left(x_i^{(k)} - \mu_B^{(k)}\right)^2$",
               Inches(5.6), Inches(2.42), tmp, fontsize=27)
    text(s, M, Inches(3.75), W - 2 * M, Inches(0.4),
         [("2.  NORMALISE — subtract the mean, divide by the standard deviation",
           15, True, ACCENT, SANS)])
    math_block(s, r"$\hat{x}_i^{(k)} = \frac{x_i^{(k)} - \mu_B^{(k)}}"
                  r"{\sqrt{\left(\sigma_B^{(k)}\right)^2 + \epsilon}}$",
               M + Inches(0.5), Inches(4.22), tmp, fontsize=27)
    text(s, M, Inches(5.55), W - 2 * M, Inches(0.4),
         [("3.  RESTORE a scale the network chooses for itself",
           15, True, ACCENT, SANS)])
    math_block(s, r"$y_i^{(k)} = \gamma^{(k)}\,\hat{x}_i^{(k)} + \beta^{(k)}$",
               M + Inches(0.5), Inches(6.02), tmp, fontsize=27)
    text(s, Inches(7.3), Inches(5.95), Inches(5.2), Inches(1.0),
         [("Step 2 throws information away; step 3 hands the control back as "
           "two learned numbers. If the best thing really is the original "
           "scale, the network can recover it.", 14, False, MUTED, SANS)],
         spacing=1.1)

    flow_slide(
        prs, "rung 2a · batchnorm · the maths  (3 of 3)",
        "What k is in a convolutional layer",
        [("lead", "Take stage 1 of our network: Conv2d(3 -> 32) on a batch of "
                  "128 CIFAR images. Its output is a 4-D tensor."),
         ("eq", r"$x \;\in\; \mathbb{R}^{\,N \times C \times H \times W}"
                r"\;=\; \mathbb{R}^{\,128 \times 32 \times 32 \times 32}$", 27),
         ("text", "k indexes C, the CHANNELS — not pixels, not images. "
                  "So d = 32 separate normalisations, one per channel."),
         ("eq", r"$\mu_B^{(k)} \ \mathrm{averages}\ "
                r"N \!\times\! H \!\times\! W \;=\; "
                r"128 \times 32 \times 32 \;=\; 131\,072\ \mathrm{numbers}$", 23),
         ("gap", 0.05),
         ("text", "Each channel is one learned filter, so this asks: across "
                  "every image and every position, what scale does this filter "
                  "output? That is the quantity worth pinning down."),
         ("note", "Consequence: the statistics depend on the other images in "
                  "the batch. That is why BatchNorm behaves differently in "
                  "train() and eval() — and why tiny batches hurt it."),
         ],
        tmp,
    )

    table_slide(
        prs, "rung 2a · batchnorm · the maths",
        "And the whole thing costs 160 parameters",
        ["stage", "channels", "γ + β added", "conv biases removed", "net"],
        [["1: Conv(3 → 32)", "32", "64", "−32", "+32"],
         ["2: Conv(32 → 64)", "64", "128", "−64", "+64"],
         ["3: Conv(64 → 64)", "64", "128", "−64", "+64"],
         ["total", "", "320", "−160", ("+160",)]],
        note="γ and β are 2 parameters per channel. The conv biases go because "
             "BatchNorm subtracts any constant the conv adds — β does that job "
             "now, so bias=False is bookkeeping, not a trick. "
             "188,810 → 188,970: whatever BatchNorm buys, it does not buy it "
             "with capacity.",
        widths=[2.0, 1.0, 1.2, 1.6, 0.8],
    )

    content_slide(
        prs, "rung 2a · batchnorm · why it should help",
        "The real prize is not accuracy — it is a usable learning rate",
        [("A gradient step of size lr means something different at every layer "
          "when their activations live on different scales.", "lead"),
         "",
         "Normalising puts every layer on the same footing, which conditions the "
         "loss surface: one learning rate is now roughly the right size "
         "everywhere. Raise it, and the network no longer blows up.",
         "",
         "So expect a modest accuracy gain at the same learning rate — and a "
         "much wider range of learning rates that still train.",
         "",
         (">> That is what makes rung 2b possible. The rungs compound.", "lead"),
         ],
    )

    tr2a, va2a, te2a, gap2a = b("2a")
    figure_slide(
        prs, "rung 2a · batchnorm · the evidence",
        f"val {pct(va1)} → {pct(va2a)}  ({delta('1', '2a')} points) for 160 parameters",
        F["bn"],
        "Same data, same learning rate, same 30 epochs. We also swept the "
        "learning rate: without BatchNorm the high end broke training; with it, "
        "3e-3 trained cleanly.",
    )

    code_slide(
        prs, "rung 2a · batchnorm · the code", "Three lines in the block builder",
        "part7_improving_cnn.py  ·  build_model(bn=...)",
        ["def block(c_in, c_out):",
         ("    layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=not bn)]",),
         ("    if bn:",),
         ("        layers.append(nn.BatchNorm2d(c_out))",),
         "    return layers + [nn.ReLU(), nn.MaxPool2d(2)]",
         "",
         "layers = block(3, 32) + block(32, 64) + block(64, 64) + [nn.Flatten()]",
         ],
        note="bias=not bn is the parameter bookkeeping from the maths slide: "
             "with BatchNorm the conv bias cannot affect the output, so we do "
             "not allocate it. Note BatchNorm2d goes BEFORE the ReLU — it "
             "normalises the pre-activations, which is exactly what the "
             "activation probe measures.",
    )

    # ------------------------------------------------ rung 2b: onecycle
    content_slide(
        prs, "rung 2b · onecycle · the idea",
        "One learning rate for all 30 epochs is a compromise",
        [("A constant learning rate has to be two things at once:", "lead"),
         "",
         "· small enough that the first steps, from a random initialisation, "
         "do not wreck the network;",
         "· small enough at the end that the weights can settle instead of "
         "bouncing around the minimum.",
         "",
         "Whatever single value satisfies both is too small for the middle of "
         "training, where you want to cover ground.",
         "",
         ("Rung 2b: warm up to a high learning rate, then anneal it to nearly "
          "zero. Same optimiser, same everything else.", "note"),
         ],
    )

    flow_slide(
        prs, "rung 2b · onecycle · the maths",
        "Two cosine halves, joined at the peak",
        [("lead", "Let p be the fraction of training completed, and pct_start "
                  "= 0.25 the fraction spent warming up. Both phases are "
                  "cosines:"),
         ("eq", r"$\mathrm{warm\!-\!up}\ (p < 0.25):\quad \eta = \eta_0 + "
                r"(\eta_{\max} - \eta_0)\,\frac{1 - \cos(\pi\,p/0.25)}{2}$", 23),
         ("eq", r"$\mathrm{anneal}\ (p \geq 0.25):\quad \eta = \eta_{\min} + "
                r"(\eta_{\max} - \eta_{\min})\,"
                r"\frac{1 + \cos\!\left(\pi\,\frac{p-0.25}{0.75}\right)}{2}$", 23),
         ("gap", 0.08),
         ("text", "PyTorch derives the endpoints from the peak: "
                  "η₀ = η_max / 25 and η_min = η₀ / 10⁴."),
         ("eq", r"$\eta_{\max} = 3\times10^{-3} \;\Rightarrow\; "
                r"\eta_0 = 1.2\times10^{-4}, \quad "
                r"\eta_{\min} = 1.2\times10^{-8}$", 23),
         ("note", "scheduler.step() runs after every BATCH — 10,560 times, not "
                  "30 — so the curve is smooth. cycle_momentum=False, because "
                  "cycling AdamW's β₁ would be a second change riding along "
                  "with the first."),
         ],
        tmp,
    )

    figure_slide(
        prs, "rung 2b · onecycle · the maths",
        "What that looks like over 30 epochs",
        F["lr"],
        "Sampled once per epoch, so the peak reads at epoch 8 rather than 7.5. "
        "Against rung 2a's flat 1e-3: OneCycle spends the middle of training "
        "well above it, and the end far below.",
        height=Inches(3.9),
    )

    content_slide(
        prs, "rung 2b · onecycle · why it should help",
        "Each phase of the cycle does a different job",
        [("WARM-UP — a random network produces large, badly-aimed gradients. "
          "Starting small keeps the first few hundred steps from destroying the "
          "initialisation.", "lead"),
         "",
         ("HIGH MIDDLE — big steps cover distance fast, and the noise they add "
          "acts as a mild regulariser: the weights cannot settle into a sharp, "
          "narrow minimum.", "lead"),
         "",
         ("ANNEAL TO ZERO — once it is in the right basin, shrinking the step "
          "lets it walk down to the bottom instead of rattling across it.", "lead"),
         "",
         ("Only safe at a peak of 3e-3 because BatchNorm is already there.",
          "note"),
         ],
    )

    tr2b, va2b, te2b, gap2b = b("2b")
    figure_slide(
        prs, "rung 2b · onecycle · the evidence",
        f"val {pct(va2a)} → {pct(va2b)}  ({delta('2a', '2b')} points) for zero parameters",
        F["cycle"],
        "Same architecture, same data, same 30 epochs. Only the learning-rate "
        "schedule moved. We swept the peak; 3e-3 and 1e-2 landed within noise "
        "of each other, so we kept 3e-3.",
    )

    code_slide(
        prs, "rung 2b · onecycle · the code", "Build it once, step it every batch",
        "part7_improving_cnn.py  ·  train()",
        ["optimizer = torch.optim.AdamW(model.parameters(), lr=lr, "
         "weight_decay=5e-4)",
         "",
         ("scheduler = torch.optim.lr_scheduler.OneCycleLR(",),
         ("    optimizer, max_lr=lr, epochs=epochs,",),
         ("    steps_per_epoch=len(train_loader),   # -> 352",),
         ("    pct_start=0.25, cycle_momentum=False)",),
         "",
         "for images, labels in train_loader:",
         "    ...",
         "    optimizer.step()",
         ("    scheduler.step()        # every BATCH, not every epoch",),
         ],
        note="steps_per_epoch x epochs is how OneCycle knows the length of the "
             "cycle; get it wrong and the schedule ends in the wrong place. "
             "cycle_momentum=False stops it also cycling AdamW's beta1 — that "
             "would be a second change riding along with the first.",
    )

    # ------------------------------------------------ checkpoint
    content_slide(
        prs, "checkpoint", "Still a ceiling — so the model is next",
        [(f"After three rungs:  train {pct(tr2b)},  val {pct(va2b)},  "
          f"gap {gap2b:.3f}.", "lead"),
         "",
         "The gap is 3 points. The model still cannot fit its own training "
         "images, so this is the same diagnosis as part 6 — a ceiling, not "
         "overfitting.",
         "",
         "But the two cheap fixes are spent: all 45,000 images are in use, and "
         "training is healthy. What is left is the model itself.",
         "",
         ("Part 6's architecture: three conv layers, at most 64 channels, and "
          "70% of its weights sitting in a fully-connected head.", "note"),
         ],
    )

    # ============================================== ACT III — part 8
    section_slide(
        prs, "part 8", "Capacity",
        "Same rules, same validation split, same recipe. Only the architecture "
        "moves — first deeper, then wider.",
    )

    # ------------------------------------------------ rung 3b: deeper
    content_slide(
        prs, "rung 3b · deeper · the idea",
        "The network does see the whole image. Look at where.",
        [("The final feature map is 64 channels on a 4×4 grid — a coarse, "
          "low-frequency summary of the entire picture. Coverage is not the "
          "problem.", "lead"),
         "",
         "But each of those 16 positions summarises only a 22×22 patch. "
         "Nothing in the convolutional stack ever relates two things more than "
         "22 px apart. All longer-range structure is left to one dense layer:",
         "",
         ("      Linear(1024 → 128)   =   131,200 parameters   =   70% of the "
          "network", "mono"),
         "",
         "That layer has no weight sharing — a dog's head top-left and the same "
         "head bottom-right are separate weights — and it combines the 16 "
         "positions in a single linear step, with no nonlinearity between them.",
         "",
         ("Rung 3b: two convs per stage. Widths unchanged, head unchanged.",
          "note"),
         ],
        top=Inches(1.92),
    )

    s = blank(prs)
    header(s, "rung 3b · deeper · the maths", "What one unit in the last stage sees")
    math_block(s, r"$r_l \;=\; r_{l-1} \;+\; (k_l - 1)\prod_{i<l} s_i$",
               M, Inches(1.82), tmp, fontsize=24)
    text(s, Inches(5.2), Inches(1.88), Inches(7.3), Inches(0.9),
         [("Each layer widens the patch a unit responds to. The product is why "
           "pooling dominates: every 2×2 pool doubles the stride, so a 3×3 conv "
           "above one pool adds 4 px, above two pools 8 px.",
           14.5, False, MUTED, SANS)], spacing=1.12)
    pic = s.shapes.add_picture(F["rf"], Emu(0), Inches(2.72), height=Inches(3.5))
    pic.left = int((W - pic.width) / 2)
    caption(s, "Blue: the receptive field after each layer. Orange: the final "
               "one. Two convs per stage is the first point at which a single "
               "unit integrates the whole frame.", y=Inches(6.42))

    s = blank(prs)
    header(s, "rung 3b · deeper · the maths",
           "Operator by operator — the shapes match, the reach does not")
    pic = s.shapes.add_picture(F["shapes"], Emu(0), Inches(1.80),
                               height=Inches(4.75))
    pic.left = int((W - pic.width) / 2)
    caption(s, "Generated by walking both networks, so these are the shapes "
               "PyTorch actually produces.", y=Inches(6.66))

    tr3b, va3b, te3b, gap3b = b("3b")
    figure_slide(
        prs, "rung 3b · deeper · the evidence",
        f"val {pct(va2b)} → {pct(va3b)}  ({delta('2b', '3b')} points) for 83k parameters",
        F["deep"],
        f"train {pct(tr2b)} → {pct(tr3b)}: the ceiling lifts. "
        f"gap {gap2b:.3f} → {gap3b:.3f} — the first sign of the other problem "
        f"coming back.",
    )

    code_slide(
        prs, "rung 3b · deeper · the code", "The architecture becomes two loops",
        "part8_bigger_cnn.py  ·  build_model(widths, convs)",
        ["layers, c_in = [], 3",
         "for width in widths:                      # widths = (32, 64, 64)",
         ("    for _ in range(convs):                # convs = 2   <- rung 3b",),
         "        layers += [nn.Conv2d(c_in, width, 3, padding=1, bias=False),",
         "                   nn.BatchNorm2d(width), nn.ReLU()]",
         "        c_in = width",
         "    layers.append(nn.MaxPool2d(2))        # one pool per STAGE",
         "",
         "layers += [nn.Flatten(), nn.Dropout(0.5),",
         "           nn.Linear(c_in * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 10)]",
         ],
        note="The pool stays outside the inner loop, so depth grows without "
             "changing the spatial sizes — the head still sees 4x4. With "
             "convs=1 and the default widths this rebuilds rung 2b layer for "
             "layer, which is what makes the comparison fair.",
    )

    # ------------------------------------------------ rung 3c: wider
    content_slide(
        prs, "rung 3c · wider · the idea",
        "Depth is composition; width is vocabulary",
        [("Depth decides how many times features get combined. Width decides "
          "how many different features each level can hold at all.", "lead"),
         "",
         "Part 6's widths were 32, 64, 64 — and that last stage, 64 channels on "
         "a 4×4 grid, is what the entire classifier head has to work from. "
         "1,024 numbers to describe an image.",
         "",
         "Notice the third stage does not even widen over the second. There was "
         "no reason for that; it was a starter architecture.",
         "",
         ("Rung 3c: widths 32-64-64 → 64-128-256. Depth stays at two convs "
          "per stage.", "note"),
         ],
    )

    s = blank(prs)
    header(s, "rung 3c · wider · the maths",
           "Same depth, same sizes — only the number of features moves")
    pic = s.shapes.add_picture(F["widths"], Emu(0), Inches(1.78),
                               height=Inches(4.55))
    pic.left = int((W - pic.width) / 2)
    math_block(s, r"$\mathrm{params} = k^2 C_{\mathrm{in}} C_{\mathrm{out}}"
                  r"\ \propto\ C^2 \ \Rightarrow\ \mathrm{double\ every\ width},"
                  r"\ \mathrm{pay}\ 4\times\ \mathrm{per\ layer}$",
               M, Inches(6.52), tmp, fontsize=21, color="#1c222b")

    content_slide(
        prs, "rung 3c · wider · why it should help",
        "Read the shape column: that is what \u2018more features\u2019 means",
        [("Every spatial position carries a vector of feature values, one per "
          "channel. Widening lengthens that vector.", "lead"),
         "",
         ("  stage 1   32 \u2192 64 features per position, over a 32\u00d732 grid",
          "mono"),
         ("  stage 3   64 \u2192 256 features per position, over a 4\u00d74 grid",
          "mono"),
         ("  head in   1,024 \u2192 4,096 numbers describing the image", "mono"),
         "",
         "The last stage was the bottleneck: 64 numbers per position is a "
         "narrow pipe, and everything the classifier is allowed to use has to "
         "fit through it. More channels means more edge orientations held early "
         "and more object parts held late.",
         "",
         ("And expect the train/val gap to widen. Capacity is exactly what "
          "overfitting is made of.", "note"),
         ],
        top=Inches(1.92),
    )

    tr3c, va3c, te3c, gap3c = b("3c")
    figure_slide(
        prs, "rung 3c · wider · the evidence",
        f"val {pct(va3b)} → {pct(va3c)}  ({delta('3b', '3c')} points) for 6× the parameters",
        F["wide"],
        f"train {pct(tr3b)} → {pct(tr3c)}. The ceiling is gone — the model now "
        f"fits its training set almost perfectly. gap {gap3b:.3f} → {gap3c:.3f}.",
    )

    code_slide(
        prs, "rung 3c · wider · the code", "Same builder, one tuple changed",
        "part8_bigger_cnn.py  ·  LADDER",
        ["LADDER = [",
         "    arch_spec(\"3b_deeper\", \"3b: 2 convs per stage\", convs=2),",
         ("    arch_spec(\"3c_wider\",  \"3c: widths x2\", convs=2,",),
         ("              widths=(64, 128, 256)),",),
         "]",
         "",
         "def arch_spec(name, label, widths=(32, 64, 64), convs=1, epochs=30):",
         "    s = spec(name, label, n_train=45000, bn=True,",
         "             onecycle=True, lr=3e-3, epochs=epochs)",
         "    s.update(widths=list(widths), convs=convs)",
         "    return s",
         ],
        note="Every rung is a dict, and the dict is what gets logged to trackio "
             "as the run config. One change per rung is enforced by the fact "
             "that arch_spec carries rung 2b's entire recipe and only the named "
             "argument differs.",
    )

    figure_slide(
        prs, "depth vs width", "Accuracy per parameter, across all six rungs",
        F["params"],
        "Rungs 0 → 2b gained 10 points at constant size. 3b bought 5 points for "
        "44% more parameters; 3c bought 3 points for 6×.",
    )

    # ------------------------------------------------ closing
    figure_slide(
        prs, "where we are", "The ceiling is gone. The gap is back.",
        F["gap"],
        f"Rung 3c: train {pct(tr3c)}, val {pct(va3c)}, test {pct(te3c)}. "
        f"A {gap3c * 100:.0f}-point gap — the same problem part 6 opened with, "
        f"on a much better model.",
    )

    table_slide(
        prs, "the arc of week 3", "We have been here before",
        ["part", "what happened", "train", "val", "diagnosis"],
        [["6 (no dropout)", "plain CNN, 18k images", "0.98", "0.72", "overfitting"],
         ["6 (config D)", "dropout + augmentation", pct(tr0), pct(va0), "underfitting"],
         ["7", "data, BatchNorm, OneCycle", pct(tr2b), pct(va2b), "still a ceiling"],
         ["8", "deeper, then wider", pct(tr3c), (pct(va3c),), "overfitting again"]],
        note="Every fix creates the next problem. That is not a failure of the "
             "method — it is the method: diagnose, apply the matching tool, "
             "re-diagnose.",
        widths=[1.6, 2.6, 0.9, 0.9, 1.6],
    )

    content_slide(
        prs, "your turn", "Close a 7-point gap — you already know how",
        [(f"Rung 3c leaves {gap3c * 100:.0f} points on the table. Part 6's "
          f"toolbox is the right toolbox; the model is now big enough to "
          f"survive it.", "lead"),
         "",
         ("  LONGER      60 epochs instead of 30 — OneCycle rewards it", "mono"),
         ("  DECAY       weight decay 5e-4 → 1e-2 (it is inactive today)", "mono"),
         ("  SMOOTHING   label_smoothing=0.1 in CrossEntropyLoss", "mono"),
         ("  CUTOUT      RandomErasing on top of crop + flip", "mono"),
         ("  POLICIES    AutoAugment / TrivialAugmentWide", "mono"),
         ("  MIXUP       blend image pairs and their labels", "mono"),
         ("  TTA         average predictions over an image and its mirror", "mono"),
         "",
         ("One change at a time. Same 5,000 validation images. "
          "Target: 93–94%.", "note"),
         ],
        top=Inches(1.92),
    )

    # ============================================== BACKUP
    section_slide(
        prs, "backup", "The instruments",
        "Accuracy tells you where you ended up. These four tell you what "
        "happened on the way, and they are all logged to trackio.",
    )

    flat3c = flat_epoch(runs["3c"]["val_loss"])
    figure_slide(
        prs, "backup · instrument 1",
        "Validation loss sees it coming. Accuracy does not.",
        F["valloss"],
        f"Rung 3c. From epoch {flat3c} validation loss is flat while training "
        f"loss keeps falling — 0.17 to 0.04, nearly 5×. Validation accuracy "
        f"went on creeping up for {runs['3c']['best_epoch'] - flat3c} more "
        f"epochs, so accuracy alone says 'still improving'.",
        height=Inches(4.0),
    )

    content_slide(
        prs, "backup · instrument 1", "What that gap actually means",
        [("Accuracy only asks whether the top-scoring class was right. "
          "Loss asks how confident the model was.", "lead"),
         "",
         f"After epoch {flat3c} the extra training is not fixing validation "
         f"images — it is making the model more certain about training images "
         f"it already had right. That is overfitting in progress, and the "
         f"accuracy curve cannot see it.",
         "",
         ("  loss gap      epoch 18  0.19    →    epoch 30  0.29", "mono"),
         ("  accuracy gap  epoch 18  0.061   →    epoch 30  0.074", "mono"),
         "",
         ("Worth being precise: in none of our six runs did validation loss "
          "actually turn back UP within 30 epochs, so early stopping never "
          "fired. It flattens. Train for 60 epochs — the first item on your "
          "menu — and you will see the turn.", "note"),
         ],
    )

    figure_slide(
        prs, "backup · instrument 2", "Gradient norm — the dying-network light",
        F["grad"],
        "Logged every 20 batches; faint lines are raw, bold is a rolling mean. "
        "None of our runs broke, so read this as the healthy baseline: steady, "
        "order 1–8, and falling at the end of 2b as OneCycle anneals.",
    )

    content_slide(
        prs, "backup · instrument 2", "What a broken run looks like instead",
        [("This is a warning light, not a score. It is worth nothing on a good "
          "run and everything on a bad one.", "lead"),
         "",
         ("  → 0        dead ReLUs, or vanishing gradients in a deep stack.", "mono"),
         ("             The network has stopped learning; loss goes flat.", "mono"),
         "",
         ("  → huge     the learning rate is too high, or a bad batch.", "mono"),
         ("             Usually followed by NaN loss a few steps later.", "mono"),
         "",
         "Both show up in the gradient norm well before they show up in "
         "validation accuracy — which is the whole reason to log it.",
         "",
         ("This is also how you would debug a OneCycle peak set too high: the "
          "norm blows up during warm-up, not at the end.", "note"),
         ],
    )

    figure_slide(
        prs, "backup · instrument 3", "Activation scale — what BatchNorm actually fixes",
        F["act"],
        "Pre-activation standard deviation per conv stage, measured on a fixed "
        "probe batch each epoch. Without BatchNorm the scales drift apart as "
        "training moves the weights; with it they hold.",
        height=Inches(4.0),
    )

    figure_slide(
        prs, "backup · instrument 4", "Weight norm — is weight decay doing anything?",
        F["wnorm"],
        "||w|| over all parameters. It climbs steadily, which tells us the "
        "5e-4 weight decay is effectively inactive at these learning rates — "
        "which is why 'raise the decay' is on the menu for you to try.",
    )

    content_slide(
        prs, "backup", "Running it yourself",
        [("  cd week3_cnn", "mono"),
         "",
         ("  python part7_improving_cnn.py --ladder     # rungs 0 → 2b, ~15 min",
          "mono"),
         ("  python part8_bigger_cnn.py  --ladder       # rungs 3b, 3c, ~18 min",
          "mono"),
         ("  python part8_bigger_cnn.py  --story        # the staircase, from "
          "results/", "mono"),
         "",
         ("  python part7_improving_cnn.py --data-curve # rung 1 in depth", "mono"),
         ("  python part7_improving_cnn.py --lr-sweep   # rung 2a in depth", "mono"),
         "",
         ("  trackio show --project \"week3-cnn\"          # every curve, live",
          "mono"),
         "",
         ("Each run writes checkpoints/<rung>.pt and results/<rung>.json. "
          "--story redraws the whole ladder from those JSONs without "
          "retraining.", "note"),
         ],
        top=Inches(1.92),
    )

    prs.save(OUT)
    return OUT, len(prs.slides.__iter__.__self__._sldIdLst)


if __name__ == "__main__":
    path, n = build()
    print(f"wrote {path}  ({n} slides)")
