"""Build the Week 4 slide deck from the measured results.

Same contract as week3_cnn/slides/build_slides.py, and the same rule: nothing
on a slide is typed in by hand if it can be computed.

  * ARCHITECTURE DIAGRAMS are drawn by running the real models with forward
    hooks, so every "32 x 96 x 96" on a slide is a shape PyTorch produced.
  * PARAMETER COUNTS come from count_params() on those same models.
  * TEST METRICS come from evaluating checkpoints/*.pt on the test split, here,
    at build time. They are cached in slides/measured.json; delete it to force
    a re-measure.
  * TRAINING CURVES come from results/training_curves.json, which the parts
    write via common.save_history().

Two tables cannot be recomputed from a checkpoint, because the runs that
produced them saved no weights: Part 2's waist sweep and Part 3's 500-label
pretraining comparison. Both are transcribed into WAIST and LOW_LABEL below,
each marked with the command that produces them. Their parameter counts are
still computed live, which catches the most likely kind of drift.

    pip install python-pptx
    python slides/build_slides.py               # -> slides/week4_unet.pptx
    python slides/build_slides.py --n-test 200  # quicker measuring pass

The slide kit (blank/header/bullets/code_slide/flow_slide/...) is copied from
Week 3 rather than imported, so each week's folder stays self-contained and a
change to one deck cannot silently restyle the other.
"""

import argparse
import json
import os
import sqlite3
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import torch
import torch.nn as nn
import torch.nn.functional as F
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
WEEK4 = os.path.dirname(HERE)
if WEEK4 not in sys.path:                      # so this runs from the repo root too
    sys.path.insert(0, WEEK4)

from common import (CLASS_COLORS, CLASS_NAMES, IMAGE_SIZE, N_CLASSES,
                    colorize_batch, count_params, evaluate_seg, get_device,
                    load_pet, make_loaders, weights_path)
from part1_classifier_to_segmenter import Constant, fcn
from part2a_cnn_autoencoder import ConvAutoencoder, psnr
from part2b_waist_sweep import VariableWaistAutoencoder
from part3_encoder_decoder import SegAutoencoder
from part4_unet import UNet
from part5_unet_visualize import BAND_LABELS, accuracy_by_distance

RESULTS = os.path.join(WEEK4, "results")
CURVES = os.path.join(RESULTS, "training_curves.json")
CACHE = os.path.join(HERE, "measured.json")
TRACKIO_DB = os.path.expanduser("~/.cache/huggingface/trackio/week4-unet.db")
OUT = os.path.join(HERE, "week4_unet.pptx")

# ------------------------------------------------------------------ palette
# Identical to Week 3's deck: this is the next session of the same course, and
# a student should not have to re-learn what the colours mean. One cool ink and
# one warm accent, separated in both hue and lightness.

INK = RGBColor(0x1C, 0x22, 0x2B)
MUTED = RGBColor(0x6B, 0x75, 0x82)
ACCENT = RGBColor(0xC2, 0x55, 0x1E)
BLUE = RGBColor(0x1F, 0x4E, 0x79)
PAPER = RGBColor(0xFA, 0xF9, 0xF7)
RULE = RGBColor(0xD8, 0xD3, 0xCB)

C_WARM = "#c2551e"      # the new thing / the thing that wins
C_COOL = "#1f4e79"      # the baseline it is being compared against
C_GREY = "#6b7582"      # the oldest/weakest series, always also dashed
#
# Checked with the palette validator rather than by eye. On this surface the
# three separate cleanly for colour-vision deficiency (worst pair ΔE 13.1
# protan, against a target of 8) and for normal vision (worst 15.9), and all
# three clear 3:1 contrast. Two of its style checks fail -- the navy is darker
# and less saturated than the reference design system's ramp step -- and we
# keep it anyway, because it is Week 3's deck colour and a student should not
# have to re-learn what the colours mean between sessions. Where a third
# series appears, the grey one is dashed as well, so identity never rests on
# the closest pair of hues alone.
C_GRID = "#d8d3cb"
PAPER_HEX = "#faf9f7"

SANS = "Helvetica Neue"
MONO = "Menlo"

W, H = Inches(13.333), Inches(7.5)
M = Inches(0.85)

# --------------------------------------------------------- measured elsewhere
#
# python part2b_waist_sweep.py
# (poolings, waist, MSE, PSNR dB). Params are recomputed below from
# VariableWaistAutoencoder(depth=...), so a change to the model shows up as a
# mismatch rather than as a stale slide.
WAIST = [(2, 24, 0.0015, 28.19),
         (3, 12, 0.0040, 23.97),
         (4, 6, 0.0076, 21.21)]

# python part3_encoder_decoder.py --pretrained --n-train 500 --skip-baseline
# (mIoU with 500 labels), against the same pair trained on all 3,680.
LOW_LABEL = {"encoder-decoder": 0.535, "+ AE pretraining": 0.575}

# The checkpoints the deck scores, in the order they appear.
MODELS = [
    ("part1 FCN", fcn, "part3_fcn"),
    ("encoder-decoder", SegAutoencoder, "part3_encdec"),
    ("U-Net, skips OFF", lambda: UNet(use_skips=False), "part4_unet_skips_off"),
    ("U-Net, skips ON", lambda: UNet(use_skips=True), "part4_unet_skips_on"),
]


# ------------------------------------------------------------------- data


def load_curves():
    with open(CURVES) as f:
        return json.load(f)["runs"]


def trackio_series(run_name, key):
    """One logged metric from the trackio database, in step order.

    Every part logs its per-epoch measurements under the label it prints, so
    these are the same runs that wrote the checkpoints. trackio reuses a run
    name across invocations, so we take the most recent run_id carrying that
    name -- otherwise an aborted earlier attempt would be interleaved with the
    run we mean.
    """
    if not os.path.exists(TRACKIO_DB):
        return [], []
    con = sqlite3.connect(f"file:{TRACKIO_DB}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT run_id FROM metrics WHERE run_name = ? ORDER BY id DESC LIMIT 1",
            (run_name,)).fetchone()
        if row is None:
            return [], []
        rows = con.execute(
            "SELECT json_extract(metrics,'$.epoch'), json_extract(metrics,?) "
            "FROM metrics WHERE run_id = ? AND json_extract(metrics,?) IS NOT NULL "
            "ORDER BY id", (f"$.{key}", row[0], f"$.{key}")).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows], [r[1] for r in rows]


def load_model(build, stem, device):
    path = weights_path(stem)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}\nrun the parts first (see week4_unet/README notes)")
    model = build()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


def measure(n_test, force=False):
    """Score every checkpoint on the test split. Cached in slides/measured.json."""
    if os.path.exists(CACHE) and not force:
        with open(CACHE) as f:
            cached = json.load(f)
        if cached.get("n_test") == n_test:
            print(f"  using cached measurements from {CACHE}")
            return cached
        print("  cache was built with a different --n-test; re-measuring")

    device = get_device()
    print(f"  measuring on {device} ({n_test} test images)")
    torch.manual_seed(0)
    test_ds = load_pet("test", n_subset=n_test)
    _, loader = make_loaders(test_ds, test_ds)

    out = {"n_test": n_test, "device": str(device), "seg": {}, "recon": {}}

    acc, iou, miou = evaluate_seg(Constant(cls=1).to(device), loader)
    out["seg"]["constant"] = dict(label='always "background"', params=0, acc=acc,
                                  iou=iou.tolist(), miou=miou)

    for label, build, stem in MODELS:
        model = load_model(build, stem, device)
        acc, iou, miou = evaluate_seg(model, loader)
        out["seg"][stem] = dict(label=label, params=count_params(model), acc=acc,
                                iou=iou.tolist(), miou=miou)
        print(f"    {label:<20} mIoU {miou:.3f}   border {iou[2]:.3f}")

    model = load_model(SegAutoencoder, "part3_pretrained", device)
    acc, iou, miou = evaluate_seg(model, loader)
    out["seg"]["part3_pretrained"] = dict(label="+ AE pretraining",
                                          params=count_params(model), acc=acc,
                                          iou=iou.tolist(), miou=miou)

    # Part 2's autoencoders are scored on reconstruction, not on IoU.
    recon_ds = load_pet("test", n_subset=n_test, segmentation=False)
    _, recon_loader = make_loaders(recon_ds, recon_ds)
    for stem, mode in [("part2_ae_convt", "convt"), ("part2_ae_upsample", "upsample")]:
        model = load_model(lambda m=mode: ConvAutoencoder(mode=m), stem, device)
        total, n = 0.0, 0
        with torch.no_grad():
            for x, y in recon_loader:
                x, y = x.to(device), y.to(device)
                total += torch.mean((model(x) - y) ** 2).item() * x.size(0)
                n += x.size(0)
        mse = total / n
        out["recon"][stem] = dict(params=count_params(model), mse=mse, psnr=psnr(mse))
        print(f"    {stem:<20} MSE {mse:.4f}   PSNR {psnr(mse):.2f} dB")

    # Part 5's curve: accuracy against distance to the nearest class boundary.
    out["distance"] = {}
    for label, stem in [("skips OFF", "part4_unet_skips_off"),
                        ("skips ON", "part4_unet_skips_on")]:
        model = load_model(lambda s=stem: UNet(use_skips=s.endswith("on")), stem, device)
        out["distance"][label] = accuracy_by_distance(model, loader, device)
        print(f"    {label:<20} by distance {['%.3f' % v for v in out['distance'][label]]}")

    # Class balance, straight off the labels -- the reason mIoU exists.
    counts = torch.zeros(N_CLASSES)
    for _, y in loader:
        counts += torch.bincount(y.flatten(), minlength=N_CLASSES).float()
    out["class_share"] = (counts / counts.sum()).tolist()

    with open(CACHE, "w") as f:
        json.dump(out, f, indent=1)
    print(f"  wrote {CACHE}")
    return out


@torch.no_grad()
def sample_batch(n=6, seed=0):
    """One fixed batch of test images, used by every picture in the deck."""
    torch.manual_seed(seed)
    ds = load_pet("test", n_subset=200)
    _, loader = make_loaders(ds, ds)
    x, y = next(iter(loader))
    return x[:n], y[:n]


@torch.no_grad()
def predict(stem, build, x, device):
    model = load_model(build, stem, device)
    return model(x.to(device)).argmax(dim=1).cpu()


# --------------------------------------------------------- shape walking
#
# Every architecture diagram is drawn from a real forward pass. The node list
# is (name, channels, spatial, level): level is how many poolings deep it sits,
# which is what puts the U in the U-Net.


def walk(model, names, device="cpu", size=IMAGE_SIZE):
    """Run one input through `model` and record each named submodule's output."""
    log, hooks = [], []
    for name in names:
        mod = model
        for attr in name.split("."):
            mod = mod[int(attr)] if attr.isdigit() else getattr(mod, attr)
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, n=name: log.append((n, tuple(o.shape[1:])))))
    with torch.no_grad():
        model.to(device)(torch.zeros(1, 3, size, size, device=device))
    for h in hooks:
        h.remove()
    return log


def levels_of(size):
    """96 -> 0, 48 -> 1, 24 -> 2, 12 -> 3."""
    return int(np.log2(IMAGE_SIZE // size))


def fcn_nodes():
    log = dict(walk(fcn(), ["0", "4", "8", "12", "13"]))
    return [
        ("Conv+BN+ReLU", log["0"], "enc"),
        ("Conv+BN+ReLU", log["4"], "enc"),
        ("Conv+BN+ReLU", log["8"], "enc"),
        ("Conv 1x1", log["12"], "waist"),
        ("Upsample x8", log["13"], "free"),
    ]


def ae_nodes():
    # Same eight attribute names as the U-Net below -- part 2a and part 4 name
    # their blocks identically, which is what makes the two diagrams comparable.
    names = ["enc1", "enc2", "enc3", "bottleneck", "dec3", "dec2", "dec1", "head"]
    log = dict(walk(ConvAutoencoder(), names))
    return [("enc1", log["enc1"], "enc"), ("enc2", log["enc2"], "enc"),
            ("enc3", log["enc3"], "enc"), ("bottleneck", log["bottleneck"], "waist"),
            ("dec3", log["dec3"], "dec"), ("dec2", log["dec2"], "dec"),
            ("dec1", log["dec1"], "dec"), ("head+Sigmoid", log["head"], "out")]


def unet_nodes():
    names = ["enc1", "enc2", "enc3", "bottleneck", "dec3", "dec2", "dec1", "head"]
    log = dict(walk(UNet(), names))
    return [("enc1", log["enc1"], "enc"), ("enc2", log["enc2"], "enc"),
            ("enc3", log["enc3"], "enc"), ("bottleneck", log["bottleneck"], "waist"),
            ("dec3", log["dec3"], "dec"), ("dec2", log["dec2"], "dec"),
            ("dec1", log["dec1"], "dec"), ("head", log["head"], "out")]


# ---------------------------------------------------------------- figures


def save(fig, path, pad=0.08):
    fig.savefig(path, dpi=200, facecolor=PAPER_HEX, bbox_inches="tight",
                pad_inches=pad)
    plt.close(fig)
    return path


def style(ax, xlabel="", ylabel=""):
    ax.set_facecolor(PAPER_HEX)
    ax.set_xlabel(xlabel, fontsize=11, color=C_GREY)
    ax.set_ylabel(ylabel, fontsize=11, color=C_GREY)
    ax.grid(True, color=C_GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_GREY, labelsize=10)


BOX_COLORS = {"enc": C_COOL, "dec": "#3f7fb5", "waist": C_GREY,
              "out": C_WARM, "free": "#b0b6bd"}


def draw_arch(ax, nodes, skips=(), gap=1.65):
    """Boxes on a U: x = position in the network, y = how many poolings deep.

    Box width encodes channels (log scale, so 3 and 256 both fit) and box
    height encodes spatial size, which is what makes the shape of the network
    the shape of the picture.
    """
    pos = {}
    for i, (name, (ch, h_, w_), kind) in enumerate(nodes):
        lvl = levels_of(h_)
        x, y = i * gap, -lvl * 1.25
        bw = 0.16 * np.log2(ch) + 0.22
        bh = 0.30 + 0.85 * (h_ / IMAGE_SIZE) ** 0.5
        ax.add_patch(plt.Rectangle((x - bw / 2, y - bh / 2), bw, bh,
                                   facecolor=BOX_COLORS[kind], edgecolor="none",
                                   alpha=0.92, zorder=3))
        ax.text(x, y + bh / 2 + 0.16, name, ha="center", fontsize=10.5,
                color="#1c222b")
        ax.text(x, y - bh / 2 - 0.30, f"{ch}x{h_}x{w_}", ha="center", fontsize=10,
                family="monospace", color=C_GREY)
        pos[i] = (x, y, bw, bh)

    for i in range(len(nodes) - 1):
        x0, y0, bw0, _ = pos[i]
        x1, y1, bw1, _ = pos[i + 1]
        ax.annotate("", xy=(x1 - bw1 / 2 - 0.04, y1), xytext=(x0 + bw0 / 2 + 0.04, y0),
                    arrowprops=dict(arrowstyle="-|>", color=C_GREY, lw=1.4,
                                    shrinkA=0, shrinkB=0), zorder=2)

    # Skips run straight across, at the level they belong to: an encoder map and
    # the decoder block it feeds are at the same height, which is the whole
    # reason the picture is a U.
    for a, b in skips:
        (x0, y0, bw0, _), (x1, _, bw1, _) = pos[a], pos[b]
        ax.annotate("", xy=(x1 - bw1 / 2 - 0.03, y0), xytext=(x0 + bw0 / 2 + 0.03, y0),
                    arrowprops=dict(arrowstyle="-|>", color=C_WARM, lw=2.2,
                                    linestyle=(0, (5, 2)), shrinkA=0, shrinkB=0),
                    zorder=1)
        ax.text((x0 + x1) / 2, y0 + 0.12, "cat", ha="center", fontsize=11,
                color=C_WARM, family="monospace", zorder=5)
    ax.set_xlim(-1.0, (len(nodes) - 1) * gap + 1.0)
    ax.axis("off")
    return pos


def fig_arch_fcn(path):
    fig, ax = plt.subplots(figsize=(11.8, 4.8), facecolor=PAPER_HEX)
    nodes = fcn_nodes()
    pos = draw_arch(ax, nodes)
    ax.set_ylim(-5.4, 1.5)
    ax.text(0, 1.1, "input  3 x 96 x 96", fontsize=12.5, color=C_COOL, ha="center",
            weight="bold")
    ax.text(pos[len(nodes) - 1][0], 1.1, "output  3 x 96 x 96", fontsize=12.5,
            color=C_WARM, ha="center", weight="bold")
    # The empty lower-left quadrant is the only place a note does not land on
    # the staircase.
    ax.text(-0.7, -3.75, "the decision is made down HERE, on a 12x12 grid\n"
                         "-- one vote per 8x8 block of photo.\n\n"
                         "Upsample then stretches it back: 0 parameters,\n"
                         "nothing learned, no detail added.",
            fontsize=11.5, color=C_GREY, ha="left", va="top", linespacing=1.5)
    return save(fig, path)


def fig_arch_ae(path):
    fig, ax = plt.subplots(figsize=(12.6, 5.0), facecolor=PAPER_HEX)
    nodes = ae_nodes()
    pos = draw_arch(ax, nodes)
    ax.set_ylim(-5.6, 1.2)
    ax.text(0, 0.95, "3 x 96 x 96", fontsize=12, color=C_COOL, ha="center")
    ax.text(pos[len(nodes) - 1][0], 0.95, "3 x 96 x 96", fontsize=12, color=C_WARM,
            ha="center")
    ax.text(pos[3][0], -4.55, "everything the decoder ever learns about this\n"
                              "photo has to fit through here",
            fontsize=11.5, color=C_GREY, ha="center", va="top", linespacing=1.5)
    return save(fig, path)


def fig_arch_unet(path):
    fig, ax = plt.subplots(figsize=(12.6, 5.2), facecolor=PAPER_HEX)
    nodes = unet_nodes()
    pos = draw_arch(ax, nodes, skips=[(0, 6), (1, 5), (2, 4)])
    ax.set_ylim(-5.8, 1.5)
    ax.text(pos[3][0], 1.15, "torch.cat([up, skip], dim=1)", fontsize=13.5,
            color=C_WARM, ha="center", weight="bold", family="monospace")
    ax.text(pos[3][0], -4.75, "same encoder, same 12x12 waist, same decoder as Part 2.\n"
                              "The three dashed arrows are the entire difference.",
            fontsize=11.5, color=C_GREY, ha="center", va="top", linespacing=1.5)
    return save(fig, path)


def fig_task(path, x, y, share):
    """What the label actually is, and why one number will not score it."""
    fig = plt.figure(figsize=(12.2, 4.2), facecolor=PAPER_HEX)
    grid = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 0.55, 2.3], hspace=0.12,
                            wspace=0.08)
    col = colorize_batch(y)
    for c in range(4):
        for r, batch in enumerate([x, col]):
            ax = fig.add_subplot(grid[r, c])
            ax.imshow(batch[c].permute(1, 2, 0).clamp(0, 1))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                ax.text(-0.08, 0.5, ["photo\n3 x 96 x 96", "trimap\n96 x 96 int64"][r],
                        transform=ax.transAxes, ha="right", va="center",
                        fontsize=10.5, color="#1c222b", linespacing=1.4)
    ax = fig.add_subplot(grid[:, 5])
    ax.set_facecolor(PAPER_HEX)
    names = ["pet", "background", "border"]
    colors = ["#e57d21", "#40516f", "#9aa3ae"]
    bars = ax.barh(range(3), share, color=colors, height=0.55, zorder=3)
    for i, (b, v) in enumerate(zip(bars, share)):
        ax.text(v + 0.012, i, f"{v * 100:.0f}%", va="center", fontsize=11,
                color="#1c222b")
    ax.set_yticks(range(3))
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, max(share) * 1.25)
    ax.set_title("share of all test pixels", fontsize=11, color=C_GREY, loc="left")
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.tick_params(axis="y", length=0, colors="#1c222b")
    return save(fig, path)


def fig_grid(path, rows, titles, suptitle=None, n=6):
    fig, axes = plt.subplots(len(rows), n, figsize=(1.52 * n + 2.0, 1.62 * len(rows)),
                             facecolor=PAPER_HEX)
    axes = np.asarray(axes).reshape(len(rows), n)
    for r, (batch, title) in enumerate(zip(rows, titles)):
        for c in range(n):
            axes[r, c].imshow(batch[c].permute(1, 2, 0).clamp(0, 1))
            axes[r, c].axis("off")
        axes[r, 0].text(-0.08, 0.5, title, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=10.5, color="#1c222b")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, color="#1c222b")
    fig.tight_layout()
    return save(fig, path)


def fig_bars(path, seg, stems, labels):
    """mIoU and border IoU, the two numbers the week argues about."""
    fig, ax = plt.subplots(figsize=(10.8, 4.2), facecolor=PAPER_HEX)
    ys = np.arange(len(stems))
    miou = [seg[s]["miou"] for s in stems]
    border = [seg[s]["iou"][2] for s in stems]
    ax.barh(ys - 0.18, miou, height=0.32, color=C_COOL, label="mean IoU", zorder=3)
    ax.barh(ys + 0.18, border, height=0.32, color=C_WARM, label="border-class IoU",
            zorder=3)
    for yy, v in list(zip(ys - 0.18, miou)) + list(zip(ys + 0.18, border)):
        ax.text(v + 0.008, yy, f"{v:.3f}", va="center", fontsize=10, color="#1c222b")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, max(miou) * 1.18)
    style(ax, "IoU  (higher is better)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, colors="#1c222b")
    ax.legend(frameon=False, fontsize=10.5, loc="lower right")
    return save(fig, path)


def fig_curves(path, curves, keys, labels):
    """Training loss per epoch, from the runs that wrote the checkpoints."""
    fig, ax = plt.subplots(figsize=(9.6, 4.3), facecolor=PAPER_HEX)
    colors = [C_GREY, C_COOL, C_WARM]
    for (key, label), color in zip(zip(keys, labels), colors):
        r = curves[key]
        ep = range(1, len(r["loss"]) + 1)
        style_ = "--" if color == C_GREY else "-"
        ax.plot(ep, r["loss"], style_, lw=2.6, color=color, label=label)
        ax.annotate(f"{r['loss'][-1]:.3f}", xy=(len(r["loss"]), r["loss"][-1]),
                    xytext=(7, -3), textcoords="offset points", fontsize=11.5,
                    color=color, weight="bold")
    ax.set_xlim(0.6, len(curves[keys[0]]["loss"]) + 1.9)
    style(ax, "epoch", "training cross-entropy")
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    return save(fig, path)


def fig_waist(path):
    """The control experiment: smaller waist, more parameters, worse picture."""
    fig, ax = plt.subplots(figsize=(9.8, 4.3), facecolor=PAPER_HEX)
    xs = [count_params(VariableWaistAutoencoder(depth=d)) for d, _, _, _ in WAIST]
    ys = [db for _, _, _, db in WAIST]
    ax.plot(xs, ys, "-", color=C_GRID, lw=2.0, zorder=1)
    for (depth, waist, mse, db), p in zip(WAIST, xs):
        last = depth == 4
        color = C_WARM if last else C_COOL
        ax.scatter(p, db, s=150, color=color, zorder=3)
        # The rightmost label has to hang to the LEFT or it runs off the axes.
        ax.annotate(f"{waist}x{waist} waist\n{p:,} params",
                    (p, db), xytext=(-14, 10) if last else (12, 6),
                    textcoords="offset points", fontsize=11, color=color,
                    ha="right" if last else "left", linespacing=1.35)
    ax.set_xscale("log")
    ax.set_xlim(3e5, 1.3e7)
    ax.set_ylim(19.5, 30.0)
    style(ax, "parameters (log scale)", "reconstruction PSNR (dB, higher is better)")
    ax.annotate("16.6x the weights,\n7 dB worse",
                xy=(xs[0], ys[0]), xytext=(30, -46), textcoords="offset points",
                fontsize=12.5, color=C_WARM, ha="left", weight="bold",
                linespacing=1.4)
    return save(fig, path)


def fig_pretrain(path, seg):
    """Self-supervised pretraining, at two label budgets."""
    fig, ax = plt.subplots(figsize=(9.4, 4.2), facecolor=PAPER_HEX)
    groups = ["500 labels", "all 3,680 labels"]
    scratch = [LOW_LABEL["encoder-decoder"], seg["part3_encdec"]["miou"]]
    pre = [LOW_LABEL["+ AE pretraining"], seg["part3_pretrained"]["miou"]]
    xs = np.arange(2)
    ax.bar(xs - 0.19, scratch, 0.36, color=C_COOL, label="from scratch", zorder=3)
    ax.bar(xs + 0.19, pre, 0.36, color=C_WARM, label="from Part 2's autoencoder",
           zorder=3)
    for x, v in list(zip(xs - 0.19, scratch)) + list(zip(xs + 0.19, pre)):
        ax.text(x, v + 0.008, f"{v:.3f}", ha="center", fontsize=11, color="#1c222b")
    for i, (a, b) in enumerate(zip(scratch, pre)):
        ax.text(i, max(a, b) + 0.055, f"{b - a:+.3f}", ha="center", fontsize=13,
                color=C_WARM if b > a else C_GREY, weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_ylim(0, 0.80)
    style(ax, "", "mean IoU")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    return save(fig, path)


def fig_distance(path, dist):
    """Part 5's figure: the gain lives at the boundary and nowhere else."""
    off, on = dist["skips OFF"], dist["skips ON"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1), facecolor=PAPER_HEX)
    xs = np.arange(len(BAND_LABELS))

    ax = axes[0]
    ax.plot(xs, off, "-o", lw=2.4, color=C_COOL, label="skips OFF")
    ax.plot(xs, on, "-o", lw=2.8, color=C_WARM, label="skips ON")
    ax.set_xticks(xs)
    ax.set_xticklabels(BAND_LABELS)
    style(ax, "distance to nearest class boundary (px)", "pixel accuracy")
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    ax.set_title("away from edges, the same model", fontsize=12, color="#1c222b",
                 loc="left")

    ax = axes[1]
    gain = [a - b for a, b in zip(on, off)]
    ax.bar(xs, gain, 0.55, color=[C_WARM if g > 0.002 else C_GREY for g in gain],
           zorder=3)
    span = max(gain) - min(0, min(gain))
    for x, g in zip(xs, gain):
        # Negative bars get their label above the zero line, where there is room
        # -- below it the label lands on the x tick labels.
        ax.text(x, max(g, 0) + span * 0.03, f"{g:+.3f}", ha="center", fontsize=11,
                color="#1c222b", va="bottom")
    ax.axhline(0, color=C_GREY, lw=1.0)
    ax.set_ylim(min(0, min(gain)) - span * 0.10, max(gain) + span * 0.16)
    ax.set_xticks(xs)
    ax.set_xticklabels(BAND_LABELS)
    style(ax, "distance to nearest class boundary (px)", "accuracy gained from skips")
    ax.grid(axis="x", visible=False)
    ax.set_title("the whole gain, within a few pixels of an edge", fontsize=12,
                 color="#1c222b", loc="left")
    return save(fig, path)


def fig_trackio(path, runs, panels, colors=None):
    """Logged metrics, per epoch, for several runs -- straight from trackio.

    `panels` is a list of (metric key, axis label, panel title). Returns None
    if the database has nothing for these runs, so the deck still builds on a
    machine that has never run the parts.
    """
    colors = colors or [C_GREY, C_COOL, C_WARM]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.9 * len(panels), 4.2),
                             facecolor=PAPER_HEX)
    axes = np.atleast_1d(axes)
    drawn = 0
    for ax, (key, ylabel, title) in zip(axes, panels):
        ends = []
        for (run_name, label), color in zip(runs, colors):
            ep, vals = trackio_series(run_name, key)
            if not ep:
                continue
            drawn += 1
            ax.plot(ep, vals, "--o" if color == C_GREY else "-o", lw=2.6, ms=4.0,
                    color=color, label=label)
            ends.append((vals[-1], ep[-1], color))
        # Two runs can finish within a hair of each other (0.417 and 0.400),
        # and their end labels would then print on top of one another. Walk
        # them in value order and push each one clear of the last.
        ends.sort()
        # Space them by a fraction of the PANEL's range, not of the spread
        # between the end values -- three runs that finish close together have
        # almost no spread, which is exactly when the labels collide.
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.055
        prev = None
        for value, last_ep, color in ends:
            y = value if prev is None else max(value, prev + gap)
            ax.annotate(f"{value:.3f}", xy=(last_ep + 0.35, y), fontsize=11,
                        color=color, weight="bold", va="center", ha="left")
            prev = y
        if drawn:
            ax.set_xlim(0.5, max(ep) + 2.6)
        style(ax, "epoch", ylabel)
        ax.set_title(title, fontsize=12, color="#1c222b", loc="left")
    if not drawn:
        plt.close(fig)
        return None
    axes[0].legend(frameon=False, fontsize=10.5, loc="lower right")
    return save(fig, path)


def fig_instrument(path, run_name):
    """Why this week logs IoU: watch accuracy sit still while the border moves."""
    series = [("test/acc", "pixel accuracy", C_GREY),
              ("test/mIoU", "mean IoU", C_COOL),
              ("test/iou_border", "border-class IoU", C_WARM)]
    fig, ax = plt.subplots(figsize=(9.8, 4.3), facecolor=PAPER_HEX)
    for key, label, color in series:
        ep, vals = trackio_series(run_name, key)
        if not ep:
            plt.close(fig)
            return None
        ax.plot(ep, vals, "--o" if color == C_GREY else "-o", lw=2.6, ms=4.5,
                color=color, label=label)
        ax.annotate(f"{vals[-1]:.3f}", xy=(ep[-1], vals[-1]), xytext=(8, -4),
                    textcoords="offset points", fontsize=11.5, color=color,
                    weight="bold")
    ax.set_xlim(0.5, max(ep) + 2.4)
    ax.set_ylim(0, 1.0)
    style(ax, "epoch", "")
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    return save(fig, path)


def fig_zoom(path, x, y, preds, titles, box=44):
    """Magnify the busiest boundary in the batch -- Part 4's zoom_figure."""
    border = (y == 2).float().unsqueeze(1)
    dens = F.avg_pool2d(border, kernel_size=box, stride=4)
    _, _, gh, gw = dens.shape
    best = int(dens.flatten().argmax())
    idx, rc = best // (gh * gw), best % (gh * gw)
    top = min((rc // gw) * 4, IMAGE_SIZE - box)
    left = min((rc % gw) * 4, IMAGE_SIZE - box)

    rows = [x, colorize_batch(y)] + [colorize_batch(p) for p in preds]
    labels = ["photo", "truth"] + titles
    fig, axes = plt.subplots(1, len(rows), figsize=(2.15 * len(rows), 2.9),
                             facecolor=PAPER_HEX)
    for ax, batch, label in zip(axes, rows, labels):
        ax.imshow(batch[idx, :, top:top + box, left:left + box]
                  .permute(1, 2, 0).clamp(0, 1))
        ax.set_title(label, fontsize=11, color="#1c222b")
        ax.axis("off")
    fig.tight_layout()
    return save(fig, path)


def fig_errors(path, x, y, preds, titles, n=5):
    """Wrong pixels in red, over a dimmed photo -- with the colour key drawn in.

    Two different colour schemes share this figure and they mean different
    things, so the bottom strip spells both out. The 'truth' row is a class
    map: CLASS_COLORS straight out of common.py, one colour per class. The
    'errors' rows are not class maps at all -- they are the photo turned grey
    and dimmed to 55%, with every pixel the model got wrong painted pure red.
    """
    rows, labels = [x, colorize_batch(y)], ["photo", "truth"]
    for pred, title in zip(preds, titles):
        wrong = (pred != y).float()
        grey = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1) * 0.55
        overlay = grey.clone()
        overlay[:, 0] = torch.maximum(overlay[:, 0], wrong)
        overlay[:, 1] = overlay[:, 1] * (1 - wrong)
        overlay[:, 2] = overlay[:, 2] * (1 - wrong)
        rows.append(overlay)
        labels.append(f"errors: {title}")

    nr = len(rows)
    fig = plt.figure(figsize=(1.52 * n + 2.0, 1.62 * nr + 0.78), facecolor=PAPER_HEX)
    gs = fig.add_gridspec(nr + 1, n, height_ratios=[1.62] * nr + [0.60])
    for r, (batch, title) in enumerate(zip(rows, labels)):
        for c in range(n):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(batch[c].permute(1, 2, 0).clamp(0, 1))
            ax.axis("off")
            if c == 0:
                ax.text(-0.08, 0.5, title, transform=ax.transAxes, ha="right",
                        va="center", fontsize=10.5, color="#1c222b")

    # the key: same two schemes, named
    key = fig.add_subplot(gs[nr, :])
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.axis("off")

    def strip(loc, lead, items):
        """One row of the key. matplotlib lays the swatches out, because it is
        the only thing here that knows how wide the labels render."""
        handles = [Patch(alpha=0.0)] + [
            Patch(facecolor=c, edgecolor="#6b7582", lw=0.8) for c, _ in items]
        labels = [lead] + [n_ for _, n_ in items]
        leg = key.legend(handles, labels, loc=loc, ncol=len(handles),
                         frameon=False, fontsize=10, handlelength=1.3,
                         handletextpad=0.5, columnspacing=1.5,
                         borderpad=0.0, borderaxespad=0.0)
        for t in leg.get_texts():
            t.set_color("#1c222b")
        leg.get_texts()[0].set_color("#6b7582")
        return leg

    truth_key = [(tuple(c.tolist()), n_)
                 for c, n_ in zip(CLASS_COLORS, CLASS_NAMES)]
    key.add_artist(strip("upper left", "truth row — class:", truth_key))
    strip("lower left", "error rows — pixel:",
          [("#ff0000", "model was wrong"),
           ("#8c8c8c", "model was right (photo, dimmed)")])
    fig.tight_layout()
    return save(fig, path)


def fig_skipmaps(path, x, device):
    """The three tensors a skip connection actually carries."""
    model = load_model(lambda: UNet(use_skips=True), "part4_unet_skips_on", device)
    with torch.no_grad():
        s1 = model.enc1(x.to(device))
        s2 = model.enc2(model.pool(s1))
        s3 = model.enc3(model.pool(s2))

    n_show = 6
    levels = [("skip 1\n32 x 96 x 96", s1), ("skip 2\n64 x 48 x 48", s2),
              ("skip 3\n128 x 24 x 24", s3)]
    fig, axes = plt.subplots(3, n_show + 1, figsize=(1.45 * (n_show + 1), 4.9),
                             facecolor=PAPER_HEX)
    for r, (title, feat) in enumerate(levels):
        axes[r, 0].imshow(x[0].permute(1, 2, 0))
        axes[r, 0].axis("off")
        axes[r, 0].text(-0.1, 0.5, title, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=10, color="#1c222b",
                        linespacing=1.4)
        f = feat[0].cpu()
        order = f.flatten(1).var(dim=1).argsort(descending=True)[:n_show]
        for c in range(n_show):
            chan = f[order[c]]
            chan = (chan - chan.min()) / (chan.max() - chan.min() + 1e-8)
            axes[r, c + 1].imshow(chan, cmap="magma")
            axes[r, c + 1].axis("off")
    fig.tight_layout()
    return save(fig, path)


def fig_upsample(path, x, device):
    """What Upsample(x8) does, on a real feature map from the trained FCN.

    The left panel is drawn with nearest-neighbour interpolation so each of the
    144 cells is visibly a cell. The right panel is the same tensor after the
    bilinear Upsample the model actually applies -- bigger, smoother, and
    carrying not one number more than the left one did.
    """
    model = load_model(fcn, "part3_fcn", device)
    trunk = nn.Sequential(*list(model.children())[:-1])     # everything but Upsample
    with torch.no_grad():
        small = trunk(x[:1].to(device))[0].cpu()            # (3, 12, 12) logits
        big = model(x[:1].to(device))[0].cpu()              # (3, 96, 96)
    ch = int(small.flatten(1).var(dim=1).argmax())
    lo, hi = float(small[ch].min()), float(small[ch].max())

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), facecolor=PAPER_HEX)
    for ax, data, title, sub in [
        (axes[0], small[ch], "12 x 12", "what the network actually computed\n"
                                        "144 numbers"),
        (axes[1], big[ch], "96 x 96", "after Upsample(x8)\n9,216 numbers, "
                                      "0 parameters"),
    ]:
        ax.imshow(data, cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(title, fontsize=13, color="#1c222b")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(C_GRID)
        ax.set_xlabel(sub, fontsize=11, color=C_GREY, linespacing=1.4)
    fig.subplots_adjust(wspace=0.25)
    # The arrow between them, in figure coordinates.
    fig.text(0.497, 0.54, "→", fontsize=34, color=C_GREY, ha="center", va="center")
    return save(fig, path)


def fig_iou_schematic(path):
    """The textbook picture: overlap over union, as a fraction."""
    fig, ax = plt.subplots(figsize=(11.4, 4.4), facecolor=PAPER_HEX)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    S = 1.25                                          # square side

    def pair(cx, cy, fill):
        """Two offset squares; `fill` is 'overlap' or 'union'."""
        a = (cx - S * 0.7, cy - S * 0.3, S, S)        # prediction
        b = (cx - S * 0.3, cy - S * 0.7, S, S)        # ground truth
        if fill == "union":
            for r in (a, b):
                ax.add_patch(plt.Rectangle(r[:2], r[2], r[3], facecolor=C_COOL,
                                           edgecolor="none", zorder=2))
        else:
            x0, y0 = max(a[0], b[0]), max(a[1], b[1])
            x1 = min(a[0] + a[2], b[0] + b[2])
            y1 = min(a[1] + a[3], b[1] + b[3])
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       facecolor=C_WARM, edgecolor="none", zorder=2))
        for r, color in ((a, C_COOL), (b, "#1c222b")):
            ax.add_patch(plt.Rectangle(r[:2], r[2], r[3], fill=False, lw=2.0,
                                       edgecolor=color, zorder=3))

    # The fraction, laid out the way it is written: label above the bar, label
    # below it, and the picture of each region out to the right of its label.
    ax.text(0.15, 2.5, "IoU$_c$  =", fontsize=27, color="#1c222b", va="center")
    ax.plot([2.5, 6.6], [2.5, 2.5], color="#1c222b", lw=2.0)
    ax.text(4.55, 2.74, "area of overlap", fontsize=15, color=C_WARM,
            va="bottom", ha="center")
    ax.text(4.55, 2.26, "area of union", fontsize=15, color=C_COOL,
            va="top", ha="center")
    pair(7.6, 3.65, "overlap")
    pair(7.6, 1.35, "union")

    ax.add_patch(plt.Rectangle((9.3, 3.86), 0.30, 0.30, fill=False, lw=2.0,
                               edgecolor=C_COOL))
    ax.text(9.85, 4.01, "what the model predicted for class $c$", fontsize=12.5,
            color="#1c222b", va="center")
    ax.add_patch(plt.Rectangle((9.3, 3.16), 0.30, 0.30, fill=False, lw=2.0,
                               edgecolor="#1c222b"))
    ax.text(9.85, 3.31, "where class $c$ truly is", fontsize=12.5, color="#1c222b",
            va="center")
    ax.text(9.3, 2.35, "Both a miss and a false alarm land\nin the denominator, so a "
                       "class cannot be\nscored well by over-predicting it.",
            fontsize=12, color=C_GREY, va="top", linespacing=1.6)
    return save(fig, path)


def fig_iou_real(path, y, pred, idx=0):
    """The same quantity on a real mask -- computed three times, once per class.

    This is what the subscript c means, made concrete: hold one class up as
    "positive", count the three regions, divide. Then do it again for the next
    class.
    """
    t, p = y[idx], pred[idx]
    fig, axes = plt.subplots(1, N_CLASSES, figsize=(4.0 * N_CLASSES, 4.3),
                             facecolor=PAPER_HEX)
    C_TP, C_FP, C_FN = C_COOL, C_WARM, "#b9bec4"
    for c, ax in enumerate(axes):
        tc, pc = (t == c), (p == c)
        tp = int((tc & pc).sum()); fp = int((~tc & pc).sum()); fn = int((tc & ~pc).sum())
        canvas = torch.ones(3, *t.shape)
        for mask, color in ((tc & pc, C_TP), (~tc & pc, C_FP), (tc & ~pc, C_FN)):
            rgb = torch.tensor([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
            canvas[:, mask] = rgb[:, None]
        ax.imshow(canvas.permute(1, 2, 0))
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(C_GRID)
        ax.set_title(f"$c$ = {CLASS_NAMES[c]}", fontsize=13.5, color="#1c222b")
        ax.set_xlabel(f"{tp:,} / ({tp:,} + {fp:,} + {fn:,})\n"
                      f"IoU$_c$ = {tp / max(1, tp + fp + fn):.3f}",
                      fontsize=11.5, color="#1c222b", linespacing=1.6)
    handles = [plt.Line2D([], [], marker="s", ls="", ms=11, color=col, label=lab)
               for col, lab in ((C_TP, "both  (intersection)"),
                                (C_FP, "predicted, but not there"),
                                (C_FN, "there, but missed"))]
    fig.legend(handles=handles, frameon=False, fontsize=11.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.20, wspace=0.08)
    return save(fig, path)


def _slab(ax, x, y, ch, size, color, label, shape_text, n_slabs=2, drop=0.0,
          label_y=None):
    """One stage, drawn as a couple of offset slabs. Height encodes resolution.

    `drop` staggers the shape caption downwards: adjacent stages have different
    box heights, so a single caption row would have them colliding.
    """
    h = 0.34 + 1.45 * (size / IMAGE_SIZE) ** 0.55
    w = 0.13 * np.log2(max(ch, 2)) + 0.18
    for k in range(n_slabs):
        off = 0.055 * k
        ax.add_patch(plt.Rectangle((x - w / 2 + off, y - h / 2 - off), w, h,
                                   facecolor=color, edgecolor="white", lw=1.0,
                                   alpha=0.95, zorder=3 - k))
    if label:
        ax.text(x, y + h / 2 + 0.16 if label_y is None else label_y, label,
                ha="center", fontsize=11, color="#1c222b", linespacing=1.35)
    if shape_text:
        ax.text(x, -1.62 - drop, shape_text, ha="center", fontsize=9.5,
                family="monospace", color=C_GREY)
    return w


def fig_vgg_vs_ae(path):
    """The classifier you already have, beside the autoencoder -- same left half.

    Both panels are drawn with the same primitive and the same scale, because
    the point is a comparison: the encoders are identical, and only what
    happens after the waist differs. Shapes are this week's 96x96 ones for both
    so the two halves line up; Week 3's classifier did exactly this on 32x32.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2), facecolor=PAPER_HEX)
    enc = [("Conv x2", 32, 96), ("Conv x2", 64, 48), ("Conv x2", 128, 24),
           ("Conv x2", 256, 12)]

    # Two fixed rows for the captions above and two below: the boxes are
    # different heights and close together, so one row of each collides.
    top = lambda i: 1.22 + 0.34 * (i % 2)

    def encoder(ax):
        x = 0.0
        for i, (name, ch, size) in enumerate(enc):
            _slab(ax, x, 0, ch, size, C_COOL, name, f"{ch}x{size}x{size}",
                  drop=0.30 * (i % 2), label_y=top(i))
            x += 1.55
        return x

    ax = axes[0]
    ax.set_facecolor(PAPER_HEX)
    x = encoder(ax)
    # One block for the whole dense head: three cramped boxes would just be
    # three cramped labels, and the point is only that the grid is gone.
    _slab(ax, x + 0.30, 0, 36864, 1, C_GREY, "Flatten + Linear", "36,864 -> 128",
          1, label_y=top(0))
    _slab(ax, x + 1.95, 0, 10, 1, C_WARM, "Linear", "10", 1, drop=0.30,
          label_y=top(1))
    ax.text(x + 1.95, -2.45, "one label\nfor the image", ha="center", fontsize=12.5,
            color=C_WARM, weight="bold", linespacing=1.4)
    ax.set_title("a classifier  (Week 3)", fontsize=14.5, color="#1c222b", loc="left")

    ax = axes[1]
    ax.set_facecolor(PAPER_HEX)
    x = encoder(ax)
    for i, (name, ch, size) in enumerate([("up", 128, 24), ("up", 64, 48),
                                          ("up", 32, 96)]):
        _slab(ax, x, 0, ch, size, "#3f7fb5", name, f"{ch}x{size}x{size}",
              drop=0.30 * (i % 2), label_y=top(i))
        x += 1.55
    _slab(ax, x, 0, 3, 96, C_WARM, "head", "3x96x96", drop=0.30, label_y=top(1))
    ax.text(x, -2.45, "one label\nfor every pixel", ha="center", fontsize=12.5,
            color=C_WARM, weight="bold", linespacing=1.4)
    ax.set_title("an encoder-decoder  (Week 4)", fontsize=14.5, color="#1c222b",
                 loc="left")

    for ax in axes:
        ax.set_xlim(-1.0, 11.4)
        ax.set_ylim(-3.2, 2.0)
        ax.axis("off")
    fig.text(0.5, 0.045, "Identical left halves. The whole difference is what "
                         "happens after the waist.",
             ha="center", fontsize=13, color=C_GREY)
    return save(fig, path)


def fig_psnr_ladder(path, x, recon, measured_db, idx=0):
    """The same photo at several PSNRs, with the autoencoder's own number in line.

    Gaussian noise gives an exact handle on PSNR (for pixels in [0,1] the noise
    variance IS the MSE), so the ladder is built by choosing a target dB and
    solving for sigma. The last panel is not synthetic: it is Part 2's
    reconstruction, at its measured PSNR.
    """
    img = x[idx]
    targets = [32.0, 26.0, 20.0]
    panels = [(img, "original", "infinite")]
    g = torch.Generator().manual_seed(0)
    for db in targets:
        sigma = float(10 ** (-db / 20))                 # MSE = sigma^2
        noisy = (img + torch.randn(img.shape, generator=g) * sigma).clamp(0, 1)
        actual = 10 * np.log10(1.0 / float(((noisy - img) ** 2).mean()))
        panels.append((noisy, "+ noise", f"{actual:.1f} dB"))
    panels.append((recon[idx].clamp(0, 1), "Part 2's autoencoder",
                   f"{measured_db:.2f} dB"))

    fig, axes = plt.subplots(1, len(panels), figsize=(2.35 * len(panels), 3.4),
                             facecolor=PAPER_HEX)
    for ax, (im, label, db) in zip(axes, panels):
        ax.imshow(im.permute(1, 2, 0).clamp(0, 1))
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(C_GRID)
        ax.set_title(label, fontsize=11.5, color="#1c222b")
        ax.set_xlabel(db, fontsize=13, color=C_WARM if "dB" in db else C_GREY,
                      weight="bold")
    fig.subplots_adjust(wspace=0.06, bottom=0.14)
    return save(fig, path)


def _cells(ax, x0, y0, values, cell=1.0, fill=None, fontsize=12, expr=None,
           edge="#1c222b", lw=1.4):
    """Draw a matrix of numbers as a grid of boxes, top-left at (x0, y0).

    `expr` optionally supplies a string per cell (e.g. "2+4+6+4") printed small
    above the total, which is how the sum becomes visible rather than asserted.
    """
    rows, cols = values.shape
    for r in range(rows):
        for c in range(cols):
            x, y = x0 + c * cell, y0 + r * cell
            ax.add_patch(plt.Rectangle((x, y), cell, cell, facecolor=fill or "white",
                                       edgecolor=edge, lw=lw, zorder=2))
            if expr is not None and expr[r][c]:
                ax.text(x + cell / 2, y + cell * 0.32, expr[r][c], ha="center",
                        va="center", fontsize=fontsize * 0.62, color=C_GREY,
                        zorder=3)
                ax.text(x + cell / 2, y + cell * 0.70, f"{values[r, c]:g}",
                        ha="center", va="center", fontsize=fontsize, weight="bold",
                        color="#1c222b", zorder=3)
            else:
                ax.text(x + cell / 2, y + cell / 2, f"{values[r, c]:g}", ha="center",
                        va="center", fontsize=fontsize, color="#1c222b", zorder=3)
    return x0 + cols * cell, y0 + rows * cell


def fig_convt_example(path):
    """One transposed convolution, worked out by hand -- and checked by torch.

    2x2 input, 3x3 kernel, stride 1. Every number here comes from
    F.conv_transpose2d, and the four stamps below are that same operation
    decomposed: one input value at a time.
    """
    inp = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ker = torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    out = F.conv_transpose2d(inp[None, None], ker[None, None])[0, 0]

    # The same result, decomposed: input cell (i, j) stamps ker * inp[i, j] at
    # offset (i, j). Summing the stamps must reproduce `out` -- asserted below,
    # so this figure cannot quietly drift away from what PyTorch does.
    stamps = []
    for i in range(2):
        for j in range(2):
            s = torch.zeros(4, 4)
            s[i:i + 3, j:j + 3] = ker * inp[i, j]
            stamps.append(((i, j), inp[i, j].item(), s))
    assert torch.allclose(sum(s for _, _, s in stamps), out)

    expr = [["" for _ in range(4)] for _ in range(4)]
    for r in range(4):
        for c in range(4):
            terms = [f"{s[r, c]:g}" for _, _, s in stamps if s[r, c] != 0]
            expr[r][c] = "+".join(terms) if len(terms) > 1 else ""

    fig = plt.figure(figsize=(12.4, 5.6), facecolor=PAPER_HEX)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.12)

    ax = fig.add_subplot(gs[0])
    _cells(ax, 0.0, 1.0, inp, fontsize=15)
    ax.text(2.6, 2.0, "×", fontsize=22, ha="center", va="center", color=C_GREY)
    _cells(ax, 3.2, 0.5, ker, fill="#fbdcd6", fontsize=15)
    ax.text(6.8, 2.0, "=", fontsize=22, ha="center", va="center", color=C_GREY)
    _cells(ax, 7.4, 0.0, out, fontsize=13, expr=expr)
    ax.text(1.0, 0.72, "input", ha="center", fontsize=12, color="#1c222b")
    ax.text(4.7, 0.22, "kernel (learned)", ha="center", fontsize=12, color=C_WARM)
    ax.text(9.4, -0.28, "output", ha="center", fontsize=12, color="#1c222b")
    ax.set_xlim(-0.4, 11.8)
    ax.set_ylim(4.6, -0.8)
    ax.axis("off")

    ax = fig.add_subplot(gs[1])
    x = 0.0
    for n, ((i, j), v, s) in enumerate(stamps):
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[i:i + 3, j:j + 3] = True
        for r in range(4):
            for c in range(4):
                ax.add_patch(plt.Rectangle((x + c * 0.62, r * 0.62), 0.62, 0.62,
                                           facecolor="#fbdcd6" if mask[r, c] else "white",
                                           edgecolor="#1c222b", lw=1.0, zorder=2))
                ax.text(x + c * 0.62 + 0.31, r * 0.62 + 0.31, f"{s[r, c]:g}",
                        ha="center", va="center", fontsize=9.5,
                        color="#1c222b" if mask[r, c] else "#c9ccd1", zorder=3)
        ax.text(x + 1.24, -0.28, f"{v:g} × kernel,  at ({i}, {j})", ha="center",
                fontsize=11, color="#1c222b")
        x += 2.48
        if n < 3:
            ax.text(x + 0.26, 1.24, "+", fontsize=19, ha="center", va="center",
                    color=C_GREY)
            x += 0.78
    ax.text(x + 0.30, 1.24, "= the grid above", fontsize=12.5, va="center",
            color=C_WARM, weight="bold")
    ax.set_xlim(-0.3, 14.2)
    ax.set_ylim(3.1, -0.8)
    ax.axis("off")
    return save(fig, path)


def fig_convt_demo(path):
    """Where checkerboard comes from, measured rather than asserted.

    Set every weight to 1 and feed an image of ones. Each output pixel then
    reports exactly how many input cells contributed to it. A uniform map means
    every output pixel is built the same way; a patterned one is the
    checkerboard, visible before any training has happened.
    """
    ones = torch.ones(1, 1, 6, 6)
    cases = []
    with torch.no_grad():
        for label, mod in [
            ("ConvTranspose2d\nk=2, s=2", nn.ConvTranspose2d(1, 1, 2, stride=2)),
            ("ConvTranspose2d\nk=3, s=2, pad=1", nn.ConvTranspose2d(
                1, 1, 3, stride=2, padding=1, output_padding=1)),
            ("Upsample(x2) + Conv3x3", nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(1, 1, 3, padding=1))),
        ]:
            for p in mod.parameters():
                nn.init.ones_(p) if p.dim() > 1 else nn.init.zeros_(p)
            for m in mod.modules():                    # bias to zero, weights to one
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.ones_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            cases.append((label, mod(ones)[0, 0]))

    fig, axes = plt.subplots(1, len(cases), figsize=(4.1 * len(cases), 4.3),
                             facecolor=PAPER_HEX)
    for ax, (label, out) in zip(axes, cases):
        core = out[2:-2, 2:-2]                          # ignore the padded border
        ax.imshow(out, cmap="magma", interpolation="nearest", vmin=0,
                  vmax=float(out.max()))
        for (r, c), v in np.ndenumerate(out.numpy()):
            ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v < out.max() * 0.65 else "#1c222b")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(C_GRID)
        ax.set_title(label, fontsize=12.5, color="#1c222b", linespacing=1.4)
        spread = f"{core.min():.0f} to {core.max():.0f}"
        even = core.min() == core.max()
        ax.set_xlabel(f"contributions per pixel: {spread}",
                      fontsize=11.5, weight="bold",
                      color=C_GREY if even else C_WARM)
    fig.subplots_adjust(wspace=0.12, bottom=0.14)
    return save(fig, path)


# ------------------------------------------------------------------- maths
# python-pptx cannot write PowerPoint equations, so each formula is typeset
# with matplotlib mathtext and embedded as a transparent PNG at native size.

MATH_DPI = 300
_math_seq = [0]

# The two metric definitions, kept here so the slide that uses each one stays
# readable.
IOU_EQ = (r"$\mathrm{IoU}_c \;=\; "
          r"\frac{|\,\hat{y} = c \;\wedge\; y = c\,|}"
          r"{|\,\hat{y} = c \;\vee\; y = c\,|} \;=\; "
          r"\frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FP}_c + \mathrm{FN}_c}$")
MIOU_EQ = (r"$\mathrm{mIoU} \;=\; \frac{1}{C}\sum_{c=1}^{C} \mathrm{IoU}_c "
           r"\;=\; \frac{1}{3}\left(\mathrm{IoU}_{\mathrm{pet}} + "
           r"\mathrm{IoU}_{\mathrm{background}} + "
           r"\mathrm{IoU}_{\mathrm{border}}\right)$")


def math_png(expr, path, fontsize=30, color="#1c222b"):
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, expr, fontsize=fontsize, color=color)
    fig.savefig(path, dpi=MATH_DPI, transparent=True, bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    return path


def math_block(slide, expr, x, y, tmp, fontsize=30, color="#1c222b", center=False):
    _math_seq[0] += 1
    path = os.path.join(tmp, f"eq{_math_seq[0]}.png")
    math_png(expr, path, fontsize=fontsize, color=color)
    px_h, px_w = plt.imread(path).shape[:2]
    w = int(px_w / MATH_DPI * 914400)
    h = int(px_h / MATH_DPI * 914400)
    # A long equation rendered at its native size can be wider than the slide;
    # shrink it to fit rather than letting it hang over the edge.
    limit = int(W - 2 * M)
    if w > limit:
        h = int(h * limit / w)
        w = limit
    left = int((W - w) / 2) if center else x
    slide.shapes.add_picture(path, left, y, width=Emu(w), height=Emu(h))
    return Emu(w), Emu(h)


# ------------------------------------------------------------ slide kit


def blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = PAPER
    return s


def text(slide, x, y, w, h, runs_spec, align=PP_ALIGN.LEFT, spacing=1.0):
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
         [(title, 29, True, BLUE, SANS)])
    rule(slide, Inches(1.62))


def bullets(slide, items, top=Inches(2.0), size=17, width=None, left=M):
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
    """Place a figure centred, scaled to `height` -- unless that would run off
    the sides, in which case width wins. A wide figure asked for by height is
    the easiest way to push a picture past the slide edge, and the overhang is
    invisible until someone opens the deck."""
    pic = slide.shapes.add_picture(path, Emu(0), top, height=height)
    limit = W - Inches(0.5)
    if pic.width > limit:
        pic.height = int(pic.height * limit / pic.width)
        pic.width = int(limit)
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
        size, bold, color = {"lead": (18, True, INK), "text": (17, False, INK),
                             "note": (15, False, MUTED)}[kind]
        text(s, M, y, W - 2 * M, Inches(0.5), [(blk[1], size, bold, color, SANS)],
             spacing=1.15)
        chars_per_line = int((W - 2 * M) / Emu(1) / 914400 * 72 / (size * 0.50))
        lines = max(1, -(-len(blk[1]) // max(1, chars_per_line)))
        y = Emu(int(y) + Inches(lines * size * 1.15 / 72 + 0.16))
    return s


def code_slide(prs, kicker, title, source, lines, note=None, size=14.5):
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


def table_slide(prs, kicker, title, headers, rows, note=None, widths=None,
                top=Inches(2.05)):
    s = blank(prs)
    header(s, kicker, title)
    ncol = len(headers)
    widths = widths or [1.0] * ncol
    scale = (W - 2 * M) / sum(widths)
    xs, x = [], M
    for w in widths:
        xs.append((x, int(w * scale)))
        x += int(w * scale)
    for (x0, w), h in zip(xs, headers):
        text(s, x0, top, w, Inches(0.4), [(h, 13, True, MUTED, SANS)])
    rule(s, Emu(int(top) + Inches(0.45)))
    for i, row in enumerate(rows):
        ry = Emu(int(top) + Inches(0.57) + Inches(0.46) * i)
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


def figure_slide(prs, kicker, title, img, cap=None, height=Inches(4.3),
                 top=Inches(1.95)):
    s = blank(prs)
    header(s, kicker, title)
    picture(s, img, top=top, height=height)
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
        text(s, M, Inches(4.6), Inches(9.8), Inches(1.2),
             [(sub, 18, False, MUTED, SANS)], spacing=1.2)
    return s


# ------------------------------------------------------------------ deck


def build(n_test, force):
    m = measure(n_test, force=force)
    seg, recon, dist = m["seg"], m["recon"], m["distance"]
    curves = load_curves()
    device = get_device()

    tmp = tempfile.mkdtemp(prefix="week4slides_")
    f = lambda n: os.path.join(tmp, n + ".png")

    x, y = sample_batch(n=6)
    p_fcn = predict("part3_fcn", fcn, x, device)
    p_enc = predict("part3_encdec", SegAutoencoder, x, device)
    p_off = predict("part4_unet_skips_off", lambda: UNet(use_skips=False), x, device)
    p_on = predict("part4_unet_skips_on", lambda: UNet(use_skips=True), x, device)

    with torch.no_grad():
        ae_convt = load_model(lambda: ConvAutoencoder(mode="convt"),
                              "part2_ae_convt", device)
        ae_up = load_model(lambda: ConvAutoencoder(mode="upsample"),
                           "part2_ae_upsample", device)
        r_convt, r_up = ae_convt(x.to(device)).cpu(), ae_up(x.to(device)).cpu()

    F_ = {
        "task": fig_task(f("task"), x, y, m["class_share"]),
        "arch_fcn": fig_arch_fcn(f("arch_fcn")),
        "upsample": fig_upsample(f("upsample"), x, device),
        "iou_sch": fig_iou_schematic(f("iou_sch")),
        "iou_real": fig_iou_real(f("iou_real"), y, p_enc),
        "vgg_ae": fig_vgg_vs_ae(f("vgg_ae")),
        "convt": fig_convt_demo(f("convt")),
        "convt_example": fig_convt_example(f("convt_example")),
        "psnr": fig_psnr_ladder(f("psnr"), x, r_convt,
                                recon["part2_ae_convt"]["psnr"]),
        "arch_ae": fig_arch_ae(f("arch_ae")),
        "arch_unet": fig_arch_unet(f("arch_unet")),
        "pred1": fig_grid(f("pred1"), [x, colorize_batch(y), colorize_batch(p_fcn)],
                          ["photo", "truth", "part 1   Conv1x1 + Upsample(x8)"]),
        "recon": fig_grid(f("recon"), [x, r_convt, r_up],
                          ["photo", "ConvTranspose2d", "Upsample + Conv"]),
        "waist": fig_waist(f("waist")),
        "pred3": fig_grid(f("pred3"), [x, colorize_batch(y), colorize_batch(p_fcn),
                                       colorize_batch(p_enc)],
                          ["photo", "truth",
                           "part 1   Conv1x1 + Upsample(x8)",
                           "part 3   ConvTranspose2d decoder"]),
        "pretrain": fig_pretrain(f("pretrain"), seg),
        "curves3": fig_curves(f("curves3"), curves,
                              ["part3_fcn", "part3_encdec", "part3_pretrained"],
                              ["part 1   Conv1x1 + Upsample(x8)",
                               "part 3   ConvTranspose2d decoder",
                               "  + AE pretraining"]),
        "skipmaps": fig_skipmaps(f("skipmaps"), x, device),
        "pred4": fig_grid(f("pred4"), [x, colorize_batch(y), colorize_batch(p_enc),
                                       colorize_batch(p_off), colorize_batch(p_on)],
                          ["photo", "truth", "encoder-decoder", "skips OFF",
                           "skips ON"]),
        "zoom": fig_zoom(f("zoom"), x, y, [p_enc, p_off, p_on],
                         ["encoder-decoder", "skips OFF", "skips ON"]),
        "curves4": fig_curves(f("curves4"), curves,
                              ["part4_encdec", "part4_unet_skips_off",
                               "part4_unet_skips_on"],
                              ["encoder-decoder", "U-Net, skips OFF",
                               "U-Net, skips ON"]),
        "errors": fig_errors(f("errors"), x, y, [p_off, p_on],
                             ["skips OFF", "skips ON"]),
        "distance": fig_distance(f("distance"), dist),
        "tio4": fig_trackio(
            f("tio4"),
            [("encoder-decoder (part3)", "encoder-decoder"),
             ("U-Net, skips OFF", "U-Net, skips OFF"),
             ("U-Net, skips ON", "U-Net, skips ON")],
            [("test/mIoU", "mean IoU", "mean IoU, per epoch"),
             ("test/iou_border", "border-class IoU", "border class, per epoch")]),
        "tio_instr": fig_instrument(f("tio_instr"), "U-Net, skips ON"),
        "bars": fig_bars(f("bars"),
                         seg, ["part3_fcn", "part3_encdec", "part4_unet_skips_off",
                               "part4_unet_skips_on"],
                         ["part 1   classifier -> FCN", "part 3   encoder-decoder",
                          "part 4   U-Net, skips OFF", "part 4   U-Net, skips ON"]),
    }

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    q = lambda stem, k="miou": seg[stem][k]
    fmt = lambda v: f"{v:.3f}"
    n_px = IMAGE_SIZE * IMAGE_SIZE
    share = m["class_share"]

    # ---------------------------------------------------------- 1. title
    s = blank(prs)
    text(s, M, Inches(2.3), W - 2 * M, Inches(0.4),
         [("WEEK 4  ·  SEGMENTATION", 15, True, ACCENT, SANS)])
    text(s, M, Inches(2.75), W - 2 * M, Inches(1.4),
         [("One label per pixel", 54, True, BLUE, SANS)])
    text(s, M, Inches(4.1), Inches(10.2), Inches(1.6),
         [("From a classifier to a U-Net, one change at a time.", 21, False, INK, SANS),
          (f"Oxford-IIIT Pet at 96x96. Every number measured on the same "
           f"{m['n_test']} test images.", 17, False, MUTED, SANS)], spacing=1.25)
    rule(s, Inches(5.9), w=Inches(3.0))

    # ------------------------------------------------------- 2. the task
    content_slide(
        prs, "the task", "Week 3 predicted one label. This week predicts 9,216.",
        [("Same photo in. A different shape out.", "lead"),
         "",
         ("  week 3   image  ->  1 label            (which of 10 classes)", "mono"),
         (f"  week 4   image  ->  96 x 96 labels     ({n_px:,} of them, "
          f"one per pixel)", "mono"),
         "",
         "Every pixel gets its own answer: 0 = pet, 1 = background, 2 = border. "
         "The labels ship with the dataset as a trimap — a human painted them.",
         "",
         ("That one change to the output shape forces everything else in this "
          "session. Nothing here is a new kind of network; it is the same "
          "convolutions, the same loss, and the same training loop from Week 3.",
          "note"),
         ],
    )

    figure_slide(
        prs, "the task", "The data: a photo, and a label for every one of its pixels",
        F_["task"],
        f"Background is {share[1] * 100:.0f}% of all pixels and the border ring is "
        f"only {share[2] * 100:.0f}%. Hold on to those two numbers — they decide "
        f"which metric we are allowed to use.",
        height=Inches(3.9),
    )

    # -------------------------------------------------- 3. the whole story
    figure_slide(
        prs, "the whole session on one slide", "Four models, one test set",
        F_["bars"],
        "Each bar is the model of one part. The blue bar is the headline number; "
        "the orange bar is the border class, where the argument of this session "
        "actually lives.",
        height=Inches(4.1),
    )

    table_slide(
        prs, "the whole session on one slide", "The same four models, as numbers",
        ["model", "params", "acc", "mIoU", "pet", "bg", "border"],
        [[seg[k]["label"], f"{seg[k]['params']:,}", fmt(seg[k]["acc"]),
          (fmt(seg[k]["miou"]),)] + [fmt(v) for v in seg[k]["iou"]]
         for k in ["part3_fcn", "part3_encdec", "part4_unet_skips_off",
                   "part4_unet_skips_on"]],
        note=f"All four trained for 15 epochs at lr 1e-3 on all 3,680 labelled "
             f"images, and scored here on {m['n_test']} held-out test images. "
             f"The rest of the deck unpacks one row at a time: the problem, the "
             f"idea, the maths, then the evidence.",
        widths=[2.4, 1.1, 0.8, 0.8, 0.8, 0.8, 0.8],
    )

    # ============================================== ACT I — part 1
    section_slide(
        prs, "part 1", "A classifier is almost a segmenter",
        "The problem: a classifier's head throws away WHERE. Delete it, and "
        "what is left already does the job.",
    )

    content_slide(
        prs, "part 1 · the problem", "What exactly is incompatible?",
        [("Week 3's CNN ended: Conv stack -> Flatten -> Linear(1024->128) -> "
          "Linear(128->10).", "lead"),
         "",
         "Flatten is the layer that is wrong for this task, and it is wrong in a "
         "specific way: after it, 'ear-shaped thing' and 'ear-shaped thing in the "
         "top-left corner' are the same 1,024 numbers in a different order. The "
         "position is still in there, but only as an arbitrary index — and the "
         "Linear that follows has no reason to treat neighbouring indices as "
         "neighbouring pixels.",
         "",
         "It also fixes the input size forever: Linear(1024 -> 128) needs exactly "
         "1,024 numbers, so the network only accepts 32x32 images.",
         "",
         (">> Everything else a classifier does — convolutions, pooling, "
          "BatchNorm, the loss — is already what a segmenter wants.", "lead"),
         ],
        top=Inches(1.92),
    )

    code_slide(
        prs, "part 1 · the idea", "Four lines of diff, and it is a segmentation model",
        "week 3 part 6 classifier   ->   part1_classifier_to_segmenter.py",
        ["Conv -> BN -> ReLU -> MaxPool      Conv -> BN -> ReLU -> MaxPool    x3",
         ("Flatten                           (deleted)",),
         ("Linear(1024 -> 128) -> ReLU       (deleted)",),
         ("Linear(128 -> 10)                 Conv1x1(128 -> 3)",),
         ("(nothing)                         Upsample(x8)",),
         "CrossEntropyLoss on (N, 10)       CrossEntropyLoss on (N, 3, H, W)",
         ],
        note="The trunk is untouched. Even the loss is untouched — "
             "nn.CrossEntropyLoss already accepts (N, C, H, W) logits against "
             "(N, H, W) integer targets. This is the 'fully convolutional' idea, "
             "and in 2015 it was a paper.",
    )

    figure_slide(
        prs, "part 1 · the idea", "And what is that Upsample line, exactly?",
        F_["upsample"],
        "One channel of the real network's output, before and after its last "
        "layer. Upsample resizes: every one of the 9,216 numbers on the right is "
        "a weighted average of the 144 on the left. Bigger, smoother, and not "
        "one bit more detailed.",
        height=Inches(3.9),
    )

    figure_slide(
        prs, "part 1 · the model", "The shapes, from a real forward pass",
        F_["arch_fcn"],
        "Box height is spatial size, box width is channel count. Three poolings "
        "take 96 down to 12; the 1x1 conv labels that 12x12 grid; Upsample "
        "stretches it back.",
        height=Inches(4.15),
    )

    flow_slide(
        prs, "part 1 · the maths", "A 1x1 convolution IS the Linear head, applied everywhere",
        [("lead", "A Linear layer maps one vector to another. A 1x1 convolution "
                  "does exactly that, once per spatial position, with the SAME "
                  "weights at every position:"),
         ("eq", r"$y_{h,w} \;=\; W\,x_{h,w} \;+\; b, \qquad "
                r"W \in \mathbb{R}^{3 \times 128}, \quad "
                r"\forall\, (h,w) \in 12 \times 12$", 26),
         ("text", "So it costs 128 x 3 + 3 = 387 parameters, against the 131,200 "
                  "in the Linear layer it replaced — and unlike that layer, it "
                  "does not care how many positions there are."),
         ("gap", 0.08),
         ("lead", "The loss needs no change either. Cross-entropy is applied at "
                  "each pixel independently and averaged:"),
         ("eq", r"$\mathcal{L} \;=\; -\frac{1}{HW}\sum_{h,w} "
                r"\log \frac{\exp\!\left(z_{h,w,\,y_{h,w}}\right)}"
                r"{\sum_{c} \exp\!\left(z_{h,w,c}\right)}$", 26),
         ("note", "Week 2's formula, inside a sum over pixels. Segmentation is "
                  "classification run 9,216 times per image — with the whole "
                  "picture available as context."),
         ],
        tmp,
    )

    # --------------------------------------------------- the metric
    content_slide(
        prs, "part 1 · the metric", "Pixel accuracy is a trap, so we retire it here",
        [(f"{share[1] * 100:.0f}% of the pixels are background. A model that "
          f"predicts 'background' everywhere and never looks at the image scores "
          f"{fmt(seg['constant']['acc'])} pixel accuracy.", "lead"),
         "",
         "Worse, the border class — the thin ring where every interesting error "
         f"lives — is {share[2] * 100:.0f}% of the pixels. A model can ignore it "
         "completely and lose almost nothing on accuracy.",
         "",
         ("  always 'background'     acc " + fmt(seg["constant"]["acc"]) +
          "     mIoU " + fmt(seg["constant"]["miou"]), "mono"),
         "",
         "Intersection over union scores each class on its own, so a class the "
         "model never predicts scores zero no matter how rare it is.",
         ],
    )

    # Two slides: what the ratio is, then what the subscript c is.
    sl = blank(prs)
    header(sl, "part 1 · the maths", "Intersection over union")
    picture(sl, F_["iou_sch"], top=Inches(1.74), height=Inches(3.05))
    math_block(sl, IOU_EQ, M, Inches(5.00), tmp, fontsize=26, center=True)
    text(sl, M, Inches(6.30), W - 2 * M, Inches(0.8),
         [("c is a CLASS INDEX. There is no single IoU: there is one per class, "
           "and the whole calculation is run once with 'pet' held up as the "
           "positive class, once with 'background', once with 'border'.",
           15.5, False, MUTED, SANS)], spacing=1.15)

    sl = blank(prs)
    header(sl, "part 1 · the maths", "Three classes, three IoUs, and then the mean")
    picture(sl, F_["iou_real"], top=Inches(1.72), height=Inches(3.25))
    math_block(sl, MIOU_EQ, M, Inches(5.20), tmp, fontsize=23, center=True)
    text(sl, M, Inches(6.24), W - 2 * M, Inches(0.9),
         [("Unweighted, so the 12% class counts as much as the 58% one -- which "
           "is the whole reason we changed metric. Read the border panel: the "
           "same prediction that looks excellent on 'background' is mediocre "
           "there. Computed from ONE confusion matrix over the whole test set, "
           "never by averaging per-batch IoUs.", 15.5, False, MUTED, SANS)],
         spacing=1.15)

    fcn_gain = q("part3_fcn") - seg["constant"]["miou"]
    figure_slide(
        prs, "part 1 · the evidence",
        f"mIoU {fmt(seg['constant']['miou'])} -> {fmt(q('part3_fcn'))} for "
        f"{seg['part3_fcn']['params']:,} parameters",
        F_["pred1"],
        f"It finds the animal. Read the number against the "
        f"{fmt(seg['constant']['miou'])} floor, not against pixel accuracy: "
        f"+{fcn_gain:.3f} mIoU from deleting two layers.",
        height=Inches(3.5),
    )

    content_slide(
        prs, "part 1 · the limit", "Now look at the pictures, not the number",
        [("Ear tips round off. Thin legs fade out. Holes open in the middle of an "
          "animal. Nothing in the output is finer than about 8 pixels.", "lead"),
         "",
         "The reason is the one thing we did not replace. Three poolings take "
         "96x96 down to 12x12, and that is where the decision is made — one vote "
         "per 8x8 block of photo. Upsample has no parameters: it can make the map "
         "bigger, it cannot make it more detailed.",
         "",
         "And we cannot simply pool less. Pooling is what gives this model its "
         "context: at 12x12 a 3x3 kernel covers a quarter of the photo, which is "
         "how it knows it is looking at an animal at all rather than at a brown "
         "patch.",
         "",
         (">> Context and resolution pull in opposite directions. The rest of "
          "the session is one idea for having both.", "lead"),
         ],
        top=Inches(1.92),
    )

    # ============================================== ACT II — part 2
    section_slide(
        prs, "part 2", "Learn the way back up",
        "The problem: Upsample stretches, it does not sharpen. Build a decoder "
        "instead — on the easiest possible target, no labels at all.",
    )

    content_slide(
        prs, "part 2 · the idea", "Make the way up learned, and train it without labels",
        [("Replace the fixed stretch with convolutions that have weights, and "
          "train them on a task where the answer is free: the photo is its own "
          "target.", "lead"),
         "",
         ("  encoder   3 x 96 x 96   ->   256 x 12 x 12      (three poolings, as in part 1)",
          "mono"),
         ("  decoder   256 x 12 x 12 ->   3 x 96 x 96        (three upsamplings, new)",
          "mono"),
         ("  loss      MSELoss(reconstruction, original)", "mono"),
         "",
         "That is an autoencoder, and we are here for exactly two reasons: it is "
         "the shortest path to understanding the decoder half, and its failure "
         "mode is the whole argument of part 4.",
         "",
         ("The training loop does not change — Week 1 Part 3's four steps, with "
          "loss_fn(model(x), y) where y happens to be x. This is the first time "
          "in the course that no label file is involved anywhere.", "note"),
         ],
        top=Inches(1.92),
    )

    figure_slide(
        prs, "part 2 · the model", "The half you already have, and the half that is new",
        F_["vgg_ae"],
        "A VGG-style classifier and an encoder-decoder, drawn at the same scale. "
        "Box height is spatial size, box width is channel count. The encoders "
        "are the same network; a classifier then collapses the grid into one "
        "vector, and an encoder-decoder builds it back up.",
        height=Inches(4.15),
    )

    figure_slide(
        prs, "part 2 · the model", "Symmetric: every pooling on the left has an upsampling on the right",
        F_["arch_ae"],
        "Shapes from a real forward pass. Remember this picture — part 4 changes "
        "three lines of it and nothing else.",
        height=Inches(4.2),
    )

    figure_slide(
        prs, "part 2 · the maths", "ConvTranspose2d: what it actually computes",
        F_["convt_example"],
        "A 2x2 input, a 3x3 kernel. Every input value scales the WHOLE kernel and "
        "stamps it at that value's own offset; where the stamps overlap, they add. "
        "Run it backwards in your head and you have ordinary convolution -- which "
        "is exactly where the name comes from.",
        height=Inches(4.35),
        top=Inches(1.82),
    )

    content_slide(
        prs, "part 2 · the maths", "What a learned upsampler buys over a fixed one",
        [("Compare that against Part 1's Upsample(x8), which had no kernel at "
          "all.", "lead"),
         "",
         ("  IT HAS WEIGHTS      interpolation can only average the numbers", "mono"),
         ("                      already there. A learned kernel can be trained", "mono"),
         ("                      to sharpen, to fill in, to invert the blur.", "mono"),
         "",
         ("  ONE KERNEL PER      a different patch for every (in, out) channel", "mono"),
         ("  CHANNEL PAIR        pair: a crisp edge for one feature, a smooth", "mono"),
         ("                      wash for another, decided by training.", "mono"),
         "",
         ("  ONE OPERATION       it resizes and recombines features in a single", "mono"),
         ("                      step -- the resize route needs a conv after it", "mono"),
         ("                      anyway to do anything useful.", "mono"),
         "",
         ("And the catch, which the next slide measures: overlapping stamps mean "
          "some output pixels collect more terms than their neighbours whenever "
          "the stride does not divide the kernel size.", "note"),
         ],
        top=Inches(1.88),
    )

    figure_slide(
        prs, "part 2 · the maths", "Set every weight to 1, and count",
        F_["convt"],
        "Feed an image of ones through each upsampler with all weights set to 1: "
        "every output pixel then reports how many input cells reached it. The rule "
        "is simple — the stamps tile evenly only when the stride divides the kernel "
        "size. k=3 with s=2 does not, and there is the checkerboard, visible in the "
        "architecture itself before any training has happened.",
        height=Inches(4.0),
    )

    # PSNR gets its own slide: the formula is two symbols, but what a dB means
    # is not obvious until you see the same picture at several of them.
    sl = blank(prs)
    header(sl, "part 2 · the maths", "PSNR: the same MSE, on a log scale")
    math_block(sl, r"$\mathrm{PSNR} \;=\; 10\,\log_{10}\!"
                   r"\left(\frac{\mathrm{MAX}^2}{\mathrm{MSE}}\right) "
                   r"\;=\; 10\,\log_{10}\!\left(\frac{1}{\mathrm{MSE}}\right)"
                   r"\ \mathrm{dB}$",
               M, Inches(1.80), tmp, fontsize=25, center=True)
    text(sl, M, Inches(2.72), W - 2 * M, Inches(0.9),
         [("MAX is the largest possible pixel value, which is 1 here because our "
           "images live in [0, 1] — so the ratio is just 1/MSE. It carries no "
           "information MSE did not, and it is worth having anyway: a ratio in "
           "decibels is easier to feel. +3 dB means the error energy HALVED; "
           "+10 dB means it fell to a tenth. And because it is a log, the same "
           "1 dB is a much bigger visual change at 30 dB than at 20.",
           15.5, False, INK, SANS)], spacing=1.15)
    picture(sl, F_["psnr"], top=Inches(4.28), height=Inches(2.55))
    text(sl, M, H - Inches(0.62), W - 2 * M, Inches(0.5),
         [("Same photo, degraded to order. And the sting in the tail: Part 2's "
           "autoencoder scores BELOW the visibly noisy 26 dB image while looking "
           "far cleaner — PSNR measures how big the error is, not what kind.",
           14, False, MUTED, SANS)], align=PP_ALIGN.CENTER, spacing=1.1)

    d_convt, d_up = recon["part2_ae_convt"], recon["part2_ae_upsample"]
    figure_slide(
        prs, "part 2 · the evidence",
        "Both work. Both are blurry in exactly the same way.",
        F_["recon"],
        f"ConvTranspose2d: {d_convt['params']:,} params, "
        f"{d_convt['psnr']:.2f} dB.   Upsample+Conv: {d_up['params']:,} params, "
        f"{d_up['psnr']:.2f} dB. Worth having, and worth keeping in proportion — "
        f"it is also the bigger model.",
        height=Inches(3.5),
    )

    content_slide(
        prs, "part 2 · the real problem", "The waist is not a compression bottleneck",
        [("Count the numbers. The bottleneck holds MORE of them than the image "
          "does:", "lead"),
         "",
         ("  input       3 x 96 x 96   =  27,648 numbers", "mono"),
         ("  waist     256 x 12 x 12   =  36,864 numbers     (1.3x MORE)", "mono"),
         "",
         "So whatever is being lost, it is not capacity and it is not count. It "
         "is WHERE: 144 positions have to describe 9,216 pixels, so each one "
         "speaks for an 8x8 block. Everything finer than that block — a whisker, "
         "the exact edge of an ear — has no representation left to live in.",
         "",
         ("If that diagnosis is right, then widening the waist should sharpen the "
          "picture even as the parameter count FALLS. That is a testable "
          "prediction, so let us test it.", "note"),
         ],
        top=Inches(1.92),
    )

    figure_slide(
        prs, "part 2 · the evidence", "The control experiment: same model, three waists",
        F_["waist"],
        "Each extra pooling costs 2–4 dB and pays for it with four times the "
        "parameters. The prediction holds: the smallest waist has the most "
        "weights and the worst picture.",
        height=Inches(4.0),
    )

    table_slide(
        prs, "part 2 · the evidence", "Resolution is what is missing, not capacity",
        ["poolings", "waist", "params", "MSE", "PSNR (dB)"],
        [[f"{d}", f"{w}x{w}", f"{count_params(VariableWaistAutoencoder(depth=d)):,}",
          f"{mse:.4f}", (f"{db:.2f}",)] for d, w, mse, db in WAIST],
        note="python part2b_waist_sweep.py  (params recomputed "
             "live from the model). And we cannot just keep the 24x24 waist: "
             "part 1 showed we need the deep pooling for context. We need the "
             "deep waist AND the detail. Part 4 stops choosing.",
        widths=[1.0, 1.0, 1.3, 1.0, 1.2],
    )

    # ============================================== ACT III — part 3
    section_slide(
        prs, "part 3", "Point the decoder at the real task",
        "Two lines change, and both follow from one fact: the output is now a "
        "class per pixel, not a colour per pixel.",
    )

    code_slide(
        prs, "part 3 · the idea", "The same network, with its last layer re-read",
        "part3_encoder_decoder.py  ·  class SegAutoencoder(ConvAutoencoder)",
        ["class SegAutoencoder(ConvAutoencoder):",
         "    def __init__(self, n_classes=3, base=32, mode=\"convt\"):",
         "        super().__init__(in_ch=3, out_ch=n_classes, base=base, ...)",
         ("        # was: Sequential(Conv1x1(32 -> 3), Sigmoid)",),
         ("        self.head = nn.Conv2d(base, n_classes, kernel_size=1)",),
         "",
         ("loss_fn = nn.CrossEntropyLoss()      # was nn.MSELoss()",),
         ],
        note="Subclassing rather than rewriting is the point: encoder, "
             "bottleneck, upsampling and decoder are character-for-character the "
             "network that was rebuilding cat photos a minute ago. The Sigmoid "
             "goes because those 3 numbers per pixel are now logits, and "
             "CrossEntropyLoss applies log_softmax itself — squashing them first "
             "is the commonest bug here.",
    )

    figure_slide(
        prs, "part 3 · the evidence",
        f"A learned decoder is worth +{q('part3_encdec') - q('part3_fcn'):.3f} mIoU "
        f"over stretching the logits",
        F_["pred3"],
        f"Part 1's fixed Upsample(x8) {fmt(q('part3_fcn'))}  ->  Part 2's "
        f"ConvTranspose2d decoder {fmt(q('part3_encdec'))}. Same encoder, same "
        f"loss, same 15 epochs: the only thing that changed is whether the way "
        f"back up has weights. The border class more than doubles, "
        f"{fmt(seg['part3_fcn']['iou'][2])} -> {fmt(seg['part3_encdec']['iou'][2])}.",
        height=Inches(3.9),
    )

    figure_slide(
        prs, "part 3 · the evidence", "The same thing, seen during training",
        F_["curves3"],
        "Training cross-entropy per epoch, from the runs that wrote the "
        "checkpoints. The fixed-Upsample model flattens out early and stays "
        "there — it is not short of epochs, it is short of resolution, and no "
        "amount of further training buys that back.",
        height=Inches(4.0),
    )

    content_slide(
        prs, "part 3 · a free question", "Part 2 trained this body with no labels. Why start from random?",
        [("Loading part 2's autoencoder weights transfers 104 of 106 tensors — "
          "everything except the head, which predicted colours and is no use "
          "here.", "lead"),
         "",
         ("  model.load_state_dict(torch.load(\"part2_ae_convt.pt\"), strict=False)",
          "mono"),
         "",
         "strict=False is what makes it work: every encoder and decoder key "
         "matches, the head keys do not, so the body loads and the head stays "
         "random. Those weights already know what fur looks like and where edges "
         "are — nothing about that knowledge is specific to reconstruction.",
         "",
         ("This is self-supervised pretraining, and it is the reason anybody "
          "cares about autoencoders in practice. The unlabelled photo is free; "
          "the trimap a human painted by hand is not.", "note"),
         ],
        top=Inches(1.92),
    )

    sl = blank(prs)
    header(sl, "part 3 · the evidence", "It buys nothing — until labels are scarce")
    picture(sl, F_["pretrain"], top=Inches(1.72), height=Inches(3.15))
    text(sl, M, Inches(5.02), W - 2 * M, Inches(2.0),
         [("What \"500 labels\" means, because it is the hinge of this chart:",
           17, True, INK, SANS),
          ("A label here IS the ground-truth trimap — one mask a human painted, "
           "pixel by pixel, for one photo. --n-train 500 keeps a fixed random 500 "
           "of the 3,680 (photo, mask) pairs and simply never shows the model the "
           "other 3,180 masks.",
           16, False, INK, SANS),
          ("The asymmetry is the point: Part 2's autoencoder still trained on all "
           "3,680 PHOTOS, because it never needed a mask. So the orange bar is "
           "500 masks + 3,680 unlabelled photos, against the blue bar's 500 masks "
           "alone — and that is the trade every real project faces. Photographs "
           "are free; the human's afternoon is not.",
           16, False, MUTED, SANS)],
         spacing=1.14)

    content_slide(
        prs, "part 3 · what is still wrong", "It knows where the boundary roughly is, and cannot commit",
        [("The border class is the worst of the three by a wide margin: "
          f"{fmt(seg['part3_encdec']['iou'][2])}, against "
          f"{fmt(seg['part3_encdec']['iou'][0])} for the pet and "
          f"{fmt(seg['part3_encdec']['iou'][1])} for the background.", "lead"),
         "",
         "It is the same complaint as part 2's blurry whiskers, in a different "
         "currency. There the lost detail cost us texture and looked soft; here "
         "it costs labelled pixels and is measurable.",
         "",
         "And the cause has not changed: the information needed to place that "
         "boundary was destroyed at the 12x12 waist, before the decoder ever ran.",
         "",
         (">> So stop destroying it.", "lead"),
         ],
    )

    # ============================================== ACT IV — part 4
    section_slide(
        prs, "part 4", "U-Net",
        "On the way down, enc1 computed a 32 x 96 x 96 map that still had the "
        "whiskers in it. We pooled it, used it once, and threw it away.",
    )

    content_slide(
        prs, "part 4 · the idea", "Hand the encoder's detail forward, across the gap",
        [("Keep every encoder feature map. When the decoder climbs back to that "
          "resolution, staple the saved map onto its own features before "
          "continuing.", "lead"),
         "",
         ("  x = self.up3(bottleneck)            # 12x12 -> 24x24", "mono"),
         ("  x = torch.cat([x, skip3], dim=1)    # <- the whole idea", "mono"),
         "",
         "The decoder now has two sources at every resolution:",
         "",
         ("  from below, upsampled:  coarse but semantically informed", "mono"),
         ("                          \"this is a cat, lying down, facing left\"", "mono"),
         ("  from the side, skipped:  sharp but naive", "mono"),
         ("                          \"there is a strong edge at exactly this pixel\"",
          "mono"),
         "",
         "Its job shrinks from 'hallucinate the detail' to 'decide which of these "
         "sharp edges are real, given what I know about the scene'.",
         ],
        top=Inches(1.88),
    )

    figure_slide(
        prs, "part 4 · the model", "Same left half, same waist, plus three arrows",
        F_["arch_unet"],
        "The U is not decoration: because padding=1 keeps sizes fixed between "
        "pools, encoder level k and decoder level k are guaranteed to have the "
        "same height and width. The symmetry IS the alignment guarantee.",
        height=Inches(4.25),
    )

    flow_slide(
        prs, "part 4 · the maths", "cat, not add — and what that does to the channel counts",
        [("eq", r"$\mathrm{cat}:\ \mathbb{R}^{128 \times 24 \times 24} \oplus "
                r"\mathbb{R}^{128 \times 24 \times 24} \;\rightarrow\; "
                r"\mathbb{R}^{256 \times 24 \times 24} \qquad "
                r"\mathrm{add}:\ \rightarrow\ \mathbb{R}^{128 \times 24 \times 24}$",
          22),
         ("text", "Adding (the ResNet move) sums the two sources irreversibly. "
                  "Concatenating keeps both, so the next convolution can learn "
                  "how much to trust each one. Both designs exist; U-Net "
                  "concatenates."),
         ("gap", 0.05),
         ("lead", "The consequence is the bug you are about to write: every "
                  "decoder block's INPUT channel count doubles."),
         ("eq", r"$\mathrm{dec}_k:\quad "
                r"\underset{\mathrm{from\ ConvT}}{C_k} \;+\; "
                r"\underset{\mathrm{from\ skip}}{C_k} \;=\; "
                r"2C_k\ \mathrm{channels\ in} \;\rightarrow\; "
                r"C_k\ \mathrm{out} \qquad (256,\ 128,\ 64\ \mathrm{here})$", 22),
         ("note", "The error message is a channel mismatch inside Conv2d, which "
                  "points at the conv — not at the cat that caused it."),
         ],
        tmp,
    )

    content_slide(
        prs, "part 4 · why it should help", "There is now a path that never passes through the waist",
        [("Without skips, every route from input pixel to output pixel goes "
          "through the 12x12 bottleneck. The finest thing the output can express "
          "is an 8x8 block, and no amount of decoder capacity changes that — "
          "part 2's waist sweep measured exactly this.", "lead"),
         "",
         "With skips, enc1's 96x96 map reaches dec1 without ever being pooled. "
         "The network CAN represent a one-pixel edge, because there is a "
         "stride-1 path from the image to the answer.",
         "",
         (">> So the prediction is specific, and that makes it falsifiable:", "lead"),
         "",
         ("  the gain should land on the BORDER class, not spread evenly", "mono"),
         ("  the gain should appear NEAR boundaries and vanish far from them", "mono"),
         "",
         ("Both are testable. Part 5 tests the second one.", "note"),
         ],
        top=Inches(1.90),
    )

    figure_slide(
        prs, "part 4 · what a skip carries", "The three tensors the decoder is handed",
        F_["skipmaps"],
        "The six highest-variance channels at each level, from the trained "
        "model. Sharp edge and fur detectors at 96x96; at 24x24 they have "
        "stopped looking like the photo and started looking like regions.",
        height=Inches(4.15),
    )

    content_slide(
        prs, "part 4 · the experiment", "Making the comparison mean only one thing",
        [("--no-skips does not delete the skip connections. It replaces the "
          "skip half of every cat with zeros.", "lead"),
         "",
         ("  def _join(self, up, skip):", "mono"),
         ("      if not self.use_skips:", "mono"),
         ("          skip = torch.zeros_like(skip)", "mono"),
         ("      return torch.cat([up, skip], dim=1)", "mono"),
         "",
         f"Same depth, same widths, same 12x12 bottleneck, and — as the table "
         f"prints — exactly the same parameter count: "
         f"{seg['part4_unet_skips_off']['params']:,} either way.",
         "",
         ("The skipless model has just as many weights to work with. It simply "
          "receives no information through them. So whatever the difference is, "
          "it is not capacity.", "note"),
         ],
        top=Inches(1.92),
    )

    gain = q("part4_unet_skips_on") - q("part4_unet_skips_off")
    bgain = (seg["part4_unet_skips_on"]["iou"][2]
             - seg["part4_unet_skips_off"]["iou"][2])
    table_slide(
        prs, "part 4 · the evidence", "Read it twice",
        ["model", "params", "acc", "mIoU", "pet", "bg", "border"],
        [[seg[k]["label"], f"{seg[k]['params']:,}", fmt(seg[k]["acc"]),
          (fmt(seg[k]["miou"]),)] + [fmt(v) for v in seg[k]["iou"]]
         for k in ["part3_encdec", "part4_unet_skips_off", "part4_unet_skips_on"]],
        note=f"1. skips ON vs OFF: identical parameter count, +{gain:.3f} mIoU, "
             f"and the largest per-class gain is the border at +{bgain:.3f} — the "
             f"one class defined by sitting on an edge.   "
             f"2. skips OFF vs part 3: 194k MORE parameters, "
             f"{q('part4_unet_skips_off') - q('part3_encdec'):+.3f} mIoU. Below "
             f"the waist, capacity cannot buy back resolution that was already "
             f"discarded. Only a path around it can.",
        widths=[2.4, 1.1, 0.8, 0.8, 0.8, 0.8, 0.8],
    )

    if F_["tio4"]:
        figure_slide(
            prs, "part 4 · the evidence", "The same three runs, watched epoch by epoch",
            F_["tio4"],
            "Logged to trackio during training, so this is not a re-measurement "
            "-- it is the run itself. The skips are ahead from the first epoch "
            "and stay ahead, and the right-hand panel is where the gap lives.",
            height=Inches(4.0),
        )

    figure_slide(
        prs, "part 4 · the evidence", "The busiest boundary in the batch, magnified",
        F_["zoom"],
        "At thumbnail size every segmentation looks fine, so we go and find the "
        "crop with the most border pixels in it. This is what 0.033 mIoU looks "
        "like where it is spent.",
        height=Inches(3.5),
    )

    figure_slide(
        prs, "part 4 · the evidence", "Every model's errors, in red",
        F_["errors"],
        "Two colour schemes, both named in the key. Row 2 is a class map. Rows 3-4 "
        "are not: they are the photo in grey at 55% brightness with every wrong "
        "pixel painted red, so red means 'argmax ≠ label' and says nothing about "
        "which class was guessed. Both models draw the same picture — a red "
        "outline tracing the animal, because that is where errors live. Skips ON "
        "draws it thinner.",
        height=Inches(3.9),
    )

    # ------------------------------------------------- part 5 / honesty
    content_slide(
        prs, "part 4 · on the size of that win", "0.033 mIoU is real, and it is smaller than U-Net's reputation",
        [("Worth saying out loud, because a modest number is exactly the kind you "
          "should not take on trust.", "lead"),
         "",
         "Two reasons, and both are about this setup rather than about skip "
         "connections:",
         "",
         ("  · the waist here is 12x12 — a gentle squeeze for a 96x96 image", "mono"),
         (f"  · part 3's decoder is already good on its own "
          f"({fmt(q('part3_encdec'))})", "mono"),
         "",
         "Part 2's waist sweep says what to do about it: a 6x6 waist starts 7 dB "
         "worse, so the tighter the waist, the more there is for the skips to "
         "rescue. The 2015 paper went four levels deep on 572x572 microscope "
         "images, where the gap is not subtle.",
         "",
         (">> So: a small effect. Does it appear where the mechanism says it "
          "should?", "lead"),
         ],
        top=Inches(1.88),
    )

    figure_slide(
        prs, "part 5 · the evidence", "Accuracy against distance to the nearest boundary",
        F_["distance"],
        f"16+ px from any boundary the two models are the same model to within "
        f"noise ({dist['skips ON'][4] - dist['skips OFF'][4]:+.3f}) — they share "
        f"an encoder, so of course they are. The entire gain appears in the first "
        f"few pixels and decays monotonically.",
        height=Inches(3.9),
    )

    content_slide(
        prs, "part 5 · the evidence", "That is what turns a small number into a finding",
        [("A small effect that lands exactly where the mechanism predicts, and "
          "nowhere else, is a real effect.", "lead"),
         "",
         ("  distance (px)      0-1     2-3     4-7    8-15     16+", "mono"),
         ("  skips OFF      " + "".join(f"{v:>8.3f}" for v in dist["skips OFF"]),
          "mono"),
         ("  skips ON       " + "".join(f"{v:>8.3f}" for v in dist["skips ON"]),
          "mono"),
         ("  gain           " + "".join(
             f"{a - b:>+8.3f}" for a, b in zip(dist["skips ON"], dist["skips OFF"])),
          "mono"),
         "",
         "'Skip connections restore high-frequency detail' is a sentence. This is "
         "the same sentence with a y-axis.",
         "",
         (f"Note also how hard the boundary is in absolute terms: even with "
          f"skips, {(1 - dist['skips ON'][0]) * 100:.0f}% of the pixels within "
          f"1px of a class boundary are wrong, against "
          f"{(1 - dist['skips ON'][4]) * 100:.0f}% out in the open. That is where "
          f"the remaining headroom is.", "note"),
         ],
        top=Inches(1.90),
    )

    # ------------------------------------------------------------ closing
    table_slide(
        prs, "the arc of week 4", "Every part fixed what the previous one broke",
        ["part", "the problem it faced", "what changed", "mIoU"],
        [["1  FCN", "a classifier throws away WHERE",
          "Flatten+Linear -> Conv1x1", fmt(q("part3_fcn"))],
         ["2  autoencoder", "Upsample cannot add detail",
          "a learned decoder", "— (MSE)"],
         ["3  encoder-decoder", "the output is a class, not a colour",
          "logits + CrossEntropy", fmt(q("part3_encdec"))],
         ["4  U-Net", "detail died at the waist",
          "torch.cat(up, skip)", (fmt(q("part4_unet_skips_on")),)]],
        note="The tension that drives the whole session never actually goes "
             "away: pooling buys context and costs resolution. A U-Net does not "
             "resolve it — it routes around it.",
        widths=[1.5, 3.0, 2.2, 0.9],
    )

    content_slide(
        prs, "where this goes", "You have now built the standard answer for image-to-image",
        [("Swap the target, keep the network:", "lead"),
         "",
         ("  target = the clean image          ->  a denoiser", "mono"),
         ("  target = the masked-out region    ->  an inpainter", "mono"),
         ("  target = a depth map              ->  monocular depth", "mono"),
         ("  target = the noise that was added ->  a diffusion model", "mono"),
         "",
         "That last one is not a joke: the network inside Stable Diffusion is a "
         "U-Net with attention blocks bolted on. It predicts the noise in a noisy "
         "image, and the encoder-decoder-with-skips shape is there for the reason "
         "you just measured — the output is an image, and it needs to be sharp.",
         "",
         ("Whenever the output has the same spatial shape as the input, this is "
          "the architecture to reach for first.", "note"),
         ],
        top=Inches(1.92),
    )

    on = seg["part4_unet_skips_on"]
    content_slide(
        prs, "your turn · 1 of 2",
        "Where the next mIoU comes from, starting with the cheapest change",
        [(f"We stopped at mIoU {fmt(on['miou'])}. Four things to try before touching "
          "the architecture:", "lead"),
         "",
         ("  1  TRAIN LONGER       --epochs 40, plus a cosine LR decay", "mono"),
         ("     nothing has converged: train loss is still falling at the last", "mono"),
         ("     epoch (0.259 -> 0.247) and test mIoU still trending up, noisily", "mono"),
         ("     (0.705, 0.714, 0.707 over the final three). We stopped for time.", "mono"),
         "",
         ("  2  AUGMENT MORE       _PetSeg.__getitem__", "mono"),
         ("     a horizontal flip is all we do. Add random crop, scale, rotation.", "mono"),
         ("     Same rule as the flip: geometry hits the photo AND the mask.", "mono"),
         "",
         ("  3  AIM THE LOSS       CrossEntropyLoss(weight=...), or a Dice loss", "mono"),
         ("     mIoU averages the classes unweighted, so the border is a third of", "mono"),
         (f"     the score and the worst third: {on['iou'][2]:.3f} vs "
          f"{on['iou'][0]:.3f} pet, {on['iou'][1]:.3f} background.", "mono"),
         "",
         ("  4  MORE PIXELS        IMAGE_SIZE = 192", "mono"),
         ("     one constant, 4x the compute. At 96x96 the border band runs 1.8 px", "mono"),
         ("     wide (median, 3 px at p75) — the worst class is half-buried in the", "mono"),
         ("     grid. The most direct attack on item 3.", "mono"),
         ],
        top=Inches(1.80),
    )

    content_slide(
        prs, "your turn · 2 of 2", "And then the bigger swaps",
        [("Each of these changes what the network IS. One at a time, same test "
          "images:", "lead"),
         "",
         ("  5  WIDER, DEEPER      base=64, or a fourth level", "mono"),
         (f"     ours is {count_params(UNet()):,} parameters, three pools, a 12x12 waist.", "mono"),
         ("     A fourth level buys context and costs resolution (part 2: a 6x6", "mono"),
         ("     waist starts 7 dB worse) — a bet on the skips covering the loss.", "mono"),
         "",
         ("  6  PRETRAINED ENCODER torchvision ResNet-34, fine-tuned", "mono"),
         ("     what production segmentation does: keep the U, start the encoder", "mono"),
         ("     from ImageNet instead of noise. Part 3 measured the small version —", "mono"),
         ("     our autoencoder bought +0.040 mIoU at 500 labels, -0.016 at 3,680.", "mono"),
         ("     Same bet, a million images behind it, and the same caveat: it pays", "mono"),
         ("     most when labels are scarce.", "mono"),
         "",
         ("  7  DIFFERENT DECODER  dilated convs (DeepLab), or a ViT encoder", "mono"),
         ("     U-Net is 2015. Dilated convolutions widen the receptive field with", "mono"),
         ("     no pooling at all — attacking this session's tension at the root", "mono"),
         ("     instead of routing around it.", "mono"),
         ],
        top=Inches(1.80),
    )

    # ============================================== BACKUP
    section_slide(
        prs, "backup", "Running it yourself",
        "Every figure in this deck is regenerated by the script that built it, "
        "from the checkpoints the parts wrote.",
    )

    content_slide(
        prs, "backup", "How to watch a run",
        [("  trackio show --project \"week4-unet\"", "mono"),
         "",
         "Every part logs, per epoch: train_loss, test/acc, test/mIoU, and the "
         "three per-class IoUs. The run config carries the architecture, the "
         "parameter count, the learning rate and the training-set size, so the "
         "dashboard's run table sorts into a leaderboard on its own.",
         "",
         ("Two honest notes.", "lead"),
         "",
         ("  1. These per-epoch numbers are on the TEST set, so they are", "mono"),
         ("     monitoring, not selection: every part trains a fixed number of", "mono"),
         ("     epochs decided in advance and reports the last one. The moment", "mono"),
         ("     you want to CHOOSE something, carve a validation split out of", "mono"),
         ("     trainval first, the way Week 3 Part 7 does.", "mono"),
         "",
         ("  2. Measuring once per epoch changed the results, at first. See the", "mono"),
         ("     next slide -- it is a bug worth meeting once.", "mono"),
         ],
        top=Inches(1.90),
    )

    if F_["tio_instr"]:
        _, acc_c = trackio_series("U-Net, skips ON", "test/acc")
        _, bor_c = trackio_series("U-Net, skips ON", "test/iou_border")
        figure_slide(
            prs, "backup · the instrument", "Why the dashboard plots IoU and not accuracy",
            F_["tio_instr"],
            f"One run (U-Net, skips ON), three metrics, one axis. Accuracy opens "
            f"at {acc_c[0]:.3f} -- barely above the {fmt(seg['constant']['acc'])} "
            f"a model scores for predicting 'background' everywhere -- and gains "
            f"{acc_c[-1] - acc_c[0]:.3f} over 15 epochs. The border class opens at "
            f"{bor_c[0]:.3f} and gains {bor_c[-1] - bor_c[0]:.3f}, and is still "
            f"climbing at the end. Same run: one curve says 'nearly done', the "
            f"other says 'keep going'.",
            height=Inches(4.0),
        )

    content_slide(
        prs, "backup · the instrument", "The bug that instrumentation introduced",
        [("Adding the per-epoch probe made the training runs stop reproducing.",
          "lead"),
         "",
         ("  epoch 1   loss 0.6974   <- identical to the un-instrumented run", "mono"),
         ("  epoch 2   loss 0.6268   <- and from here on, different", "mono"),
         "",
         "Nothing was wrong with the measurement. Creating a DataLoader "
         "ITERATOR draws a number from the global RNG, and our training loader "
         "shuffles from that same RNG — so evaluating once per epoch re-dealt "
         "the training order from epoch 2 onwards.",
         "",
         ("  def undisturbed(fn):          # common.py", "mono"),
         ("      state = torch.get_rng_state()", "mono"),
         ("      try:     return fn(model)", "mono"),
         ("      finally: torch.set_rng_state(state)", "mono"),
         "",
         ("The check that it worked: the per-epoch losses now match the runs "
          "from before any logging existed, to four decimals.", "note"),
         ],
        top=Inches(1.88),
    )

    content_slide(
        prs, "backup", "The files",
        [("  cd week4_unet", "mono"),
         "",
         ("  python part1_classifier_to_segmenter.py     # ~2 min", "mono"),
         ("  python part2a_cnn_autoencoder.py            # ~9 min", "mono"),
         ("  python part2b_waist_sweep.py                # ~11 min", "mono"),
         ("  python part3_encoder_decoder.py --pretrained             # ~12 min",
          "mono"),
         ("  python part3_encoder_decoder.py --pretrained --n-train 500 "
          "--skip-baseline", "mono"),
         ("  python part4_unet.py                        # ~18 min", "mono"),
         ("  python part4_unet.py --shapes-only          # shapes, no training",
          "mono"),
         ("  python part5_unet_visualize.py              # take-home", "mono"),
         "",
         ("  python compare_parts.py                     # the summary figure",
          "mono"),
         ("  python slides/build_slides.py               # this deck", "mono"),
         "",
         ("Weights land in checkpoints/, figures in results/ (set "
          "WEEK4_SAVE_FIGS=1), loss curves in results/training_curves.json. "
          "The first run caches the dataset at 96x96, once, in about 23 seconds.",
          "note"),
         ],
        top=Inches(1.88),
    )

    content_slide(
        prs, "backup", "Where the numbers in this deck come from",
        [(f"Measured at build time, on {m['n_test']} held-out test images "
          f"({m['device']}):", "lead"),
         "",
         "· every mIoU, IoU and accuracy — by loading checkpoints/*.pt and "
         "running them here;",
         "· every parameter count — count_params() on the model itself;",
         "· every tensor shape in the architecture diagrams — forward hooks on a "
         "real forward pass;",
         "· the class shares — counted off the test labels.",
         "",
         "Read from results/training_curves.json, written by the parts:",
         "· the training loss curves.",
         "",
         ("Transcribed by hand, because the runs that produced them saved no "
          "weights: part 2's waist sweep and the 500-label pretraining "
          "comparison. Both are marked in build_slides.py with the command that "
          "reproduces them, and the waist sweep's parameter counts are still "
          "recomputed live.", "note"),
         ],
        top=Inches(1.88),
    )

    prs.save(OUT)
    return OUT, len(prs.slides._sldIdLst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-test", type=int, default=1000,
                        help="test images to measure on (default: 1000)")
    parser.add_argument("--force", action="store_true",
                        help="ignore slides/measured.json and re-measure")
    args = parser.parse_args()
    path, n = build(args.n_test, args.force)
    print(f"wrote {path}  ({n} slides)")


if __name__ == "__main__":
    main()
