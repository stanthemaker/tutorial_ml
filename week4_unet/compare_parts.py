"""Week 4 - the summary figure: what each part actually bought.

Not part of the teaching sequence. This is the slide at the end of the week:
every model the week produced, on the same test images, with the metrics that
go with them.

It loads the weights the parts saved rather than retraining, so run the parts
first:

    python part1_classifier_to_segmenter.py     (optional -- see note below)
    python part3_encoder_decoder.py
    python part4_unet.py
    python compare_parts.py

A note on fairness, because it decides which checkpoint gets used. Part 1
trains its FCN with its own recipe (8 epochs at lr 3e-3) and Part 3 retrains
the same architecture on Part 3's recipe (15 epochs at lr 1e-3) precisely so
that its table compares architectures and not schedules. This figure uses the
15-epoch checkpoints throughout -- part3_fcn.pt, not Part 1's -- so every bar
here differs only in architecture. That is also why the FCN scores 0.463 here
and 0.484 in Part 1's own table.

    python compare_parts.py
    python compare_parts.py --n-test 500      # quicker
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (CLASS_NAMES, colorize_batch, count_params,
                    evaluate_seg, get_device, load_pet, make_loaders,
                    predict_batch, RESULTS_DIR, weights_path)
from part1_classifier_to_segmenter import fcn
from part3_encoder_decoder import SegAutoencoder
from part4_unet import UNet

# Categorical slots 1 and 2 of the reference palette: validated for
# colour-vision deficiency separation against a white surface.
C_MIOU, C_BORDER = "#2a78d6", "#eb6834"
INK, MUTED = "#1a1a19", "#6b6a63"

# (row label, builder, checkpoint stem). All four checkpoints come from the
# same 15-epoch, lr 1e-3 recipe.
MODELS = [
    ("Part 1  classifier → FCN", fcn, "part3_fcn"),
    ("Part 3  encoder-decoder", SegAutoencoder, "part3_encdec"),
    ("Part 4  U-Net, skips OFF", lambda: UNet(use_skips=False), "part4_unet_skips_off"),
    ("Part 4  U-Net, skips ON", lambda: UNet(use_skips=True), "part4_unet_skips_on"),
]


def load(build, stem, device):
    path = weights_path(stem)
    if not os.path.exists(path):
        raise SystemExit(f"missing {stem}.pt -- run the parts first (see the docstring)")
    model = build()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-show", type=int, default=6)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = get_device()
    print(f"using device: {device}\n")

    test_ds = load_pet("test", n_subset=args.n_test)
    _, test_loader = make_loaders(test_ds, test_ds)

    results = []
    for label, build, stem in MODELS:
        model = load(build, stem, device)
        acc, iou, miou = evaluate_seg(model, test_loader)
        x, y, pred = predict_batch(model, test_loader, n=args.n_show)
        results.append(dict(label=label, params=count_params(model), acc=acc,
                            iou=iou.tolist(), miou=miou, pred=pred))
        print(f"  {label:<30} params {count_params(model):>10,}   "
              f"acc {acc:.3f}   mIoU {miou:.3f}   border {iou[2]:.3f}")

    n = args.n_show
    n_rows = len(results) + 2                       # photo + truth + models
    fig = plt.figure(figsize=(1.55 * n + 3.4, 1.62 * n_rows + 3.0))
    grid = fig.add_gridspec(nrows=n_rows + 1, ncols=n, height_ratios=[1] * n_rows + [2.1],
                            hspace=0.08, wspace=0.05, top=0.955, bottom=0.045)

    def image_row(r, batch, label, sub=None):
        for c in range(n):
            ax = fig.add_subplot(grid[r, c])
            ax.imshow(batch[c].permute(1, 2, 0).clamp(0, 1))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                ax.text(-0.06, 0.62, label, transform=ax.transAxes, ha="right",
                        va="center", fontsize=10, color=INK)
                if sub:
                    ax.text(-0.06, 0.30, sub, transform=ax.transAxes, ha="right",
                            va="center", fontsize=8.5, color=MUTED)

    image_row(0, x, "photo")
    image_row(1, colorize_batch(y), "ground truth", "pet / background / border")
    for i, r in enumerate(results):
        image_row(i + 2, colorize_batch(r["pred"]),
                  r["label"], f"mIoU {r['miou']:.3f}   border IoU {r['iou'][2]:.3f}")

    # --- the metrics panel -------------------------------------------------
    # Two series (overall quality, and the class where the week's argument
    # lives), horizontal so the model names read left-to-right at full size.
    ax = fig.add_subplot(grid[n_rows, :])
    ys = np.arange(len(results))
    h = 0.32          # leaves a visible gap between the two bars of a pair
    miou = [r["miou"] for r in results]
    border = [r["iou"][2] for r in results]

    # mIoU on top of each pair, so the panel reads in the same order as the legend
    ax.barh(ys - 0.18, miou, height=h, color=C_MIOU, label="mean IoU", zorder=3)
    ax.barh(ys + 0.18, border, height=h, color=C_BORDER, label="border-class IoU", zorder=3)
    for yy, v in zip(ys - 0.18, miou):
        ax.text(v + 0.008, yy, f"{v:.3f}", va="center", fontsize=9, color=INK)
    for yy, v in zip(ys + 0.18, border):
        ax.text(v + 0.008, yy, f"{v:.3f}", va="center", fontsize=9, color=INK)

    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in results], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(miou) * 1.16)
    ax.set_xlabel("IoU  (intersection over union, higher is better)",
                  fontsize=9.5, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d0")
    ax.grid(axis="x", color="#ececE6", zorder=0)
    ax.set_axisbelow(True)
    # Upper right of the panel is empty (Part 1's bars are the shortest), so
    # the legend costs no space and collides with nothing.
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")

    fig.suptitle("Week 4: one label per pixel, four architectures, one test set",
                 fontsize=13, color=INK, y=0.985)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "week4_comparison.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"\n  saved {out}")

    head = f"{'model':<30} {'params':>10} {'acc':>7} {'mIoU':>7}"
    head += "".join(f"{c:>9}" for c in ["pet", "bg", "border"])
    print("\n" + "=" * len(head))
    print(head)
    print("-" * len(head))
    for r in results:
        line = f"{r['label']:<30} {r['params']:>10,} {r['acc']:>7.3f} {r['miou']:>7.3f}"
        line += "".join(f"{v:>9.3f}" for v in r["iou"])
        print(line)
    print("=" * len(head))


if __name__ == "__main__":
    main()
