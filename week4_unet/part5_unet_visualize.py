"""Week 4 - Part 5: look at what the skips actually carry.

Part 4 made an argument out of one number: switching the skip connections on
is worth 0.033 mIoU at an identical parameter count, and the largest share of
it goes to the border class. That is a modest number, and a modest number is
exactly the kind you should not take on trust. This file is the Week 3 Part
4/5 habit applied to it -- stop reading the table, go and look at whether the
gain is where the story says it should be.

Three figures, three questions.

  1. WHAT IS IN A SKIP?  We plot the feature maps that leave enc1, enc2 and
     enc3 -- the three tensors the decoder gets handed. They are not abstract:
     the 96x96 maps are edge and fur-texture detectors that fire on outlines,
     the 24x24 maps have stopped looking like the photo and started looking
     like regions. Fine and sharp at the top, coarse and meaningful at the
     bottom -- which is exactly the pair of things Part 1 said you could not
     have at once.

  2. WHERE DO THE TWO MODELS DIFFER?  We plot each model's wrong pixels in
     red. Both models draw the same picture: a red outline tracing the animal,
     because that is where segmentation errors live. Skips ON draws it
     thinner. The point is what you do NOT see -- neither model is scattering
     mistakes across the middle of the dog or the middle of the sky. The two
     models fail in the same place, and one fails less there.

  3. HOW WRONG, AND HOW FAR FROM THE EDGE?  Accuracy as a function of distance
     to the nearest class boundary. This is the figure that settles it: 16px
     from any boundary the two models are identical to within noise, and the
     entire gap appears in the first few pixels, decaying monotonically. A
     small effect landing exactly where the mechanism predicts is a real
     effect. That is "skip connections restore high-frequency detail",
     measured rather than asserted.

Uses the weights Part 4 saved. Run part4_unet.py first, or pass --train to
train the two models here.

MEASURED RESULTS (M1 Pro, on Part 4's two 15-epoch models, about 2 minutes):

    distance (px)      0-1     2-3     4-7    8-15     16+
    skips OFF        0.648   0.878   0.940   0.964   0.953
    skips ON         0.696   0.896   0.951   0.969   0.952
    gain            +0.048  +0.018  +0.012  +0.005  -0.001

That last column is the control and it is the one to read first: 16 or more
pixels from any boundary, the two models are the same model to within noise.
They share an encoder, so of course they are. Every point of Part 4's 0.033
mIoU is bought within a few pixels of an edge, and the gain decays smoothly
with distance. "Skip connections restore high-frequency detail" is a sentence;
this is the same sentence with a y-axis.

Note also how hard the boundary is in absolute terms: even with skips, better
than 30% of the pixels within 1px of a class boundary are wrong, against 5%
out in the open. That is where the remaining headroom in this week is.

    python part5_unet_visualize.py
    python part5_unet_visualize.py --train     # if part3 weights are missing

Every run logs its per-epoch curves to trackio. Watch them live, or compare
runs after the fact, with:

    trackio show --project "week4-unet"

Set WEEK4_NO_TRACKIO=1 to turn the logging off.
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (N_CLASSES, colorize_batch, finish, get_device,
                    load_pet, make_loaders, tracked, train, weights_path)
from part4_unet import UNet

MAX_DIST = 32
BANDS = [(0, 2), (2, 4), (4, 8), (8, 16), (16, MAX_DIST + 1)]
BAND_LABELS = ["0-1", "2-3", "4-7", "8-15", "16+"]


def boundary_distance(mask, max_d=MAX_DIST):
    """For every pixel, how far is the nearest class boundary? (in pixels)

    Pure PyTorch, no scipy, and it is a nice use of a tool from Week 3: a 3x3
    max-pool over the one-hot mask reports which classes appear in each
    pixel's neighbourhood. If more than one does, that pixel sits on a
    boundary. Dilating that set one ring at a time then labels distance 1, 2,
    3, ... -- a poor man's distance transform in six lines.
    """
    onehot = F.one_hot(mask, N_CLASSES).permute(0, 3, 1, 2).float()
    present = F.max_pool2d(onehot, 3, stride=1, padding=1).sum(dim=1)
    cur = present > 1.0                                  # (N, H, W) bool

    dist = torch.full_like(mask, max_d)
    for d in range(max_d):
        dist = torch.where(cur & (dist == max_d), torch.full_like(dist, d), dist)
        if d + 1 < max_d:
            cur = F.max_pool2d(cur.float().unsqueeze(1), 3, stride=1,
                               padding=1).squeeze(1) > 0
    return dist


@torch.no_grad()
def accuracy_by_distance(model, loader, device):
    """Pixel accuracy inside each distance band. -> list of floats."""
    model.eval()
    correct = torch.zeros(len(BANDS))
    total = torch.zeros(len(BANDS))
    for x, y in loader:
        pred = model(x.to(device)).argmax(dim=1).cpu()
        dist = boundary_distance(y)
        ok = (pred == y)
        for i, (lo, hi) in enumerate(BANDS):
            in_band = (dist >= lo) & (dist < hi)
            correct[i] += (ok & in_band).sum()
            total[i] += in_band.sum()
    return (correct / total.clamp(min=1)).tolist()


def figure_skips(model, x, device):
    """Figure 1: the three tensors the decoder gets handed."""
    model.eval()
    with torch.no_grad():
        s1 = model.enc1(x.to(device))
        s2 = model.enc2(model.pool(s1))
        s3 = model.enc3(model.pool(s2))

    n_show = 6
    levels = [("skip 1  (32 x 96 x 96)", s1), ("skip 2  (64 x 48 x 48)", s2),
              ("skip 3  (128 x 24 x 24)", s3)]
    fig, axes = plt.subplots(3, n_show + 1, figsize=(1.5 * (n_show + 1), 5.0))
    for r, (title, feat) in enumerate(levels):
        axes[r, 0].imshow(x[0].permute(1, 2, 0))
        axes[r, 0].axis("off")
        axes[r, 0].text(-0.1, 0.5, title, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=8)
        # The channels with the most variance are the ones doing something;
        # a flat channel is just a dead unit and makes a boring picture.
        f = feat[0].cpu()
        order = f.flatten(1).var(dim=1).argsort(descending=True)[:n_show]
        for c in range(n_show):
            chan = f[order[c]]
            chan = (chan - chan.min()) / (chan.max() - chan.min() + 1e-8)
            axes[r, c + 1].imshow(chan, cmap="viridis")
            axes[r, c + 1].axis("off")
    fig.suptitle("What the decoder is handed: sharp at the top, semantic at the bottom",
                 fontsize=11)
    fig.tight_layout()
    return fig


def figure_errors(models, x, y, device, n=5):
    """Figure 2: wrong pixels in red, one row per model."""
    rows = [x, colorize_batch(y)]
    titles = ["photo", "truth"]
    for label, model in models.items():
        model.eval()
        with torch.no_grad():
            pred = model(x.to(device)).argmax(dim=1).cpu()
        wrong = (pred != y).float()                       # (N, H, W)
        # Grey photo underneath so the red marks are readable on top of it.
        grey = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1) * 0.6
        overlay = grey.clone()
        overlay[:, 0] = torch.maximum(overlay[:, 0], wrong)
        overlay[:, 1] = overlay[:, 1] * (1 - wrong)
        overlay[:, 2] = overlay[:, 2] * (1 - wrong)
        rows.append(overlay)
        titles.append(f"errors: {label}")

    n = min(n, x.size(0))
    fig, axes = plt.subplots(len(rows), n, figsize=(1.6 * n, 1.8 * len(rows)))
    for r in range(len(rows)):
        for c in range(n):
            axes[r, c].imshow(rows[r][c].permute(1, 2, 0).clamp(0, 1))
            axes[r, c].axis("off")
        axes[r, 0].text(-0.12, 0.5, titles[r], transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)
    fig.suptitle("The same mistakes, thinner: every error hugs a boundary", fontsize=11)
    fig.tight_layout()
    return fig


def figure_distance(curves):
    """Figure 3: accuracy vs distance to the nearest boundary."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    xs = range(len(BAND_LABELS))
    for label, acc in curves.items():
        ax.plot(xs, acc, marker="o", label=label)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(BAND_LABELS)
    ax.set_xlabel("distance to nearest class boundary (px)")
    ax.set_ylabel("pixel accuracy")
    ax.set_title("Both models are fine away from edges")
    ax.legend()
    ax.grid(alpha=0.3)

    on, off = curves["skips ON"], curves["skips OFF"]
    ax2.bar(list(xs), [a - b for a, b in zip(on, off)], color="#e07d21")
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(BAND_LABELS)
    ax2.set_xlabel("distance to nearest class boundary (px)")
    ax2.set_ylabel("accuracy gain from skips")
    ax2.set_title("The whole gain is within a few pixels of an edge")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def load_or_train(use_skips, train_loader, device, epochs, do_train):
    stem = f"part4_unet_skips_{'on' if use_skips else 'off'}"
    path = weights_path(stem)
    model = UNet(use_skips=use_skips).to(device)
    if os.path.exists(path) and not do_train:
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"  loaded {stem}.pt")
        return model
    print(f"=== training {stem} ===")
    torch.manual_seed(0)
    with tracked(f"part5 {stem}", config=dict(part=5, arch=stem, use_skips=use_skips,
                                              epochs=epochs, lr=1e-3)) as run:
        train(model, train_loader, device, nn.CrossEntropyLoss(), epochs=epochs,
              lr=1e-3, run=run)
    torch.save(model.state_dict(), path)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true",
                        help="retrain both models instead of loading part4's weights")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = get_device()
    print(f"using device: {device}\n")

    train_ds = load_pet("trainval", augment=True)
    test_ds = load_pet("test", n_subset=1000)
    train_loader, test_loader = make_loaders(train_ds, test_ds)

    models = {
        "skips OFF": load_or_train(False, train_loader, device, args.epochs, args.train),
        "skips ON": load_or_train(True, train_loader, device, args.epochs, args.train),
    }
    print()

    x, y = next(iter(test_loader))
    x, y = x[:6], y[:6]

    figure_skips(models["skips ON"], x, device)
    figure_errors(models, x, y, device)

    curves = {label: accuracy_by_distance(m, test_loader, device)
              for label, m in models.items()}
    figure_distance(curves)

    head = f"{'distance (px)':<16}" + "".join(f"{b:>8}" for b in BAND_LABELS)
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for label, acc in curves.items():
        print(f"{label:<16}" + "".join(f"{a:>8.3f}" for a in acc))
    gain = [a - b for a, b in zip(curves["skips ON"], curves["skips OFF"])]
    print(f"{'gain':<16}" + "".join(f"{g:>+8.3f}" for g in gain))
    print("=" * len(head))

    print("\nThe right-hand column is the control: 16 or more pixels from any")
    print("boundary -- the middle of the animal, the middle of the sky -- the")
    print("two models are the same model to within noise. They share an")
    print("encoder, so of course they are.")
    print("\nThe gain is concentrated in the first few pixels and decays")
    print("monotonically with distance. A small effect that appears exactly")
    print("where the mechanism predicts is a real effect: 'skip connections")
    print("restore high-frequency detail' is a sentence, and this plot is the")
    print("same sentence with a y-axis.")
    print("\nOne more thing to take away, and it is the headroom left in this")
    print("week: even with the skips, roughly 30% of the pixels within 1px of")
    print("a boundary are still wrong, against about 5% out in the open. The")
    print("boundary is where segmentation is hard, and we have improved it")
    print("rather than solved it.")

    finish("part5")


if __name__ == "__main__":
    main()
