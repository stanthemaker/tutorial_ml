"""Week 4 - Part 4b: Segmentation -- what U-Net was actually invented for.

U-Net comes from a 2015 paper on segmenting cells in microscope images. Every
part of the design we have been building up exists to serve this task:

  * The output is an image, so we need a decoder.        (Part 1)
  * The output is spatial, so we need convolutions.      (Part 2)
  * The output must be pixel-accurate at the boundaries, so we need skip
    connections.                                          (Part 3)

Here we run it on Oxford-IIIT Pet, which ships a per-pixel "trimap" label:
every pixel is 0 = pet, 1 = background, or 2 = border. Predicting that map is
segmentation.

TWO things change from Part 3, and both follow from one fact: the output is
now a *class per pixel*, not a colour per pixel.

  1. The head produces 3 channels and NO Sigmoid. Those 3 numbers per pixel
     are logits -- one score per class -- exactly like the 10 logits at the
     end of Week 3's classifier. Squashing them to [0,1] would be wrong for
     the same reason it would have been wrong there.

  2. The loss is CrossEntropyLoss, not MSELoss. And it is the *same*
     CrossEntropyLoss from Week 2 and 3: nn.CrossEntropyLoss accepts
     (N, C, H, W) logits against (N, H, W) integer targets and simply applies
     the usual formula at every pixel independently. So segmentation really is
     "classification, run 9216 times per image, with the whole picture as
     context".

The metric changes too, for a reason worth stating. Pixel accuracy is a bad
score here: roughly two thirds of the pixels are background, so a model that
outputs "background" everywhere already scores ~65%. We report mean IoU
(intersection over union) per class instead, which gives the tiny border class
nowhere to hide.

This is also the file where the skip connections' contribution becomes
visually unambiguous, because segmentation errors concentrate exactly where
detail lives: along the outline of the animal.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from common import get_device, count_params, load_pet, make_loaders, save_weights
from part3_unet import UNet

IMAGE_SIZE = 96
N_CLASSES = 3
CLASS_NAMES = ["pet", "background", "border"]


class UNetSegmenter(UNet):
    """Part 3's U-Net with a classification head instead of a pixel head.

    Subclassing rather than rewriting is the point: the body of the network --
    encoder, bottleneck, upsampling, concatenation -- is identical. Only the
    last 1x1 conv changes, and only because the meaning of the output changed.
    """

    def __init__(self, in_ch=3, n_classes=N_CLASSES, base=32, use_skips=True):
        super().__init__(in_ch=in_ch, out_ch=n_classes, base=base, use_skips=use_skips)
        # Replace the Conv1x1 + Sigmoid with a bare Conv1x1: raw logits.
        self.head = nn.Conv2d(base, n_classes, kernel_size=1)


def train_segmenter(model, loader, device, epochs=20, lr=1e-3):
    """The same loop again. Two lines differ from train_denoiser."""
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()   # <- was MSELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for image, mask in loader:
            image, mask = image.to(device), mask.to(device)
            logits = model(image)              # (N, 3, 96, 96)
            loss = loss_fn(logits, mask)       # <- target is (N, 96, 96) int64
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
        print(f"  epoch {epoch + 1}/{epochs}  CE {losses[-1]:.4f}")
    return losses


@torch.no_grad()
def evaluate(model, loader, device):
    """Pixel accuracy and per-class IoU.

    IoU for a class = (pixels where prediction and truth both say the class)
                    / (pixels where either one does).

    A model that never predicts "border" gets IoU 0 for border no matter how
    many background pixels it gets right -- which is exactly the pressure we
    want on the metric.
    """
    model.eval()
    inter = torch.zeros(N_CLASSES)
    union = torch.zeros(N_CLASSES)
    correct = total = 0

    for image, mask in loader:
        image, mask = image.to(device), mask.to(device)
        pred = model(image).argmax(dim=1)
        correct += (pred == mask).sum().item()
        total += mask.numel()
        for c in range(N_CLASSES):
            p, t = (pred == c), (mask == c)
            inter[c] += (p & t).sum().item()
            union[c] += (p | t).sum().item()

    iou = inter / union.clamp(min=1)
    return correct / total, iou


def mask_to_rgb(mask):
    """Colour a class-index map so it can go through the same plotting path.

    pet = orange, background = dark blue, border = white. Returned as a
    (3, H, W) float tensor in [0, 1], i.e. it looks like any other image to
    matplotlib.
    """
    palette = torch.tensor([[0.90, 0.55, 0.15],
                            [0.15, 0.20, 0.40],
                            [1.00, 1.00, 1.00]])
    return palette[mask.long()].permute(2, 0, 1)


def show_segmentation(images, truths, predictions, titles, n=6):
    """Rows of: photo, ground-truth mask, then one row per model."""
    rows = [images] + [torch.stack([mask_to_rgb(m) for m in batch])
                       for batch in [truths] + predictions]
    row_titles = ["photo", "true mask"] + titles

    fig, axes = plt.subplots(len(rows), n, figsize=(1.5 * n, 1.7 * len(rows)))
    axes = axes.reshape(len(rows), n)
    for r, (batch, title) in enumerate(zip(rows, row_titles)):
        for c in range(n):
            axes[r, c].imshow(batch[c].permute(1, 2, 0).clamp(0, 1))
            axes[r, c].axis("off")
        axes[r, 0].text(-0.15, 0.5, title, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)
    fig.suptitle("Pet segmentation: orange = pet, navy = background, white = border")
    fig.tight_layout()


def main():
    torch.manual_seed(0)
    device = get_device()
    print(f"using device: {device}\n")

    train_ds = load_pet("trainval", image_size=IMAGE_SIZE, n_subset=2000,
                        segmentation=True)
    test_ds = load_pet("test", image_size=IMAGE_SIZE, n_subset=400,
                       segmentation=True)
    train_loader, test_loader = make_loaders(train_ds, test_ds, batch_size=32)
    print(f"train {len(train_ds)} / test {len(test_ds)} images, "
          f"{IMAGE_SIZE}x{IMAGE_SIZE}, {N_CLASSES} classes per pixel\n")

    # Baseline to keep the numbers honest: predict the single most common
    # class everywhere. Any model that cannot beat this has learned nothing.
    counts = torch.zeros(N_CLASSES)
    for _, mask in train_loader:
        counts += torch.bincount(mask.flatten(), minlength=N_CLASSES).float()
    shares = counts / counts.sum()
    for name, s in zip(CLASS_NAMES, shares):
        print(f"  class share  {name:<11} {s:.1%}")
    # Predicting one class everywhere gives that class IoU = its own share
    # (intersection = its pixels, union = all pixels) and 0 for every other,
    # so the mean IoU of that baseline is share / 3.
    top = shares.max().item()
    print(f"  -> predicting '{CLASS_NAMES[shares.argmax()]}' everywhere scores "
          f"{top:.1%} pixel accuracy but only {top / N_CLASSES:.3f} mean IoU.\n")

    predictions, titles, scores = [], [], {}
    images = truths = None

    for label, use_skips, stem in [
        ("U-Net, skips OFF", False, "part4_seg_skips_off"),
        ("U-Net, skips ON", True, "part4_seg_skips_on"),
    ]:
        torch.manual_seed(0)
        model = UNetSegmenter(use_skips=use_skips)
        print(f"=== {label} -- {count_params(model):,} parameters ===")
        train_segmenter(model, train_loader, device, epochs=20)
        save_weights(model, stem)

        acc, iou = evaluate(model, test_loader, device)
        scores[label] = (acc, iou)
        print(f"  pixel accuracy {acc:.2%}   mean IoU {iou.mean():.3f}")
        for name, v in zip(CLASS_NAMES, iou):
            print(f"    IoU {name:<11} {v:.3f}")
        print()

        image, mask = next(iter(test_loader))
        images, truths = image[:6], mask[:6]
        with torch.no_grad():
            model.eval()
            predictions.append(model(images.to(device)).argmax(dim=1).cpu())
        titles.append(label)

    print("=" * 66)
    print(f"{'model':<20} {'pixel acc':>10} {'mean IoU':>10} {'border IoU':>12}")
    print("-" * 66)
    for label, (acc, iou) in scores.items():
        print(f"{label:<20} {acc:>9.2%} {iou.mean():>10.3f} {iou[2]:>12.3f}")
    print("=" * 66)
    print("\nWatch the border column in particular. The border class is a")
    print("1-2 pixel ribbon around the animal -- the most detail-dependent")
    print("thing in the dataset, and the first thing a bottleneck destroys.")
    print("It is also why Ronneberger et al. needed skip connections for cell")
    print("boundaries in 2015: separating two touching cells is entirely a")
    print("question of getting a thin line in exactly the right place.")

    show_segmentation(images, truths, predictions, titles, n=6)
    plt.show()


if __name__ == "__main__":
    main()
