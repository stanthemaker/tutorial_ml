"""Week 4 - Part 4: U-Net. Stop throwing the encoder's detail away.

Part 3 left us with one observation and one question.

Observation: the encoder-decoder gets the animal right and the outline wrong.
The border class sits near zero and the edges are soft.

Question: on the way down, enc1 computed a 32 x 96 x 96 feature map that still
had the whiskers and the exact edge of the ear in it. We pooled it, used it
once, and discarded it. Why does the decoder not get to see it?

A U-Net is that question, answered literally. Keep every encoder feature map,
and when the decoder climbs back to that resolution, staple the saved map onto
the decoder's own features before continuing:

    x = self.up3(bottleneck)            # 12x12 -> 24x24
    x = torch.cat([x, skip3], dim=1)    # <-- the whole idea

The decoder now has two sources at every resolution:

  * from below, upsampled: coarse but semantically informed
    ("this is a cat, lying down, facing left")
  * from the side, via the skip: sharp but naive
    ("there is a strong edge at exactly this pixel")

and its job shrinks from "hallucinate the detail" to "decide which of these
sharp edges are real, given what I know about the scene". That is a far easier
job, and it is why the boundary comes back.

THREE THINGS THAT TRIP PEOPLE UP, in the order they will trip you:

  1. cat, not add. torch.cat([up, skip], dim=1) stacks along the CHANNEL axis:
     64 channels + 64 channels = 128 channels, both kept separately. Adding
     them (the ResNet move) would sum them into 64 and mix the two sources
     irreversibly. Both designs exist in practice; U-Net concatenates, so the
     next conv can learn how much to trust each source.

  2. Because of (1), every decoder block's INPUT channel count is doubled.
     dec3 takes 256 in, not 128. This is the most common bug in a hand-written
     U-Net, and the error message -- a channel mismatch inside Conv2d --
     points at the conv, not at the cat that caused it.

  3. Sizes must line up exactly or the cat throws. padding=1 on every 3x3
     conv means resolution changes ONLY at the pool and upsample steps, so
     encoder level k and decoder level k are guaranteed to match. The "U" is
     not decoration -- the symmetry IS the alignment guarantee.

THE ABLATION. One run trains the same graph twice. UNet(use_skips=False)
builds the identical network with the skip half of every cat replaced by zeros. Same depth, same widths, same bottleneck, and --
as the table prints -- exactly the same parameter count. So the comparison is
skips against no skips, not a big model against a small one.

MEASURED RESULTS (M1 Pro, 15 epochs, all 3,680 labels, about 18 minutes):

    model                      params     acc    mIoU     pet      bg  border
    encoder-decoder (part3)  1,734,947   0.878   0.688   0.774   0.872   0.417
    U-Net, skips OFF         1,928,483   0.870   0.674   0.761   0.859   0.400
    U-Net, skips ON          1,928,483   0.887   0.707   0.787   0.881   0.455

Read it twice.

1. skips ON vs skips OFF: identical parameter count, identical depth, identical
   widths, identical bottleneck. The only difference is whether the encoder's
   high-resolution feature maps reach the decoder, and it is worth 0.033 mIoU.
   The largest per-class gain is the border, +0.055 -- the one class that is
   DEFINED by being near an edge. Part 5 pins this down properly: the gain is
   +0.048 for pixels within 1px of a boundary, and decays to zero by 16px away.

2. skips OFF vs Part 3's encoder-decoder: 194k MORE parameters, 0.014 LOWER
   mIoU. Below the bottleneck, capacity cannot buy back resolution that was
   already discarded -- only a path AROUND the bottleneck can. That is what a
   skip connection is, and it is why the U is shaped like a U.

ON THE SIZE OF THAT WIN, because honesty is worth more here than drama. 0.033
mIoU is real, reproducible and mechanism-confirmed, and it is smaller than the
U-Net's reputation would suggest. Two reasons, both about this setup rather
than about skip connections:

  * the waist here is 12x12, a gentle squeeze for a 96x96 image;
  * Part 3's decoder is already good on its own (0.688).

Part 2's waist sweep is the hint for what to do about it: a 6x6 waist starts
7 dB worse, so the tighter the waist, the more there is for the skips to
rescue. Adding a fourth level to this network and re-running the ablation is
the exercise to do next. The original paper went four levels deep on 572x572
microscope images, where the gap is not subtle.

    python part4_unet.py
    python part4_unet.py --epochs 30
    python part4_unet.py --shapes-only     # print the tensor shapes and exit
    python part4_unet.py --eval            # reload the saved weights, no training

Every run logs its per-epoch curves to trackio. Watch them live, or compare
runs after the fact, with:

    trackio show --project "week4-unet"

Set WEEK4_NO_TRACKIO=1 to turn the logging off.
"""

import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from common import (
    CLASS_SHORT,
    IMAGE_SIZE,
    N_CLASSES,
    DoubleConv,
    colorize_batch,
    count_params,
    evaluate_seg,
    finish,
    get_device,
    load_pet,
    load_weights,
    make_loaders,
    predict_batch,
    print_metrics,
    save_history,
    save_weights,
    seg_probe,
    show_grid,
    tracked,
    train,
)
from part3_encoder_decoder import SegAutoencoder


class UNet(nn.Module):
    """Part 3's network, plus three torch.cat calls.

        enc1  DoubleConv(3 -> 32)      32 x 96 x 96  ──────────────────┐
              MaxPool(2)                                               │ s1
        enc2  DoubleConv(32 -> 64)     64 x 48 x 48  ───────┐          │
              MaxPool(2)                                    │ s2       │
        enc3  DoubleConv(64 -> 128)   128 x 24 x 24  ──┐    │          │
              MaxPool(2)                               │ s3 │          │
        bott  DoubleConv(128 -> 256)  256 x 12 x 12    │    │          │
              ConvT(256 -> 128)       128 x 24 x 24    │    │          │
        dec3  DoubleConv(256 -> 128)  <- cat(128,128) ─┘    │          │
              ConvT(128 -> 64)         64 x 48 x 48         │          │
        dec2  DoubleConv(128 -> 64)   <- cat(64, 64) ───────┘          │
              ConvT(64 -> 32)          32 x 96 x 96                    │
        dec1  DoubleConv(64 -> 32)    <- cat(32, 32) ──────────────────┘
        head  Conv1x1(32 -> 3)         3 x 96 x 96    logits, no Sigmoid

    Compare that against Part 2's diagram: same left half, same bottleneck,
    same upsampling. The only edits are the three cat arrows and the three
    decoder input counts they force -- 256, 128, 64, each twice what the
    ConvTranspose above it produced. That doubling is gotcha (2) in the module
    docstring, and it is the bug you are about to write.

    The bottleneck is still 256 x 12 x 12, deliberately: if the detail comes
    back, it cannot be because we gave the model an easier squeeze.

    (The original 2015 paper went four levels deep on 572x572 microscope
    images and used no padding, so it had to crop every skip to make the
    shapes match. Same structure; we stop earlier because our images are
    smaller, and pad so the shapes match by construction.)
    """

    def __init__(self, in_ch=3, n_classes=N_CLASSES, base=32, use_skips=True):
        super().__init__()
        self.use_skips = use_skips
        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(in_ch, base)  # 96x96
        self.enc2 = DoubleConv(base, base * 2)  # 48x48
        self.enc3 = DoubleConv(base * 2, base * 4)  # 24x24
        self.bottleneck = DoubleConv(base * 4, base * 8)  # 12x12

        # Each ConvTranspose halves the channel count; the cat then doubles it
        # straight back, which is why every dec block takes 2x what up gives.
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)  # base*8 = 128 up + 128 skip
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)  # base*4 =  64 up +  64 skip
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)  # base*2 =  32 up +  32 skip

        # A 1x1 conv is a per-pixel linear layer across channels: 32 feature
        # values at each pixel -> 3 class scores, with no spatial mixing. Raw
        # logits, exactly as in Part 3.
        self.head = nn.Conv2d(base, n_classes, kernel_size=1)

    def _join(self, up, skip):
        """cat along channels -- or cat with zeros when running the ablation.

        Zeroing rather than removing keeps the channel counts, and therefore
        the parameter count, identical. The skipless model has exactly as many
        weights to work with; it just receives no information through them.
        """
        if not self.use_skips:
            skip = torch.zeros_like(skip)
        return torch.cat([up, skip], dim=1)  # dim=1 is the channel axis

    def forward(self, x):
        # TODO:
        # ---- encoder: keep every feature map on the way down ----
        # s1 = self.enc1(x)                     #  32 x 96 x 96  <- finest detail
        # s2 = self.enc2(self.pool(s1))         #  64 x 48 x 48
        # s3 = self.enc3(self.pool(s2))         # 128 x 24 x 24
        # b = self.bottleneck(self.pool(s3))    # 256 x 12 x 12  <- same as Part 3

        # # ---- decoder: upsample, re-attach the matching encoder map ----
        # d3 = self.dec3(self._join(self.up3(b), _))    # 128 x 24 x 24
        # d2 = self.dec2(self._join(self.up2(d3), _))   #  64 x 48 x 48
        # d1 = self.dec1(self._join(self.up1(d2), _))   #  32 x 96 x 96
        return self.head(d1)


def describe_shapes(model, device):
    """Print the real tensor shapes from one forward pass.

    Cheap, and it turns the ASCII diagram above into something verified rather
    than believed. If you change the architecture, run this first.
    """
    hooks, log = [], []
    for name in [
        "enc1",
        "enc2",
        "enc3",
        "bottleneck",
        "up3",
        "dec3",
        "up2",
        "dec2",
        "up1",
        "dec1",
        "head",
    ]:
        hooks.append(
            getattr(model, name).register_forward_hook(
                lambda m, i, o, n=name: log.append(
                    (n, i[0].shape[1], tuple(o.shape[1:]))
                )
            )
        )
    with torch.no_grad():
        model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device))
    for h in hooks:
        h.remove()

    print(f"U-Net shapes for one 3 x {IMAGE_SIZE} x {IMAGE_SIZE} input:")
    print(f"  {'layer':<11} {'in_ch':>6}  ->  {'output (C, H, W)':<18}")
    for name, in_ch, shape in log:
        note = ""
        if name.startswith("dec"):
            note = f"  <- cat({in_ch // 2} up + {in_ch // 2} skip) = {in_ch} in"
        print(f"  {name:<11} {in_ch:>6}  ->  {str(shape):<18}{note}")
    print()


def zoom_figure(rows, titles, truth, box=44):
    """Crop the busiest boundary in the batch, so 'sharper' stops being an opinion.

    At thumbnail size every segmentation looks fine, and a crop of the middle
    of a dog shows three models agreeing that a dog is a dog. The argument of
    this file lives on the outline, so we go and find one: an average-pool over
    the border-class mask scores every possible box by how much boundary it
    contains, and we crop the winner.
    """
    border = (truth == 2).float().unsqueeze(1)  # (N, 1, H, W)
    dens = torch.nn.functional.avg_pool2d(border, kernel_size=box, stride=4)
    flat = dens.flatten()
    n, _, gh, gw = dens.shape
    best = int(flat.argmax())
    idx, r, c = best // (gh * gw), (best % (gh * gw)) // gw, (best % (gh * gw)) % gw
    top, left = min(r * 4, IMAGE_SIZE - box), min(c * 4, IMAGE_SIZE - box)

    fig, axes = plt.subplots(1, len(rows), figsize=(2.6 * len(rows), 3.0))
    for ax, batch, title in zip(axes, rows, titles):
        ax.imshow(
            batch[idx, :, top : top + box, left : left + box]
            .permute(1, 2, 0)
            .clamp(0, 1)
        )
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("The busiest boundary in the batch, magnified")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument(
        "--shapes-only",
        action="store_true",
        help="print the shape table and exit, no training",
    )
    parser.add_argument(
        "--skip-part3", action="store_true", help="only run the skips ON/OFF ablation"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="skip training: load the checkpoints a previous run "
        "saved and print the same table and figure",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = get_device()
    print(f"using device: {device}\n")

    describe_shapes(UNet().to(device), device)
    if args.shapes_only:
        return

    train_ds = load_pet("trainval", n_subset=args.n_train, augment=True)
    test_ds = load_pet("test", n_subset=1000)
    train_loader, test_loader = make_loaders(train_ds, test_ds)

    loss_fn = nn.CrossEntropyLoss()
    rows, titles, table = [], [], []

    # Built lazily so the seed reset below applies to each model's init as
    # well as to the batch shuffling -- otherwise part of what we would be
    # comparing is which random draw each model happened to get.
    contenders = [
        ("U-Net, skips OFF", lambda: UNet(use_skips=False), "part4_unet_skips_off"),
        ("U-Net, skips ON", lambda: UNet(use_skips=True), "part4_unet_skips_on"),
    ]
    if not args.skip_part3:
        contenders.insert(
            0, ("encoder-decoder (part3)", SegAutoencoder, "part4_encdec")
        )

    for label, build, stem in contenders:
        torch.manual_seed(0)  # same init draw for every contender
        model = build()
        print(f"=== {label} -- {count_params(model):,} parameters ===")
        # The label budget is part of the run's identity: without it in the
        # name, a --n-train 500 run would overwrite the full-data weights.
        key = f"{stem}_n{args.n_train}" if args.n_train else stem
        if args.eval:
            load_weights(model, key)
            model.to(device)
        else:
            with tracked(
                label,
                config=dict(
                    part=4,
                    arch=stem,
                    params=count_params(model),
                    skips=isinstance(model, UNet) and model.use_skips,
                    epochs=args.epochs,
                    lr=1e-3,
                    n_train=len(train_ds),
                ),
            ) as run:
                history = train(
                    model,
                    train_loader,
                    device,
                    loss_fn,
                    epochs=args.epochs,
                    lr=1e-3,
                    run=run,
                    eval_fn=seg_probe(test_loader),
                )

        acc, iou, miou = evaluate_seg(model, test_loader)
        print_metrics("test", acc, iou, miou)
        print()
        table.append((label, count_params(model), acc, iou, miou))
        if not args.eval:
            save_weights(model, key)
            save_history(
                key, label.strip(), count_params(model), history, (acc, iou, miou)
            )

        x, y, pred = predict_batch(model, test_loader)
        if not rows:
            rows += [x, colorize_batch(y)]
            titles += ["photo", "truth"]
        rows.append(colorize_batch(pred))
        titles.append(label)

    head = f"{'model':<24} {'params':>10} {'acc':>7} {'mIoU':>7}"
    head += "".join(f"{n:>9}" for n in CLASS_SHORT)
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for label, params, acc, iou, miou in table:
        line = f"{label:<24} {params:>10,} {acc:>7.3f} {miou:>7.3f}"
        line += "".join(f"{v:>9.3f}" for v in iou)
        print(line)
    print("=" * len(head))

    print("\nTwo readings, and the second is the important one.")
    print("\n1. skips ON vs skips OFF. Same depth, same widths, the same")
    print("   parameter count, the same 12x12 bottleneck. The only difference")
    print("   is whether the encoder's high-resolution feature maps reach the")
    print("   decoder. Check which class moved most: the border, the one class")
    print("   defined by sitting on an edge. Part 5 measures that properly.")
    print("\n2. skips OFF vs the plain encoder-decoder. The skipless U-Net is")
    print("   the LARGER model and it scores LOWER. Below the bottleneck, extra")
    print("   capacity cannot recover resolution that was already discarded.")
    print("   Only a path around the bottleneck can -- and that is all a skip")
    print("   connection is.")
    print("\nThe honest footnote: this gain is real but modest, because a")
    print("12x12 waist is a gentle squeeze and part 3's decoder was already")
    print("good. Part 2's waist sweep says a 6x6 waist starts 7 dB worse --")
    print("so a deeper U-Net has more to rescue. That is the next exercise.")
    print("\nWhere this goes next: the same U-Net, with the target swapped, is")
    print("a denoiser, an inpainter, a depth estimator, and the backbone inside")
    print("a diffusion model. The encoder-decoder-with-skips shape is the")
    print("standard answer whenever the output is an image.")

    show_grid(
        rows, titles, suptitle="Segmentation: the skip connections are the difference"
    )
    zoom_figure(rows, titles, y)
    finish("part4")


if __name__ == "__main__":
    main()
