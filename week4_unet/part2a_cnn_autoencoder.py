"""Week 4 - Part 2: the decoder. Learning to make an image, not a label.

Part 1 ended in a deadlock: pooling buys context and costs resolution, and
its way of getting the resolution back -- stretch the 12x12 logits 8x with
bilinear interpolation -- is a fixed formula that learns nothing.

So make the way back up *learned*. That is a decoder, and this file builds one
on the easiest possible target: no labels at all, the photo is its own answer.

    encoder:  3 x 96 x 96  ->  128 x 12 x 12        (three poolings, as in part 1)
    decoder:  128 x 12 x 12 ->  3 x 96 x 96         (three upsamplings, new)
    loss:     MSELoss(reconstruction, original)

That is an autoencoder, and we are here for exactly two reasons: it is the
shortest path to understanding the decoder half, and its failure mode is the
whole argument of Part 4.

THE TRAINING LOOP DOES NOT CHANGE. Week 1 Part 3's four steps, with
`loss_fn(model(x), y)` where y happens to be x. The supervision signal comes
from the data itself, which is the first time in this course that no label
file is involved anywhere.

TWO WAYS TO UPSAMPLE, and the file runs both:

  ConvTranspose2d(k=2, s=2)   learns a 2x2 patch to paint for each input cell.
                              Fast, standard, and prone to CHECKERBOARD
                              artifacts when the kernel size is not divisible
                              by the stride (k=3, s=2 is the classic offender)
                              because neighbouring output pixels then get
                              different numbers of contributions.

  Upsample(2) + Conv2d(3x3)   resize first (a fixed formula, no parameters),
                              then let a normal conv clean it up. Slightly
                              slower, no checkerboard by construction. This is
                              what most modern code does.

What this model CANNOT do is the point of the week. Everything the decoder
knows has to fit through 128 x 12 x 12, so whiskers, fur texture and the exact
outline of an ear are gone before the decoder starts. It returns a smooth,
plausible, slightly melted cat. That is not a bug in the training -- it is
what a bottleneck is.

MEASURED RESULTS (M1 Pro, 15 epochs, full trainval, about 9 minutes total):

    decoder                     params      MSE   PSNR (dB)
    ConvTranspose2d          1,734,947   0.0040       23.97
    Upsample + Conv2d        1,949,987   0.0030       25.22

Upsample + Conv2d wins by 1.25 dB -- worth having, and worth keeping in
proportion: it is also the bigger model, because a 3x3 conv carries more
weights than a 2x2 transposed conv. Look at the two reconstructions side by
side and you will struggle to say which is which.

That is the finding. Both are blurry, in the same way, for a reason that has
nothing to do with the upsampler -- and the next file proves it. Part 2b
trains this same autoencoder with 2, 3 and 4 poolings and watches the picture
get worse as the waist gets smaller, while the parameter count goes UP.

    python part2a_cnn_autoencoder.py
    python part2a_cnn_autoencoder.py --epochs 30
    python part2a_cnn_autoencoder.py --eval  # reload the saved weights, no training
    python part2b_waist_sweep.py            # the control experiment

Every run logs its per-epoch curves to trackio. Watch them live, or compare
runs after the fact, with:

    trackio show --project "week4-unet"

Set WEEK4_NO_TRACKIO=1 to turn the logging off.
"""

import argparse

import torch
import torch.nn as nn

from common import (DoubleConv, count_params, finish, get_device,
                    load_pet, load_weights, make_loaders, save_weights,
                    show_grid, tracked, train, undisturbed)


def up_block(in_ch, out_ch, mode):
    """One resolution-doubling step, in either of the two standard styles."""
    if mode == "convt":
        # kernel_size=2, stride=2: each input cell paints its own 2x2 tile,
        # tiles never overlap, so this particular setting is checkerboard-free.
        # To see the artifact the docstring warns about, use
        #     kernel_size=3, stride=2, padding=1, output_padding=1
        # -- 3 is not divisible by 2, so the tiles overlap unevenly and some
        # output pixels collect more contributions than their neighbours. The
        # padding args are only there to keep the output exactly 2x; without
        # them a k=3 transposed conv turns 12x12 into 25x25, not 24x24.
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
    )


class ConvAutoencoder(nn.Module):
    """Encoder down to 12x12, decoder back up to 96x96. No shortcuts anywhere.

        enc1  DoubleConv(3 -> 32)      32 x 96 x 96
              MaxPool(2)
        enc2  DoubleConv(32 -> 64)     64 x 48 x 48
              MaxPool(2)
        enc3  DoubleConv(64 -> 128)   128 x 24 x 24
              MaxPool(2)
        bott  DoubleConv(128 -> 256)  256 x 12 x 12   <- everything fits here
        up3   up(256 -> 128)          128 x 24 x 24
        dec3  DoubleConv(128 -> 128)
        up2   up(128 -> 64)            64 x 48 x 48
        dec2  DoubleConv(64 -> 64)
        up1   up(64 -> 32)             32 x 96 x 96
        dec1  DoubleConv(32 -> 32)
        head  Conv1x1(32 -> 3) -> Sigmoid

    Every label on the left is an attribute name, so the diagram and __init__
    can be read side by side. Part 4's U-Net uses the same names for the same
    blocks -- put the two diagrams next to each other and the only differences
    are three cat arrows and the channel counts they force.

    The shape is symmetric on purpose: every pooling on the left has a
    matching upsampling on the right, so the output lands on exactly the input
    size with no cropping.

    Sigmoid on the head because the target is pixels in [0, 1], which is the
    range ToTensor produced. Match the two or the model spends its whole
    budget fighting its own output layer. (In Part 3 the target stops being
    pixels, and the Sigmoid goes.)
    """

    def __init__(self, in_ch=3, out_ch=3, base=32, mode="convt"):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # ---- encoder: three blocks, one pooling after each ----
        self.enc1 = DoubleConv(in_ch, base)                   #  32 x 96 x 96
        self.enc2 = DoubleConv(base, base * 2)                #  64 x 48 x 48
        self.enc3 = DoubleConv(base * 2, base * 4)            # 128 x 24 x 24
        self.bottleneck = DoubleConv(base * 4, base * 8)      # 256 x 12 x 12

        # ---- decoder: one upsampling per pooling, in reverse ----
        # Each up_block halves the channels and doubles the resolution, so the
        # decoder retraces the encoder's shapes exactly backwards.
        self.up3 = up_block(base * 8, base * 4, mode)         # 128 x 24 x 24
        self.dec3 = DoubleConv(base * 4, base * 4)
        self.up2 = up_block(base * 4, base * 2, mode)         #  64 x 48 x 48
        self.dec2 = DoubleConv(base * 2, base * 2)
        self.up1 = up_block(base * 2, base, mode)             #  32 x 96 x 96
        self.dec1 = DoubleConv(base, base)

        self.head = nn.Sequential(nn.Conv2d(base, out_ch, 1), nn.Sigmoid())

    def forward(self, x):
        # ---- down. Each block looks at the image, then throws away half the
        #      resolution. Note what happens to e1, e2, e3: nothing. They are
        #      computed, used once, and dropped. That is Part 4's whole point.
        e1 = self.enc1(x)                      #  32 x 96 x 96  <- finest detail
        e2 = self.enc2(self.pool(e1))          #  64 x 48 x 48
        e3 = self.enc3(self.pool(e2))          # 128 x 24 x 24
        b = self.bottleneck(self.pool(e3))     # 256 x 12 x 12  <- the waist

        # ---- up. Everything below here is rebuilt from b and nothing else.
        #      Worth noticing: b holds MORE numbers (256x12x12 = 36,864) than
        #      the output has pixels (3x96x96 = 27,648). The squeeze is not
        #      capacity, it is RESOLUTION -- one waist cell per 8x8 block of
        #      input, so nothing narrower than 8 pixels has a place to live.
        d3 = self.dec3(self.up3(b))            # 128 x 24 x 24
        d2 = self.dec2(self.up2(d3))           #  64 x 48 x 48
        d1 = self.dec1(self.up1(d2))           #  32 x 96 x 96
        return self.head(d1)


def psnr(mse):
    """Peak signal-to-noise ratio in dB, for pixels in [0, 1]: 10*log10(1/MSE).

    Carries no information MSE doesn't. It is here because image papers quote
    it, and because "+3 dB" is easier to feel than "0.0041 vs 0.0082" -- +3 dB
    means the error energy halved.
    """
    return float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


@torch.no_grad()
def eval_mse(model, loader):
    device = next(model.parameters()).device
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += torch.mean((model(x) - y) ** 2).item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def reconstruct(model, loader, n=6):
    device = next(model.parameters()).device
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n]
    return x, model(x.to(device)).cpu()


def recon_probe(loader):
    """-> a per-epoch measurement function for train(eval_fn=...).

    The segmentation parts log IoU; this file has no classes to score, so it
    logs the two numbers it does have. PSNR is a restatement of MSE, but it is
    the one that reads on a chart.
    """
    @undisturbed
    def probe(model):
        mse = eval_mse(model, loader)
        model.train()               # eval_mse left it in eval mode
        return {"test/mse": mse, "test/psnr": psnr(mse)}
    return probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--eval", action="store_true",
                        help="skip training: load part2_ae_convt.pt and "
                             "part2_ae_upsample.pt and print the same table")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = get_device()
    print(f"using device: {device}\n")

    # segmentation=False: the dataset hands back (photo, photo) and the masks
    # are never read. No labels are involved in this file at all.
    train_ds = load_pet("trainval", n_subset=args.n_train, segmentation=False)
    test_ds = load_pet("test", n_subset=1000, segmentation=False)
    train_loader, test_loader = make_loaders(train_ds, test_ds)

    loss_fn = nn.MSELoss()      # the target is pixels, so squared error
    rows, titles, table = [], [], []

    for label, mode in [("ConvTranspose2d", "convt"), ("Upsample + Conv2d", "upsample")]:
        torch.manual_seed(0)
        model = ConvAutoencoder(mode=mode)
        print(f"=== {label} decoder -- {count_params(model):,} parameters ===")
        # The training-set size is part of the run's identity, as in parts 3
        # and 4: a quick --n-train run must not overwrite the full-data weights
        # that part 3's pretraining loads.
        key = f"part2_ae_{mode}_n{args.n_train}" if args.n_train else f"part2_ae_{mode}"
        if args.eval:
            load_weights(model, key)
            model.to(device)
        else:
            with tracked(label, config=dict(part=2, arch=mode,
                                            params=count_params(model),
                                            poolings=3,
                                            epochs=args.epochs, lr=1e-3,
                                            n_train=len(train_ds))) as run:
                train(model, train_loader, device, loss_fn, epochs=args.epochs,
                      lr=1e-3, run=run, eval_fn=recon_probe(test_loader))

        mse = eval_mse(model, test_loader)
        print(f"  test MSE {mse:.4f}   PSNR {psnr(mse):.2f} dB\n")
        table.append((label, count_params(model), mse, psnr(mse)))
        if not args.eval:
            save_weights(model, key)

        x, recon = reconstruct(model, test_loader)
        if not rows:
            rows.append(x)
            titles.append("original")
        rows.append(recon)
        titles.append(label)

    print("=" * 56)
    print(f"{'decoder':<22} {'params':>10} {'MSE':>8} {'PSNR (dB)':>11}")
    print("-" * 56)
    for label, params, mse, db in table:
        print(f"{label:<22} {params:>10,} {mse:>8.4f} {db:>11.2f}")
    print("=" * 56)

    print("\nUpsample + Conv2d comes out ahead, and it is also the bigger of")
    print("the two models -- a 3x3 conv carries more weights than a 2x2")
    print("transposed conv. Now look at the two rows of pictures and try to")
    print("say which is which. The difference the numbers report is real and")
    print("it is not what you are looking at.")
    print("\nWhat you are looking at is in both rows equally. The dog is the")
    print("right dog, in the right pose, the right colour, against the right")
    print("background -- and it is made of smooth paint. Find an eye, or the")
    print("texture of the fur, or a whisker: gone in both.")
    print("\nThat is the bottleneck doing exactly what it is built to do. Every")
    print("pixel of the output is reconstructed from 256 x 12 x 12 numbers, and")
    print("a whisker three pixels wide left no trace in them. Run")
    print("part2b_waist_sweep.py to watch that claim get tested: more")
    print("poolings, worse pictures, four times the parameters each step.")
    print("Capacity is not the thing that is missing.")
    print("\nSo here is the thing worth being annoyed about: on the way down,")
    print("the first encoder block computed a 32 x 96 x 96 feature map that")
    print("still HAD the whiskers in it. We used it once, pooled it, and threw")
    print("it away. Part 4 is that one sentence, turned into an architecture.")

    show_grid(rows, titles, suptitle="Reconstruction through a 12x12 waist: shape survives, detail does not")
    finish("part2")


if __name__ == "__main__":
    main()
