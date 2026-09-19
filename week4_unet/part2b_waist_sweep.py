"""Week 4 - Part 2b: the control experiment. Is it the waist, or the capacity?

Part 2a ended on a claim, and a claim is not evidence. Both decoders returned
the same smooth, melted cat, and Part 2a asserted that the blur comes from the
bottleneck -- that detail dies at the 12x12 waist and no decoder can invent it
back.

That is testable. If the waist is the cause, then changing ONLY the waist
should change the picture, and the ranking should follow the waist rather than
anything else. So train the same autoencoder three times, with two, three and
four poolings, and change nothing else:

    poolings    waist       params      MSE     PSNR
           2    24x24      421,795   0.0015    28.19
           3    12x12    1,734,947   0.0040    23.97
           4      6x6    6,982,691   0.0076    21.21

(M1 Pro, 15 epochs each, full trainval, about 11 minutes for all three.)

READ THE PARAMETER COLUMN AND THE PSNR COLUMN TOGETHER. They run in opposite
directions. The 6x6 waist has 16.6x the weights of the 24x24 waist and
reconstructs 7 dB worse. Each extra pooling costs 2-4 dB and pays FOUR TIMES
the parameters for the privilege, because every level added is a whole stage
with twice the channels of the one above it.

That is what makes this a control rather than a demo. If capacity were the
thing that was missing, the ranking would come out backwards. It does not.
Resolution is what is missing, and once it is gone at the waist, no amount of
decoder gets it back.

Which sets up the question the rest of the week answers. We cannot simply keep
the 24x24 waist and go home: Part 1's demo showed we need the deep pooling for
CONTEXT, to know that this is an animal at all. We need the deep waist AND the
detail, and so far those have been a trade. Part 4 stops trading.

    python part2b_waist_sweep.py
    python part2b_waist_sweep.py --depths 2 3 4 5 --epochs 5
    python part2b_waist_sweep.py --eval     # reload the saved weights, no training

Every run logs its per-epoch curves to trackio, one run per waist:

    trackio show --project "week4-unet"

Set WEEK4_NO_TRACKIO=1 to turn the logging off.
"""

import argparse

import torch
import torch.nn as nn

from common import (IMAGE_SIZE, DoubleConv, count_params, get_device,
                    load_pet, load_weights, make_loaders, save_weights, tracked,
                    train)
from part2a_cnn_autoencoder import (ConvAutoencoder, eval_mse, psnr,
                                    recon_probe, up_block)


class VariableWaistAutoencoder(nn.Module):
    """Part 2a's autoencoder with the number of levels turned into a knob.

    Read part2a.ConvAutoencoder first. It spells its three levels out by name
    -- enc1, enc2, enc3, up3, dec3, and so on -- because that is the version
    you learn the architecture from, and every name there matches a line of
    the diagram in its docstring.

    This class is the same network with those names replaced by lists, and it
    exists for exactly one reason: this experiment varies the NUMBER of levels,
    and you cannot vary something that is written out by hand. That is the
    whole trade. Anywhere else, prefer the explicit version.

    depth=3 must therefore be the same network as Part 2a's, not merely a
    similar one -- otherwise the sweep's middle row would not be comparable to
    Part 2a's result. main() asserts it.

        depth=2   3 -> 32 -> 64          waist 128 @ 24x24
        depth=3   3 -> 32 -> 64 -> 128   waist 256 @ 12x12   <- Part 2a
        depth=4   3 -> 32 -> 64 -> 128 -> 256    waist 512 @ 6x6
    """

    def __init__(self, depth=3, in_ch=3, out_ch=3, base=32, mode="convt"):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.depth = depth

        # depth=3 -> [32, 64, 128]: one channel count per encoder level.
        widths = [base * 2 ** i for i in range(depth)]

        # Built in exactly Part 2a's order -- encoders, bottleneck, then each
        # (up, dec) pair -- so that the same seed produces the same weights.
        self.encoders = nn.ModuleList()          # enc1, enc2, enc3, ...
        ch = in_ch
        for w in widths:
            self.encoders.append(DoubleConv(ch, w))
            ch = w

        self.bottleneck = DoubleConv(ch, ch * 2)  # the waist
        ch = ch * 2

        self.ups = nn.ModuleList()               # up3, up2, up1, ...
        self.decoders = nn.ModuleList()          # dec3, dec2, dec1, ...
        for w in reversed(widths):
            self.ups.append(up_block(ch, w, mode))
            self.decoders.append(DoubleConv(w, w))
            ch = w

        self.head = nn.Sequential(nn.Conv2d(ch, out_ch, 1), nn.Sigmoid())

    def forward(self, x):
        for enc in self.encoders:         # each level: convolve, then halve
            x = self.pool(enc(x))
        x = self.bottleneck(x)            # the waist; everything must fit here
        for up, dec in zip(self.ups, self.decoders):
            x = dec(up(x))                # each level: double, then convolve
        return self.head(x)

    def waist(self, image_size=IMAGE_SIZE):
        """Side length of the bottleneck feature map: one halving per level."""
        return image_size // 2 ** self.depth


def assert_matches_part2a():
    """depth=3 is Part 2a's network, weight for weight.

    Cheap to check and worth checking: if the two ever drift apart, the middle
    row of the sweep stops being comparable to Part 2a's headline number, and
    nothing about the output would tell you.
    """
    torch.manual_seed(0)
    explicit = ConvAutoencoder(mode="convt")
    torch.manual_seed(0)
    looped = VariableWaistAutoencoder(depth=3, mode="convt")

    assert count_params(explicit) == count_params(looped), "parameter counts differ"
    explicit.eval()
    looped.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert torch.equal(explicit(x), looped(x)), "same seed, different output"
    print(f"  depth=3 matches part 2a exactly "
          f"({count_params(looped):,} parameters, identical outputs)\n")


def waist_sweep(train_loader, test_loader, device, epochs, depths,
                n_train=None, evaluate_only=False):
    """Train the same autoencoder at each depth and tabulate the result.

    Each depth's weights go to checkpoints/part2b_waist<side>.pt, so --eval can
    rebuild the table later without paying for the training again.
    """
    print("=== waist sweep: how much does the bottleneck cost? ===")
    rows = []
    for depth in depths:
        torch.manual_seed(0)
        model = VariableWaistAutoencoder(depth=depth, mode="convt")
        waist, params = model.waist(), count_params(model)
        key = f"part2b_waist{waist}" + (f"_n{n_train}" if n_train else "")
        print(f"\n--- {depth} poolings -> {waist}x{waist} waist, "
              f"{params:,} parameters ---")
        if evaluate_only:
            load_weights(model, key)
            model.to(device)
        else:
            with tracked(f"waist {waist}x{waist}",
                         config=dict(part=2, sweep="waist", poolings=depth,
                                     waist=waist, params=params, epochs=epochs,
                                     lr=1e-3)) as run:
                train(model, train_loader, device, nn.MSELoss(), epochs=epochs,
                      lr=1e-3, run=run, eval_fn=recon_probe(test_loader))
            save_weights(model, key)
        mse = eval_mse(model, test_loader)
        print(f"  test MSE {mse:.4f}   PSNR {psnr(mse):.2f} dB")
        rows.append((depth, waist, params, mse, psnr(mse)))

    print("\n" + "=" * 58)
    print(f"{'poolings':>8} {'waist':>8} {'params':>11} {'MSE':>8} {'PSNR':>8}")
    print("-" * 58)
    for depth, waist, params, mse, db in rows:
        print(f"{depth:>8} {f'{waist}x{waist}':>8} {params:>11,} {mse:>8.4f} {db:>8.2f}")
    print("=" * 58)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4],
                        help="poolings per run; each one halves the waist")
    parser.add_argument("--eval", action="store_true",
                        help="skip training: load part2b_waist<side>.pt for each "
                             "depth and print the same table")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = get_device()
    print(f"using device: {device}\n")
    assert_matches_part2a()

    # segmentation=False: (photo, photo) pairs. No labels here either.
    train_ds = load_pet("trainval", n_subset=args.n_train, segmentation=False)
    test_ds = load_pet("test", n_subset=1000, segmentation=False)
    train_loader, test_loader = make_loaders(train_ds, test_ds)

    rows = waist_sweep(train_loader, test_loader, device, args.epochs, args.depths,
                       n_train=args.n_train, evaluate_only=args.eval)

    best = max(rows, key=lambda r: r[4])
    worst = min(rows, key=lambda r: r[4])
    print(f"\nThe {best[1]}x{best[1]} waist reconstructs {best[4] - worst[4]:.1f} dB better "
          f"than the {worst[1]}x{worst[1]} waist, using")
    print(f"{worst[2] / best[2]:.1f}x FEWER parameters to do it. Capacity is not the thing")
    print("that is missing. Resolution is, and the waist is where it goes --")
    print("Part 4 keeps the deep waist and gets the detail back anyway.")


if __name__ == "__main__":
    main()
