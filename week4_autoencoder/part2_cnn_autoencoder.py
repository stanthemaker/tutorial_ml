"""Week 4 - Part 2: Give the decoder back its second dimension.

Part 1's very first line of work was `x.view(-1, 784)`. Week 3 spent a whole
session on why that is wasteful for the encoder: a Conv2d reuses one small
kernel everywhere and keeps neighbouring pixels neighbours. So the encoder
half is a solved problem -- we already know how to write it.

The new question is the decoder. Conv + pool make images *smaller*. To get
back from 7x7 to 28x28 we need the opposite operation, and there are exactly
two ways people do it:

  (a) ConvTranspose2d(stride=2)          -- a learned upsample
  (b) Upsample(scale_factor=2) + Conv2d  -- a fixed upsample, then a learned
                                            conv to clean it up

This file builds both, trains them side by side on MNIST and CIFAR-10, and
compares them against Part 1's MLP.

What ConvTranspose2d actually does (it is NOT "deconvolution" -- it does not
invert anything). A normal conv takes a patch and produces one number.
A transposed conv takes one number and produces a patch: it multiplies the
whole kernel by that single input value, stamps the result into the output at
a stride-2 offset, and *sums* wherever the stamps overlap.

    input 1x1 -> kernel 4x4, stride 2:  one input pixel paints a 4x4 block
    input 2x2 -> output 4x4:            blocks land 2 apart and overlap by 2

That overlap is the whole story of the checkerboard artifact. Where kernel
size is not divisible by stride, some output pixels get contributions from
more stamps than their neighbours, so they come out systematically brighter,
and you see a regular grid pattern in the output. kernel=4/stride=2 divides
evenly (used below); kernel=3/stride=2 does not, and we show that case too.

Output size for a transposed conv (the mirror of Week 3 Part 1's formula):

    out = (in - 1) * stride - 2 * padding + kernel + output_padding

Check it: (7 - 1) * 2 - 2 * 1 + 4 + 0 = 14.  And (14 - 1) * 2 - 2 + 4 = 28.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from common import (get_device, count_params, load_mnist, load_cifar10,
                    make_loaders, show_grid, psnr, eval_mse, sample_batch,
                    save_weights)
from part1_mlp_autoencoder import build_autoencoder


# --------------------------------------------------------------------------
# the models
# --------------------------------------------------------------------------

def build_conv_ae(in_ch=1, base=16, latent_ch=32, upsample="convt"):
    """Conv encoder + upsampling decoder, two levels deep.

    Shape walk-through for MNIST (in_ch=1):

        input                          1 x 28 x 28
        Conv(1  -> 16, k3, s2, p1)    16 x 14 x 14   downsample by striding
        Conv(16 -> 32, k3, s2, p1)    32 x  7 x  7   <- the bottleneck
        ConvT(32 -> 16, k4, s2, p1)   16 x 14 x 14   upsample
        ConvT(16 -> 1,  k4, s2, p1)    1 x 28 x 28
        Sigmoid                                      pixels back in [0, 1]

    CIFAR-10 (32x32) runs the same code and lands on 32x8x8 in the middle.

    Two notes on the choices:

      * We downsample with stride=2 convolutions here instead of MaxPool.
        Both halve the image; the strided conv *learns* how to summarise each
        2x2 neighbourhood while MaxPool hardcodes "keep the brightest". Part 3
        switches to MaxPool so you have seen both -- neither is wrong.

      * kernel_size=4 with stride=2 in the decoder, not 3. 4 is divisible by
        2, so every output pixel receives the same number of overlapping
        stamps and there is no checkerboard. Try changing it to 3 (or run
        compare_upsampling below) to see the grid pattern appear.
    """
    encoder = nn.Sequential(
        nn.Conv2d(in_ch, base, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(base, latent_ch, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
    )

    if upsample == "convt":
        decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, base, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base, in_ch, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    elif upsample == "convt_k3":
        # Deliberately broken-looking: kernel 3 is not divisible by stride 2.
        # output_padding=1 is needed to hit the right size, and the uneven
        # overlap is what produces the checkerboard.
        decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, base, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base, in_ch, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid(),
        )
    elif upsample == "resize":
        # The artifact-free alternative: enlarge with plain interpolation
        # (no learned weights, no overlap, so no checkerboard), then let a
        # stride-1 conv decide what the enlarged map should actually contain.
        decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(latent_ch, base, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base, in_ch, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
    else:
        raise ValueError(f"unknown upsample mode: {upsample}")

    return nn.Sequential(encoder, decoder)


# --------------------------------------------------------------------------
# a picture of one transposed convolution, before we train anything
# --------------------------------------------------------------------------

def demo_transpose_mechanics():
    """Stamp one kernel with a hand-set ConvTranspose2d and look at the result.

    No training, no dataset -- a 6x6 input of ones and a kernel of ones, so
    the output value at each position is literally "how many stamps landed
    here".

    Read the *interior* of the printed maps, ignoring the outermost ring
    (every setting is uneven at the border simply because the image ends
    there, and that edge effect is not what we are talking about):

        kernel 4, stride 2 -> interior is a flat field of 4s
        kernel 3, stride 2 -> interior alternates 1, 2, 1, 2, ...

    That alternating pattern IS the checkerboard, and it is there before a
    single gradient step. Training can partly compensate by shrinking the
    over-covered weights, but nothing forces it to, so the pattern usually
    survives into the output image.
    """
    x = torch.ones(1, 1, 6, 6)
    print("ConvTranspose2d: one input pixel paints a whole kernel-sized block.")
    print("With an all-ones kernel the output value = number of overlapping stamps.\n")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    axes[0].imshow(x[0, 0], cmap="gray", vmin=0, vmax=4)
    axes[0].set_title("input\n6x6 of ones")

    for ax, k in zip(axes[1:], (4, 3)):
        pad, outpad = (1, 0) if k == 4 else (1, 1)
        ct = nn.ConvTranspose2d(1, 1, kernel_size=k, stride=2,
                                padding=pad, output_padding=outpad, bias=False)
        with torch.no_grad():
            ct.weight.fill_(1.0)
            out = ct(x)
        ax.imshow(out[0, 0], cmap="gray")
        divides = "divides" if k % 2 == 0 else "does NOT divide"
        ax.set_title(f"kernel {k}, stride 2 -> {tuple(out.shape[-2:])}\n"
                     f"kernel {divides} stride")

        # drop the outer ring: we care about the tiling pattern, not the edge
        interior = out[0, 0, 2:-2, 2:-2]
        counts = sorted(interior.flatten().unique().tolist())
        verdict = "uniform -> no artifact" if len(counts) == 1 else \
                  "alternating -> CHECKERBOARD"
        print(f"  kernel {k}, stride 2: output {tuple(out.shape[-2:])}, "
              f"interior overlap counts {counts}  <- {verdict}")
        print(f"{interior.int()}\n")

    for ax in axes:
        ax.axis("off")
    fig.suptitle("Why kernel size should be divisible by stride\n"
                 "(brightness = how many overlapping stamps hit that pixel)")
    fig.tight_layout()


# --------------------------------------------------------------------------
# training (same loop as Part 1, but no flattening anywhere)
# --------------------------------------------------------------------------

def train(model, loader, device, epochs=10, lr=1e-3, quiet=False, tag=""):
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, _ in loader:
            x = x.to(device)          # note: NO .view() -- images stay 2D
            recon = model(x)
            loss = loss_fn(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
        if not quiet:
            print(f"  {tag}epoch {epoch + 1}/{epochs}  MSE {losses[-1]:.5f}")
    return losses


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------

def compare_upsampling(train_loader, test_loader, in_ch, device, epochs, name):
    """Train all three decoders on the same data and compare reconstructions.

    Read the table and the pictures together, because they disagree, and the
    disagreement IS the lesson:

      * 'ConvT k3' frequently posts the BEST MSE of the three while wearing a
        visible grid. A pixel-wise loss barely charges for the checkerboard --
        the over-bright pixels are only slightly over-bright, and they are
        spread evenly -- so nothing in the objective pushes the model to
        remove it. This is why the artifact survives training, and why you
        have to look at outputs rather than trusting the loss.

      * 'Upsample + Conv' usually scores WORST here, which surprises people
        who have read that it is the recommended fix. Two honest reasons: it
        has fewer parameters than the k4 decoder (a 3x3 stride-1 conv is
        smaller than a 4x4 transposed conv), and nearest-neighbour
        interpolation hands the conv a blocky input to clean up. What it buys
        is a guarantee: with no overlapping stamps, a checkerboard is
        structurally impossible rather than merely discouraged.

    So there is no free lunch to announce -- there is a trade to understand.
    """
    print(f"\n[{name}] three decoders, same encoder, same budget")
    rows, titles = [], []
    for mode, label in [("convt", "ConvT k4 (k % s == 0)"),
                        ("convt_k3", "ConvT k3 (checkerboard)"),
                        ("resize", "Upsample + Conv")]:
        torch.manual_seed(0)
        model = build_conv_ae(in_ch=in_ch, upsample=mode).to(device)
        train(model, train_loader, device, epochs=epochs, quiet=True)
        mse = eval_mse(model, test_loader)
        clean, _, recon = sample_batch(model, test_loader, n=8)
        if not rows:
            rows.append(clean)
            titles.append("original")
        rows.append(recon)
        titles.append(label)
        print(f"  {label:<26} test MSE {mse:.5f}  "
              f"PSNR {psnr(recon, clean):5.2f} dB  "
              f"params {count_params(model):,}")

    print("  note: the checkerboard decoder often wins on MSE. Judge this one")
    print("        with your eyes -- zoom into row 3 and look for the grid.")
    show_grid(rows, titles, n=8,
              suptitle=f"{name}: how the decoder upsamples "
                       f"(look for the grid pattern in row 3)")


def run_dataset(name, train_ds, test_ds, in_ch, image_dim, device, epochs):
    train_loader, test_loader = make_loaders(train_ds, test_ds, batch_size=128)

    torch.manual_seed(0)
    model = build_conv_ae(in_ch=in_ch, upsample="convt").to(device)
    print(f"\n=== {name}: CNN autoencoder ===")
    losses = train(model, train_loader, device, epochs=epochs)
    save_weights(model, f"part2_{name.lower().replace('-', '')}_ae")

    mse = eval_mse(model, test_loader)
    clean, _, recon = sample_batch(model, test_loader, n=8)
    print(f"{name} CNN AE   test MSE {mse:.5f}  PSNR {psnr(recon, clean):.2f} dB  "
          f"params {count_params(model):,}")

    # The headline comparison. We build Part 1's actual model rather than
    # quoting a number, so the two counts cannot drift apart. The MLP needed a
    # separate weight for every (pixel, hidden unit) pair; the CNN reuses a
    # handful of kernels across the whole image. Fewer parameters AND a lower
    # loss is what "the architecture matches the data" buys you.
    n_pixels = in_ch * image_dim * image_dim
    mlp, _, _ = build_autoencoder(n_pixels, hidden_dim=128, latent_dim=32)
    print(f"  for reference, Part 1's MLP on {name} used {count_params(mlp):,} "
          f"parameters ({count_params(mlp) / count_params(model):.0f}x more)")

    show_grid([clean, recon], ["original", "CNN AE"], n=8,
              suptitle=f"{name}: convolutional encoder + transposed-conv decoder")

    plt.figure()
    plt.plot(losses, marker="o", color="tab:orange")
    plt.title(f"{name} CNN autoencoder training loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.tight_layout()

    return train_loader, test_loader


def main():
    device = get_device()
    print(f"using device: {device}\n")

    # (0) the mechanism, on paper, before any training
    demo_transpose_mechanics()

    # (1) MNIST
    m_train, m_test = run_dataset(
        "MNIST",
        load_mnist(train=True, n_subset=20000),
        load_mnist(train=False, n_subset=2000),
        in_ch=1, image_dim=28, device=device, epochs=10)
    compare_upsampling(m_train, m_test, in_ch=1, device=device, epochs=8, name="MNIST")

    # (2) CIFAR-10 -- the dataset the MLP autoencoder could not handle
    c_train, c_test = run_dataset(
        "CIFAR-10",
        load_cifar10(train=True, n_subset=20000),
        load_cifar10(train=False, n_subset=2000),
        in_ch=3, image_dim=32, device=device, epochs=15)
    compare_upsampling(c_train, c_test, in_ch=3, device=device, epochs=12, name="CIFAR-10")

    print("\nTakeaway: same bottleneck idea, fewer parameters, better pictures --")
    print("because conv layers assume neighbouring pixels are related and the MLP")
    print("had to learn that from scratch. CIFAR-10 in particular goes from mush")
    print("to recognisable.")
    print("\nBut note what is still missing on CIFAR-10: edges are soft, textures")
    print("are smoothed away. So far we have only asked the model to COPY its")
    print("input, which hides that weakness. demo_why_skip_connections.py gives it")
    print("a task where the answer is not sitting in the input.")

    plt.show()


if __name__ == "__main__":
    main()
