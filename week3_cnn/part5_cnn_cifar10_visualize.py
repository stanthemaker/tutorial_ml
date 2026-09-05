"""Week 3 - Part 5: How filters change with depth (and what "RGB" means deeper in).

part4 looked at *one* layer -- the first conv of the MNIST CNN -- and showed it
learned edge/stroke detectors. The obvious next question: what do the *later*
layers learn? The standard claim is that depth builds a hierarchy: early layers
fire on edges and colour blobs, middle layers on textures and corners, late
layers on object parts. Here we sample three depths of the part3 CIFAR-10 CNN
and put their kernels side by side so you can judge that claim yourself.

We pick 3 conv layers and 3 random filters from each. Every filter gets two
things drawn for it:

  * its 3 input-channel slices -- 3 x 3 x 3 = 27 little kernel images, and
  * its response on one real test photo -- 3 x 3 = 9 feature maps.

The second half is what makes the first half readable. A 3x3 kernel on its own
is nine numbers; you can stare at a red/blue square all day and not know
whether it matters, or what it fires on. The response map is the filter's
*answer* on real data -- bright red where it found its pattern -- so the pair
"here is the kernel, here is where it fired" is the actual evidence. This is
the same move part4 made on MNIST, now repeated at three depths.

A caveat worth understanding, because it *is* the lesson of this file. A conv
kernel has shape (in_channels, kH, kW), so "show the filter's R, G and B" only
means something literal at the very first layer, where in_channels == 3 and
those three slices really are the red, green and blue channels of the image.
Deeper down, layer 2's input is not a picture -- it is the 32 feature maps that
layer 1 produced, and layer 3's input is 64 maps from layer 2. So we keep
picking 3 input slices per filter, but they are "which earlier feature map does
this kernel listen to", not colours. That shift is the whole point: after the
first layer a CNN stops looking at pixels and starts looking at *its own
previous answers*.

Reading the plots: one figure per depth. The far-left panel is the CIFAR-10
test photo everything in that figure responds to, titled with its true label,
and each row after it goes kernel -> kernel -> kernel -> response for one
filter. Every 3x3 kernel is drawn on a red/blue diverging scale,
symmetric about zero and shared across the three slices of one filter so they
are directly comparable. Red = positive weight (this input excites the filter),
blue = negative (it suppresses it), white = roughly ignored. The last column
uses the same colours for the response map, so red there means "this filter
fired positively on this part of the photo". Watch the maps shrink as you go
deeper -- 32x32, then 16x16, then 8x8 -- because each pooling layer halves the
resolution. That shrinking is why late filters can only respond to big, blobby
things: one pixel of the last map covers an 8x8 patch of the original photo.

This script only *looks* at a model -- it never trains one. Pass the weights
part3 saved as the one required argument:

    python part3_cnn_cifar10.py                     # trains, writes part3.pt
    python part5_cnn_cifar10_visualize.py part3.pt

Requiring the checkpoint is deliberate. A quick-trained stand-in would still
draw 27 pretty squares, and you would have no way to tell that you were reading
structure into a barely-trained model. Better to refuse to run.
"""

import argparse
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from part3_cnn_cifar10 import (CLASSES, build_model, denormalize, get_device,
                               load_cifar10)

N_FILTERS = 3  # filters sampled per depth
N_SLICES = 3   # input-channel slices shown per filter ("RGB" at layer 1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "checkpoint",
        metavar="MODEL.pt",
        help="weights to visualise -- run part3_cnn_cifar10.py to produce part3.pt",
    )
    args = parser.parse_args()
    if not os.path.exists(args.checkpoint):
        parser.error(
            f"{args.checkpoint} not found -- run 'python part3_cnn_cifar10.py' "
            "first to train a model and save its weights"
        )
    return args


def load_cnn(path, device):
    """Load part3's architecture and fill it with the checkpoint's weights."""
    cnn = build_model().to(device)
    # map_location=device so weights saved on any device load onto ours.
    state = torch.load(path, map_location=device)
    try:
        cnn.load_state_dict(state)
    except RuntimeError as err:
        # Almost always an older checkpoint from a different architecture.
        raise SystemExit(
            f"{path} does not match part3's model -- retrain with "
            f"'python part3_cnn_cifar10.py'\n\n{err}"
        )
    print(f"loaded weights from {path}")
    return cnn


def find_conv_layers(model):
    """Return [(index, layer), ...] for every Conv2d in the Sequential."""
    return [(i, m) for i, m in enumerate(model) if isinstance(m, nn.Conv2d)]


def pick_depths(convs, n=3):
    """Choose n conv layers spread across the network (first, middle, last).

    part3's model has exactly 3 conv layers so this returns all of them, but
    spacing them out keeps the script honest if you deepen the model later.
    """
    if len(convs) <= n:
        return convs
    step = (len(convs) - 1) / (n - 1)
    return [convs[round(k * step)] for k in range(n)]


def pick_test_image(generator):
    """One real CIFAR-10 test photo -- the input every response map comes from."""
    test_ds = load_cifar10(train=False)
    idx = torch.randint(len(test_ds), (1,), generator=generator).item()
    image, label = test_ds[idx]
    return image, CLASSES[label], idx


def feature_maps(cnn, image, index):
    """Every filter's response at conv `index`, for one image.

    nn.Sequential slicing does the work: cnn[:index + 1] is exactly the stack
    of layers ending at this conv, so running the image through it returns that
    conv's raw output -- before the ReLU that follows. We show it pre-ReLU on
    purpose: the blue (negative) regions are real, and seeing them makes it
    obvious what the next line of the model throws away, since ReLU clamps
    every one of them to zero.
    """
    device = next(cnn.parameters()).device
    with torch.no_grad():
        return cnn[: index + 1](image.unsqueeze(0).to(device))[0].cpu()


def slice_labels(in_channels, chosen):
    """Name the input slices: real colours at layer 1, feature maps deeper in."""
    if in_channels == 3:
        return [f"{name} channel" for name in ("R", "G", "B")]
    return [f"input map {c}" for c in chosen.tolist()]


def plot_depth(index, conv, depth_rank, generator, maps, photo):
    """One figure per depth: the input photo, then 3 filters x (3 slices + response)."""
    image, label, idx = photo
    weights = conv.weight.detach().cpu()  # (out_channels, in_channels, kH, kW)
    out_channels, in_channels = weights.shape[0], weights.shape[1]
    size = maps.shape[-1]  # response resolution at this depth: 32, 16, then 8

    filters = torch.randperm(out_channels, generator=generator)[:N_FILTERS]
    # Columns: the input photo, then this filter's 3 kernel slices, then its
    # response. Reading a row left to right is "here is the picture, here is
    # what the filter looks for, here is where it found it".
    n_cols = 1 + N_SLICES + 1
    fig = plt.figure(figsize=(2.5 * n_cols, 2.5 * N_FILTERS + 0.8))
    grid = fig.add_gridspec(N_FILTERS, n_cols)

    # The photo spans all three rows: it is the same input for every filter,
    # and drawing it once keeps the comparison honest.
    ax_photo = fig.add_subplot(grid[:, 0])
    ax_photo.imshow(denormalize(image))  # undo the (x-0.5)/0.5 normalisation
    ax_photo.set_title(f"input: {label}\n(test image #{idx})", fontsize=9)
    ax_photo.axis("off")

    axes = [[fig.add_subplot(grid[r, c]) for c in range(1, n_cols)]
            for r in range(N_FILTERS)]

    for row, f in enumerate(filters):
        kernel = weights[f]  # (in_channels, kH, kW)
        # At layer 1 in_channels == 3, so this is exactly R, G, B in order.
        # Deeper, we sample 3 of the many incoming feature maps.
        if in_channels == 3:
            chosen = torch.arange(3)
        else:
            chosen = torch.randperm(in_channels, generator=generator)[:N_SLICES]
            chosen = chosen.sort().values  # ascending reads more naturally

        # One symmetric colour scale per filter: the three slices of a filter
        # are then directly comparable, and 0 always sits at white.
        limit = kernel[chosen].abs().max().item() or 1e-8

        for col, (c, label) in enumerate(zip(chosen, slice_labels(in_channels, chosen))):
            ax = axes[row][col]
            ax.imshow(kernel[c], cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.set_title(f"filter {f.item()} - {label}", fontsize=8)
            ax.axis("off")

        # The payoff column. Its own scale, not the kernel's: weights and
        # activations are different quantities and share no units, so forcing
        # them onto one colour bar would only wash the map out.
        response = maps[f]
        r_limit = response.abs().max().item() or 1e-8
        ax = axes[row][N_SLICES]
        ax.imshow(response, cmap="RdBu_r", vmin=-r_limit, vmax=r_limit)
        ax.set_title(f"filter {f.item()} response ({size}x{size})", fontsize=8)
        ax.axis("off")

    kind = "R/G/B of the image" if in_channels == 3 else "3 sampled input maps"
    fig.suptitle(
        f"Depth {depth_rank}: Conv2d[{index}]   {in_channels} -> {out_channels} channels\n"
        f"{N_FILTERS} random filters x {kind}, plus each filter's response\n"
        "red = positive (weight, or activation), blue = negative",
        fontsize=10,
    )
    # Explicit margins rather than tight_layout: these axes hold square images
    # with a fixed aspect, and tight_layout reserves the suptitle's headroom by
    # squeezing the grid, which pushed the bottom row off the canvas.
    fig.subplots_adjust(top=0.84, bottom=0.03, left=0.02, right=0.98,
                        hspace=0.22, wspace=0.15)


def main():
    args = parse_args()
    # A dedicated generator keeps the "random" filter picks and the test image
    # reproducible run to run, so you can compare two checkpoints on the same
    # filters and the same photo.
    generator = torch.Generator().manual_seed(1)

    device = get_device()
    print(f"using device: {device}")
    cnn = load_cnn(args.checkpoint, device)

    photo = pick_test_image(generator)
    print(f"response maps computed on test image #{photo[2]} ({photo[1]})")

    convs = find_conv_layers(cnn)
    depths = pick_depths(convs, n=3)
    print(f"visualising {len(depths)} of {len(convs)} conv layers:")

    for rank, (index, conv) in enumerate(depths, start=1):
        c_in, c_out = conv.in_channels, conv.out_channels
        maps = feature_maps(cnn, photo[0], index)
        print(f"  depth {rank}: model[{index}]  Conv2d({c_in} -> {c_out})"
              f"  response {tuple(maps.shape)}")
        plot_depth(index, conv, rank, generator, maps, photo)

    n_kernels = len(depths) * N_FILTERS * N_SLICES
    n_maps = len(depths) * N_FILTERS
    print(f"{n_kernels} kernel slices + {n_maps} response maps "
          f"across {len(depths)} figures")

    plt.show()


if __name__ == "__main__":
    main()
