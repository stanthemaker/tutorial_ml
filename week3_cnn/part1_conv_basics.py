"""Week 3 - Part 1: What a convolution actually does.

Before we stack conv layers into a network, let's see the two building blocks
on their own, on a tiny picture we fully understand.

Three ideas, in order:

  1. A convolution is a small window (the "kernel") that slides across the
     image. At each position it multiplies the pixels under it by the kernel
     weights and sums them into one output number. That is it. We'll hand-set
     the kernel to a classic edge detector (Sobel) and *watch* it light up the
     edges of a synthetic shape -- no training involved. Later layers learn
     their kernels, but the mechanism is exactly this.

  2. Output size. A conv/pool shrinks the image, and there is a formula:

         out = floor((in + 2*padding - kernel) / stride) + 1

     We print it for a few settings so the numbers stop being mysterious --
     these are the shapes you have to track when you design a CNN.

  3. Receptive field. One output pixel of the first conv "sees" only a
     kernel-sized patch of the input. Stack a second conv (or a pool) on top
     and each output pixel now sees a *larger* patch, because it combines
     neighbours that each already summarised their own patch. This growing
     window is how a deep CNN goes from detecting edges to detecting whole
     objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def make_shape(size=28):
    """A synthetic grayscale image: a bright square on a dark background.

    Deliberately simple so the edges are obvious -- when the edge detector
    fires, we know exactly which pixels it should have found.
    """
    img = torch.zeros(1, 1, size, size)  # (batch, channel, height, width)
    img[:, :, 8:20, 8:20] = 1.0
    return img


def conv_output_size(in_size, kernel, stride, padding):
    """The shape formula every CNN designer memorises."""
    return (in_size + 2 * padding - kernel) // stride + 1


def print_size_table():
    """Show how kernel / stride / padding change the output height & width."""
    print("output size = floor((in + 2*padding - kernel) / stride) + 1\n")
    print(f"{'in':>4} {'kernel':>7} {'stride':>7} {'pad':>4} -> {'out':>4}")
    settings = [
        (28, 3, 1, 0),  # plain 3x3: shrinks by 2
        (28, 3, 1, 1),  # 3x3 with pad 1: size preserved (the usual choice)
        (28, 5, 1, 0),  # bigger kernel: shrinks more
        (28, 2, 2, 0),  # 2x2 stride 2: MaxPool -- halves the image
        (28, 3, 2, 1),  # strided conv: also roughly halves
    ]
    for in_size, k, s, p in settings:
        out = conv_output_size(in_size, k, s, p)
        print(f"{in_size:>4} {k:>7} {s:>7} {p:>4} -> {out:>4}")
    print()


def sobel_kernels():
    """Two hand-built 3x3 edge detectors: vertical and horizontal edges.

    These are the weights a Conv2d would normally *learn*. We set them by hand
    so the very first thing students see is a convolution doing something
    recognisable.
    """
    gx = torch.tensor([[-1.0, 0.0, 1.0],
                       [-2.0, 0.0, 2.0],
                       [-1.0, 0.0, 1.0]])
    gy = gx.t().clone()
    return gx, gy


def main():
    torch.manual_seed(0)

    # Note: this file stays on CPU on purpose. There is no training here -- just
    # a single forward pass of a hand-set filter on one 28x28 image -- so a GPU
    # (Mac MPS / NVIDIA CUDA) would add copying overhead for no speedup. The
    # training scripts (part2, part3, the demos) use get_device() to pick a GPU.

    # (2) the numbers first, so the shapes below are no surprise
    print_size_table()

    img = make_shape()

    # (1) a real convolution. One input channel, two output channels (one per
    # Sobel kernel), 3x3 window, padding 1 so the output stays 28x28.
    conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1, bias=False)
    gx, gy = sobel_kernels()
    with torch.no_grad():
        conv.weight[0, 0] = gx  # channel 0 detects vertical edges
        conv.weight[1, 0] = gy  # channel 1 detects horizontal edges
    with torch.no_grad():
        edges = conv(img)  # (1, 2, 28, 28)

    print(f"input  shape: {tuple(img.shape)}")
    print(f"conv   shape: {tuple(edges.shape)}   (2 channels = 2 edge maps)")

    # MaxPool: slide a 2x2 window with stride 2, keep the max in each window.
    # It halves height and width -- the standard way a CNN downsamples.
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    pooled = pool(F.relu(edges))
    print(f"pool   shape: {tuple(pooled.shape)}   (halved by 2x2 stride-2 pool)")

    # (3) receptive field, made concrete. Stack a second 3x3 conv on top of the
    # first. A single pixel of the first conv's output saw a 3x3 input patch;
    # a single pixel of the second conv's output combines a 3x3 block of those,
    # so it effectively sees a 5x5 input patch. Deeper -> wider view.
    rf1 = 3
    rf2 = rf1 + (3 - 1)  # each extra 3x3 conv (stride 1) adds (kernel-1) to the field
    print(f"\nreceptive field after 1 conv (3x3): {rf1}x{rf1} input pixels")
    print(f"receptive field after 2 convs (3x3): {rf2}x{rf2} input pixels")

    # Visualise it all: the input, the two Sobel edge maps, and one pooled map.
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(img[0, 0], cmap="gray")
    axes[0].set_title("input\n(bright square)")
    axes[1].imshow(edges[0, 0], cmap="gray")
    axes[1].set_title("conv ch0\n(vertical edges)")
    axes[2].imshow(edges[0, 1], cmap="gray")
    axes[2].set_title("conv ch1\n(horizontal edges)")
    axes[3].imshow(pooled[0, 0], cmap="gray")
    axes[3].set_title("after ReLU + 2x2 pool\n(14x14, downsampled)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("A convolution slides a small kernel to detect local patterns (edges)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
