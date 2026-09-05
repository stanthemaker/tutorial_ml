"""Week 3 - Part 4: Seeing what the CNN learned (filter visualisation).

Same spirit as Week 2's part4_mnist_pca_verify.py. There we used PCA to *see*
why a linear model was enough -- the point was to make an abstract claim
visible. Here we do the same for the CNN's central claim: that its first conv
layer learns local pattern detectors (edges, strokes, blobs) rather than
memorising whole images.

Two views:

  1. The learned kernels themselves. part1 set an edge detector (Sobel) by
     hand; here we plot the 8 kernels the network *learned* in part2. You'll
     see they look like little oriented edge / stroke detectors -- nobody told
     them to; that structure fell out of training.

  2. The feature maps. We push one real digit through the first conv layer and
     show each channel's response. Bright regions are where that kernel found
     its pattern in the image. This is the "eyes on the evidence" step: the
     filters from view (1), applied to an actual digit.

This script only *looks* at a model -- it never trains one. Pass the weights
part2 saved as the one required argument:

    python part2_cnn_mnist.py                     # trains, writes part2.pt
    python part4_cnn_mnist_visualize.py part2.pt

Requiring the checkpoint is deliberate (part5 works the same way). A
quick-trained stand-in would still draw eight plausible-looking kernels, and
you would have no way to tell that you were reading structure into a barely
trained model. Better to refuse to run.
"""

import argparse
import os

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from part2_cnn_mnist import load_mnist, build_model, get_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "checkpoint",
        metavar="MODEL.pt",
        help="weights to visualise -- run part2_cnn_mnist.py to produce part2.pt",
    )
    args = parser.parse_args()
    if not os.path.exists(args.checkpoint):
        parser.error(
            f"{args.checkpoint} not found -- run 'python part2_cnn_mnist.py' "
            "first to train a model and save its weights"
        )
    return args


def load_cnn(path, device):
    """Load part2's architecture and fill it with the checkpoint's weights."""
    cnn = build_model().to(device)
    # map_location=device so weights saved on any device load onto ours.
    state = torch.load(path, map_location=device)
    try:
        cnn.load_state_dict(state)
    except RuntimeError as err:
        # Almost always an older checkpoint from a different architecture.
        raise SystemExit(
            f"{path} does not match part2's model -- retrain with "
            f"'python part2_cnn_mnist.py'\n\n{err}"
        )
    print(f"loaded weights from {path}")
    return cnn


def plot_filters(conv):
    """Plot the first conv layer's learned kernels as little images.

    conv.weight has shape (out_channels, in_channels, kH, kW). For MNIST
    in_channels is 1, so each of the 8 output channels is a single 3x3 image.
    """
    weights = conv.weight.detach().cpu()  # (8, 1, 3, 3); .cpu() for matplotlib
    n = weights.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 2))
    for i, ax in enumerate(axes):
        ax.imshow(weights[i, 0], cmap="gray")
        ax.set_title(f"kernel {i}", fontsize=8)
        ax.axis("off")
    fig.suptitle("First-layer conv kernels the CNN *learned* (compare part1's hand-set Sobel)")
    fig.tight_layout()


def plot_feature_maps(cnn, image):
    """Show one digit and the first conv layer's response on each channel."""
    device = next(cnn.parameters()).device
    first_conv = cnn[0]  # the Conv2d(1, 8, 3) at the start of build_model()
    with torch.no_grad():
        maps = first_conv(image.unsqueeze(0).to(device))[0].cpu()  # (8, 28, 28)

    n = maps.shape[0]
    fig, axes = plt.subplots(1, n + 1, figsize=(1.5 * (n + 1), 2))
    axes[0].imshow(image.squeeze(), cmap="gray")
    axes[0].set_title("input digit", fontsize=8)
    axes[0].axis("off")
    for i in range(n):
        axes[i + 1].imshow(maps[i], cmap="gray")
        axes[i + 1].set_title(f"ch {i}", fontsize=8)
        axes[i + 1].axis("off")
    fig.suptitle("Each kernel's response on a real digit (bright = pattern found here)")
    fig.tight_layout()


def main():
    args = parse_args()
    torch.manual_seed(0)  # only the test digit picked below is random

    device = get_device()
    print(f"using device: {device}")
    cnn = load_cnn(args.checkpoint, device)

    # (1) the learned kernels on their own
    plot_filters(cnn[0])

    # (2) the same kernels applied to a real test digit
    test_ds = load_mnist(train=False)
    image, _ = next(iter(DataLoader(test_ds, batch_size=1, shuffle=True)))
    plot_feature_maps(cnn, image[0])

    plt.show()


if __name__ == "__main__":
    main()
