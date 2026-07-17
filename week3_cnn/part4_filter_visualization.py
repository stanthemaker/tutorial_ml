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

Like Week 2's part4, this file reloads the model trained in part2 if its
weights are on disk (mnist_cnn.pt); otherwise it trains a quick CNN so the
script still runs standalone.
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from part2_cnn_mnist import load_mnist, build_model, get_device


def get_trained_cnn(device):
    """Reuse part2's trained weights if present; otherwise train a quick CNN."""
    cnn = build_model().to(device)
    if os.path.exists("mnist_cnn.pt"):
        # map_location=device so weights saved on any device load onto ours.
        cnn.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
        print("loaded trained model from mnist_cnn.pt")
        return cnn

    print(
        "mnist_cnn.pt not found -- training a quick CNN (run part2 first "
        "to reuse its weights)"
    )
    train_loader = DataLoader(
        load_mnist(train=True, n_subset=15000), batch_size=128, shuffle=True
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    for _ in range(3):  # a few epochs is plenty for the filters to take shape
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = loss_fn(cnn(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
    torch.manual_seed(0)

    device = get_device()
    print(f"using device: {device}")
    cnn = get_trained_cnn(device)

    # (1) the learned kernels on their own
    plot_filters(cnn[0])

    # (2) the same kernels applied to a real test digit
    test_ds = load_mnist(train=False)
    image, _ = next(iter(DataLoader(test_ds, batch_size=1, shuffle=True)))
    plot_feature_maps(cnn, image[0])

    plt.show()


if __name__ == "__main__":
    main()
