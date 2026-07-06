"""Week 2 - Part 3c: Why did removing the ReLU barely hurt? (PCA).

Big idea: part3b showed that collapsing the MNIST MLP to a linear model only
cost a few percent accuracy, even though the same move destroyed the sine-wave
fit in demo_linear_limits.py. The reason: MNIST digit classes are already
nearly *linearly separable* in raw 784-dim pixel space.

Linearly separable means a single hyperplane `Wx + b` can (nearly) correctly
separate the classes just by ranking scores -- no bending required. Why is
this so much easier in high dimensions? A 2D line has only 2 free parameters
(slope + intercept), so on make_moons its interleaved shape defeats any
straight line no matter what. A 784-dim hyperplane has 784 free parameters --
far more flexibility to carve up the space. And crucially, MNIST's classes
have genuinely near-linear structure, whereas moons' interleaving is
genuinely non-linear (more dimensions wouldn't save a straight line there).

We can't draw 784 dimensions, so we use PCA to project down to 2D just to
*look* at how separated the classes are: once for the raw pixels, once for the
trained MLP's hidden-layer representation.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

from part3_mlp_mnist import load_mnist, build_model


def get_sample(n=2000):
    """Grab n flattened test images + labels as plain tensors."""
    ds = load_mnist(train=False)
    loader = DataLoader(ds, batch_size=n, shuffle=True)
    images, labels = next(iter(loader))
    return images.view(-1, 784), labels


def get_trained_mlp():
    """Reuse part3's trained weights if present; otherwise train a quick MLP
    so this file still runs standalone (consistent with part1/part2/demo)."""
    mlp = build_model()
    if os.path.exists("mnist_mlp.pt"):
        mlp.load_state_dict(torch.load("mnist_mlp.pt"))
        print("loaded trained model from mnist_mlp.pt")
        return mlp

    print(
        "mnist_mlp.pt not found -- training a quick MLP (run part3 first "
        "to reuse its weights)"
    )
    train_loader = DataLoader(
        load_mnist(train=True, n_subset=15000), batch_size=128, shuffle=True
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
    for _ in range(3):  # a few epochs is plenty to see the effect
        for images, labels in train_loader:
            logits = mlp(images.view(-1, 784))
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return mlp


def main():
    torch.manual_seed(0)
    X, y = get_sample(n=2000)
    mlp = get_trained_mlp()

    # Hidden-layer representation: run the sample through just the first two
    # layers -- Linear(784,128) + ReLU -- i.e. mlp[:2], stopping BEFORE the
    # final classification layer. This is the 128-dim space the classifier
    # actually sees; the ReLU is the only non-linearity in the whole network.
    with torch.no_grad():
        hidden = mlp[:2](X).numpy()

    # PCA down to 2D so we can plot each 784-dim / 128-dim point on a page.
    raw_2d = PCA(n_components=2).fit_transform(X.numpy())
    hidden_2d = PCA(n_components=2).fit_transform(hidden)

    # In part2, we could literally draw the decision boundary because the
    # input was 2D. MNIST is 784-dim, so we can't draw that directly -- but
    # PCA lets us compress it down to 2D just to *look* at how well-separated
    # the classes already are.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    labels = y.numpy()

    sc = axes[0].scatter(raw_2d[:, 0], raw_2d[:, 1], c=labels, cmap="tab10", s=8)
    axes[0].set_title("Raw pixel space (PCA)")
    # Notice the left plot already shows fairly separated clusters, even
    # though it's just raw pixels with no learning involved -- this is the
    # visual evidence for why the ReLU-free model in part3b still hit ~90%+:
    # the classes were already close to linearly separable.

    axes[1].scatter(hidden_2d[:, 0], hidden_2d[:, 1], c=labels, cmap="tab10", s=8)
    axes[1].set_title("Hidden layer representation (PCA)")
    # The right plot shows tighter, more separated clusters -- this is where
    # the extra ~6-8 percentage points from part3's ReLU come from: the
    # non-linearity cleans up the remaining overlap (e.g. between 4/9 or 3/5/8).

    # shared colorbar mapping colors -> digit labels 0-9
    cbar = fig.colorbar(sc, ax=axes, ticks=range(10), fraction=0.046, pad=0.04)
    cbar.set_label("digit label")
    plt.show()


if __name__ == "__main__":
    main()
