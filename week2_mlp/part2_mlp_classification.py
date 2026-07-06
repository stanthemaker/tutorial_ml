"""Week 2 - Part 2: The same MLP, now doing classification.

Big idea: the training loop is identical. To switch from regression to
classification we only change the *data* and the *loss function*:
    - data : two interleaving half-moons (make_moons) with integer labels
    - loss : nn.CrossEntropyLoss (the classification version of MSE)

The network outputs 2 numbers (one score per class); argmax picks the winner.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


def main():
    torch.manual_seed(0)
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    X_np, y_np = make_moons(n_samples=300, noise=0.2, random_state=0)

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)

    clf = nn.Sequential(
        nn.Linear(2, 16),  # 2 input features
        nn.ReLU(),
        nn.Linear(16, 2),  # 2 classes -> 2 outputs
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=0.01)

    losses = []  # track loss so we can plot the curve
    for epoch in range(500):  # same 4-step loop, classification flavour
        logits = clf(X)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    pred = clf(X).argmax(dim=1)
    acc = (pred == y).float().mean()
    print(f"accuracy: {acc:.2%}")

    plot_results(clf, X, y, pred, losses, acc)


def plot_results(clf, X, y, pred, losses, acc):
    """Three views of the trained classifier: raw data, decision boundary,
    and the training loss curve."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1) the raw data: two interleaving half-moons, coloured by true label
    axes[0].scatter(
        X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=15, edgecolor="k", linewidth=0.3
    )
    axes[0].set_title("Data (true labels)")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")

    # 2) decision boundary: run the model over a dense grid and shade each
    #    region by the class it predicts, then overlay the actual points.
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = torch.meshgrid(
        torch.linspace(x_min, x_max, 300),
        torch.linspace(y_min, y_max, 300),
        indexing="xy",
    )
    grid = torch.stack([xx.ravel(), yy.ravel()], dim=1)
    with torch.no_grad():
        zz = clf(grid).argmax(dim=1).reshape(xx.shape)
    axes[1].contourf(xx, yy, zz, alpha=0.3, cmap="coolwarm", levels=1)
    # colour the points by whether the model got them right
    correct = pred == y
    axes[1].scatter(
        X[correct, 0],
        X[correct, 1],
        c=y[correct],
        cmap="coolwarm",
        s=15,
        edgecolor="k",
        linewidth=0.3,
        label="correct",
    )
    axes[1].scatter(
        X[~correct, 0],
        X[~correct, 1],
        marker="x",
        c="black",
        s=40,
        label="misclassified",
    )
    axes[1].set_title(f"Decision boundary (acc {acc:.1%})")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    axes[1].legend(loc="upper right", fontsize=8)

    # 3) training loss curve: cross-entropy falling as the model learns
    axes[2].plot(losses, color="tab:purple")
    axes[2].set_title("Training loss")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("cross-entropy")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
