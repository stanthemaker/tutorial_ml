"""Week 2 - Part 3: The same MLP, now classifying MNIST digits.

Big idea: nothing new. This is the *exact same recipe* as
part2_mlp_classification.py (Linear -> ReLU -> Linear, CrossEntropyLoss, the
same 4-step training loop). The only thing that changes is the data: instead
of 2 input numbers (a point on the moons plot) we now feed in 784 numbers
(the 28x28 pixels of a handwritten digit), and instead of 2 classes we have
10 (the digits 0-9).

The problem is structurally identical to part2: find a boundary in a
multi-dimensional space that separates the classes. In part2 that space was
2D and we could literally draw the decision boundary. Here the space has 784
dimensions, so we *can't* draw it directly -- we'll come back to *seeing*
this space later, using PCA (see part3c_mnist_pca_verify.py).
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_mnist(train, n_subset=None):
    """Load MNIST as flattened 784-dim vectors.

    Each 28x28 image is flattened to a length-784 vector (see build_model's
    comment for why). To keep runtime reasonable on CPU we optionally
    subsample the training set -- fewer images means faster epochs at the
    cost of a little accuracy, which is a fine tradeoff for a teaching demo.
    """
    ds = datasets.MNIST(
        root="./data", train=train, download=True, transform=transforms.ToTensor()
    )
    if n_subset is not None and n_subset < len(ds):
        # deterministic subset so results are reproducible
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))
        ds = Subset(ds, idx[:n_subset].tolist())
    return ds


def build_model():
    # Same shape as part2_mlp_classification.py's `clf`, just wider:
    # 784 in (flattened pixels), 128 hidden, 10 classes out.
    # The problem is identical: find a boundary in a multi-dimensional space
    # that separates the classes -- the space just has 784 dimensions
    # instead of 2.
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def evaluate(model, loader):
    """Test accuracy, computed exactly like part2: argmax + mean of correct."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            x = images.view(-1, 784)
            pred = model(x).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    torch.manual_seed(0)

    # subsample train to ~20k images so each epoch is quick on CPU
    train_ds = load_mnist(train=True, n_subset=20000)
    test_ds = load_mnist(train=False)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    mlp = build_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

    losses = []  # average loss per epoch, for the curve
    for epoch in range(15):  # same 4-step loop, now over mini-batches
        mlp.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            x = images.view(-1, 784)  # 28x28 image -> 784-dim vector
            logits = mlp(x)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
        print(f"epoch {epoch + 1}/15  loss {losses[-1]:.4f}")

    acc = evaluate(mlp, test_loader)
    print(f"MLP (with ReLU) test accuracy: {acc:.2%}")

    # Save the trained weights so part3c can reload this exact model to
    # extract hidden-layer activations (instead of retraining from scratch).
    torch.save(mlp.state_dict(), "mnist_mlp.pt")

    # Only a loss curve here: unlike part2 we can't draw a decision boundary,
    # because the input lives in 784 dimensions and a plot has 2. We'll come
    # back to *seeing* that space with PCA in part3c_mnist_pca_verify.py.
    plt.plot(losses, marker="o", color="tab:purple")
    plt.title(f"MNIST MLP training loss (test acc {acc:.1%})")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy")
    plt.show()


if __name__ == "__main__":
    main()
