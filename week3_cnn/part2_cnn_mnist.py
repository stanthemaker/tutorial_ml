"""Week 3 - Part 2: The same MNIST task, now with a CNN.

Big idea: nothing about the *problem* changes -- same digits, same 10 classes,
same CrossEntropyLoss, the same 4-step training loop as Week 2's
part3_mlp_mnist.py. The only thing that changes is the *model*: we swap the
`Flatten -> Linear -> ReLU -> Linear` MLP for a stack of
`Conv -> ReLU -> Pool` blocks followed by a small classifier head.

Two things to compare against the MLP:

  * Parameters. The MLP spent ~200k weights just on its first 784->256 matrix,
    because it wired every pixel to every hidden unit. The CNN reuses one small
    kernel across the whole image, so it reaches similar (or better) accuracy
    with far fewer parameters. We print both counts.

  * Structure. The CNN never flattens the image until the very end, so it keeps
    the 2D layout -- neighbouring pixels stay neighbours. That is why it handles
    shifted/local patterns that tripped up the MLP in demo_mlp_limits.py.

We reuse the same three views as Week 2: loss curve, sample predictions, and a
confusion matrix. The trained weights are saved to week3_cnn/part2.pt so
part4 can reload this exact model to visualise its filters.
"""

import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Checkpoints land next to this file, never in the current working directory,
# so part4 finds them at the same path however you launched training.
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "part2.pt")


def get_device():
    """Pick the fastest available device: NVIDIA GPU, then Mac GPU, then CPU.

    Two GPU backends, checked with two flags:
      * use_windows_gpu -- an NVIDIA card via CUDA (Windows/Linux).
      * use_mac_gpu     -- Apple-silicon (M1/M2/...) GPU via Metal (MPS).
    Everything still works on CPU; this just uses a GPU when there is one.
    """
    use_windows_gpu = torch.cuda.is_available()
    use_mac_gpu = torch.backends.mps.is_available()
    if use_windows_gpu:
        return torch.device("cuda")
    if use_mac_gpu:
        return torch.device("mps")
    return torch.device("cpu")


def load_mnist(train, n_subset=None):
    """Load MNIST as (1, 28, 28) image tensors.

    Note the difference from Week 2: we do NOT flatten to 784. A CNN wants the
    2D image shape so its kernels can slide across height and width. As before
    we optionally subsample the training set to keep CPU epochs quick.
    """
    ds = datasets.MNIST(
        root="./data", train=train, download=True, transform=transforms.ToTensor()
    )
    if n_subset is not None and n_subset < len(ds):
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))
        ds = Subset(ds, idx[:n_subset].tolist())
    return ds


def build_model():
    """A small CNN: two Conv -> ReLU -> Pool blocks, then a linear classifier.

    Shape walk-through (input is 1x28x28):
        Conv2d(1, 8, 3, padding=1)   -> 8 x 28 x 28   (padding keeps the size)
        MaxPool2d(2)                 -> 8 x 14 x 14   (halved)
        Conv2d(8, 16, 3, padding=1)  -> 16 x 14 x 14
        MaxPool2d(2)                 -> 16 x  7 x  7
        Flatten                      -> 16*7*7 = 784  (only flatten at the end!)
        Linear(784, 10)              -> 10 class scores

    Each Conv2d learns its own kernels (unlike part1, where we set Sobel by
    hand) -- part4 will visualise what the first layer ended up learning.
    """
    return nn.Sequential(
        # nn.Conv2d(_, _, kernel_size=_, padding=_),
        # nn.ReLU(),
        # nn.MaxPool2d(_),
        # nn.Conv2d(_, _, kernel_size=_, padding=_),
        # nn.ReLU(),
        # nn.MaxPool2d(_),
        # nn.Flatten(),
        # nn.Linear(_, 10),
    )


def evaluate(model, loader):
    device = next(model.parameters()).device
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images).argmax(
                dim=1
            )  # no flatten -- CNN takes images directly
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total


def collect_predictions(model, loader):
    device = next(model.parameters()).device
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds.append(model(images.to(device)).argmax(dim=1).cpu())
            trues.append(labels)
    return torch.cat(preds), torch.cat(trues)  # back on CPU for sklearn / numpy


def plot_sample_predictions(model, test_ds, n=10):
    """Show n test digits with the CNN's predicted label (red = wrong)."""
    device = next(model.parameters()).device
    model.eval()
    fig, axes = plt.subplots(2, n // 2, figsize=(1.4 * (n // 2), 3.2))
    for ax, (image, true) in zip(axes.ravel(), test_ds):
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device)).argmax(dim=1).item()
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"pred {pred}", color="green" if pred == true else "red")
        ax.axis("off")
    fig.suptitle("Test digits and the CNN's prediction (red = wrong)")
    fig.tight_layout()


def plot_confusion(preds, trues):
    cm = confusion_matrix(trues.numpy(), preds.numpy(), labels=range(10))
    disp = ConfusionMatrixDisplay(cm, display_labels=range(10))
    disp.plot(cmap="Blues", colorbar=False)
    disp.ax_.set_title("MNIST CNN confusion matrix (rows = true, cols = predicted)")


def main():
    torch.manual_seed(0)

    device = get_device()
    print(f"using device: {device}")

    train_ds = load_mnist(train=True, n_subset=20000)
    test_ds = load_mnist(train=False)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    cnn = build_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

    # For contrast: the MLP from Week 2 had ~200k parameters. Print the CNN's.
    n_params = sum(p.numel() for p in cnn.parameters())
    print(f"CNN parameters: {n_params:,}  (Week 2's MLP had ~203,000)")

    losses = []
    for epoch in range(8):  # same 4-step loop, now over mini-batches of images
        cnn.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = cnn(images)  # feed images straight in -- no .view(-1, 784)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
        print(f"epoch {epoch + 1}/8  loss {losses[-1]:.4f}")

    acc = evaluate(cnn, test_loader)
    print(f"Test accuracy: {acc:.2%}")

    # Save so part4 can reload this exact model to visualise its first-layer
    # filters. part4 *requires* this file -- it never trains a stand-in.
    torch.save(cnn.state_dict(), WEIGHTS)
    print(f"saved weights to {WEIGHTS}")

    plt.figure()
    plt.plot(losses, marker="o", color="tab:green")
    plt.title(f"MNIST CNN training loss (test acc {acc:.1%}, {n_params:,} params)")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy")

    plot_sample_predictions(cnn, test_ds, n=10)

    preds, trues = collect_predictions(cnn, test_loader)
    plot_confusion(preds, trues)

    plt.show()


if __name__ == "__main__":
    main()
