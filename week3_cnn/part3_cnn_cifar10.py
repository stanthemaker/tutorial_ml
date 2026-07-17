"""Week 3 - Part 3: CIFAR-10, where spatial structure finally matters.

Big idea: MNIST was almost too easy -- Week 2's part4 showed its classes are
nearly linearly separable in raw pixel space, so even a plain linear model did
well. CIFAR-10 is a different animal: 32x32 *colour* photos of real objects
(plane, car, bird, cat, ...) with textured backgrounds, varied lighting, and
the object sitting anywhere in the frame. Raw pixels no longer line up class by
class, so the MLP/PCA tricks from Week 2 fall apart here.

This is exactly the setting a CNN is built for. Sliding kernels pick up local
patterns (edges, colour blobs, textures), pooling makes them robust to small
shifts, and stacking blocks grows the receptive field until later layers can
respond to whole object parts. The code is the *same recipe* as part2 -- just
3 input channels instead of 1, and a couple more conv blocks because the task
is harder.

Note: CIFAR-10 is ~170 MB and CPU training is slower than MNIST. We subsample
and keep epochs modest so it finishes in a few minutes -- enough to land well
above chance (10%) and comfortably past where a same-size MLP plateaus.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CLASSES = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_device():
    """Pick the fastest available device: NVIDIA GPU, then Mac GPU, then CPU.

    Two GPU backends, checked with two flags:
      * use_windows_gpu -- an NVIDIA card via CUDA (Windows/Linux).
      * use_mac_gpu     -- Apple-silicon (M1/M2/...) GPU via Metal (MPS).
    CIFAR training is the heaviest job in Week 3, so a GPU helps most here.
    Everything still works on CPU regardless.
    """
    use_windows_gpu = torch.cuda.is_available()
    use_mac_gpu = torch.backends.mps.is_available()
    if use_windows_gpu:
        return torch.device("cuda")
    if use_mac_gpu:
        return torch.device("mps")
    return torch.device("cpu")


def load_cifar10(train, n_subset=None):
    """Load CIFAR-10 as (3, 32, 32) tensors, normalised to roughly [-1, 1].

    Three channels now (R, G, B) instead of MNIST's one -- colour is a real
    cue here (skies are blue, frogs are green). Normalising keeps the inputs
    centred, which helps training converge faster.
    """
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    ds = datasets.CIFAR10(root="./data", train=train, download=True, transform=tf)
    if n_subset is not None and n_subset < len(ds):
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))
        ds = Subset(ds, idx[:n_subset].tolist())
    return ds


def build_model():
    """A slightly deeper CNN for the harder task. Same block as part2, repeated.

    Shape walk-through (input is 3x32x32):
        Conv(3, 32) + Pool   -> 32 x 16 x 16
        Conv(32, 64) + Pool  -> 64 x  8 x  8
        Conv(64, 64) + Pool  -> 64 x  4 x  4
        Flatten              -> 64*4*4 = 1024
        Linear(1024, 128) -> ReLU -> Linear(128, 10)
    """
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def evaluate(model, loader):
    device = next(model.parameters()).device
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images).argmax(dim=1)
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


def denormalize(img):
    """Undo the (x-0.5)/0.5 normalisation and put channels last, for imshow."""
    return (img * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0)


def plot_sample_predictions(model, test_ds, n=10):
    device = next(model.parameters()).device
    model.eval()
    fig, axes = plt.subplots(2, n // 2, figsize=(1.6 * (n // 2), 3.6))
    for ax, (image, true) in zip(axes.ravel(), test_ds):
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device)).argmax(dim=1).item()
        ax.imshow(denormalize(image))
        color = "green" if pred == true else "red"
        ax.set_title(f"{CLASSES[pred]}", color=color, fontsize=9)
        ax.axis("off")
    fig.suptitle("Test images and the CNN's prediction (red = wrong)")
    fig.tight_layout()


def plot_confusion(preds, trues):
    cm = confusion_matrix(trues.numpy(), preds.numpy(), labels=range(10))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    disp.plot(cmap="Blues", colorbar=False, xticks_rotation=45)
    disp.ax_.set_title("CIFAR-10 CNN confusion matrix (rows = true, cols = predicted)")


def main():
    torch.manual_seed(0)

    device = get_device()
    print(f"using device: {device}")

    train_ds = load_cifar10(train=True, n_subset=20000)
    test_ds = load_cifar10(train=False)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    cnn = build_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

    n_params = sum(p.numel() for p in cnn.parameters())
    print(f"CNN parameters: {n_params:,}")

    losses = []
    for epoch in range(20):
        cnn.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = cnn(images)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
        print(f"epoch {epoch + 1}/20  loss {losses[-1]:.4f}")

    acc = evaluate(cnn, test_loader)
    print(f"Test accuracy: {acc:.2%}  (chance is 10%)")

    torch.save(cnn.state_dict(), "cifar_cnn.pt")

    plt.figure()
    plt.plot(losses, marker="o", color="tab:orange")
    plt.title(f"CIFAR-10 CNN training loss (test acc {acc:.1%})")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy")

    plot_sample_predictions(cnn, test_ds, n=10)

    preds, trues = collect_predictions(cnn, test_loader)
    plot_confusion(preds, trues)

    plt.show()


if __name__ == "__main__":
    main()
