"""Week 3 - Demo: the limit of an MLP on CIFAR-10 (and *seeing* why).

Companion to demo_mlp_limits.py. That file used shifted MNIST to expose one
specific MLP weakness -- no translation invariance. This file makes a broader
point on harder data: on real colour photos an MLP barely gets off the ground,
and if we crack it open the reason is visible in its own weights.

The setup is the exact Week 2 recipe -- flatten the image to one long vector,
send it through Linear -> ReLU -> Linear -- but the image is now a 32x32 RGB
CIFAR-10 photo (plane, car, bird, cat, ...). Two things follow:

  1. Accuracy plateaus low. Even with well over a million parameters the MLP
     lands around ~45-50% (chance is 10%, and the Week 3 CNN clears 70%+).
     Flattening throws away the 2D layout the moment training starts.

  2. We can *see* the failure. The first Linear layer holds one weight per
     input pixel per hidden neuron, so each neuron's weights fold back into a
     32x32x3 colour "template" -- the picture that neuron looks for. On natural
     images these come out as blurry, position-locked colour blobs: the MLP is
     trying to memorise whole-image templates at fixed absolute positions. A
     cat in the top-left and the same cat in the bottom-right look like two
     unrelated inputs to it.

That is the whole motivation for the CNN in part3_cnn_cifar10.py: slide a small
filter so position stops mattering, and reuse it everywhere so you need far
fewer weights.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CLASSES = [
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_device():
    """Pick the fastest available device: NVIDIA GPU, then Mac GPU, then CPU.

    Two GPU backends, checked with two flags:
      * use_windows_gpu -- an NVIDIA card via CUDA (Windows/Linux).
      * use_mac_gpu     -- Apple-silicon (M1/M2/...) GPU via Metal (MPS).
    The biggest Week 3 win is here, since CIFAR training is the heaviest job.
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
    """Load CIFAR-10 as flat 3072-vectors' worth of (3, 32, 32) tensors.

    We keep the tensors as (3, 32, 32) here and let the model's Flatten turn
    them into 3072-long vectors -- the same "throw away the 2D shape" step the
    MLP always does. Normalising to roughly [-1, 1] helps training converge.
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(root="./data", train=train, download=True, transform=tf)
    if n_subset is not None and n_subset < len(ds):
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))
        ds = Subset(ds, idx[:n_subset].tolist())
    return ds


def build_mlp():
    """The Week 2 recipe again: flatten, one hidden layer, classify.

    Input is 3*32*32 = 3072 numbers. That first Linear(3072, 512) alone is
    3072 * 512 = 1,572,864 weights -- the parameter count explodes precisely
    because every pixel (and colour channel) is wired to every hidden unit.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3072, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
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


def plot_learned_templates(model, n=16):
    """Fold the first layer's weights back into 32x32x3 colour templates.

    build_mlp starts with Flatten, so model[1] is the Linear(3072, 512). Its
    weight has shape (512, 3072): one row per hidden neuron, one number per
    input value. After Flatten, PyTorch orders those 3072 values channels-first
    (3 x 32 x 32), so we reshape to (-1, 3, 32, 32) and move the channel axis
    last for imshow -- reshaping straight to (-1, 32, 32, 3) would scramble the
    colours.

    Each template is the picture its neuron responds to most. On CIFAR they come
    out as soft, position-locked colour blobs -- the visible signature of an MLP
    trying to template-match whole images instead of detecting local patterns.
    """
    W = model[1].weight.detach().cpu()               # (512, 3072); .cpu() for matplotlib
    templates = W.reshape(-1, 3, 32, 32)             # channels-first, per neuron
    templates = templates.permute(0, 2, 3, 1)        # -> (512, 32, 32, 3) for imshow

    fig, axes = plt.subplots(4, n // 4, figsize=(1.5 * (n // 4), 6))
    for i, ax in enumerate(axes.ravel()):
        t = templates[i]
        # Per-neuron min-max stretch to [0, 1] so the faint structure is visible
        # (raw weights are tiny and would show up as flat grey otherwise).
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        ax.imshow(t)
        ax.set_title(f"neuron {i}", fontsize=8)
        ax.axis("off")
    fig.suptitle(
        "What the MLP's first layer learned: one 32x32x3 template per hidden neuron\n"
        "blurry, position-locked colour blobs -- whole-image templating, not local features"
    )
    fig.tight_layout()


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
        ax.set_title(CLASSES[pred], color="green" if pred == true else "red", fontsize=9)
        ax.axis("off")
    fig.suptitle("Test photos and the MLP's guess (red = wrong)")
    fig.tight_layout()


def main():
    torch.manual_seed(0)

    device = get_device()
    print(f"using device: {device}")

    train_ds = load_cifar10(train=True, n_subset=20000)
    test_ds = load_cifar10(train=False)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    mlp = build_mlp().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

    # The count that makes the point: over 1.5M weights, almost all in the first
    # 3072 -> 512 matrix -- and it still won't be enough on natural images.
    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"MLP parameters: {n_params:,}")

    losses = []
    for epoch in range(15):
        mlp.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = loss_fn(mlp(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
        print(f"epoch {epoch + 1}/15  loss {losses[-1]:.4f}")

    acc = evaluate(mlp, test_loader)
    print(f"\nTest accuracy: {acc:.2%}  (chance is 10%; the Week 3 CNN clears 70%+)")

    # View 1: the loss curve and how low the ceiling is.
    plt.figure()
    plt.plot(losses, marker="o", color="crimson")
    plt.title(f"CIFAR-10 MLP training loss (test acc {acc:.1%}, {n_params:,} params)")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy")

    # View 2: sample guesses, so the low number has faces attached to it.
    plot_sample_predictions(mlp, test_ds, n=10)

    # View 3: crack the MLP open and see the position-locked templates it learned.
    plot_learned_templates(mlp)

    plt.show()


if __name__ == "__main__":
    main()
