"""Week 3 - Demo: the limit of an MLP on images.

Same role as Week 2's demo_linear_limits.py, one level up. There, a straight
line failed on a curvy wave and that failure motivated the MLP. Here the MLP
itself hits its own wall, and that failure motivates the CNN.

The wall is *translation*. An MLP flattens the 28x28 image into 784 numbers
and wires every pixel to every hidden unit. Pixel (0,0) and pixel (5,5) are
just two unrelated entries in a long vector -- the network has no built-in
notion that they are neighbours, or that a "3" is a "3" no matter *where* in
the frame it sits. So if we train on centred digits and then slide each test
digit a few pixels sideways, accuracy falls off a cliff: to the MLP the
shifted image is a brand-new pattern it never saw.

We show two things:
  1. Train an MLP on normal (centred) MNIST, then test it on the *same*
     digits shifted by a few pixels. Accuracy collapses.
  2. Print the parameter count. Almost all of it lives in that first
     784 -> hidden matrix, and it grows with the number of input pixels.
     A bigger image would blow this up further.

The CNN fixes both: a small filter slides across the image (built-in
translation handling) and *reuses* the same weights everywhere (far fewer
parameters). That is the whole pitch for the rest of Week 3.
"""

import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Checkpoints land next to this file, never in the current working directory.
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "part1b.pt")


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


def load_mnist(train):
    return datasets.MNIST(
        root="./data", train=train, download=True, transform=transforms.ToTensor()
    )


def build_mlp():
    # The exact recipe from Week 2 part3: flatten to 784, one hidden layer.
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


def shift_batch(images, dx, dy):
    """Slide a batch of images by (dx, dy) pixels, filling the edge with zeros.

    torch.roll wraps pixels around the border, so we roll and then blank out
    the wrapped-in strip -- the digit really translates instead of teleporting
    from the opposite side.
    """
    shifted = torch.roll(images, shifts=(dy, dx), dims=(2, 3))
    if dy > 0:
        shifted[:, :, :dy, :] = 0
    elif dy < 0:
        shifted[:, :, dy:, :] = 0
    if dx > 0:
        shifted[:, :, :, :dx] = 0
    elif dx < 0:
        shifted[:, :, :, dx:] = 0
    return shifted


def plot_learned_templates(model, n=16):
    """Reshape the first layer's weights back into 28x28 images.

    build_mlp starts with Flatten, so model[1] is the Linear(784, 256). Its
    weight has shape (256, 784): one row of 784 numbers per hidden neuron, i.e.
    one weight for every input pixel. Fold each row back into a 28x28 grid and
    it becomes that neuron's "template" -- the picture it responds most
    strongly to (bright = looks for ink here, dark = looks for background).

    This is the MLP counterpart of Week 2 part4's PCA pictures and Week 3
    part4's conv filters: a way to *see* what the model learned. The giveaway
    is that these templates are whole-digit, centred blobs -- each neuron is
    tuned to ink at fixed absolute positions. That is exactly why sliding the
    digit a few pixels breaks them: the template no longer lines up.
    """
    W = model[1].weight.detach().cpu()    # (256, 784); .cpu() so matplotlib can read it
    templates = W.reshape(-1, 28, 28)     # one 28x28 picture per hidden neuron

    fig, axes = plt.subplots(4, n // 4, figsize=(1.5 * (n // 4), 6))
    for i, ax in enumerate(axes.ravel()):
        # Symmetric scale so 0 (no preference) maps to mid-grey and the
        # positive/negative weights are comparable across neurons.
        vmax = templates[i].abs().max()
        ax.imshow(templates[i], cmap="bwr", vmin=-vmax, vmax=vmax)
        ax.set_title(f"neuron {i}", fontsize=8)
        ax.axis("off")
    fig.suptitle(
        "What the MLP's first layer learned: one 28x28 template per hidden neuron\n"
        "(red = looks for ink here, blue = looks for background) -- all tied to fixed positions"
    )
    fig.tight_layout()


def evaluate(model, loader, dx=0, dy=0):
    device = next(model.parameters()).device  # run on whatever device the model is on
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = shift_batch(images, dx, dy).to(device)
            labels = labels.to(device)
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    torch.manual_seed(0)

    device = get_device()
    print(f"using device: {device}")

    train_loader = DataLoader(load_mnist(train=True), batch_size=128, shuffle=True)
    test_loader = DataLoader(load_mnist(train=False), batch_size=256)

    mlp = build_mlp().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

    for epoch in range(5):  # a few epochs is plenty to reach ~97% on centred data
        mlp.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = loss_fn(mlp(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch + 1}/5 done")

    # Keep the weights so the shift experiment below can be re-run (or the
    # templates re-plotted) without paying for another five epochs.
    torch.save(mlp.state_dict(), WEIGHTS)
    print(f"saved weights to {WEIGHTS}")

    # How many learnable numbers is that? Almost all of them sit in the first
    # 784 -> 256 matrix (784 * 256 = 200,704 weights). That count is tied to the
    # number of input pixels -- it is the price of wiring every pixel to every
    # hidden unit.
    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"\nMLP parameters: {n_params:,}")

    # Accuracy vs how far we slide the test digits. Same digits, same labels --
    # only the position changes.
    shifts = [0, 1, 2, 3, 4, 5]
    accs = [evaluate(mlp, test_loader, dx=s, dy=s) for s in shifts]
    print("\nshift (px)  accuracy")
    for s, a in zip(shifts, accs):
        print(f"   {s:>2}        {a:.2%}")

    # Left: accuracy falling as the digit moves -- the MLP is not translation
    # invariant. Right: a single digit at each shift, so students can see the
    # test images really are the same digit, just relocated.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot([s for s in shifts], [a * 100 for a in accs], marker="o", color="crimson")
    axes[0].set_title(f"MLP accuracy collapses when digits shift\n({n_params:,} parameters)")
    axes[0].set_xlabel("shift (pixels, diagonal)")
    axes[0].set_ylabel("test accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(alpha=0.3)

    sample, _ = next(iter(test_loader))
    one = sample[:1]
    inner = axes[1].inset_axes([0, 0, 1, 1])
    inner.axis("off")
    grid = torch.cat([shift_batch(one, s, s)[0, 0] for s in shifts], dim=1)
    inner.imshow(grid, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title("The same digit, shifted 0 -> 5 px (what the MLP is tested on)")

    fig.tight_layout()

    # Third view: crack the MLP open and look at what its first layer learned.
    plot_learned_templates(mlp)

    plt.show()


if __name__ == "__main__":
    main()
