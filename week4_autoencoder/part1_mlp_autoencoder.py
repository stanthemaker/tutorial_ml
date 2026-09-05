"""Week 4 - Part 1: What if the output has to be an image too?

Every model so far ended in `Linear(..., 10)`: image in, one label out. The
last layer's job was to throw away everything except the class. This week the
output is an image again, which means the network needs a second half that
*builds* rather than *summarises*.

The simplest version of that idea is an autoencoder:

    x  --encoder-->  z  --decoder-->  x_hat        (and we ask x_hat ~= x)
   784               32               784

The middle vector z is the "bottleneck". Because it is much smaller than the
input, the encoder cannot copy the image through -- it has to decide what is
worth keeping. That forced choice is the entire lesson.

Three things to notice while reading:

  1. The training loop is the Week 1 Part 3 loop, unchanged. Forward, loss,
     zero_grad + backward, step. The *only* edit is `loss_fn(recon, x)`
     instead of `loss_fn(pred, y)`.

  2. We never touch the labels. This is our first unsupervised task: the
     supervision signal is the input itself. `for x, _ in loader` -- the `_`
     is the label we are deliberately ignoring.

  3. The decoder ends in Sigmoid because ToTensor gives pixels in [0, 1], so
     the output range must be [0, 1] as well.

We run the whole thing twice, on MNIST and on CIFAR-10, because the contrast
sets up Part 2. MNIST reconstructs beautifully from 32 numbers; CIFAR-10 --
same architecture, same bottleneck ratio -- comes back as coloured mush. The
reason is not "CIFAR is harder": it is that `x.view(-1, 3072)` on the first
line threw away the 2D structure, and photos need it far more than digits do.
"""

import math

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from common import (get_device, count_params, load_mnist, load_cifar10,
                    make_loaders, show_grid, save_weights)


def build_autoencoder(in_dim, hidden_dim=128, latent_dim=32):
    """Encoder in_dim -> hidden -> latent, decoder back out again.

    Note the symmetry: the decoder is the encoder read right-to-left. That is
    a convention, not a requirement -- but it makes the "compress then
    reconstruct" story legible, and it is the shape we will keep all week.

    The encoder has NO activation on its last layer: z is a free-form vector
    of features, and squashing it would throw away information for nothing.
    The decoder's last layer DOES get a Sigmoid, because those numbers are
    pixels and pixels live in [0, 1].
    """
    encoder = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, latent_dim),
    )
    decoder = nn.Sequential(
        nn.Linear(latent_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, in_dim),
        nn.Sigmoid(),
    )
    return nn.Sequential(encoder, decoder), encoder, decoder


def train_autoencoder(model, loader, device, epochs=10, lr=1e-3, quiet=False):
    """The Week 1 Part 3 training loop with exactly one line changed."""
    model.to(device)
    loss_fn = nn.MSELoss()  # pixel-wise squared error: "how far off is each pixel"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, _ in loader:                 # the label is ignored -- unsupervised
            x = x.to(device)
            flat = x.view(x.size(0), -1)    # (N, C, H, W) -> (N, C*H*W). The 2D
                                            # structure dies here; Part 2 fixes it.
            recon = model(flat)
            loss = loss_fn(recon, flat)     # <-- target IS the input
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
        if not quiet:
            print(f"  epoch {epoch + 1}/{epochs}  MSE {losses[-1]:.5f}")
    return losses


@torch.no_grad()
def reconstruct(model, batch, device):
    """Run a (N, C, H, W) batch through the flat model and reshape back."""
    model.eval()
    shape = batch.shape
    flat = batch.to(device).view(shape[0], -1)
    return model(flat).view(shape).cpu()


# --------------------------------------------------------------------------
# experiment 1: how small can the bottleneck get?
# --------------------------------------------------------------------------

@torch.no_grad()
def flat_test_mse(model, loader, device):
    """Reconstruction MSE over the whole test set, not just the 8 shown images.

    Worth the extra loop: on 8 images the numbers are noisy enough that a
    larger bottleneck can appear to lose to a smaller one, which muddles the
    exact trend the experiment exists to demonstrate.
    """
    model.eval()
    total, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        flat = x.view(x.size(0), -1)
        total += torch.mean((model(flat) - flat) ** 2).item() * x.size(0)
        n += x.size(0)
    return total / n


def bottleneck_sweep(train_loader, test_loader, in_dim, device,
                     dims=(128, 32, 8, 2), epochs=8, name=""):
    """Train one autoencoder per bottleneck size and stack the results.

    This is the experiment that makes "bottleneck" concrete. Watch what
    degrades first as the latent shrinks: identity goes before shape. At 8
    dims a 4 may come back looking like a 9 -- the model kept "roundish digit
    with a stem" and dropped the part that distinguishes them.
    """
    test_batch, _ = next(iter(test_loader))
    test_batch = test_batch[:8]

    rows, titles = [test_batch], ["original"]
    print(f"\n[{name}] bottleneck sweep")
    for d in dims:
        torch.manual_seed(0)
        model, _, _ = build_autoencoder(in_dim, latent_dim=d)
        train_autoencoder(model, train_loader, device, epochs=epochs, quiet=True)
        rows.append(reconstruct(model, test_batch, device))
        titles.append(f"z = {d}")
        mse = flat_test_mse(model, test_loader, device)
        print(f"  latent {d:>3}  ({in_dim / d:5.1f}x compression)  "
              f"MSE {mse:.5f}  PSNR {10 * math.log10(1 / mse):5.2f} dB")
    show_grid(rows, titles, n=8,
              suptitle=f"{name}: reconstruction as the bottleneck shrinks")


# --------------------------------------------------------------------------
# experiment 2: what does a 2-D latent space look like?
# --------------------------------------------------------------------------

def plot_latent_2d(train_loader, test_loader, in_dim, device, epochs=10):
    """Squeeze MNIST into 2 numbers and plot them, coloured by digit.

    This is the direct sequel to week2_mlp/part4_mnist_pca_verify.py. There we
    projected MNIST onto its first two principal components; here we *learn*
    two coordinates instead. Same picture type, different machinery.

    The point worth making at the plot: a linear autoencoder -- this same
    network with every ReLU deleted -- spans the same subspace as PCA. It is
    literally the same solution reached by gradient descent instead of an
    eigendecomposition. The ReLUs are what let this version bend the space and
    pull apart clusters PCA leaves overlapping (4/9 and 3/5/8 are the usual
    suspects). We are not deriving that here, just naming it.
    """
    torch.manual_seed(0)
    model, encoder, _ = build_autoencoder(in_dim, latent_dim=2)
    print("\n[MNIST] training a 2-D bottleneck for the latent map")
    train_autoencoder(model, train_loader, device, epochs=epochs, quiet=True)

    encoder.eval()
    zs, ys = [], []
    with torch.no_grad():
        for x, y in test_loader:
            zs.append(encoder(x.to(device).view(x.size(0), -1)).cpu())
            ys.append(y)
    z = torch.cat(zs)
    y = torch.cat(ys)

    plt.figure(figsize=(6.5, 5.5))
    sc = plt.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10", s=4, alpha=0.6)
    plt.colorbar(sc, ticks=range(10), label="true digit")
    plt.title("MNIST squeezed into 2 learned numbers\n"
              "(same plot as week2 part4's PCA -- but learned, and non-linear)")
    plt.xlabel("latent dim 0")
    plt.ylabel("latent dim 1")
    plt.tight_layout()


def run_dataset(name, train_ds, test_ds, image_shape, device, epochs=10):
    """Train one reference autoencoder (latent 32) and show its reconstructions."""
    in_dim = math.prod(image_shape)
    train_loader, test_loader = make_loaders(train_ds, test_ds, batch_size=128)

    torch.manual_seed(0)
    model, _, _ = build_autoencoder(in_dim, latent_dim=32)
    print(f"\n=== {name}: MLP autoencoder {in_dim} -> 128 -> 32 -> 128 -> {in_dim} ===")
    print(f"parameters: {count_params(model):,}")
    losses = train_autoencoder(model, train_loader, device, epochs=epochs)
    save_weights(model, f"part1_{name.lower().replace('-', '')}_ae")

    test_batch, _ = next(iter(test_loader))
    test_batch = test_batch[:8]
    recon = reconstruct(model, test_batch, device)
    mse = flat_test_mse(model, test_loader, device)
    print(f"{name} test MSE {mse:.5f}  PSNR {10 * math.log10(1 / mse):.2f} dB")

    show_grid([test_batch, recon], ["original", "z = 32"], n=8,
              suptitle=f"{name}: {in_dim} pixels -> 32 numbers -> {in_dim} pixels")

    plt.figure()
    plt.plot(losses, marker="o")
    plt.title(f"{name} MLP autoencoder training loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.tight_layout()

    return train_loader, test_loader, in_dim


def main():
    device = get_device()
    print(f"using device: {device}")

    # ---- MNIST: the clean version of the story ----------------------------
    mnist_train = load_mnist(train=True, n_subset=20000)
    mnist_test = load_mnist(train=False, n_subset=2000)
    m_train_loader, m_test_loader, m_dim = run_dataset(
        "MNIST", mnist_train, mnist_test, (1, 28, 28), device, epochs=10)

    bottleneck_sweep(m_train_loader, m_test_loader, m_dim, device,
                     dims=(128, 32, 8, 2), epochs=8, name="MNIST")

    plot_latent_2d(m_train_loader, m_test_loader, m_dim, device, epochs=10)

    # ---- CIFAR-10: the same architecture meeting a real photograph --------
    # 3072 -> 32 is a 96x compression, versus MNIST's 24x, and photographs
    # carry far more detail per pixel than pen strokes. Expect blurry blobs
    # with roughly the right colours -- that failure is the argument for
    # Part 2, not a bug to fix here.
    cifar_train = load_cifar10(train=True, n_subset=20000)
    cifar_test = load_cifar10(train=False, n_subset=2000)
    c_train_loader, c_test_loader, c_dim = run_dataset(
        "CIFAR-10", cifar_train, cifar_test, (3, 32, 32), device, epochs=15)

    # More epochs than the MNIST sweep: a 256-d latent has more capacity to
    # fit and needs longer before the ordering "wider latent = lower error"
    # actually shows up in the numbers.
    bottleneck_sweep(c_train_loader, c_test_loader, c_dim, device,
                     dims=(256, 64, 16), epochs=20, name="CIFAR-10")

    print("\nTakeaway: the same 784 -> 32 trick that nails MNIST turns CIFAR-10")
    print("into coloured mush. The first thing this model does is .view(-1, 3072),")
    print("which tells it that pixel (0,0) and pixel (0,1) are unrelated inputs.")
    print("Part 2 keeps the 2D structure on both halves of the network.")

    plt.show()


if __name__ == "__main__":
    main()
