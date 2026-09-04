"""Week 4 - shared plumbing for every part.

Nothing conceptually new lives here. It is the code we would otherwise
copy-paste into six files: device selection (same helper as Week 3), dataset
loading, a parameter counter, the reconstruction-quality metric, and the
three-row comparison figure the whole week is built around.

The *training loops* deliberately stay inside each part -- that loop is the
thing we keep pointing at ("it is the same four steps as Week 1 Part 3"), so
hiding it in a helper would defeat the point.
"""

import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Anchored to this file, not to the working directory, so the scripts behave
# the same whether you run them from the repo root (`python
# week4_autoencoder/part3_unet.py`, as the README suggests) or from inside
# week4_autoencoder/. Otherwise "./data" would point at two different places
# and torchvision would helpfully re-download 800MB of cat photos.
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Trained weights are anchored the same way, so a .pt always lands in
# week4_autoencoder/ rather than wherever you happened to run python from.
WEIGHTS_DIR = os.path.dirname(os.path.abspath(__file__))


def save_weights(model, name):
    """Write a model's weights to week4_autoencoder/<name>.pt.

    Most parts here train two or three models to compare them, so the name
    carries the variant ("part3_unet_skips_on"), not just the part number.
    The sweeps -- bottleneck sizes, upsampling styles -- are deliberately not
    saved: they exist to produce one plot each and are never reloaded.
    """
    path = os.path.join(WEIGHTS_DIR, f"{name}.pt")
    torch.save(model.state_dict(), path)
    print(f"  saved weights to {path}")
    return path


# --------------------------------------------------------------------------
# device
# --------------------------------------------------------------------------

def get_device():
    """Pick the fastest available device: NVIDIA GPU, then Mac GPU, then CPU.

    Identical to Week 3's helper. Week 4 trains image-to-image models, which
    are heavier than a classifier, so a GPU helps more than it did before --
    but everything still runs on CPU if you shrink the subset sizes.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------
# Every dataset here is returned with pixels in [0, 1] (ToTensor's default
# range). That is not an accident: the decoders in this week all end in a
# Sigmoid, whose output range is exactly [0, 1]. Match the two or the model
# spends its whole budget fighting its own output layer.

def _subsample(ds, n_subset, seed=0):
    if n_subset is None or n_subset >= len(ds):
        return ds
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))
    return Subset(ds, idx[:n_subset].tolist())


def load_mnist(train, n_subset=None):
    """MNIST as (1, 28, 28) tensors in [0, 1]."""
    ds = datasets.MNIST(
        root=DATA_ROOT, train=train, download=True, transform=transforms.ToTensor()
    )
    return _subsample(ds, n_subset)


def load_cifar10(train, n_subset=None):
    """CIFAR-10 as (3, 32, 32) tensors in [0, 1].

    Note we do NOT normalise with mean/std the way a classifier would. For a
    classifier the pixel scale is just a preprocessing detail; here the pixels
    *are* the prediction target, so we keep them in the plain [0, 1] range the
    Sigmoid can produce and matplotlib can display.
    """
    ds = datasets.CIFAR10(
        root=DATA_ROOT, train=train, download=True, transform=transforms.ToTensor()
    )
    return _subsample(ds, n_subset)


def load_pet(split, image_size=96, n_subset=None, segmentation=False):
    """Oxford-IIIT Pet: real photographs, resized to a square.

    Two ways to use it, both of which we need this week:

      * segmentation=False -> the dataset yields (image, image). We ignore the
        label entirely and use the photo as its own target, which is what a
        denoising / inpainting autoencoder wants.
      * segmentation=True  -> yields (image, mask) where mask is a (H, W)
        LongTensor of class indices 0/1/2 = pet / background / border. This is
        the task U-Net was invented for (part4_unet_segmentation.py).

    Why photos and not MNIST for the U-Net parts: a digit is mostly flat black
    and white, so a blurry reconstruction still looks passable. Fur, whiskers
    and grass are pure high-frequency detail -- exactly what a bottleneck
    destroys -- so the skip connections have something visible to rescue.

    First run downloads ~800MB into DATA_ROOT.
    """
    img_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # -> (3, H, W) in [0, 1]
    ])

    if not segmentation:
        # target_types=[] makes the dataset return the image alone; the
        # wrapper below turns it into the (input, target) pair a training loop
        # expects, with the target being the image itself.
        base = datasets.OxfordIIITPet(
            root=DATA_ROOT, split=split, target_types=[],
            transform=img_tf, download=True,
        )
        return _subsample(_SelfTarget(base), n_subset)

    # For masks: NEAREST resize, because class indices must not be blended.
    # Interpolating a 1 and a 2 into 1.5 would invent a class that doesn't
    # exist. The trimap is stored as 1/2/3, so we subtract 1 to get 0/1/2.
    mask_tf = transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])
    base = datasets.OxfordIIITPet(
        root=DATA_ROOT, split=split, target_types="segmentation",
        transform=img_tf, target_transform=mask_tf, download=True,
    )
    return _subsample(_TrimapToClass(base), n_subset)


class _SelfTarget(torch.utils.data.Dataset):
    """Wrap a dataset of images into (image, image) pairs.

    This tiny class is the whole idea of "unsupervised" in code form: there is
    no label file anywhere, the supervision signal is the input itself.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img = self.base[i]
        if isinstance(img, (tuple, list)):  # torchvision returns a 1-tuple here
            img = img[0]
        return img, img


class _TrimapToClass(torch.utils.data.Dataset):
    """Turn the Pet trimap (uint8 values 1/2/3) into class indices 0/1/2."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, mask = self.base[i]
        return img, (mask.squeeze(0).long() - 1)


def make_loaders(train_ds, test_ds, batch_size=128):
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size),
    )


# --------------------------------------------------------------------------
# corruption: how we build "input != target" tasks out of unlabelled images
# --------------------------------------------------------------------------

def add_noise(x, sigma=0.3):
    """Add Gaussian noise and clip back into the valid pixel range.

    Called fresh on every batch, so the model never sees the same corrupted
    version of an image twice -- the noise acts as data augmentation too.
    """
    return (x + sigma * torch.randn_like(x)).clamp(0.0, 1.0)


def mask_center(x, frac=0.35):
    """Cut a square hole out of the middle of each image (inpainting input).

    Returns (masked_image, mask) where mask is 1 inside the hole. The hole is
    filled with 0.5 (mid grey) rather than 0 so the model cannot cheat by
    detecting "pure black = hole" on a dataset whose backgrounds are dark.
    """
    _, _, h, w = x.shape
    size_h, size_w = int(h * frac), int(w * frac)
    top, left = (h - size_h) // 2, (w - size_w) // 2

    mask = torch.zeros_like(x[:, :1])  # (N, 1, H, W)
    mask[:, :, top:top + size_h, left:left + size_w] = 1.0

    masked = x.clone()
    masked[:, :, top:top + size_h, left:left + size_w] = 0.5
    return masked, mask


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def psnr(pred, target):
    """Peak signal-to-noise ratio, in dB. Higher is better.

    For pixels in [0, 1] this is just MSE on a log scale:

        PSNR = 10 * log10(1 / MSE)

    It carries no information MSE doesn't -- it is here because image papers
    quote PSNR, and because "+3 dB" is easier to feel than "MSE 0.0041 vs
    0.0082". Roughly: +3 dB means the error energy halved.
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


@torch.no_grad()
def eval_mse(model, loader, corrupt=None):
    """Average MSE between the model's output and the clean target.

    `corrupt` is the function that turns a clean batch into the model's input
    (noise, masking, ...). If it is None the task is plain reconstruction and
    the input is the target.
    """
    device = next(model.parameters()).device
    model.eval()
    total, n = 0.0, 0
    for clean, _ in loader:
        clean = clean.to(device)
        x = clean if corrupt is None else corrupt(clean)
        recon = model(x)
        total += torch.mean((recon - clean) ** 2).item() * clean.size(0)
        n += clean.size(0)
    return total / n


# --------------------------------------------------------------------------
# the one figure we reuse everywhere
# --------------------------------------------------------------------------

def _to_hwc(img):
    """(C, H, W) tensor -> something imshow understands."""
    img = img.detach().cpu().clamp(0, 1)
    if img.size(0) == 1:
        return img.squeeze(0), "gray"
    return img.permute(1, 2, 0), None


def show_grid(rows, row_titles, n=8, suptitle=None):
    """One column per example, one row per version of it.

    `rows` is a list of (N, C, H, W) batches that all describe the same N
    images -- e.g. [clean, noisy, reconstruction]. Reading down a column is
    the comparison this whole week is about, so every part plots it the same
    way on purpose: differences between parts should come from the models,
    not from the plotting.
    """
    n = min(n, rows[0].size(0))
    fig, axes = plt.subplots(len(rows), n, figsize=(1.5 * n, 1.7 * len(rows)))
    axes = axes.reshape(len(rows), n)
    for r, (batch, title) in enumerate(zip(rows, row_titles)):
        for c in range(n):
            data, cmap = _to_hwc(batch[c])
            axes[r, c].imshow(data, cmap=cmap, vmin=0, vmax=1)
            axes[r, c].axis("off")
        # axis("off") hides set_ylabel too, so place the row label as free text
        axes[r, 0].text(-0.15, 0.5, title, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=10)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


@torch.no_grad()
def sample_batch(model, loader, corrupt=None, n=8):
    """Grab one batch, run the model, return (clean, model_input, recon).

    Everything comes back on the CPU, ready for show_grid.
    """
    device = next(model.parameters()).device
    model.eval()
    clean, _ = next(iter(loader))
    clean = clean[:n].to(device)
    x = clean if corrupt is None else corrupt(clean)
    recon = model(x)
    return clean.cpu(), x.cpu(), recon.cpu()


# --------------------------------------------------------------------------
# building blocks shared by part2 / part3 / part4
# --------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Conv3x3 -> BN -> ReLU, twice. The unit U-Net is built out of.

    Two design choices worth saying out loud:

      * padding=1 with a 3x3 kernel keeps height and width unchanged. Sizes
        then only ever change at the explicit pool / upsample steps, which is
        what makes the skip connections line up without any cropping. (The
        original U-Net paper used no padding and had to crop every skip.)
      * BatchNorm is not doing anything conceptually new -- it just makes the
        deeper U-Net train at the same learning rate as the shallow CNN, so
        the comparison between them stays fair.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
