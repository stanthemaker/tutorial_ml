"""Week 4 - shared plumbing for every part.

Nothing conceptually new lives here. It is the code we would otherwise
copy-paste into five files: device selection (the same helper as Week 3),
dataset loading, a parameter counter, the segmentation metrics, and the
comparison figure the whole week is built around.

The *training loops* deliberately stay inside each part -- that loop is the
thing we keep pointing at ("it is the same four steps as Week 1 Part 3"), so
hiding it in a helper would defeat the point.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import InterpolationMode

# Anchored to this file, not to the working directory, so the scripts behave
# the same whether you run them from the repo root (`python
# week4_unet/part4_unet.py`, as the README suggests) or from inside
# week4_unet/. Otherwise "./data" would point at two different places and
# torchvision would helpfully re-download 800MB of cat photos.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")
# Same layout as week3_cnn/: trained weights in checkpoints/, figures in
# results/, the deck and its build script in slides/.
CKPT_DIR = os.path.join(HERE, "checkpoints")

# The task, fixed for the whole week. The Pet trimap stores 1 = foreground,
# 2 = background, 3 = "not classified" (the fuzzy ring around the animal); we
# subtract 1 so the classes are the usual 0..C-1 that CrossEntropyLoss wants.
N_CLASSES = 3
CLASS_NAMES = ["pet", "background", "border"]
# Short forms, for table headers where "background" would not fit.
CLASS_SHORT = ["pet", "bg", "border"]
IMAGE_SIZE = 96

# One colour per class, for turning a (H, W) map of indices into something you
# can look at. Pet = orange, background = blue-grey, border = white.
CLASS_COLORS = torch.tensor([
    [0.90, 0.49, 0.13],
    [0.25, 0.32, 0.45],
    [1.00, 1.00, 1.00],
])


def weights_path(name):
    """week4_unet/checkpoints/<name>.pt -- the one place weights live."""
    return os.path.join(CKPT_DIR, f"{name}.pt")


def save_history(stem, label, params, history, metrics=None):
    """Record one run's loss curve and final metrics in results/training_curves.json.

    Week 3 wrote a JSON per run and rebuilt its slide deck from those files, so
    the deck could not drift away from the numbers. Same idea here, in one
    file: slides/build_slides.py reads this to draw the training curves, and
    re-running a part overwrites only its own entry.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "training_curves.json")
    doc = {"runs": {}}
    if os.path.exists(path):
        with open(path) as f:
            doc = json.load(f)
    entry = {"label": label, "params": params, "loss": list(history)}
    if metrics is not None:
        acc, iou, miou = metrics
        entry["test"] = {"acc": round(acc, 4), "miou": round(miou, 4),
                         "iou": [round(v, 4) for v in iou.tolist()]}
    doc.setdefault("runs", {})[stem] = entry
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    return path


def save_weights(model, name):
    """Write a model's weights to week4_unet/checkpoints/<name>.pt."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = weights_path(name)
    torch.save(model.state_dict(), path)
    print(f"  saved weights to {path}")
    return path


def load_weights(model, name):
    """--eval: fill a freshly built model from checkpoints/<name>.pt.

    Strict on purpose. A key that does not match means the checkpoint was
    written by a different version of the model, and loading it anyway would
    hand you a half-random network with no error -- the one way to get a wrong
    number that looks exactly like a right one.
    """
    path = weights_path(name)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run the same command without "
                         "--eval once to train and save it")
    try:
        # map_location: a checkpoint saved on a GPU box still loads on a laptop.
        model.load_state_dict(torch.load(path, map_location="cpu"))
    except RuntimeError as err:
        raise SystemExit(f"{path} does not match this model (it was saved by an "
                         f"older version) -- retrain without --eval\n\n{err}")
    print(f"  loaded {name}.pt (skipping training)")
    return model


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
# dataset
# --------------------------------------------------------------------------

def _subsample(ds, n_subset, seed=0):
    if n_subset is None or n_subset >= len(ds):
        return ds
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))
    return Subset(ds, idx[:n_subset].tolist())


def _build_cache(split, image_size):
    """Resize every photo and trimap once, and keep them as uint8 tensors.

    Decoding 3,680 JPEGs and resizing them costs about 12 seconds per epoch,
    which would be more than the training step itself and would be paid again
    by every script. The whole dataset at 96x96 is 135MB as uint8, so we do
    the work once, cache it next to the images, and spend the rest of the week
    on the models.

    Three decisions in the two resize calls, and they are the ones that matter:

      1. The mask is resized with NEAREST. Bilinear would average a 1 and a 2
         into 1.5 -- a class that does not exist. (Try it: switch the
         interpolation below and print torch.unique on the result.)
      2. The mask is a map of class *indices*, not a picture. It is never
         scaled to [0, 1], never normalised, never touched by ToTensor.
      3. Photo and mask go through the same geometry, in the same order.
    """
    path = os.path.join(DATA_ROOT, f"pet_seg_{image_size}_{split}.pt")
    if os.path.exists(path):
        return torch.load(path)

    base = datasets.OxfordIIITPet(
        root=DATA_ROOT, split=split, target_types="segmentation", download=True)
    print(f"  preparing {split} cache ({len(base)} images at {image_size}px, once)...")

    size = (image_size, image_size)
    images = torch.empty(len(base), 3, image_size, image_size, dtype=torch.uint8)
    masks = torch.empty(len(base), image_size, image_size, dtype=torch.uint8)
    for i, (img, mask) in enumerate(base):
        img = TF.resize(img.convert("RGB"), size, InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size, InterpolationMode.NEAREST)
        images[i] = TF.pil_to_tensor(img)
        masks[i] = TF.pil_to_tensor(mask).squeeze(0) - 1   # 1/2/3 -> 0/1/2

    torch.save((images, masks), path)
    print(f"  cached to {path}")
    return images, masks


class _PetSeg(torch.utils.data.Dataset):
    """Oxford-IIIT Pet with its per-pixel trimap.

    Yields (image, mask): image is (3, S, S) float in [0, 1], mask is (S, S)
    int64 with values 0 = pet, 1 = background, 2 = border.

    The one thing happening at __getitem__ time is augmentation, and it is
    here to make a point: the flip is decided once per sample and then applied
    to the photo *and* the mask. Flip only the photo and every label is wrong
    -- which shows up as "my model just won't converge", not as an error.
    """

    def __init__(self, split, image_size=IMAGE_SIZE, augment=False):
        self.images, self.masks = _build_cache(split, image_size)
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img, mask = self.images[i], self.masks[i]
        if self.augment and torch.rand(1).item() < 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)   # one decision, both tensors
        return img.float() / 255.0, mask.long()


class _SelfTarget(torch.utils.data.Dataset):
    """Wrap the segmentation dataset into (image, image) pairs.

    This tiny class is what "no labels" looks like in code: the mask is thrown
    away and the photo becomes its own target. Part 1 trains on this.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, _ = self.base[i]
        return img, img


def load_pet(split, image_size=IMAGE_SIZE, n_subset=None, segmentation=True,
             augment=False):
    """Oxford-IIIT Pet, the one dataset for the whole week.

      * segmentation=True  -> (photo, mask), the task U-Net was invented for.
      * segmentation=False -> (photo, photo), no labels at all, for Part 1's
        autoencoder.

    Why photos and not MNIST: a digit is mostly flat black and white, so a
    blurry reconstruction still looks passable and a coarse mask would still
    look right. Fur, whiskers and grass are pure high-frequency detail --
    exactly what a bottleneck destroys -- so the skip connections have
    something visible to rescue.

    First run downloads ~800MB into DATA_ROOT.
    """
    ds = _PetSeg(split, image_size=image_size, augment=augment)
    if not segmentation:
        ds = _SelfTarget(ds)
    return _subsample(ds, n_subset)


def make_loaders(train_ds, test_ds, batch_size=32):
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size),
    )


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
# Pixel accuracy is the obvious metric and it is a trap: roughly 60% of the
# pixels in this dataset are background, so "predict background everywhere"
# already scores 0.58 without looking at the image. Worse, the border class --
# the thin ring where all the interesting errors are -- is only 12% of the
# pixels, so a model can ignore it entirely and lose almost nothing.
#
# IoU (intersection over union) fixes that by scoring each class on its own:
#
#       IoU_c = (pixels where prediction == c AND truth == c)
#               ---------------------------------------------
#               (pixels where prediction == c  OR truth == c)
#
# A class the model never predicts scores 0, no matter how rare it is. We
# report the per-class numbers *and* their mean (mIoU), and the per-class
# numbers are where Part 3's argument gets settled.

@torch.no_grad()
def confusion(model, loader, n_classes=N_CLASSES):
    """Accumulate an (n_classes, n_classes) truth-by-prediction count matrix.

    One pass over the loader, counting every pixel once. Every metric below is
    a two-line function of this matrix, which is why we build it rather than
    averaging per-batch IoUs (averaging ratios over batches is subtly wrong --
    a batch with no border pixels has no defined border IoU).
    """
    device = next(model.parameters()).device
    model.eval()
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for x, y in loader:
        pred = model(x.to(device)).argmax(dim=1).cpu()   # (N, C, H, W) -> (N, H, W)
        k = y.flatten() * n_classes + pred.flatten()
        cm += torch.bincount(k, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    return cm


def metrics_from_confusion(cm):
    """-> (pixel_accuracy, per_class_iou, mean_iou)."""
    inter = cm.diag().float()
    union = cm.sum(dim=1).float() + cm.sum(dim=0).float() - inter
    iou = inter / union.clamp(min=1)
    return (inter.sum() / cm.sum()).item(), iou, iou.mean().item()


def evaluate_seg(model, loader):
    """The one call every part uses: -> (pixel_acc, per_class_iou, mIoU)."""
    return metrics_from_confusion(confusion(model, loader))


def print_metrics(label, acc, iou, miou, width=22):
    per_class = "  ".join(f"{n} {v:.3f}" for n, v in zip(CLASS_NAMES, iou))
    print(f"  {label:<{width}} acc {acc:.3f}   mIoU {miou:.3f}   [{per_class}]")


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def colorize(mask):
    """(H, W) class indices -> (3, H, W) RGB, so masks can go in show_grid."""
    return CLASS_COLORS[mask.cpu().long()].permute(2, 0, 1)


def colorize_batch(masks):
    """(N, H, W) -> (N, 3, H, W)."""
    return torch.stack([colorize(m) for m in masks])


def _to_hwc(img):
    """(C, H, W) tensor -> something imshow understands."""
    img = img.detach().cpu().float().clamp(0, 1)
    if img.size(0) == 1:
        return img.squeeze(0), "gray"
    return img.permute(1, 2, 0), None


def show_grid(rows, row_titles, n=6, suptitle=None):
    """One column per example, one row per version of it.

    `rows` is a list of (N, 3, H, W) batches that all describe the same N
    images -- e.g. [photo, truth, prediction]. Reading down a column is the
    comparison this whole week is about, so every part plots it the same way
    on purpose: differences between parts should come from the models, not
    from the plotting.
    """
    n = min(n, rows[0].size(0))
    fig, axes = plt.subplots(len(rows), n, figsize=(1.5 * n, 1.75 * len(rows)))
    axes = np.asarray(axes).reshape(len(rows), n)
    for r, (batch, title) in enumerate(zip(rows, row_titles)):
        for c in range(n):
            data, cmap = _to_hwc(batch[c])
            axes[r, c].imshow(data, cmap=cmap, vmin=0, vmax=1)
            axes[r, c].axis("off")
        # axis("off") hides set_ylabel too, so place the row label as free text
        axes[r, 0].text(-0.15, 0.5, title, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


@torch.no_grad()
def predict_batch(model, loader, n=6):
    """One test batch -> (photos, true masks, predicted masks), all on CPU."""
    device = next(model.parameters()).device
    model.eval()
    x, y = next(iter(loader))
    x, y = x[:n], y[:n]
    pred = model(x.to(device)).argmax(dim=1).cpu()
    return x, y, pred


def finish(prefix):
    """Show the figures -- or, with WEEK4_SAVE_FIGS set, write them to disk.

    Every part ends with `finish("part3")`. Interactively that is just
    plt.show(); set WEEK4_SAVE_FIGS=1 to drop PNGs in results/ instead, which
    is how the figures in the slides get made.
    """
    if os.environ.get("WEEK4_SAVE_FIGS"):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        for i in plt.get_fignums():
            path = os.path.join(RESULTS_DIR, f"{prefix}_fig{i}.png")
            plt.figure(i).savefig(path, dpi=110, bbox_inches="tight")
            print(f"  saved figure to {path}")
        plt.close("all")
    else:
        plt.show()


# --------------------------------------------------------------------------
# building blocks shared by part1 / part2 / part3
# --------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Conv3x3 -> BN -> ReLU, twice. The unit U-Net is built out of.

    Two design choices worth saying out loud:

      * padding=1 with a 3x3 kernel keeps height and width unchanged. Sizes
        then only ever change at the explicit pool / upsample steps, which is
        what makes the skip connections line up without any cropping. (The
        original U-Net paper used no padding and had to crop every skip.)
      * BatchNorm is not doing anything conceptually new -- it is Week 3 Part
        7's trick again, and it is here so the deeper U-Net trains at the same
        learning rate as the shallow autoencoder. The comparison between them
        then stays about architecture.
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


def train(model, loader, device, loss_fn, epochs=10, lr=1e-3, label="",
          run=None, eval_fn=None):
    """Week 1 Part 3's loop, unchanged. Only `loss_fn` differs between parts.

    Kept here because all five scripts would otherwise repeat it verbatim --
    but read it once per part anyway. The whole point of this week is that
    predicting 9,216 labels per image needs no new training machinery: the
    same four steps (zero_grad, forward, backward, step) still run the show.

    `run` and `eval_fn` are instrumentation and change nothing about the
    optimisation: at the end of each epoch, eval_fn(model) returns a dict of
    measurements and run.log() sends them to trackio. Both default to None, so
    the loop below is exactly the loop from Week 1 either way.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        history.append(total / n)

        line = f"  {label}epoch {epoch + 1:2d}/{epochs}   loss {history[-1]:.4f}"
        row = {"epoch": epoch + 1, "train_loss": history[-1]}
        if eval_fn is not None:
            measured = eval_fn(model)
            row.update(measured)
            line += "   " + "  ".join(f"{k.split('/')[-1]} {v:.3f}"
                                      for k, v in measured.items()
                                      if k in HEADLINE)
        if run is not None:
            run.log(row)
        print(line)
    return history


# --------------------------------------------------------------------------
# trackio
# --------------------------------------------------------------------------
# Week 3 Part 7 introduced trackio and made one argument with it: accuracy
# tells you where a run ended up, and the instruments tell you what happened
# on the way. Week 4 keeps the habit and swaps the instruments, because the
# thing worth watching is no longer a single accuracy.
#
# What each part logs, per epoch:
#
#   train_loss        the number the optimiser is actually minimising
#   test/acc          pixel accuracy -- logged mainly so you can watch it fail
#                     to move while the interesting metric climbs
#   test/mIoU         the headline
#   test/iou_border   the argument of the whole week, on its own axis
#
# ON THE TEST SET, because it deserves saying out loud after Week 3 spent a
# session on validation discipline: these per-epoch numbers are MONITORING,
# not selection. Every part trains for a fixed number of epochs decided in
# advance and reports the last one, so nothing is ever chosen by looking at
# this curve. The moment you want to choose something -- early stopping,
# picking the best epoch, tuning the learning rate -- carve a validation split
# out of trainval first, the way Week 3 Part 7 does, and log that instead.

TRACKIO_PROJECT = "week4-unet"

# Metrics echoed on the per-epoch print line; everything else goes to trackio
# only, so the terminal output stays readable.
HEADLINE = {"test/mIoU", "test/psnr"}


class _NullRun:
    """What you get when trackio is missing or switched off: a no-op.

    Every part stays runnable with nothing installed but torch, which is the
    whole reason the logging goes through an object instead of through
    `import trackio` at the top of five files.
    """

    def log(self, row):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _TrackioRun:
    def __init__(self, name, config, project):
        self.name, self.config, self.project = name, config, project
        self.step = 0

    def __enter__(self):
        import trackio
        self._trackio = trackio
        trackio.init(project=self.project, name=self.name, config=self.config)
        return self

    def log(self, row):
        self.step += 1
        self._trackio.log(row, step=self.step)

    def __exit__(self, *exc):
        self._trackio.finish()
        return False


def tracked(name, config=None, project=TRACKIO_PROJECT):
    """A logging context for one run:  with tracked("unet_skips_on") as run: ...

    Returns a no-op if trackio is not installed, or if WEEK4_NO_TRACKIO is set
    in the environment -- handy when you are iterating and do not want fifty
    junk runs in the dashboard.

        trackio show --project "week4-unet"
    """
    if os.environ.get("WEEK4_NO_TRACKIO"):
        return _NullRun()
    try:
        import trackio  # noqa: F401
    except ImportError:
        print("  (trackio not installed -- skipping logging)")
        return _NullRun()
    return _TrackioRun(name, config or {}, project)


def undisturbed(fn):
    """Run a measurement without moving the global random number generator.

    This is not a nicety, it is a bug fix, and the bug is worth knowing about
    because it will bite you in your own code.

    Creating a DataLoader ITERATOR draws one number from the global RNG (torch
    uses it to seed the workers). Our training loader shuffles, and shuffling
    also draws from the global RNG -- so the moment we started evaluating on
    the test loader once per epoch, the training set was shuffled differently
    from then on. Epoch 1 matched the un-instrumented run to four decimals and
    epoch 2 did not, which is exactly the signature.

    Nothing about the measurement was wrong. Measuring changed the experiment.
    Saving the RNG state around the probe puts that right, and the check that
    it worked is that the per-epoch losses match the runs from before any of
    this logging existed.
    """
    def wrapped(model):
        state = torch.get_rng_state()
        try:
            return fn(model)
        finally:
            torch.set_rng_state(state)
    return wrapped


def seg_probe(loader):
    """-> a function that scores a model the way this week scores models.

    Handed to train(eval_fn=...), so the segmentation metrics appear once per
    epoch as curves rather than only once at the end as a table.
    """
    @undisturbed
    def probe(model):
        acc, iou, miou = evaluate_seg(model, loader)
        model.train()          # evaluate_seg left it in eval mode
        return {"test/acc": acc, "test/mIoU": miou,
                "test/iou_pet": iou[0].item(), "test/iou_bg": iou[1].item(),
                "test/iou_border": iou[2].item()}
    return probe
