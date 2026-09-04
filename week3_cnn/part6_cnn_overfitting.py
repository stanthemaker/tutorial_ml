"""Week 3 - Part 6: Overfitting -- how to see it, and what actually fixes it.

part3 trained a CNN on CIFAR-10 and reported one number at the end: 68% test
accuracy. That number hides the story. Measure the same checkpoint on the data
it trained on and you get:

    train accuracy  100.0%
    test  accuracy   68.3%

The model did not fail to learn. It learned its 20,000 training photos
*perfectly* -- every single one, by heart -- and that knowledge carried over to
barely two thirds of photos it had never seen. A 32-point gap between train and
test is the signature of overfitting.

part3 had no way to show you this, because the only curve it plotted was
training loss. Training loss always goes down. It is the one number guaranteed
to tell you nothing about generalisation, and watching it is how people convince
themselves a memorising model is a good model.

So this file does two things.

(1) IT MAKES THE GAP VISIBLE. Every epoch we measure accuracy twice -- on
    images the model is fitting, and on held-out images it never trains on --
    and plot both. The moment the two lines separate is the moment memorisation
    starts. On the baseline that happens around epoch 10, long before the 100
    epochs part3 spent.

(2) IT LETS YOU SWITCH THE TWO STANDARD FIXES ON AND OFF, so you can see what
    each is worth rather than take it on faith:

      --augment     random crop + horizontal flip on the training images
      --regularize  dropout before the classifier head, plus weight decay

    Neither changes the amount of data or the architecture's capacity. They
    change what the model is *allowed* to do with them.

(3) IT ALWAYS KEEPS THE BEST CHECKPOINT, which is the third fix and the only
    one that is free. Every epoch we snapshot the weights if validation
    accuracy improved, and hand those weights back at the end. The baseline's
    validation peaks at epoch 18 and then *declines* -- epochs 19 to 30 make
    the model measurably worse. part3 ran 100 of them and kept the last.

Four configurations, all trained on the identical 18,000 images:

    python part6_cnn_overfitting.py                        # A: baseline
    python part6_cnn_overfitting.py --augment              # B
    python part6_cnn_overfitting.py --regularize           # C
    python part6_cnn_overfitting.py --augment --regularize # D
    python part6_cnn_overfitting.py --compare              # all four, one figure

THE THREE-WAY SPLIT, and why it matters more than any of the above.

    train       18,000 images   weights are fitted on these
    validation   2,000 images   held out; drives the per-epoch curve
    test        10,000 images   CIFAR-10's separate test files

The validation set is carved out of CIFAR-10's *training* pool, never out of
the test files. That is deliberate and it is the most important line in this
script. You will run this file many times, flipping flags and comparing curves.
Every one of those decisions is made by looking at validation accuracy. If
validation came from the test set, you would be choosing your settings using
the test set -- and your final test number would be an optimistic fiction,
because you had effectively fitted to it by hand, one command at a time.

So the test set is loaded exactly once, at the end, after training is over and
every choice has already been made -- including which epoch's weights to keep,
which is decided on validation accuracy alone. Until that line runs, no model in this file
has ever seen a test file. That is the only thing that makes the final number
mean what it claims to mean.

Note also that train and validation come from the same pool but use *different
transforms*: augmentation is applied to training images only. Randomly cropping
your validation images would just make the yardstick noisy. And the "train
accuracy" we plot is measured on clean, un-augmented training images, so that
the gap compares like with like across configs.
Measured results, 30 epochs each on this exact split (your numbers will move a
little with hardware, but the ordering is stable):

    config                       epoch   train     val    test     gap
    A: baseline                     18   0.902   0.675   0.668   0.227
    B: + augmentation               29   0.772   0.731   0.737   0.041
    C: + dropout & weight decay     25   0.891   0.722   0.724   0.170
    D: both                         30   0.771   0.744   0.732   0.027

Each row is the epoch validation picked, so train/val/test all describe the
same weights. Read the gap column, not just the test column. Left to run all 30
epochs the baseline ends at train 0.976 / val 0.650 -- a 33-point gap, and the
same collapse part3 showed. Config D ends at 0.771 / 0.744, a gap of under 3.

What changes between them is not how well the model *can* fit -- every config
drives training accuracy up. What changes is how much of that fit survives
contact with new images. Augmentation is worth roughly twice what dropout and
weight decay are worth here, which is worth sitting with: the fix that tells
the model a true fact about the world (a mirrored dog is a dog) beats the fix
that merely handicaps it.

One honest caveat: B and D are still improving at epoch 30 (their best epochs
are 29 and 30, i.e. they never turned over). Regularisation slows training
down, which is the trade -- it takes longer to fit data it is no longer allowed
to memorise. Run them with --epochs 60 and both go higher. The baseline will
not; it peaked at 18 and has nowhere left to go.

What does NOT fix this, and is worth trying so you believe it: training longer
(that is what created the gap), or using a bigger model. Shrinking the model
does close the gap, but by dragging train accuracy down to meet test accuracy
-- a worse model that merely looks more honest.
"""

import argparse
import copy
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from part3_cnn_cifar10 import evaluate, get_device

# Anchored to this file, so data and checkpoints land in week3_cnn/ no matter
# which directory you launched python from.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data")

# 18,000 + 2,000 = the same 20,000-image budget part3 used, so the comparison
# against part3's numbers is honest. We are not fixing overfitting by quietly
# adding data -- every config below sees exactly these 18,000 photos.
N_TRAIN = 18000
N_VAL = 2000
N_PROBE = 2000   # training images re-measured cleanly, for the train curve
SPLIT_SEED = 0   # fixed: every config must get the identical split


def build_transform(augment):
    """Eval preprocessing, optionally with the standard CIFAR augmentations.

    RandomCrop(32, padding=4) pads the image with 4 black pixels on each side
    and cuts a random 32x32 window back out, so the object shifts by up to 4
    pixels in any direction. RandomHorizontalFlip mirrors it half the time.

    Both encode a fact about the world that the model has no way to know
    otherwise: a dog shifted three pixels left is still a dog, and so is a dog
    facing the other way. The model now sees a slightly different version of
    each photo every epoch, which makes memorising individual images much
    harder without making the actual task any easier.

    (Vertical flips are NOT included on purpose -- an upside-down car is not a
    normal photo of a car, so that augmentation would teach a falsehood.)
    """
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if not augment:
        return transforms.Compose([transforms.ToTensor(), normalize])
    # Crop and flip act on the PIL image, so they come before ToTensor.
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


def make_splits(augment):
    """Split CIFAR-10's training pool into train / validation / train-probe.

    Two dataset objects over the same files, with different transforms: the
    augmented one for training, the clean one for anything we measure. Subset
    then picks disjoint index ranges out of a single shuffled permutation, so
    no image is ever in both train and validation.

    The probe set is the first 2,000 *training* images with augmentation off.
    That is what the "train" curve is measured on. Measuring it on augmented
    batches instead would make an augmented run look artificially worse at
    memorising, and the whole point is to compare gaps between configs.
    """
    augmented = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True,
                                 transform=build_transform(augment))
    clean = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True,
                             transform=build_transform(False))

    order = torch.randperm(len(clean), generator=torch.Generator().manual_seed(SPLIT_SEED))
    train_idx = order[:N_TRAIN].tolist()
    val_idx = order[N_TRAIN:N_TRAIN + N_VAL].tolist()

    return (Subset(augmented, train_idx),
            Subset(clean, val_idx),
            Subset(clean, train_idx[:N_PROBE]))


def load_test():
    """CIFAR-10's separate test files. Called once, at the very end."""
    return datasets.CIFAR10(root=DATA_ROOT, train=False, download=True,
                            transform=build_transform(False))


def build_model(dropout=0.0):
    """part3's architecture exactly, with one optional Dropout layer.

    With dropout=0.0 this is byte-for-byte part3's model: 188,810 parameters.
    Nothing about the model's capacity changes between configs -- that is the
    control that makes the experiment worth running.

    Where the Dropout sits is the interesting part. Count the parameters:

        Conv(3 -> 32)         896     0.5%
        Conv(32 -> 64)     18,496     9.8%
        Conv(64 -> 64)     36,928    19.6%
        Linear(1024 -> 128) 131,200   69.5%   <-- here
        Linear(128 -> 10)    1,290     0.7%

    Nearly seventy percent of the model is one fully-connected layer. The conv
    stack -- the part part5 showed learning genuine edge detectors, the part
    that generalises -- is a minority of the weights. Memorisation capacity is
    concentrated in that head, so that is where the dropout goes: right after
    Flatten, before the 131k-parameter layer.

    Dropout zeroes half of its inputs at random on each training batch, which
    stops the head from building a fragile chain of co-adapted features that
    only works when all 1024 inputs are present. At eval time it is disabled
    automatically by model.eval(), which is why every measurement below calls
    it.
    """
    layers = [
        nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 10)]
    return nn.Sequential(*layers)


def train(model, train_loader, probe_loader, val_loader, epochs, weight_decay, lr=1e-3):
    """part3's training loop plus two measurements per epoch.

    The optimizer is AdamW rather than Adam. With weight_decay=0.0 the two are
    identical, so the baseline is unchanged; with weight_decay > 0 AdamW applies
    the decay correctly (plain Adam folds it into the gradient, where the
    adaptive step size then partly cancels it out).

    Weight decay is a steady pull of every weight toward zero. A weight only
    survives if the loss keeps pushing back on it, so weights that exist purely
    to nail one awkward training image get shrunk away.
    """
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"loss": [], "train": [], "val": [], "best_epoch": 1}
    best_val, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = loss_fn(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        history["loss"].append(epoch_loss / len(train_loader))
        # Two accuracies, same size, same transforms. The only difference is
        # whether the model was fitted on those images. Their gap IS the
        # overfitting, plotted directly.
        history["train"].append(evaluate(model, probe_loader))
        history["val"].append(evaluate(model, val_loader))

        # Early stopping, in its most useful form: rather than halting the run,
        # keep a copy of the best-validating weights and hand those back at the
        # end. The baseline's validation accuracy peaks around epoch 18 and then
        # *declines* -- training past that point actively destroys a model.
        # Keeping the snapshot costs 188k floats and makes the last 12 epochs
        # harmless instead of harmful.
        if history["val"][-1] > best_val:
            best_val = history["val"][-1]
            history["best_epoch"] = epoch + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        print(f"  epoch {epoch + 1:3d}/{epochs}  loss {history['loss'][-1]:.4f}"
              f"  train {history['train'][-1]:.3f}  val {history['val'][-1]:.3f}"
              f"  gap {history['train'][-1] - history['val'][-1]:+.3f}"
              f"{'  <- best' if history['best_epoch'] == epoch + 1 else ''}")

    # The model we keep is the one validation liked best, not the one that
    # happened to be in memory when the loop ran out. Note that this choice is
    # made entirely on validation data -- the test set is still unopened.
    model.load_state_dict(best_state)
    print(f"  restored epoch {history['best_epoch']} (val {best_val:.3f})")
    return history


CONFIGS = {
    "baseline": dict(augment=False, regularize=False, label="A: baseline (part3)"),
    "aug":      dict(augment=True,  regularize=False, label="B: + augmentation"),
    "reg":      dict(augment=False, regularize=True,  label="C: + dropout & weight decay"),
    "aug_reg":  dict(augment=True,  regularize=True,  label="D: both"),
}

DROPOUT = 0.5
WEIGHT_DECAY = 5e-4


def config_key(augment, regularize):
    return {(False, False): "baseline", (True, False): "aug",
            (False, True): "reg", (True, True): "aug_reg"}[(augment, regularize)]


def run_config(key, args, device):
    """Train one configuration end to end and return its history and model."""
    cfg = CONFIGS[key]
    print(f"\n=== {cfg['label']} "
          f"(augment={cfg['augment']}, regularize={cfg['regularize']}) ===")

    train_ds, val_ds, probe_ds = make_splits(cfg["augment"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)
    probe_loader = DataLoader(probe_ds, batch_size=512)

    # Reseed before building: every config starts from identical random weights
    # and sees identical batch ordering, so any difference in the curves comes
    # from the flags and nothing else.
    torch.manual_seed(0)
    model = build_model(dropout=DROPOUT if cfg["regularize"] else 0.0).to(device)
    if key == "baseline":
        n = sum(p.numel() for p in model.parameters())
        print(f"parameters: {n:,}  (identical to part3)")

    history = train(model, train_loader, probe_loader, val_loader,
                    epochs=args.epochs,
                    weight_decay=WEIGHT_DECAY if cfg["regularize"] else 0.0)

    if not args.no_save:
        path = os.path.join(HERE, f"part6_{key}.pt")
        torch.save(model.state_dict(), path)
        print(f"  saved weights to {path}")
    return history, model


def plot_gap(ax, history, title):
    """Train vs validation accuracy, with the gap between them shaded."""
    epochs = range(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], color="tab:blue", label="train (seen)")
    ax.plot(epochs, history["val"], color="tab:orange", label="validation (unseen)")
    ax.fill_between(epochs, history["val"], history["train"],
                    color="tab:red", alpha=0.15, label="overfitting gap")
    best = history["best_epoch"]
    ax.axvline(best, color="tab:green", linestyle="--", linewidth=1,
               label=f"best val (epoch {best})")
    gap = history["train"][best - 1] - history["val"][best - 1]
    ax.set_title(f"{title}\nbest val {history['val'][best - 1]:.3f} "
                 f"at epoch {best}, gap {gap:.3f}", fontsize=10)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")


def plot_single(history, label):
    """The headline figure: the misleading curve next to the honest one."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(range(1, len(history["loss"]) + 1), history["loss"],
                 color="tab:green")
    axes[0].set_title("What part3 plotted: training loss\n"
                      "(it goes down; it always goes down)", fontsize=10)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].grid(alpha=0.3)
    plot_gap(axes[1], history, "What actually matters: seen vs unseen")
    fig.suptitle(label)
    fig.tight_layout()


def plot_compare(results):
    """One panel per config, plus an overlay of the four validation curves."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, history) in zip(axes.ravel(), results.items()):
        plot_gap(ax, history, CONFIGS[key]["label"])
    fig.suptitle("Same architecture, same 18,000 images, same epochs -- "
                 "only the regularisation differs")
    fig.tight_layout()

    plt.figure(figsize=(7, 4.5))
    for key, history in results.items():
        plt.plot(range(1, len(history["val"]) + 1), history["val"],
                 marker="o", markersize=3, label=CONFIGS[key]["label"])
    plt.title("Validation accuracy: the only curve worth optimising")
    plt.xlabel("epoch")
    plt.ylabel("accuracy on unseen images")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--augment", action="store_true",
                        help="config B: random crop + horizontal flip on training images")
    parser.add_argument("--regularize", action="store_true",
                        help="config C: dropout 0.5 before the head + weight decay 5e-4")
    parser.add_argument("--compare", action="store_true",
                        help="train all four configs back to back and plot them together")
    parser.add_argument("--epochs", type=int, default=30,
                        help="epochs per config (default 30 -- the gap opens by epoch 12)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-save", action="store_true",
                        help="skip writing part6_<config>.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"using device: {device}")
    print(f"train {N_TRAIN} / validation {N_VAL} images, "
          f"both from CIFAR-10's training pool")
    print("the test files stay unopened until every model has finished training")

    keys = list(CONFIGS) if args.compare else [config_key(args.augment, args.regularize)]
    results = {}
    models = {}
    for key in keys:
        results[key], models[key] = run_config(key, args, device)

    # ---- the test set, opened once, after every decision has been made ----
    print("\nloading the held-out test files for the first time...")
    test_loader = DataLoader(load_test(), batch_size=512)

    # Every row describes the epoch validation selected, so train/val/test all
    # refer to the same set of weights.
    print(f"\n{'config':<32}{'epoch':>7}{'train':>8}{'val':>8}{'test':>8}{'gap':>8}")
    print("-" * 71)
    for key in keys:
        history = results[key]
        best = history["best_epoch"]
        train_acc, val_acc = history["train"][best - 1], history["val"][best - 1]
        test_acc = evaluate(models[key], test_loader)
        print(f"{CONFIGS[key]['label']:<32}{best:>7}"
              f"{train_acc:>8.3f}{val_acc:>8.3f}"
              f"{test_acc:>8.3f}{train_acc - val_acc:>8.3f}")

    # Validation tracks test closely because both are unseen -- that is the
    # whole reason a validation set works as a stand-in for the real thing.
    print("\nvalidation and test track each other because both are unseen data.")
    print("that is what lets you tune on validation and still trust the test number.")

    if args.compare:
        plot_compare(results)
    else:
        plot_single(results[keys[0]], CONFIGS[keys[0]]["label"])

    plt.show()


if __name__ == "__main__":
    main()
