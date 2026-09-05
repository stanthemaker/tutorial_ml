"""Week 4 - Part 1: ResNet -- the same CNN, but the layers learn a correction.

Week 3 ended with a stack of convolutions: Conv -> ReLU -> Pool, three times,
then a fully-connected head. Part 6 pushed that model as far as it goes on
CIFAR-10 -- augmentation, dropout, weight decay, best-checkpoint selection --
and landed around 73% test accuracy. The obvious next move is to go deeper:
more conv layers, a bigger receptive field, richer features.

That move does not work with the Week 3 block. Past roughly eight or ten
stacked convolutions a plain CNN gets *worse*, and not because it overfits --
its training accuracy drops too. That is the "degradation problem" the ResNet
paper opened with, and it is the reason this file exists.

THE ONE NEW IDEA: a layer that starts as a no-op.

    plain block:     out = F(x)
    residual block:  out = F(x) + x

`F` is the same pair of convolutions either way. The only change is the `+ x`,
called a skip (or identity, or shortcut) connection, and it changes what the
block has to learn. A plain block must reproduce everything useful in `x` and
then improve on it -- if it cannot, it destroys information. A residual block
starts from `x` for free and only has to learn the *difference* it wants to
make. Learning nothing (F = 0) leaves the input untouched, so adding depth can
no longer make the model worse by accident.

The backward pass tells the same story. Differentiating `out = F(x) + x` gives
`dout/dx = dF/dx + 1`. That `+ 1` is a gradient path straight from the loss to
every earlier layer that does not pass through any weight matrix, so the
gradient reaching layer 3 of a 14-layer net is not the product of eleven small
numbers. Depth stops being something the optimizer has to survive.

WHAT ELSE IS NEW HERE (both are standard ResNet furniture, not big ideas):

  * BatchNorm2d after every convolution. It renormalises each channel over the
    batch, which keeps activations from drifting as depth grows and lets us use
    a healthy learning rate. Like Dropout, it behaves differently in train and
    eval mode -- another reason every measurement below calls model.eval().
  * Global average pooling instead of Flatten -> Linear(1024, 128). Part 6
    measured that 70% of its parameters sat in that one head, and that the head
    was where the memorisation lived. Averaging each final feature map down to
    a single number gives a 64-value vector and a 650-parameter classifier.

THE ARCHITECTURE (He et al.'s CIFAR ResNet, with n=2 blocks per stage):

    Conv3x3(3 -> 16) + BN + ReLU                        16 x 32 x 32
    stage 1:  2 residual blocks, 16 channels            16 x 32 x 32
    stage 2:  2 residual blocks, 32 channels, stride 2  32 x 16 x 16
    stage 3:  2 residual blocks, 64 channels, stride 2  64 x  8 x  8
    global average pool                                 64
    Linear(64 -> 10)                                    10

Depth counts as 6n + 2 = 14 weighted layers, against Week 3's 5. Parameter
count lands at 175,258, just *under* part 6's 188,810 -- so this is not a bigger
model, it is a differently wired one of the same size. That is the comparison
worth making: nearly three times the depth for slightly fewer weights.

EVERYTHING ELSE IS DELIBERATELY PART 6'S SETUP, unchanged:

    train       18,000 images   the same split, the same seed
    validation   2,000 images   drives the per-epoch curve and picks the epoch
    test        10,000 images   loaded once, at the very end

Augmentation (random crop + horizontal flip) is on by default here rather than
behind a flag, because part 6 already established what it is worth. The test
files stay unopened until training is over and the checkpoint has been chosen
on validation accuracy alone.

    python part1_resnet_cifar10.py                 # part 6's budget, 30 epochs
    python part1_resnet_cifar10.py --sgd           # the paper's optimizer
    python part1_resnet_cifar10.py --full-data     # 45,000 images, 60 epochs
    python part1_resnet_cifar10.py --blocks 3      # ResNet-20 instead of -14
    python part1_resnet_cifar10.py --no-augment    # see the gap re-open
    python part1_resnet_cifar10.py --load part1_resnet.pt   # skip training

MEASURED RESULTS (your numbers will move a little with hardware, but the
ordering is stable):

    run                          images  epochs  optimizer      train    val   test
    Week 3 part 6 D (aug+reg)    18,000      30  AdamW const    0.771  0.744  0.732
    default                      18,000      30  AdamW const    0.849  0.787  0.784
    --sgd                        18,000      30  SGD + cosine   0.934  0.846  0.839
    --full-data                  45,000      60  SGD + cosine   0.975  0.904  0.904

Row 2 is the one that answers Week 3: same images, same epochs, same optimizer,
+5.2 points of test accuracy for 13,552 *fewer* parameters than part 6's CNN.
The gain is not capacity -- it is being able to use depth at all.

Rows 3 and 4 answer the question that immediately follows, which is "the ResNet
paper reports 91.25%, why is this 78%?" Nothing about that gap is architectural.
It decomposes into exactly two things, and the flags let you buy them back one
at a time:

  +5.5 points from the OPTIMIZER AND SCHEDULE alone (--sgd). Identical images,
       identical epochs, identical runtime -- SGD at lr 0.1 annealed to zero on
       a cosine curve instead of AdamW held at 1e-3 forever.

  +6.5 points from the DATA BUDGET and a longer schedule (--full-data). Part 6
       capped training at 18,000 images so its four configs were comparable,
       and this file inherited that cap to stay comparable with part 6. CIFAR-10
       actually ships 50,000.

Together: 0.784 -> 0.904, on the same 175,258 parameters.

The tell was visible in the default run's own log all along. Its training
accuracy peaked at 0.849 -- nowhere near the ~0.98 a converged ResNet reaches.
That model was not overfitting, it was UNDERFITTING: still climbing when the
epochs ran out, still taking 1e-3-sized steps when it should have been taking
1e-5-sized ones. Part 6 trained you to read a train/val gap as the danger sign.
Here the danger sign is the opposite -- a training accuracy that never got high
enough to *have* a gap worth discussing.

Two smaller things separate --full-data from the paper's 91.25%: it uses
ResNet-14 rather than ResNet-20 (try --blocks 3), and 60 epochs rather than
164. Neither is worth the runtime on a laptop for the point being made.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Anchored to this file, so data and checkpoints land in week4_resnet/ no
# matter which directory you launched python from.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data")
WEIGHTS = os.path.join(HERE, "part1_resnet.pt")

CLASSES = ["plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Identical to part 6, on purpose: same 20,000-image budget, same seed, so the
# two rows of the results table describe models trained on the same photos.
N_TRAIN = 18000
N_VAL = 2000
N_PROBE = 2000   # training images re-measured cleanly, for the train curve
SPLIT_SEED = 0


def get_device():
    """Pick the fastest available device: NVIDIA GPU, then Mac GPU, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# data -- part 6's three-way split, unchanged
# --------------------------------------------------------------------------

def build_transform(augment):
    """Eval preprocessing, optionally with part 6's two augmentations."""
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if not augment:
        return transforms.Compose([transforms.ToTensor(), normalize])
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


def make_splits(augment, n_train=N_TRAIN, n_val=N_VAL):
    """Split CIFAR-10's training pool into train / validation / train-probe.

    Two dataset objects over the same files with different transforms: the
    augmented one for training, the clean one for anything we measure. The
    probe is the first 2,000 training images with augmentation off, so the
    "train" curve and the "validation" curve differ only in whether the model
    was fitted on those images.

    n_train defaults to part 6's 18,000. --full-data raises it to 45,000 with
    a 5,000-image validation set -- the split the ResNet paper actually used.
    Both draw from the same shuffled permutation, so the 18,000-image run is a
    strict subset of the 45,000-image one.
    """
    augmented = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True,
                                 transform=build_transform(augment))
    clean = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True,
                             transform=build_transform(False))

    order = torch.randperm(len(clean),
                           generator=torch.Generator().manual_seed(SPLIT_SEED))
    train_idx = order[:n_train].tolist()
    val_idx = order[n_train:n_train + n_val].tolist()

    return (Subset(augmented, train_idx),
            Subset(clean, val_idx),
            Subset(clean, train_idx[:N_PROBE]))


def load_test():
    """CIFAR-10's separate test files. Called once, at the very end."""
    return datasets.CIFAR10(root=DATA_ROOT, train=False, download=True,
                            transform=build_transform(False))


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Two 3x3 convolutions, plus the input added back on: out = F(x) + x.

    This is the entire idea of the week, and it is four lines of forward().
    Note what is NOT here: no extra parameters for the skip (except in the
    downsampling case below), no gating, no attention. Just an addition.

    Ordering matters and is easy to get wrong. The second BatchNorm comes
    *before* the addition and the final ReLU comes *after* it:

        conv -> bn -> relu -> conv -> bn -> (+ x) -> relu

    Putting the ReLU before the addition would clip the block's output to
    non-negative values, so `F(x)` could only ever add to the signal, never
    subtract from it -- half the corrections it might want to make would be
    unavailable.

    THE SHAPE PROBLEM. `F(x) + x` only type-checks if `F(x)` and `x` have the
    same shape, and at the start of each stage they do not: we halve the
    spatial size (stride=2) and double the channels. The standard fix, and the
    one used here, is a 1x1 convolution with the same stride on the shortcut,
    which is the cheapest possible way to reshape `x` -- 512 parameters where
    the block's own convolutions cost ~14,000. When shapes already match, the
    shortcut is nn.Identity(): literally nothing happens to x.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # bias=False on every conv above: the BatchNorm that follows has its
        # own shift parameter, so a conv bias would be redundant (and promptly
        # subtracted away by the normalisation).
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # <- the skip connection
        return F.relu(out)


class ResNet(nn.Module):
    """He et al.'s CIFAR ResNet: a stem, three stages, and a 650-param head.

    `n_blocks` is the paper's n. Depth is 6n + 2 weighted layers, so n=2 gives
    ResNet-14 (the default here, 175,258 parameters -- the same budget as Week
    3's five-layer CNN) and n=3 gives the paper's ResNet-20.

    Each stage doubles the channels and halves the resolution, which keeps the
    work per stage roughly constant: a quarter as many pixels, twice as many
    channels, four times the parameters per convolution. Only the first block
    of a stage does the halving; the rest keep the shape and are pure identity
    shortcuts.
    """

    def __init__(self, n_blocks=2, num_classes=10, widths=(16, 32, 64)):
        super().__init__()
        w1, w2, w3 = widths
        self.stem = nn.Sequential(
            nn.Conv2d(3, w1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.ReLU(),
        )
        self.stage1 = self._make_stage(w1, w1, n_blocks, stride=1)
        self.stage2 = self._make_stage(w1, w2, n_blocks, stride=2)
        self.stage3 = self._make_stage(w2, w3, n_blocks, stride=2)
        # Global average pooling: each 8x8 feature map collapses to its mean,
        # turning 64 x 8 x 8 into 64 numbers. Part 6's Flatten fed a 1024-wide
        # vector into a 131k-parameter Linear; this feeds 64 into a 650-
        # parameter one, and the classifier stops being where the model hides
        # its memorised training images.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w3, num_classes)

    @staticmethod
    def _make_stage(in_channels, out_channels, n_blocks, stride):
        """n_blocks residual blocks; only the first one changes shape."""
        blocks = [ResidualBlock(in_channels, out_channels, stride=stride)]
        blocks += [ResidualBlock(out_channels, out_channels, stride=1)
                   for _ in range(n_blocks - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# --------------------------------------------------------------------------
# training -- part 6's loop, including the best-checkpoint rule
# --------------------------------------------------------------------------

def evaluate(model, loader):
    device = next(model.parameters()).device
    model.eval()   # switches BatchNorm to its running statistics
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train(model, train_loader, probe_loader, val_loader, epochs,
          sgd=False, lr=None, weight_decay=5e-4):
    """The Week 1 Part 3 loop, with two measurements and a snapshot per epoch.

    Two recipes, because the difference between them is worth five points:

    AdamW, lr 1e-3, constant (the default) is part 6's optimizer, kept so that
    the architecture is the only thing that changes between the two rows of
    the results table.

    --sgd switches to what the ResNet paper actually used: SGD with Nesterov
    momentum 0.9 and a *much* larger learning rate, 0.1, annealed to zero on a
    cosine curve. Two things are going on there. A big learning rate early
    keeps the optimizer bouncing across the loss surface instead of diving
    into the first sharp minimum it finds; annealing it to nearly zero at the
    end then lets the model settle precisely into a wide one. A constant lr
    does neither, which is why the default run's validation curve wobbles for
    its last ten epochs instead of converging -- it is still taking 1e-3-sized
    steps when it should be taking 1e-5-sized ones.

    BatchNorm is what makes lr=0.1 survivable at all; try it on Week 3's plain
    CNN and it diverges.
    """
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss()
    if sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr or 0.1,
                                    momentum=0.9, nesterov=True,
                                    weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr or 1e-3,
                                      weight_decay=weight_decay)
    # T_max = epochs, so lr reaches ~0 exactly as training ends. With AdamW at
    # a constant lr this scheduler is not used at all.
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if sgd else None)

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

        if scheduler is not None:
            scheduler.step()

        history["loss"].append(epoch_loss / len(train_loader))
        history["train"].append(evaluate(model, probe_loader))
        history["val"].append(evaluate(model, val_loader))

        # Keep the best-validating weights rather than whatever happens to be
        # in memory when the loop ends. Decided on validation only; the test
        # files are still unopened at this point.
        if history["val"][-1] > best_val:
            best_val = history["val"][-1]
            history["best_epoch"] = epoch + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        print(f"  epoch {epoch + 1:3d}/{epochs}  loss {history['loss'][-1]:.4f}"
              f"  train {history['train'][-1]:.3f}  val {history['val'][-1]:.3f}"
              f"  gap {history['train'][-1] - history['val'][-1]:+.3f}"
              f"{'  <- best' if history['best_epoch'] == epoch + 1 else ''}")

    model.load_state_dict(best_state)
    print(f"  restored epoch {history['best_epoch']} (val {best_val:.3f})")
    return history


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def plot_history(history, title):
    """Training loss next to the seen-vs-unseen curves from part 6."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    epochs = range(1, len(history["loss"]) + 1)

    axes[0].plot(epochs, history["loss"], color="tab:green")
    axes[0].set_title("training loss", fontsize=10)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["train"], color="tab:blue", label="train (seen)")
    ax.plot(epochs, history["val"], color="tab:orange", label="validation (unseen)")
    ax.fill_between(epochs, history["val"], history["train"],
                    color="tab:red", alpha=0.15, label="overfitting gap")
    best = history["best_epoch"]
    ax.axvline(best, color="tab:green", linestyle="--", linewidth=1,
               label=f"best val (epoch {best})")
    gap = history["train"][best - 1] - history["val"][best - 1]
    ax.set_title(f"seen vs unseen\nbest val {history['val'][best - 1]:.3f} "
                 f"at epoch {best}, gap {gap:.3f}", fontsize=10)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(title)
    fig.tight_layout()


def plot_confusion(model, test_loader):
    """Which classes it still confuses -- the same figure as Week 3 part 3."""
    device = next(model.parameters()).device
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            preds.append(model(images.to(device)).argmax(dim=1).cpu())
            trues.append(labels)
    cm = confusion_matrix(torch.cat(trues).numpy(), torch.cat(preds).numpy(),
                          labels=range(10))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    disp.plot(cmap="Blues", colorbar=False, xticks_rotation=45)
    disp.ax_.set_title("ResNet confusion matrix (rows = true, cols = predicted)")


# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--blocks", type=int, default=2,
                        help="residual blocks per stage (n); depth is 6n+2, "
                             "so 2 -> ResNet-14 (default), 3 -> ResNet-20")
    parser.add_argument("--epochs", type=int, default=None,
                        help="default 30, or 60 with --full-data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sgd", action="store_true",
                        help="the paper's recipe: SGD 0.1 + momentum + cosine "
                             "decay, instead of part 6's constant-lr AdamW")
    parser.add_argument("--full-data", action="store_true",
                        help="train on 45,000 images (val 5,000) instead of "
                             "part 6's 18,000; implies --sgd and 60 epochs")
    parser.add_argument("--no-augment", action="store_true",
                        help="drop random crop + flip, to watch the gap re-open")
    parser.add_argument("--load", metavar="PATH", default=None,
                        help="path to a saved .pt state_dict; skips training")
    parser.add_argument("--save", metavar="PATH", default=WEIGHTS,
                        help="where to write the trained weights")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    augment = not args.no_augment
    depth = 6 * args.blocks + 2

    # --full-data is the "take the handicap off" switch: it is pointless to
    # pay for 2.5x the images and then hobble them with the constant-lr
    # recipe, so it turns --sgd on and doubles the schedule.
    sgd = args.sgd or args.full_data
    n_train, n_val = (45000, 5000) if args.full_data else (N_TRAIN, N_VAL)
    epochs = args.epochs if args.epochs is not None else (60 if args.full_data else 30)

    print(f"using device: {device}")
    torch.manual_seed(0)
    model = ResNet(n_blocks=args.blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet-{depth}: {n_params:,} parameters "
          f"({depth} weighted layers, vs part 6's 188,810 in 5)")

    history = None
    if args.load is not None:
        model.load_state_dict(torch.load(args.load, map_location=device))
        print(f"loaded weights from {args.load} (skipping training)")
    else:
        print(f"train {n_train} / validation {n_val} images from CIFAR-10's "
              f"training pool, augment={augment}, "
              f"optimizer={'SGD 0.1 + cosine' if sgd else 'AdamW 1e-3 constant'}, "
              f"epochs={epochs}")
        train_ds, val_ds, probe_ds = make_splits(augment, n_train, n_val)
        history = train(
            model,
            DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
            DataLoader(probe_ds, batch_size=512),
            DataLoader(val_ds, batch_size=512),
            epochs=epochs,
            sgd=sgd,
        )
        if not args.no_save:
            torch.save(model.state_dict(), args.save)
            print(f"  saved weights to {args.save}")

    # ---- the test set, opened once, after every decision has been made ----
    print("\nloading the held-out test files for the first time...")
    test_loader = DataLoader(load_test(), batch_size=512)
    test_acc = evaluate(model, test_loader)

    if history is not None:
        best = history["best_epoch"]
        train_acc, val_acc = history["train"][best - 1], history["val"][best - 1]
        print(f"\n{'model':<24}{'params':>10}{'epoch':>7}"
              f"{'train':>8}{'val':>8}{'test':>8}{'gap':>8}")
        print("-" * 73)
        print(f"{'week3 part6 D':<24}{188810:>10,}{30:>7}"
              f"{0.771:>8.3f}{0.744:>8.3f}{0.732:>8.3f}{0.027:>8.3f}")
        row = f"ResNet-{depth}" + (" full" if args.full_data else (" sgd" if sgd else ""))
        print(f"{row:<24}{n_params:>10,}{best:>7}"
              f"{train_acc:>8.3f}{val_acc:>8.3f}{test_acc:>8.3f}"
              f"{train_acc - val_acc:>8.3f}")
    else:
        print(f"Test accuracy: {test_acc:.2%}  (chance is 10%)")

    if history is not None:
        plot_history(history, f"ResNet-{depth} on CIFAR-10 ({n_params:,} params, "
                              f"{n_train:,} images, "
                              f"{'SGD + cosine' if sgd else 'AdamW constant lr'})")
    plot_confusion(model, test_loader)
    plt.show()


if __name__ == "__main__":
    main()
