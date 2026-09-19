"""Week 3 - Part 7: From 73% to 83% --

Three rungs raise the ceiling, and the method matters more than the rungs:

    ONE CHANGE PER RUNG.  Each rung differs from the one above it in exactly
                          one respect, so each gain has exactly one cause.
    ONE FIXED YARDSTICK.  Every run in this file -- every rung, every sweep,
                          every control -- is scored on the same 5,000
                          validation images.
    MEASURE THE CONFOUND. When a change drags a second change along with it,
                          run the control that separates them.

    python part7_improving_cnn.py --ladder       # rungs 0 -> 2b
    python part7_improving_cnn.py --data-curve   # rung 1 in depth
    python part7_improving_cnn.py --lr-sweep     # rung 2a in depth

    python part7_improving_cnn.py                               # rung 0 alone
    python part7_improving_cnn.py --full-data                   # rung 1 alone
    python part7_improving_cnn.py --full-data --bn              # rung 2a alone
    python part7_improving_cnn.py --full-data --bn --onecycle   # rung 2b alone

MEASURED RESULTS (M1 Pro, 30 epochs, the ladder takes about 15 minutes):

    rung                          images  epoch   train     val    test     gap
    0: part6 config D (18k)       18,000     28   0.765   0.734   0.732   0.031
    1: + full data (45k)          45,000     28   0.812   0.790   0.782   0.022
    2a: + BatchNorm               45,000     30   0.835   0.811   0.802   0.024
    2b: + OneCycle LR             45,000     28   0.867   0.832   0.826   0.034

"""

import argparse
import copy
import json
import math
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import trackio
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from part3_cnn_cifar10 import evaluate, get_device
from part6_cnn_overfitting import (
    CHECKPOINTS,
    DATA_ROOT,
    DROPOUT,
    RESULTS,
    WEIGHT_DECAY,
    build_transform,
    load_test,
    plot_gap,
)

TRACKIO_PROJECT = "week3-cnn"  # parts 7 and 8 share one project

# CIFAR-10's training pool is 50,000 images. The last 5,000 of a fixed shuffle
# are validation for EVERY run in this file; training takes the first n_train.
# So the 18,000-image run and the 45,000-image run are judged on the identical
# yardstick, and the 18,000 images are a subset of the 45,000 -- stage 1 adds
# data, it does not swap it. (With SPLIT_SEED 0 those first 18,000 are exactly
# the images part6 trained on.)
N_POOL = 50000
N_VAL = 5000
N_FULL = N_POOL - N_VAL  # 45,000
N_PART6 = 18000
N_PROBE = 2000
SPLIT_SEED = 0

BASE_LR = 1e-3  # part6's constant learning rate
ONECYCLE_PEAK_LR = 3e-3  # chosen from --lr-sweep, see the docstring


# --------------------------------------------------------------------------- data


def make_splits(n_train):
    """Train (augmented) / validation / train-probe, carved from one shuffle.

    Same mechanics as part6: two dataset objects over the same files, one with
    augmentation for training and one clean for everything we measure. The only
    difference is that validation now sits at the *end* of the shuffle and never
    moves, whatever n_train is.
    """
    augmented = datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=build_transform(True)
    )
    clean = datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=build_transform(False)
    )

    order = torch.randperm(N_POOL, generator=torch.Generator().manual_seed(SPLIT_SEED))
    train_idx = order[:n_train].tolist()
    val_idx = order[N_FULL:].tolist()

    return (
        Subset(augmented, train_idx),
        Subset(clean, val_idx),
        Subset(clean, train_idx[:N_PROBE]),
    )


# -------------------------------------------------------------------------- model


def build_model(bn=False, dropout=DROPOUT):
    """part6's config-D network, with optional BatchNorm after every conv.

    With bn=False this is exactly part6's build_model(dropout=0.5).

    With bn=True each block becomes Conv -> BatchNorm -> ReLU -> Pool. BatchNorm
    takes every channel, subtracts its mean over the batch (and over all spatial
    positions), divides by its standard deviation, then applies a learned scale
    gamma and shift beta:

        y = gamma * (x - mean_batch) / std_batch + beta

    Two consequences worth spelling out:

      * The conv's bias becomes dead weight. Whatever constant the conv adds,
        the "- mean_batch" subtracts straight back out, and beta does the job
        instead. So bias=False -- not a trick, just not storing a parameter that
        cannot affect the output.

      * BN costs almost nothing: 2 parameters per channel (gamma, beta), minus
        the conv biases we dropped. 188,810 -> 188,970 parameters. Whatever it
        buys, it does not buy with capacity.
    """

    def block(c_in, c_out):
        layers = [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm2d(c_out))
        return layers + [nn.ReLU(), nn.MaxPool2d(2)]

    layers = block(3, 32) + block(32, 64) + block(64, 64) + [nn.Flatten()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 10)]
    return nn.Sequential(*layers)


class ActivationProbe:
    """Records what goes INTO each conv block's ReLU: the pre-activations.

    That is the signal BatchNorm is supposed to control. Without BN it is the
    raw conv output, whose scale depends on every weight upstream; with BN it
    is the normalised-then-rescaled version. Logging its std per layer, per
    epoch, lets you watch whether the scale each layer receives stays put or
    drifts as training moves the weights under it.
    """

    def __init__(self, model):
        self.stats = {}
        # The ReLUs that directly follow a conv (or its BatchNorm), in order --
        # not the one inside the fully-connected head.
        modules = list(model.modules())
        relus = [
            m
            for prev, m in zip(modules, modules[1:])
            if isinstance(m, nn.ReLU) and isinstance(prev, (nn.Conv2d, nn.BatchNorm2d))
        ]
        for i, relu in enumerate(relus, start=1):
            relu.register_forward_hook(self._hook(f"conv{i}"))
        self.enabled = False

    def _hook(self, name):
        def hook(module, inputs, output):
            if self.enabled:
                x = inputs[0]
                self.stats[name] = (x.mean().item(), x.std().item())

        return hook

    def measure(self, model, images):
        """One forward pass in eval mode on a fixed batch; returns log-ready dict."""
        model.eval()
        self.enabled = True
        with torch.no_grad():
            model(images)
        self.enabled = False
        out = {}
        for name, (mean, std) in self.stats.items():
            out[f"act_std/{name}"] = std
            out[f"act_mean/{name}"] = mean
        return out


def evaluate_loss(model, loader, loss_fn):
    """One eval-mode pass; returns (accuracy, mean loss).

    Accuracy alone hides the moment overfitting starts. A model that is losing
    its grip gets MORE CONFIDENT about its wrong answers before it gets more of
    them wrong, so validation loss turns upward while validation accuracy is
    still flat or creeping up. Both numbers come from the pass we already make
    each epoch, so the second one is free.
    """
    device = next(model.parameters()).device
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += loss_fn(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total, loss_sum / total


def total_norm(tensors):
    """L2 norm of the given tensors laid end to end -- one number for the lot."""
    return torch.sqrt(sum((t.detach() ** 2).sum() for t in tensors)).item()


# ----------------------------------------------------------------------- training


def train(
    model,
    train_loader,
    probe_loader,
    val_loader,
    *,
    epochs,
    lr,
    onecycle,
    run_name,
    group,
    config,
    project=TRACKIO_PROJECT,
    log_every=20,
):
    """part6's loop, plus an optional per-batch LR schedule and trackio logging.

    Everything is logged against the global batch count (trackio's `step`),
    and every row also carries a fractional `epoch`. In the dashboard, switch
    the x-axis to `epoch` when comparing runs with different dataset sizes --
    a 45k-image epoch is 2.5x as many steps as an 18k-image one.
    """
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    scheduler = None
    if onecycle:
        # OneCycle: warm up from lr/25 to `lr` over the first 25% of training,
        # then cosine-anneal down to lr/(25*1e4) -- effectively zero -- by the
        # last batch. It is stepped every BATCH, not every epoch, so the curve
        # is smooth.
        #
        # cycle_momentum=False: by default OneCycle also cycles Adam's beta1
        # opposite to the LR. That is a second change riding along with the
        # first; we turn it off so this stage measures the schedule and nothing
        # else.
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.25,
            cycle_momentum=False,
        )

    probe = ActivationProbe(model)
    probe_images = next(iter(probe_loader))[0][:512].to(device)

    trackio.init(project=project, name=run_name, group=group, config=config)

    history = {
        "loss": [],
        "train": [],
        "val": [],
        "lr": [],
        "best_epoch": 1,
        "train_loss": [],
        "val_loss": [],
        "weight_norm": [],
        "diverged": False,
    }
    start = time.time()
    best_val, best_state = -1.0, None
    step, steps_per_epoch = 0, len(train_loader)
    for epoch in range(epochs):
        model.train()
        epoch_loss, window_loss, window_n = 0.0, 0.0, 0
        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            loss = loss_fn(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            # The size of the whole gradient, in one number. It is the clearest
            # warning light in this file: without BatchNorm at a high learning
            # rate it spikes and then collapses towards zero as the layers die.
            grad_norm = (
                total_norm(p.grad for p in model.parameters() if p.grad is not None)
                if (step + 1) % log_every == 0
                else None
            )
            optimizer.step()
            lr_now = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()
            step += 1

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                # A learning rate too large for the network sends the loss to
                # inf/NaN, and there is no coming back from NaN weights. Stop
                # here rather than burn the remaining epochs.
                history["diverged"] = True
                break
            epoch_loss += loss_value
            window_loss += loss_value
            window_n += 1

            row = {}
            if step % log_every == 0:
                row.update(
                    {
                        "epoch": step / steps_per_epoch,
                        "batch_loss": window_loss / window_n,
                        "lr": lr_now,
                        "grad_norm": grad_norm,
                    }
                )
                window_loss, window_n = 0.0, 0
            if batch == steps_per_epoch - 1:
                history["lr"].append(lr_now)
            if row:
                trackio.log(row, step=step)

        if history["diverged"]:
            print(f"  epoch {epoch + 1:3d}/{epochs}  loss went non-finite -- diverged")
            trackio.log({"epoch": step / steps_per_epoch, "diverged": 1}, step=step)
            break

        train_acc, train_loss = evaluate_loss(model, probe_loader, loss_fn)
        val_acc, val_loss = evaluate_loss(model, val_loader, loss_fn)
        history["loss"].append(epoch_loss / steps_per_epoch)
        history["train"].append(train_acc)
        history["val"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        # Total weight size. Weight decay is supposed to hold this down; if the
        # curve is flat, the decay you set is not actually doing anything.
        history["weight_norm"].append(total_norm(model.parameters()))

        row = {
            "epoch": epoch + 1,
            "epoch_loss": history["loss"][-1],
            # train_loss and val_loss are measured the same way: eval mode,
            # no augmentation, no dropout. Only then is the pair comparable.
            "train_loss": train_loss,
            "val_loss": val_loss,
            "loss_gap": val_loss - train_loss,
            "weight_norm": history["weight_norm"][-1],
            "train_acc": train_acc,
            "val_acc": val_acc,
            "gap": train_acc - val_acc,
        }
        row.update(probe.measure(model, probe_images))
        trackio.log(row, step=step)

        if history["val"][-1] > best_val:
            best_val = history["val"][-1]
            history["best_epoch"] = epoch + 1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        print(
            f"  epoch {epoch + 1:3d}/{epochs}  loss {history['loss'][-1]:.4f}"
            f"  lr {history['lr'][-1]:.2e}"
            f"  train {history['train'][-1]:.3f}  val {history['val'][-1]:.3f}"
            f"  gap {history['train'][-1] - history['val'][-1]:+.3f}"
            f"{'  <- best' if history['best_epoch'] == epoch + 1 else ''}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  restored epoch {history['best_epoch']} (val {best_val:.3f})")
    else:
        history["val"], history["train"] = [0.1], [0.1]  # never finished an epoch

    history["minutes"] = (time.time() - start) / 60
    # Summary values: one row per run in trackio's run table, so the table
    # sorts into a leaderboard. Everything in `config` is a column there too.
    best = history["best_epoch"] - 1
    trackio.log(
        {
            "best_val_acc": max(history["val"]),
            "best_epoch": history["best_epoch"],
            "best_train_acc": history["train"][best],
            "final_gap": history["train"][best] - history["val"][best],
            "params": config["params"],
            "train_minutes": history["minutes"],
        },
        step=step,
    )
    trackio.finish()
    return history


# ---------------------------------------------------------------------- run specs


def spec(name, label, n_train=N_PART6, bn=False, onecycle=False, lr=BASE_LR, epochs=30):
    return dict(
        name=name,
        label=label,
        n_train=n_train,
        bn=bn,
        onecycle=onecycle,
        lr=lr,
        epochs=epochs,
    )


# The ladder. Each rung changes exactly one thing relative to the rung above it.
LADDER = [
    spec("0_config_d", "0: part6 config D (18k)"),
    spec("1_full_data", "1: + full data (45k)", n_train=N_FULL),
    spec("2a_batchnorm", "2a: + BatchNorm", n_train=N_FULL, bn=True),
    spec(
        "2b_onecycle",
        "2b: + OneCycle LR",
        n_train=N_FULL,
        bn=True,
        onecycle=True,
        lr=ONECYCLE_PEAK_LR,
    ),
]


def run(
    s,
    group,
    device,
    save=False,
    model_fn=None,
    project=TRACKIO_PROJECT,
    prefix="part7",
    load=False,
):
    """Train one spec end to end. Returns (history, model).

    model_fn(s) builds the network; the default is this file's build_model.
    part8 passes its own, and its own trackio project and checkpoint prefix.

    load=True is --eval: the same (history, model) pair, read back from the
    checkpoint and results/ JSON a previous training run wrote, so every table
    and figure downstream works unchanged without retraining anything.
    """
    print(f"\n=== {s['label']} ===")
    if load:
        return load_run(s, device, model_fn=model_fn, prefix=prefix)
    print(
        f"    images {s['n_train']:,}  bn={s['bn']}  onecycle={s['onecycle']}"
        f"  lr={s['lr']:g}  epochs={s['epochs']}"
    )

    train_ds, val_ds, probe_ds = make_splits(s["n_train"])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)
    probe_loader = DataLoader(probe_ds, batch_size=512)

    torch.manual_seed(0)
    model = (model_fn(s) if model_fn else build_model(bn=s["bn"])).to(device)
    config = {k: v for k, v in s.items() if k not in ("name", "label")}
    config["params"] = sum(p.numel() for p in model.parameters())

    history = train(
        model,
        train_loader,
        probe_loader,
        val_loader,
        epochs=s["epochs"],
        lr=s["lr"],
        onecycle=s["onecycle"],
        run_name=f"{group}/{s['name']}",
        group=group,
        config=config,
        project=project,
    )

    save_history(history, s, f"{prefix}_{s['name']}")
    if save and not history["diverged"]:
        os.makedirs(CHECKPOINTS, exist_ok=True)
        path = os.path.join(CHECKPOINTS, f"{prefix}_{s['name']}.pt")
        torch.save(model.state_dict(), path)
        print(f"  saved weights to {path}")
    return history, model


def load_run(s, device, model_fn=None, prefix="part7"):
    """--eval: one finished run, rebuilt from disk instead of retrained.

    Weights come from checkpoints/<key>.pt, the per-epoch curve from
    results/<key>.json. The checkpoint holds the best-validation epoch (train()
    restores it before run() saves), which is the epoch the history's
    best_epoch points at -- so the table rows line up exactly.
    """
    key = f"{prefix}_{s['name']}"
    path = os.path.join(CHECKPOINTS, f"{key}.pt")
    for needed in (path, os.path.join(RESULTS, f"{key}.json")):
        if not os.path.exists(needed):
            raise SystemExit(
                f"{needed} not found -- run the same command without --eval "
                "once to train and save it"
            )
    model = (model_fn(s) if model_fn else build_model(bn=s["bn"])).to(device)
    # map_location: a checkpoint saved on a GPU box still loads on a laptop.
    model.load_state_dict(torch.load(path, map_location=device))
    _, history = load_history(key)
    print(f"  loaded {path} and its history (skipping training)")
    return history, model


def save_history(history, s, key):
    """Write the run's per-epoch numbers to results/<key>.json.

    The checkpoint holds the weights; this holds the CURVE. Without it, any
    figure that compares runs means retraining every one of them, which is an
    hour of GPU time to move a legend. A few kB of JSON instead, and it is
    small enough to commit, so the numbers quoted in these docstrings have a
    source you can check.
    """
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{key}.json")
    with open(path, "w") as f:
        json.dump({"spec": s, **history}, f, indent=1)
    print(f"  saved history to {path}")


def load_history(key):
    """Read back one results/<key>.json. Returns (spec, history)."""
    with open(os.path.join(RESULTS, f"{key}.json")) as f:
        data = json.load(f)
    return data.pop("spec"), data


def record_test(key, test_acc):
    """Add the test score to a saved run, once it has finally been measured."""
    path = os.path.join(RESULTS, f"{key}.json")
    with open(path) as f:
        data = json.load(f)
    data["test"] = test_acc
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def best_row(history):
    b = history["best_epoch"] - 1
    return history["train"][b], history["val"][b]


# -------------------------------------------------------------------------- modes


def run_ladder(args, device):
    """Stages 0 -> 2b, then the test set once, then the figures."""
    results = []
    for s in LADDER:
        s = dict(s, epochs=args.epochs)
        history, model = run(
            s, "ladder", device, save=not args.no_save, load=args.eval
        )
        results.append((s, history, model))

    print("\nloading the held-out test files for the first time...")
    test_loader = DataLoader(load_test(), batch_size=512)

    print(
        f"\n{'stage':<28}{'images':>8}{'epoch':>7}{'train':>8}{'val':>8}"
        f"{'test':>8}{'gap':>8}"
    )
    print("-" * 75)
    rows = []
    for s, history, model in results:
        train_acc, val_acc = best_row(history)
        test_acc = evaluate(model, test_loader)
        record_test(f"part7_{s['name']}", test_acc)
        rows.append((s["label"], train_acc, val_acc, test_acc))
        print(
            f"{s['label']:<28}{s['n_train']:>8,}{history['best_epoch']:>7}"
            f"{train_acc:>8.3f}{val_acc:>8.3f}{test_acc:>8.3f}"
            f"{train_acc - val_acc:>8.3f}"
        )

    bn_model = next(m for s, h, m in reversed(results) if s["bn"])
    eval_mode_demo(bn_model, device)

    # Figure 1: the ladder itself.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.27
    xs = range(len(rows))
    for i, (key, color) in enumerate(
        [("train", "tab:blue"), ("val", "tab:orange"), ("test", "tab:green")]
    ):
        vals = [r[i + 1] for r in rows]
        bars = ax.bar(
            [x + (i - 1) * width for x in xs], vals, width, label=key, color=color
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(list(xs), [r[0] for r in rows], fontsize=8)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("One change per rung, same validation set throughout")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    # Figure 2: per-stage gap panels, same as part6's --compare.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (s, history, _) in zip(axes.ravel(), results):
        plot_gap(ax, history, s["label"])
        ax.set_ylim(0.4, 1.02)
    fig.tight_layout()


def run_data_curve(args, device):
    """Stage-0 settings at several dataset sizes, plus an equal-compute control."""
    sizes = [5000, 10000, N_PART6, N_FULL]
    points = []
    for n in sizes:
        s = spec(f"n{n}", f"config D, {n:,} images", n_train=n, epochs=args.epochs)
        history, _ = run(s, "data-curve", device)
        points.append((n, *best_row(history)))

    # The control. 45k images x 30 epochs is 2.5x the gradient steps of
    # 18k x 30. Give 18k the same number of steps and see how much of stage 1's
    # gain was really just "trained longer".
    control_epochs = args.epochs * N_FULL // N_PART6
    s = spec(
        f"n{N_PART6}_equal_steps",
        f"config D, {N_PART6:,} images, {control_epochs} epochs",
        epochs=control_epochs,
    )
    history, _ = run(s, "data-curve", device)
    control = best_row(history)

    print(f"\n{'images':>8}{'epochs':>8}{'train':>8}{'val':>8}{'gap':>8}")
    for n, tr, va in points:
        print(f"{n:>8,}{args.epochs:>8}{tr:>8.3f}{va:>8.3f}{tr - va:>8.3f}")
    print(
        f"{N_PART6:>8,}{control_epochs:>8}{control[0]:>8.3f}{control[1]:>8.3f}"
        f"{control[0] - control[1]:>8.3f}   <- same steps as 45k x {args.epochs}"
    )

    plt.figure(figsize=(7, 4.5))
    ns = [p[0] for p in points]
    plt.plot(ns, [p[1] for p in points], "o-", color="tab:blue", label="train")
    plt.plot(ns, [p[2] for p in points], "o-", color="tab:orange", label="validation")
    plt.plot(
        [N_PART6],
        [control[1]],
        "s",
        color="tab:red",
        markersize=8,
        label=f"validation, 18k x {control_epochs} epochs (equal steps)",
    )
    plt.xscale("log")
    plt.minorticks_off()
    plt.xticks(ns, [f"{n // 1000}k" for n in ns])
    plt.xlabel("training images (log scale)")
    plt.ylabel("accuracy")
    plt.title("Learning curve: same model, more data")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()


def run_lr_sweep(args, device):
    """Constant LR, with and without BN, short runs on the full data."""
    lrs = [1e-3, 3e-3, 1e-2, 3e-2]
    table = {False: [], True: []}
    for bn in (False, True):
        for lr in lrs:
            s = spec(
                f"{'bn' if bn else 'plain'}_lr{lr:g}",
                f"{'BN' if bn else 'no BN'}, constant lr {lr:g}",
                n_train=N_FULL,
                bn=bn,
                lr=lr,
                epochs=args.sweep_epochs,
            )
            history, _ = run(s, "lr-sweep", device)
            table[bn].append(
                None
                if history["diverged"] and len(history["val"]) <= 1
                else max(history["val"])
            )

    print(f"\nbest validation accuracy after {args.sweep_epochs} epochs, constant lr")
    print(f"{'lr':>8}{'no BN':>10}{'BN':>10}")
    for i, lr in enumerate(lrs):
        cells = [
            "diverged" if v is None else f"{v:.3f}"
            for v in (table[False][i], table[True][i])
        ]
        print(f"{lr:>8g}{cells[0]:>10}{cells[1]:>10}")

    plt.figure(figsize=(7, 4.5))
    for bn, color in ((False, "tab:gray"), (True, "tab:purple")):
        ys = [0.1 if v is None else v for v in table[bn]]
        plt.plot(
            lrs, ys, "o-", color=color, label="with BatchNorm" if bn else "no BatchNorm"
        )
    plt.axhline(0.1, color="black", linewidth=0.8, linestyle=":")
    plt.text(lrs[0], 0.12, "chance (10%)", fontsize=8)
    plt.xscale("log")
    plt.xlabel("constant learning rate (log scale)")
    plt.ylabel(f"best val accuracy, {args.sweep_epochs} epochs")
    plt.title("BatchNorm widens the range of learning rates that work")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()


def eval_mode_demo(model, device):
    """What happens if a BN model is evaluated without model.eval().

    In train mode BN normalises with the CURRENT batch's statistics. In eval
    mode it uses running averages accumulated during training. Leave BN in
    train mode at test time and each image's prediction depends on which other
    images happen to share its batch.

    We flip only the BN layers into train mode (dropout stays off) so the
    damage is attributable to BN alone, and work on a copy, because train-mode
    BN also overwrites the running averages as a side effect.
    """
    _, val_ds, _ = make_splits(N_FULL)
    labels = torch.tensor([val_ds.dataset.targets[i] for i in val_ds.indices])
    by_class = Subset(val_ds, torch.argsort(labels, stable=True).tolist())

    def acc(loader, bn_train):
        m = copy.deepcopy(model)
        m.eval()
        if bn_train:
            for layer in m.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.train()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                pred = m(x.to(device)).argmax(1).cpu()
                correct += (pred == y).sum().item()
                total += len(y)
        return correct / total

    print("\nforgetting model.eval() on a BatchNorm model (validation set):")
    print(
        f"  model.eval(), as it should be          "
        f"{acc(DataLoader(val_ds, batch_size=512), False):.3f}"
    )
    print(
        f"  BN in train mode, shuffled batches 512  "
        f"{acc(DataLoader(val_ds, batch_size=512, shuffle=True), True):.3f}"
    )
    print(
        f"  BN in train mode, shuffled batches 16   "
        f"{acc(DataLoader(val_ds, batch_size=16, shuffle=True), True):.3f}"
    )
    print(
        f"  BN in train mode, batches of one class  "
        f"{acc(DataLoader(by_class, batch_size=128), True):.3f}"
    )


# --------------------------------------------------------------------------- main


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="stage 1: train on all 45,000 non-validation images",
    )
    parser.add_argument(
        "--bn", action="store_true", help="stage 2a: BatchNorm after every conv"
    )
    parser.add_argument(
        "--onecycle",
        action="store_true",
        help=f"stage 2b: OneCycle LR schedule (peak {ONECYCLE_PEAK_LR:g})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=f"learning rate (constant), or the OneCycle peak "
        f"(default {BASE_LR:g}, or {ONECYCLE_PEAK_LR:g} with --onecycle)",
    )
    parser.add_argument(
        "--ladder",
        action="store_true",
        help="run stages 0, 1, 2a, 2b back to back and compare",
    )
    parser.add_argument(
        "--data-curve",
        action="store_true",
        help="stage 1 in depth: accuracy vs dataset size, plus an equal-steps control",
    )
    parser.add_argument(
        "--lr-sweep",
        action="store_true",
        help="stage 2a in depth: constant-LR sweep with and without BN",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sweep-epochs", type=int, default=8)
    parser.add_argument(
        "--no-save", action="store_true", help="skip writing part7_<run>.pt"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="skip training: load part7_<run>.pt and results/part7_<run>.json, "
        "then print the same table and figures (with --ladder or a single stage)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"using device: {device}")
    print(f"validation: the same {N_VAL:,} images for every run")
    print(f'watch live:  trackio show --project "{TRACKIO_PROJECT}"')

    if args.eval and (args.data_curve or args.lr_sweep):
        raise SystemExit(
            "--eval needs saved weights, and the sweeps never save any -- "
            "use --eval with --ladder or a single stage"
        )

    if args.ladder:
        run_ladder(args, device)
    elif args.data_curve:
        run_data_curve(args, device)
    elif args.lr_sweep:
        run_lr_sweep(args, device)
    else:
        lr = (
            args.lr
            if args.lr is not None
            else (ONECYCLE_PEAK_LR if args.onecycle else BASE_LR)
        )
        parts = [f"n{N_FULL if args.full_data else N_PART6}"]
        parts += [p for p, on in (("bn", args.bn), ("onecycle", args.onecycle)) if on]
        s = spec(
            "-".join(parts) + f"-lr{lr:g}",
            " + ".join(parts),
            n_train=N_FULL if args.full_data else N_PART6,
            bn=args.bn,
            onecycle=args.onecycle,
            lr=lr,
            epochs=args.epochs,
        )
        history, model = run(
            s, "single", device, save=not args.no_save, load=args.eval
        )

        print("\nloading the held-out test files for the first time...")
        test_acc = evaluate(model, DataLoader(load_test(), batch_size=512))
        record_test(f"part7_{s['name']}", test_acc)
        train_acc, val_acc = best_row(history)
        print(
            f"\nepoch {history['best_epoch']}  train {train_acc:.3f}  "
            f"val {val_acc:.3f}  test {test_acc:.3f}  gap {train_acc - val_acc:.3f}"
        )
        if args.bn:
            eval_mode_demo(model, device)

        fig, ax = plt.subplots(figsize=(6, 4.2))
        plot_gap(ax, history, s["label"])
        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
