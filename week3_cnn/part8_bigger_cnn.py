"""Week 3 - Part 8: Capacity -- a model big enough for the data.

Part 7 finished at train 0.867, validation 0.832. The gap is 3.4 points, so this
is still a CEILING, not overfitting: the model cannot fit its own training
images. Data is all used and training is now healthy, so what is left is the
model. Part 6's architecture is three conv layers, at most 64 channels, and 70%
of its weights in a fully-connected head.

Two more rungs, on part 7's rules -- one change each, same 5,000 validation
images, same recipe (45k images, BatchNorm, OneCycle peak 3e-3, 30 epochs). Only
the architecture moves:

    3b   DEEPER   two convs per stage instead of one
    3c   WIDER    and the channels doubled, 32-64-64 -> 64-128-256

    python part8_bigger_cnn.py --ladder        # trains rungs 3b and 3c
    python part8_bigger_cnn.py --story         # all six rungs from results/
    python part8_bigger_cnn.py --depth-sweep   # 1, 2, 3, 4 convs per stage
    python part8_bigger_cnn.py --convs 2 --widths 64 128 256   # one model

MEASURED RESULTS (M1 Pro, 30 epochs; --ladder takes about 18 minutes):

    rung                         params    min   train     val    test     gap
    2b: part7 final model       188,970    4.9   0.867   0.832   0.826   0.034
    3b: 2 convs per stage       272,234    6.9   0.934   0.883   0.881   0.051
    3c: widths x2             1,672,010   11.4   0.986   0.915   0.914   0.071

Rung 2b is part 7's -- the defaults in build_model() rebuild it layer for layer
-- so --ladder does not retrain it; --story reads it back from results/.

"""

import argparse
import os

import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

from part3_cnn_cifar10 import evaluate, get_device
from part6_cnn_overfitting import DROPOUT, load_test, plot_gap
from part7_improving_cnn import (
    N_FULL,
    ONECYCLE_PEAK_LR,
    TRACKIO_PROJECT,
    best_row,
    load_history,
    record_test,
    run,
    spec,
)


def build_model(widths=(32, 64, 64), convs=1):
    """A family of plain CNNs: three stages, each ending in a 2x2 max-pool.

        stage 1:  convs x [Conv -> BN -> ReLU], widths[0] channels, 32x32 -> 16x16
        stage 2:  convs x [Conv -> BN -> ReLU], widths[1] channels, 16x16 ->  8x8
        stage 3:  convs x [Conv -> BN -> ReLU], widths[2] channels,  8x8  ->  4x4
        head:     Flatten -> Dropout -> Linear(16 * widths[2] -> 128) -> ReLU
                  -> Linear(128 -> 10)

    The head is part 6's, unchanged on every rung. With the defaults this is
    part 7's stage-2b network exactly -- same layers in the same order, so the
    same random initial weights and 188,970 parameters. The two knobs are depth
    (convs per stage) and width (channels per stage).
    """
    layers, c_in = [], 3
    for width in widths:
         # TODO:
    #     for _ in range(convs):
    #         layers += [
    #             three elemments ?         
    #         ]
    #         c_in = width
    #     layers.append(nn.MaxPool2d(2))
    # layers += [
    #     two elements ? 
    #     nn.Linear(c_in * 4 * 4, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 10),
    # ]
    return nn.Sequential(*layers)


def arch_spec(name, label, widths=(32, 64, 64), convs=1, epochs=30):
    """part7's stage-2b training recipe, plus the two architecture knobs."""
    s = spec(
        name,
        label,
        n_train=N_FULL,
        bn=True,
        onecycle=True,
        lr=ONECYCLE_PEAK_LR,
        epochs=epochs,
    )
    s.update(widths=list(widths), convs=convs)
    return s


def model_from_spec(s):
    return build_model(widths=s["widths"], convs=s["convs"])


def train_spec(s, group, device, save, load=False):
    history, model = run(
        s,
        group,
        device,
        save=save,
        model_fn=model_from_spec,
        project=TRACKIO_PROJECT,
        prefix="part8",
        load=load,
    )
    return history, model, sum(p.numel() for p in model.parameters())


# Two more rungs on part7's ladder, each changing exactly one thing. Rung 2b is
# part7's -- the defaults above rebuild it layer for layer -- so it is not
# re-trained here; --story reads it back from results/.
LADDER = [
    arch_spec("3b_deeper", "3b: 2 convs per stage", convs=2),
    arch_spec("3c_wider", "3c: widths x2", convs=2, widths=(64, 128, 256)),
]

# The whole week-3 story, in the order it was measured: results/<key>.json.
STORY = [
    "part7_0_config_d",
    "part7_1_full_data",
    "part7_2a_batchnorm",
    "part7_2b_onecycle",
    "part8_3b_deeper",
    "part8_3c_wider",
]

SWEEP_WIDTHS = (64, 128, 256)  # rung 3c's widths


def run_ladder(args, device):
    rungs = [
        s for s in LADDER if args.rungs is None or s["name"].split("_")[0] in args.rungs
    ]
    results = []
    for s in rungs:
        history, model, params = train_spec(
            dict(s, epochs=args.epochs),
            "ladder",
            device,
            save=not args.no_save,
            load=args.eval,
        )
        results.append((s, history, model, params))

    print("\nloading the held-out test files for the first time...")
    test_loader = DataLoader(load_test(), batch_size=512)

    print(
        f"\n{'rung':<24}{'params':>11}{'min':>6}{'epoch':>7}{'train':>8}"
        f"{'val':>8}{'test':>8}{'gap':>8}"
    )
    print("-" * 80)
    for s, history, model, params in results:
        train_acc, val_acc = best_row(history)
        test_acc = evaluate(model, test_loader)
        record_test(f"part8_{s['name']}", test_acc)
        print(
            f"{s['label']:<24}{params:>11,}{history['minutes']:>6.1f}"
            f"{history['best_epoch']:>7}{train_acc:>8.3f}{val_acc:>8.3f}"
            f"{test_acc:>8.3f}{train_acc - val_acc:>8.3f}"
        )

    fig, axes = plt.subplots(
        1, len(results), figsize=(5 * len(results), 4), squeeze=False
    )
    for ax, (s, history, _, params) in zip(axes[0], results):
        plot_gap(ax, history, f"{s['label']} ({params:,} params)")
        ax.set_ylim(0.4, 1.02)
    fig.tight_layout()


def run_depth_sweep(args, device):
    """Does more depth keep helping? Convs per stage at rung 3c's widths.

    Validation only: this is a decision about the architecture, so the test
    set stays closed.
    """
    rows = []
    for convs in args.convs_list:
        s = arch_spec(
            f"convs{convs}",
            f"{convs} convs per stage ({3 * convs} conv layers)",
            widths=SWEEP_WIDTHS,
            convs=convs,
            epochs=args.epochs,
        )
        history, _, params = train_spec(s, "depth-sweep", device, save=False)
        rows.append((convs, params, history["minutes"], *best_row(history)))

    print(
        f"\n{'convs/stage':>12}{'layers':>8}{'params':>11}{'min':>6}"
        f"{'train':>8}{'val':>8}{'gap':>8}"
    )
    for convs, params, minutes, train_acc, val_acc in rows:
        print(
            f"{convs:>12}{3 * convs:>8}{params:>11,}{minutes:>6.1f}"
            f"{train_acc:>8.3f}{val_acc:>8.3f}{train_acc - val_acc:>8.3f}"
        )

    layers = [3 * r[0] for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(layers, [r[3] for r in rows], "o-", color="tab:blue", label="train")
    plt.plot(layers, [r[4] for r in rows], "o-", color="tab:orange", label="validation")
    plt.xticks(layers)
    plt.xlabel(
        f"conv layers (widths {'-'.join(map(str, SWEEP_WIDTHS))}, no skip connections)"
    )
    plt.ylabel("accuracy")
    plt.title("Depth: does stacking more convolutions keep helping?")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()


def run_story():
    """The six-rung staircase, drawn from results/ -- nothing is re-trained.

    Every run writes its per-epoch numbers to results/<key>.json, so the figure
    that ends week 3 costs a second to redraw instead of an hour of GPU. If a
    rung is missing, the message tells you which command produces it.
    """
    rungs = []
    for key in STORY:
        try:
            s, history = load_history(key)
        except FileNotFoundError:
            src = "part7_improving_cnn.py" if key.startswith("part7") else __file__
            raise SystemExit(
                f"missing results/{key}.json -- run: "
                f"python {os.path.basename(src)} --ladder"
            )
        rungs.append((s, history))

    print(
        f"\n{'rung':<28}{'params':>11}{'min':>6}{'epoch':>7}{'train':>8}"
        f"{'val':>8}{'test':>8}{'gap':>8}"
    )
    print("-" * 84)
    rows = []
    for s, history in rungs:
        train_acc, val_acc = best_row(history)
        test_acc = history.get("test")
        rows.append((s["label"], train_acc, val_acc, test_acc))
        print(
            f"{s['label']:<28}{s['params']:>11,}{history['minutes']:>6.1f}"
            f"{history['best_epoch']:>7}{train_acc:>8.3f}{val_acc:>8.3f}"
            f"{'--' if test_acc is None else format(test_acc, '.3f'):>8}"
            f"{train_acc - val_acc:>8.3f}"
        )

    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.27
    xs = range(len(rows))
    for i, (key, color) in enumerate(
        [("train", "tab:blue"), ("val", "tab:orange"), ("test", "tab:green")]
    ):
        vals = [(r[i + 1] or 0) for r in rows]
        bars = ax.bar(
            [x + (i - 1) * width for x in xs], vals, width, label=key, color=color
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(list(xs), [r[0] for r in rows], fontsize=8)
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("accuracy")
    ax.set_title(
        "Week 3: make it fit (rungs 1-3c), then make it generalise (your turn)"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ladder", action="store_true", help="train rungs 3b and 3c back to back"
    )
    parser.add_argument(
        "--story",
        action="store_true",
        help="redraw all six rungs from results/ -- no training",
    )
    parser.add_argument(
        "--rungs",
        nargs="+",
        default=None,
        help="with --ladder, only these rungs, e.g. --rungs 3b",
    )
    parser.add_argument(
        "--depth-sweep",
        action="store_true",
        help=f"convs per stage at widths {SWEEP_WIDTHS}",
    )
    parser.add_argument(
        "--convs-list",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="with --depth-sweep, the convs-per-stage values to try",
    )
    parser.add_argument("--convs", type=int, default=1, help="convs per stage")
    parser.add_argument(
        "--widths",
        type=int,
        nargs=3,
        default=[32, 64, 64],
        help="channels in the three stages",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--no-save", action="store_true", help="skip writing part8_<run>.pt"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="skip training: load part8_<run>.pt and results/part8_<run>.json, "
        "then print the same table and figures (with --ladder or a single run)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"using device: {device}")
    print(f'watch live:  trackio show --project "{TRACKIO_PROJECT}"')

    if args.story:
        run_story()
        plt.show()
        return

    if args.eval and args.depth_sweep:
        raise SystemExit(
            "--eval needs saved weights, and --depth-sweep never saves any -- "
            "use --eval with --ladder or a single run"
        )

    if args.ladder:
        run_ladder(args, device)
    elif args.depth_sweep:
        run_depth_sweep(args, device)
    else:
        name = f"convs{args.convs}-w{'-'.join(map(str, args.widths))}"
        s = arch_spec(
            name, name, widths=args.widths, convs=args.convs, epochs=args.epochs
        )
        history, model, params = train_spec(
            s, "single", device, save=not args.no_save, load=args.eval
        )
        print(f"parameters: {params:,}")

        print("\nloading the held-out test files for the first time...")
        test_acc = evaluate(model, DataLoader(load_test(), batch_size=512))
        record_test(f"part8_{s['name']}", test_acc)
        train_acc, val_acc = best_row(history)
        print(
            f"\nepoch {history['best_epoch']}  train {train_acc:.3f}  "
            f"val {val_acc:.3f}  test {test_acc:.3f}  gap {train_acc - val_acc:.3f}"
            f"  ({history['minutes']:.1f} min)"
        )

        fig, ax = plt.subplots(figsize=(6, 4.2))
        plot_gap(ax, history, name)
        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
