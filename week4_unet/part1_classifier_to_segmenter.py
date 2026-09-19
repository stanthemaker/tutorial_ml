"""Week 4 - Part 1: a classifier is almost a segmentation model.

Week 3 ended at "image -> one of 10 labels". The last layers of that CNN were
Flatten -> Linear -> Linear, crushing the whole picture into 10 numbers.

This week the output is a *label for every pixel*:

    input   3 x 96 x 96 photo of a dog
    output  96 x 96 = 9,216 labels, one per pixel, each 0/1/2
            (0 = pet, 1 = background, 2 = border)

That sounds like a different field. It is four lines of diff. Here is Week 3
Part 6's CIFAR-10 classifier against this file's model:

    Week 3 classifier                model here
    ------------------------------   ------------------------------
    Conv -> BN -> ReLU -> MaxPool    Conv -> BN -> ReLU -> MaxPool     x3
    Flatten                          (deleted)
    Linear(1024 -> 128) -> ReLU      (deleted)
    Linear(128 -> 10)                Conv1x1(128 -> 3)
    (nothing)                        Upsample(x8)
    CrossEntropyLoss on (N, 10)      CrossEntropyLoss on (N, 3, H, W)

The trunk is untouched. Even the loss is untouched: nn.CrossEntropyLoss
accepts (N, C, H, W) logits against (N, H, W) integer targets and applies the
Week 2 formula at every pixel independently, then averages. Segmentation is
classification, run 9,216 times per image, with the whole picture as context.

WHICH PART HAD TO GO, AND WHY. Only `Flatten -> Linear`. Flattening throws away
*where* each feature was -- after it, "ear-shaped thing" and "ear-shaped thing
in the top left" are the same 1024 numbers in a different order, and the Linear
that follows has to be told the input size in advance. Everything else a
classifier does -- convolutions, pooling, BatchNorm, the loss -- is already
what a segmentation model wants.

Take the Linear head out, put a 1x1 conv in its place, and the classifier IS a
segmenter. That is the "fully convolutional" idea, and in 2015 it was a paper.
A side effect the file demonstrates at the end: with no Linear layer left,
nothing fixes the input size, so the same trained weights run on a 192x192
photo and produce a 192x192 map.

PIXEL ACCURACY IS A TRAP, so we retire it here. 58% of the pixels in this
dataset are background, so "predict background everywhere" scores 0.579
without looking at the image. We report mean IoU instead:

    IoU_c = (pixels where prediction == c AND truth == c)
            ---------------------------------------------
            (pixels where prediction == c  OR truth == c)

A class the model never predicts scores 0 no matter how rare it is, so the
constant baseline gets 0.193 instead of 0.579, and the thin border class has
nowhere to hide. mIoU is the number for the rest of the week.

MEASURED RESULTS (M1 Pro, 8 epochs, full trainval, about 2 minutes):

    model                    params    acc    mIoU     pet      bg   border
    always "background"           0  0.579   0.193   0.000   0.579    0.000
    classifier -> FCN        94,083  0.734   0.484   0.523   0.700    0.227

That is most of the way to a working segmentation model, for the cost of
deleting two layers.

NOW THE PART THAT DOES NOT WORK. Look at the pictures, not the number. The
outlines are lumpy and approximate: ear tips round off, thin legs thin out
into the background, holes open up in the middle of an animal, and the border
ring wanders. Nothing in the output is finer than about 8 pixels.

The reason is the one thing we did not replace. Three poolings take 96x96 down
to 12x12, and that is where the decision is actually made -- one vote per 8x8
block of the photo. `Upsample` then stretches those votes back to 96x96, and it
is a fixed formula with no parameters: it can make the map bigger, it cannot
make it more detailed.

And we cannot simply pool less. Pooling is what gives this model its context:
at 12x12 a 3x3 kernel covers a quarter of the photo, which is how it knows it
is looking at an animal at all rather than at a brown patch. The same operation
buys the context and destroys the resolution.

    context and resolution pull in opposite directions

The rest of the week is one idea for having both: pool all the way down for
context, then LEARN the way back up instead of stretching (Part 2), and hand
the encoder's detail forward across the gap (Part 4).

    python part1_classifier_to_segmenter.py
    python part1_classifier_to_segmenter.py --epochs 20
    python part1_classifier_to_segmenter.py --eval   # reload part1_fcn.pt, no training

Every run logs its per-epoch curves to trackio. Watch them live, or compare
runs after the fact, with:

    trackio show --project "week4-unet"

Set WEEK4_NO_TRACKIO=1 to turn the logging off.
"""

import argparse

import torch
import torch.nn as nn

from common import (
    CLASS_SHORT,
    IMAGE_SIZE,
    N_CLASSES,
    colorize_batch,
    count_params,
    evaluate_seg,
    finish,
    get_device,
    load_pet,
    load_weights,
    make_loaders,
    predict_batch,
    print_metrics,
    save_weights,
    seg_probe,
    show_grid,
    tracked,
    train,
)


class Constant(nn.Module):
    """Predicts one class everywhere. The baseline every metric must beat.

    Written as a model rather than as a special case inside the metric code, so
    that it goes through evaluate_seg untouched: the number it gets is computed
    exactly the way the real model's number is.
    """

    def __init__(self, cls=1):
        super().__init__()
        self.cls = cls
        self.dummy = nn.Parameter(torch.zeros(1))  # so .parameters() is non-empty

    def forward(self, x):
        out = torch.zeros(x.size(0), N_CLASSES, *x.shape[2:], device=x.device)
        out[:, self.cls] = 1.0
        return out


def fcn(width=32):
    """Week 3's classifier with its Flatten -> Linear head replaced by a 1x1 conv.

    Read it against the diff in the module docstring. The first nine lines are
    Week 3's trunk, unchanged: three stages of Conv -> BN -> ReLU -> MaxPool,
    doubling the channels each time, 96 -> 48 -> 24 -> 12.

    The last two lines are the whole of this week's first idea:

      Conv2d(..., 1)   a 1x1 convolution is a linear layer applied at each
                       pixel independently: 128 feature values at this location
                       -> 3 class scores at this location, no spatial mixing.
                       It is the Linear head, applied 144 times instead of once,
                       and it keeps the grid instead of flattening it.

      Upsample(x8)     the 12x12 grid of predictions, stretched back to the
                       input size. Fixed formula, zero parameters, nothing
                       learned. This is the line Part 2 replaces.

    No Sigmoid or Softmax at the end: these are raw logits, exactly like the 10
    at the end of Week 3's classifier. CrossEntropyLoss applies log_softmax
    itself, and squashing the outputs before handing them over is the most
    common bug in this file.
    """
    return nn.Sequential(
        nn.Conv2d(3, width, 3, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(width, width * 2, 3, padding=1),
        nn.BatchNorm2d(width * 2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(width * 2, width * 4, 3, padding=1),
        nn.BatchNorm2d(width * 4),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(width * 4, N_CLASSES, kernel_size=1),
        # TODO:
        # nn.Upsample(scale_factor=_, mode="_", align_corners=False),
    )


def show_shape_contract(x, y):
    """Print the shapes going into the loss. Every error in this file is here."""
    print("the shape contract:")
    print(f"  image   {tuple(x.shape)}  {x.dtype}  in [{x.min():.2f}, {x.max():.2f}]")
    print(
        f"  mask    {tuple(y.shape)}      {y.dtype}  values "
        f"{sorted(torch.unique(y).tolist())}"
    )
    print("  logits  (N, 3, 96, 96) float, raw scores -- no Sigmoid, no Softmax")
    print("  The mask has no channel axis and is not one-hot: CrossEntropyLoss")
    print("  wants the class INDEX at each pixel.\n")


@torch.no_grad()
def any_input_size(model, device, size=192):
    """A fully convolutional network does not care about the input size.

    There is no Linear layer left, so no weight matrix has the input size baked
    into it. Every layer is a convolution or a pool, and those slide over
    whatever they are given. Train at 96x96, run at 192x192.

    (Whether the *answers* are as good at a different scale is another
    question -- the filters learned what a dog looks like at one zoom level.
    But the shapes go through, and that is what a Linear head would have made
    impossible.)
    """
    model.eval()
    out = model(torch.zeros(1, 3, size, size, device=device))
    print(f"  the same weights on a {size}x{size} input -> {tuple(out.shape)}")
    print("  no Linear layer anywhere, so nothing fixed the input size.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="subset size for a quicker run (default: all 3,680)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="skip training: load checkpoints/part1_fcn.pt and "
        "print the same table and figure",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = get_device()
    print(f"using device: {device}\n")

    train_ds = load_pet("trainval", n_subset=args.n_train, augment=True)
    test_ds = load_pet("test", n_subset=1000)
    train_loader, test_loader = make_loaders(train_ds, test_ds)

    x, y = next(iter(train_loader))
    show_shape_contract(x, y)

    table = []

    baseline = Constant(cls=1).to(device)
    acc, iou, miou = evaluate_seg(baseline, test_loader)
    table.append(('always "background"', 0, acc, iou, miou))
    print_metrics('always "background"', acc, iou, miou)
    print()

    torch.manual_seed(0)
    model = fcn()
    print(f"=== classifier -> FCN -- {count_params(model):,} parameters ===")
    # Part 1's own recipe (8 epochs, lr 3e-3). Part 3 retrains the same
    # architecture on its recipe and saves that as part3_fcn -- a separate file.
    key = f"part1_fcn_n{args.n_train}" if args.n_train else "part1_fcn"
    if args.eval:
        load_weights(model, key)
        model.to(device)
    else:
        # The same CrossEntropyLoss as Weeks 2 and 3, now scoring 9,216 pixels
        # instead of one image.
        with tracked(
            "classifier -> FCN",
            config=dict(
                part=1,
                arch="fcn",
                params=count_params(model),
                epochs=args.epochs,
                lr=args.lr,
                n_train=len(train_ds),
            ),
        ) as run:
            train(
                model,
                train_loader,
                device,
                nn.CrossEntropyLoss(),
                epochs=args.epochs,
                lr=args.lr,
                run=run,
                eval_fn=seg_probe(test_loader),
            )
        save_weights(model, key)

    acc, iou, miou = evaluate_seg(model, test_loader)
    print_metrics("test", acc, iou, miou)
    print()
    table.append(("classifier -> FCN", count_params(model), acc, iou, miou))

    any_input_size(model, device)

    head = f"{'model':<24} {'params':>9} {'acc':>7} {'mIoU':>7}"
    head += "".join(f"{n:>9}" for n in CLASS_SHORT)
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for label, params, acc, iou, miou in table:
        line = f"{label:<24} {params:>9,} {acc:>7.3f} {miou:>7.3f}"
        line += "".join(f"{v:>9.3f}" for v in iou)
        print(line)
    print("=" * len(head))

    print("\nTwo things to take from this, and they point in opposite")
    print("directions.")
    print("\n1. Deleting two layers turned an image classifier into a working")
    print("   segmentation model. Pixel accuracy would have flattered it even")
    print("   more -- note that predicting 'background' everywhere already")
    print("   scores 0.579, which is why we read the mIoU column instead.")
    print("\n2. Now look at the pictures. The outlines are lumpy: ear tips")
    print("   round off, thin legs fade into the background, holes open in")
    print("   the middle of an animal. Nothing in the output is finer than")
    print("   about 8 pixels, because the decision was made on a 12x12 grid")
    print("   -- one vote per 8x8 block of photo -- and Upsample only")
    print("   stretched it. Upsample has no parameters; it cannot add detail")
    print("   that was not there.")
    print("\nAnd we cannot just pool less: pooling is what gives this model the")
    print("context to know it is looking at an animal at all. One operation")
    print("buys the context and destroys the resolution.")
    print("\nPart 2 replaces that Upsample with something that learns.")

    x, y, pred = predict_batch(model, test_loader)
    show_grid(
        [x, colorize_batch(y), colorize_batch(pred)],
        ["photo", "truth", "FCN"],
        suptitle="A classifier with its head replaced: right animal, 8-pixel answers",
    )
    finish("part1")


if __name__ == "__main__":
    main()
