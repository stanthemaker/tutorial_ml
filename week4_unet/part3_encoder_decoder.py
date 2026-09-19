"""Week 4 - Part 3: point the decoder at the real task.

Part 2 built an encoder-decoder that turns a photo into a photo. Segmentation
wants a photo turned into a map of class indices. It is the same network. Two
lines change, and both follow from one fact: the output is now a *class* per
pixel, not a *colour* per pixel.

    head    Conv1x1(32 -> 3), and NO Sigmoid
            Those 3 numbers per pixel are logits -- one score per class --
            exactly like the 10 logits at the end of Week 3's classifier.
            Squashing them into [0, 1] would be wrong here for the same reason
            it would have been wrong there. CrossEntropyLoss applies
            log_softmax itself; a Sigmoid in front of it is the single most
            common bug in this file.

    loss    nn.CrossEntropyLoss(), not MSELoss.
            And it is the *same* CrossEntropyLoss from Weeks 2 and 3: it
            accepts (N, C, H, W) logits against (N, H, W) int64 targets and
            applies the usual formula at every pixel independently, then
            averages. Segmentation really is classification, run 9,216 times
            per image, with the whole picture as context.

Nothing else moves. Same encoder, same bottleneck, same decoder, same training
loop, same optimiser.

The shape contract is worth writing on the board, because every error message
in this file comes from getting it wrong:

    model output    (N, 3, 96, 96)   float, raw scores
    target          (N, 96, 96)      int64, values in {0, 1, 2}
    prediction      logits.argmax(dim=1) -> (N, 96, 96)

Note the target has no channel axis and is not one-hot. PyTorch wants the
index, not the vector.

MEASURED RESULTS (M1 Pro, 15 epochs, all 3,680 labels, about 12 minutes):

    model                      params     acc    mIoU     pet      bg  border
    part1 FCN                  94,083   0.746   0.463   0.490   0.720   0.179
    encoder-decoder         1,734,947   0.878   0.688   0.774   0.872   0.417
      + AE pretraining      1,734,947   0.864   0.672   0.746   0.854   0.414

A learned decoder is worth 0.225 mIoU over stretching the logits -- a bigger
jump than anything else this week. The border class more than doubles. Both
models are trained here on the same recipe so that only the architecture
differs; that recipe is not the one Part 1 used for its own table (lr 1e-3 for
15 epochs here, lr 3e-3 for 8 there), which is why the FCN scores 0.463 here
and 0.484 there. Compare within a table, not across.

--pretrained is the third row, and it answers a fair question: Part 2 trained
this exact body on photographs with no labels at all, so why start from random
numbers? Loading those weights transfers 104 of 106 tensors -- everything but
the head, which predicted colours and is no use here.

The answer, with all 3,680 labels available, is that it does not help. 0.672
against 0.688. Now run it with a realistic label budget instead:

    python part3_encoder_decoder.py --pretrained --n-train 500 --skip-baseline

    model                      params     acc    mIoU     pet      bg  border
    encoder-decoder         1,734,947   0.788   0.535   0.579   0.772   0.254
      + AE pretraining      1,734,947   0.811   0.575   0.635   0.794   0.295

With 500 labels, pretraining is worth 0.040 mIoU. That is the whole case for
self-supervised pretraining in two tables: it buys nothing when you already
have enough labels, and it buys real accuracy when you do not -- which is the
only situation in which anybody reaches for it. Photographs are free; a trimap
that a human painted by hand is not.

NOW THE PART THAT IS STILL WRONG, and it is the same complaint as Part 2's
blurry whiskers. Look at the outlines. The model knows roughly where the
boundary is and cannot commit to a pixel, because the information needed to
place it was destroyed at the 12x12 waist. In Part 2 that cost us texture and
looked soft. Here it costs labelled pixels and is measurable: the border class
is the worst of the three by a wide margin.

    python part3_encoder_decoder.py
    python part3_encoder_decoder.py --pretrained
    python part3_encoder_decoder.py --pretrained --n-train 500 --skip-baseline
    python part3_encoder_decoder.py --pretrained --eval   # reload, no training

Every run logs its per-epoch curves to trackio. Watch them live, or compare
runs after the fact, with:

    trackio show --project "week4-unet"

Set WEEK4_NO_TRACKIO=1 to turn the logging off.
"""

import argparse
import os

import torch
import torch.nn as nn

from common import (CLASS_SHORT, N_CLASSES, colorize_batch, count_params,
                    evaluate_seg, finish, get_device, load_pet, load_weights,
                    make_loaders, predict_batch, print_metrics, save_history,
                    save_weights, seg_probe, show_grid, tracked, train,
                    weights_path)
from part1_classifier_to_segmenter import fcn
from part2a_cnn_autoencoder import ConvAutoencoder


class SegAutoencoder(ConvAutoencoder):
    """Part 2's autoencoder with a classification head.

    Subclassing instead of rewriting is the point: encoder, bottleneck,
    upsampling and decoder are character-for-character the network that was
    reconstructing cat photos a minute ago. Only the last layer changes, and
    only because the meaning of the output changed.
    """

    def __init__(self, n_classes=N_CLASSES, base=32, mode="convt"):
        super().__init__(in_ch=3, out_ch=n_classes, base=base, mode=mode)
        # Was: Sequential(Conv1x1(32 -> 3), Sigmoid). Now: raw logits.
        self.head = nn.Conv2d(base, n_classes, kernel_size=1)


def pretrained_segmenter(stem="part2_ae_convt"):
    """The same network, but starting from Part 2's autoencoder weights.

    Part 2 trained this exact encoder and decoder to rebuild photographs, using
    no labels at all. Those weights already know something: what fur looks
    like, where edges are, which textures go together. Nothing about that
    knowledge is specific to reconstruction.

    So instead of starting the segmenter from random numbers, start it from
    there and let the labels only do the last bit of the work. This is
    self-supervised pretraining, and it is the reason anyone cares about
    autoencoders in practice -- the unlabelled photo is free, the trimap a
    human had to paint is not.

    `strict=False` is what makes the transfer work: every encoder and decoder
    key matches, and the head keys do not (the autoencoder's head is
    Sequential(Conv, Sigmoid) with key "head.0.weight", ours is a bare Conv
    with key "head.weight"). So the body loads and the head stays random,
    which is exactly what we want -- the old head predicted colours.

    Expect the benefit to depend on how many labels you have; see --n-train.
    """
    path = weights_path(stem)
    if not os.path.exists(path):
        raise SystemExit(f"no pretrained weights at {path}\n"
                         f"run: python part2a_cnn_autoencoder.py")
    model = SegAutoencoder()
    result = model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    loaded = len(model.state_dict()) - len(result.missing_keys)
    print(f"  loaded {loaded}/{len(model.state_dict())} tensors from {stem}.pt "
          f"(head left random: {result.missing_keys})")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--skip-baseline", action="store_true",
                        help="do not retrain part 1's FCN for comparison")
    parser.add_argument("--pretrained", action="store_true",
                        help="also train a copy initialised from part 2's autoencoder")
    parser.add_argument("--eval", action="store_true",
                        help="skip training: load the checkpoints a previous run "
                             "with the same flags saved, and print the same table")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = get_device()
    print(f"using device: {device}\n")

    # segmentation=True now: the masks we ignored in Part 2 are the target.
    train_ds = load_pet("trainval", n_subset=args.n_train, augment=True)
    test_ds = load_pet("test", n_subset=1000)
    train_loader, test_loader = make_loaders(train_ds, test_ds)

    x, y = next(iter(train_loader))
    print("the shape contract:")
    print(f"  image  {tuple(x.shape)}  {x.dtype}  in [{x.min():.2f}, {x.max():.2f}]")
    print(f"  mask   {tuple(y.shape)}      {y.dtype}  values "
          f"{sorted(torch.unique(y).tolist())}")
    print("  no channel axis on the mask, and no one-hot: CrossEntropyLoss")
    print("  wants the class index at each pixel.\n")

    loss_fn = nn.CrossEntropyLoss()
    rows, titles, table = [], [], []

    contenders = [("encoder-decoder", lambda: SegAutoencoder(), "part3_encdec")]
    if args.pretrained:
        contenders.append(("  + AE pretraining", pretrained_segmenter, "part3_pretrained"))
    if not args.skip_baseline:
        # Part 1 trained this for 8 epochs; retrain it here on this file's
        # recipe so the comparison differs only in architecture.
        contenders.insert(0, ("part1 FCN", fcn, "part3_fcn"))

    for label, build, stem in contenders:
        torch.manual_seed(0)
        # The label budget is part of the run's identity: without it in the
        # name, a --n-train 500 run would overwrite the full-data weights.
        key = f"{stem}_n{args.n_train}" if args.n_train else stem
        if args.eval:
            # Only the architecture is needed -- the weights come from the
            # checkpoint -- so the pretrained row is a plain SegAutoencoder.
            model = SegAutoencoder() if build is pretrained_segmenter else build()
            print(f"=== {label} -- {count_params(model):,} parameters ===")
            load_weights(model, key)
            model.to(device)
        else:
            model = build()
            print(f"=== {label} -- {count_params(model):,} parameters ===")
            with tracked(label.strip(), config=dict(part=3, arch=stem,
                                                    params=count_params(model),
                                                    epochs=args.epochs, lr=1e-3,
                                                    n_train=len(train_ds))) as run:
                history = train(model, train_loader, device, loss_fn,
                                epochs=args.epochs, lr=1e-3, run=run,
                                eval_fn=seg_probe(test_loader))

        acc, iou, miou = evaluate_seg(model, test_loader)
        print_metrics("test", acc, iou, miou)
        print()
        table.append((label, count_params(model), acc, iou, miou))
        if not args.eval:
            save_weights(model, key)
            save_history(key, label.strip(), count_params(model),
                         history, (acc, iou, miou))

        x, y, pred = predict_batch(model, test_loader)
        if not rows:
            rows += [x, colorize_batch(y)]
            titles += ["photo", "truth"]
        rows.append(colorize_batch(pred))
        titles.append(label)

    head = f"{'model':<24} {'params':>10} {'acc':>7} {'mIoU':>7}"
    head += "".join(f"{n:>8}" for n in CLASS_SHORT)
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for label, params, acc, iou, miou in table:
        line = f"{label:<24} {params:>10,} {acc:>7.3f} {miou:>7.3f}"
        line += "".join(f"{v:>8.3f}" for v in iou)
        print(line)
    print("=" * len(head))

    print("\nA learned decoder beats a bilinear stretch by more than anything")
    print("else this week will buy, and the pictures say why: the outlines")
    print("follow the animal instead of the 12x12 grid it was decided on.")
    if args.pretrained:
        print("\nThe pretrained row is the interesting one. With every label")
        print("available it is a wash -- there is nothing an autoencoder knows")
        print("that 3,680 labelled masks will not also teach. Re-run with")
        print("--n-train 500 and it wins: that is when the free, unlabelled")
        print("photographs are worth more than the labels you did not pay for.")
    print("\nNow look at the border column, and then at the outline of the")
    print("animal in the last row. The model knows roughly where the edge is")
    print("and cannot commit to a pixel. It is Part 2's blurry whiskers again")
    print("-- same network, same 12x12 waist, same lost high frequencies --")
    print("except that this time the blur is not a matter of taste. It is")
    print("wrong labels, in the one place a segmentation is actually used.")
    print("\nThe encoder had that detail at 96x96 and we threw it away. Part 4")
    print("stops throwing it away.")

    n_note = f" ({args.n_train} labels)" if args.n_train else ""
    show_grid(rows, titles,
              suptitle="Same network, new head and loss: segmentation through a "
                       f"12x12 waist{n_note}")
    finish(f"part3_n{args.n_train}" if args.n_train else "part3")


if __name__ == "__main__":
    main()
