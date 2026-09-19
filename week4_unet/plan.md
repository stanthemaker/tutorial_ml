# Week 4 — Segmentation with U-Net (plan)

Week 3 ended at *image → one of 10 labels*. The whole of week 4 follows from
changing the output: **a label for every pixel**.

That one change forces, in order, every concept in the week:

| the problem | the concept | where |
|---|---|---|
| a classifier's head throws away *where* | `Flatten → Linear` out, `Conv1x1` in — a fully convolutional net | part1 |
| 60% of the pixels are background | mIoU, not pixel accuracy | part1 |
| `Upsample` stretches, it does not sharpen | a learned decoder, `ConvTranspose2d` vs `Upsample+Conv` | part2a |
| detail dies at the waist, not for lack of capacity | the bottleneck, measured by a waist sweep | part2b |
| the output is a class, not a colour | `Conv1x1 → C` logits, `CrossEntropyLoss` on `(N,C,H,W)` | part3 |
| the decoder cannot invent detail it never saw | skip connections, `torch.cat` | part4 |

One session: parts 1–4, with part5 (visualization) as the take-home file.

Dataset for the entire week: **Oxford-IIIT Pet at 96×96**, cached once as uint8
(23 s, then 0.2 s/epoch). Photos, not MNIST — fur and whiskers are pure
high-frequency detail, so what a bottleneck destroys is visible to the eye.
Labels are the shipped **3-class trimap**: `0 = pet, 1 = background,
2 = border`.

## Files

### `common.py`
`get_device`, `count_params`, `load_pet(split, segmentation=)` with the
resize cache and paired hflip augmentation, `make_loaders`, confusion-matrix
`evaluate_seg` (per-class IoU + mIoU), `colorize`, `show_grid`, `DoubleConv`,
and the shared `train` loop.

### `part1_classifier_to_segmenter.py`
Week 3 part 6's classifier, with `Flatten → Linear → Linear` deleted and
`Conv1x1 → Upsample(×8)` in its place. Same trunk, same loss. Establishes:
segmentation is classification per pixel; only the Linear head was
structurally incompatible; pixel accuracy is a trap (constant baseline 0.579 →
mIoU 0.193). Ends on the limit: the decision is made on a 12×12 grid and
`Upsample` has no parameters, so nothing finer than 8 px can exist in the
output — and pooling less is not an option, because pooling is where the
context comes from.

### `part2a_cnn_autoencoder.py`
Replace the stretch with a learned decoder, on the easiest target: no labels,
the photo is its own answer. `ConvTranspose2d` vs `Upsample + Conv2d` (both
run). The model is written out level by level — `enc1/enc2/enc3`,
`bottleneck`, `up3/dec3 … up1/dec1`, `head` — using the same attribute names
part4's `UNet` uses, so the two diagrams can be read side by side.

### `part2b_waist_sweep.py`
The control experiment for the claim part2a ends on. Trains 2/3/4 poolings to
prove the remaining blur is the bottleneck and not the upsampler or the
parameter count — the smallest waist has the *most* parameters and the worst
reconstruction. This is the one place a depth knob is needed, so it is the one
place the levels are a loop (`VariableWaistAutoencoder`); `depth=3` is
asserted at runtime to be part2a's network, weight for weight.

### `part3_encoder_decoder.py`
Point that decoder at the real task. Two lines change: `Conv1x1(base → 3)` with
**no Sigmoid** (raw logits), and `CrossEntropyLoss` on `(N,C,H,W)` against
`(N,H,W)` int64. Segmentation data gotchas walked into deliberately: NEAREST
resize for masks, `long` dtype, trimap 1/2/3 → 0/1/2, augmentation applied to
image *and* mask. Beats part1 clearly — and every boundary is mush.

### `part4_unet.py`
The point of the week: `torch.cat([up, skip], dim=1)`, three times. The three
gotchas in the order they break student code (cat vs add; decoder input
channels double; padding=1 keeps the shapes aligned). `use_skips=False` zeroes
the skip half of every cat, keeping the parameter count *identical*, so the gain is
provably the skips — and it lands on the border class.

### `part5_unet_visualize.py`
Take-home. The feature maps handed to the decoder at each skip, each model's
wrong pixels in red, and accuracy plotted against distance to the nearest
boundary (a six-line max-pool distance transform, no scipy).

### `slides/build_slides.py`
Builds `slides/week4_unet.pptx`. Follows `slides/genslide_guide.txt`: a shape
diagram for every architecture, and for every concept the problem → the idea →
the maths in LaTeX → the measured evidence. Nothing is typed in by hand that
can be computed — metrics come from evaluating `checkpoints/*.pt` at build
time, shapes from forward hooks, parameter counts from the models. The two
tables that cannot (part 2's waist sweep, the 500-label comparison) are marked
in the source with the command that reproduces them.

## Layout (same as week3_cnn)
```
week4_unet/
  part1..part5, common.py, compare_parts.py
  checkpoints/   trained weights (*.pt)
  results/       figures, training_curves.json
  slides/        build_slides.py, week4_unet.pptx, genslide_guide.txt
  data -> ../data
```

## Conventions (unchanged from week 3)
* long narrative module docstring with a MEASURED RESULTS table (M1 Pro,
  wall-clock included) at the top of every file. **Every number in those tables
  is copied from a real run.**
* argparse flags for every ablation; scripts runnable bare with sane defaults.
* `WEEK4_SAVE_FIGS=1` writes the figures to `results/` instead of showing them.
* every run logs per-epoch curves to trackio (`trackio show --project
  "week4-unet"`); `WEEK4_NO_TRACKIO=1` turns it off, and the parts run without
  trackio installed. The instruments are week-4 specific: `test/mIoU` and
  `test/iou_border` rather than accuracy, because watching accuracy barely
  move while the border climbs is itself the lesson.
* `week4_todo` branch blanks: the `DoubleConv` body, the decoder channel
  arithmetic, the `torch.cat` lines, the CE-on-4-D call, and `mean_iou`.

## Dropped
* The ResNet material (old `week4_resnet/`) is out of the course; the `cat` vs
  `add` contrast in part4 is where residual connections get mentioned.
* The old `week4_autoencoder/` MLP-autoencoder ramp is replaced by part2a, which
  keeps the autoencoder only as long as it takes to build a decoder.
* Models A (1×1 pixel classifier) and B (full-resolution CNN) from the first
  draft of the demo: degenerate strawmen. The context/resolution tension is
  argued from the real model instead.

## Optional appendix if time allows
Same U-Net, target = the clean image (denoise / inpaint), ~10 lines changed —
the bridge toward diffusion.
