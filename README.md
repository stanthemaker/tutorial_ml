# tutorial_ml

A hands-on PyTorch tutorial that builds up from gradient descent by hand to a
multi-layer perceptron (MLP), then a convolutional neural network (CNN), and
finally a U-Net that labels every pixel of a photograph. Each script is
self-contained and meant to be run and read top-to-bottom.

## Contents

### Week 1 — Introduction ([week1_introduction/](week1_introduction/))
Linear regression, three ways:
- [part1_manual_gradient.py](week1_introduction/part1_manual_gradient.py) — gradient descent by hand (no autograd).
- [part2_autograd.py](week1_introduction/part2_autograd.py) — same loop, gradients from `loss.backward()`.
- [part3_nn_linear.py](week1_introduction/part3_nn_linear.py) — the idiomatic PyTorch loop (`nn.Linear` + optimizer). **This is the template** reused everywhere after.

### Week 2 — MLP ([week2_mlp/](week2_mlp/))
From a straight line to a curve, then on to real images:
- [demo_linear_limits.py](week2_mlp/demo_linear_limits.py) — why a linear model fails on curvy data (motivation for the MLP).
- [part1_mlp_regression.py](week2_mlp/part1_mlp_regression.py) — fit a sine wave with `Linear → ReLU → Linear`.
- [part2_mlp_classification.py](week2_mlp/part2_mlp_classification.py) — the same MLP doing classification on two half-moons.
- [part3_mlp_mnist.py](week2_mlp/part3_mlp_mnist.py) — the exact same recipe classifying MNIST handwritten digits (784 inputs, 10 classes).
- [part4_mnist_pca_verify.py](week2_mlp/part4_mnist_pca_verify.py) — use PCA to *see* why MNIST is nearly linearly separable.

### Week 3 — CNN ([week3_cnn/](week3_cnn/))
From a flattened MLP to convolutions that keep the image 2D:
- [demo_mlp_limits.py](week3_cnn/demo_mlp_limits.py) — an MLP trained on centred MNIST collapses when the test digits are shifted a few pixels, and its first-layer weights reveal position-locked templates (motivation for the CNN).
- [demo_mlp_limits_cifar.py](week3_cnn/demo_mlp_limits_cifar.py) — the same MLP plateaus near ~48% on CIFAR-10 colour photos despite 1.5M parameters; its learned first-layer weights fold back into blurry 32×32 colour templates that show *why*.
- [part1_conv_basics.py](week3_cnn/part1_conv_basics.py) — `nn.Conv2d` / `nn.MaxPool2d`, the output-size formula, and receptive field, shown with a hand-set edge detector on a synthetic shape.
- [part2_cnn_mnist.py](week3_cnn/part2_cnn_mnist.py) — the same MNIST task with `Conv → ReLU → Pool` instead of the MLP: similar accuracy with ~20× fewer parameters.
- [part3_cnn_cifar10.py](week3_cnn/part3_cnn_cifar10.py) — CIFAR-10 colour photos, where the Week 2 MLP/PCA tricks fail and the CNN's spatial features pay off.
- [part4_cnn_mnist_visualize.py](week3_cnn/part4_cnn_mnist_visualize.py) — visualize the first-layer conv filters the CNN *learned* on MNIST, plus their response on a real digit, echoing Week 2's PCA verification.
- [part5_cnn_cifar10_visualize.py](week3_cnn/part5_cnn_cifar10_visualize.py) — how filters change with depth: kernels and feature maps at three depths of the CIFAR-10 CNN, and what "RGB" stops meaning after layer 1.
- [part6_cnn_overfitting.py](week3_cnn/part6_cnn_overfitting.py) — part3 hits 100% train / 68% test; a held-out validation split makes the gap visible, and `--augment` / `--regularize` show what actually closes it.
- Every training script takes `--eval`: skip training, load the checkpoint a previous run saved to `checkpoints/`, and print the same test numbers. Parts 7 and 8 also reread `results/<run>.json`, so their tables and curves come back in full; their sweeps (`--data-curve`, `--lr-sweep`, `--depth-sweep`) save no weights and so have no `--eval`.

### Week 4 — Segmentation with U-Net ([week4_unet/](week4_unet/))
From one label per image to one label per pixel, on Oxford-IIIT Pet photographs:
- [part1_classifier_to_segmenter.py](week4_unet/part1_classifier_to_segmenter.py) — Week 3's classifier with `Flatten → Linear` deleted and `Conv1x1 → Upsample(×8)` in its place. Same trunk, same `CrossEntropyLoss`, now applied to 9,216 pixels instead of one image. Deleting two layers gets most of a working segmenter — and the output has no structure finer than 8 px, because the decision was made on a 12×12 grid. Retires pixel accuracy (predicting "background" everywhere scores 0.579) in favour of mean IoU.
- [part2a_cnn_autoencoder.py](week4_unet/part2a_cnn_autoencoder.py) — replace the fixed stretch with a *learned* decoder, on the easiest target there is: no labels, the photo is its own answer. `ConvTranspose2d` vs `Upsample + Conv2d`.
- [part2b_waist_sweep.py](week4_unet/part2b_waist_sweep.py) — the control experiment: 2/3/4 poolings prove the remaining blur is the bottleneck rather than the parameter count — a 6×6 waist with 16.6× the weights of a 24×24 one reconstructs 7 dB worse.
- [part3_encoder_decoder.py](week4_unet/part3_encoder_decoder.py) — aim that decoder at the real task. Two lines change: a 3-channel head with no Sigmoid, and `CrossEntropyLoss` on `(N, C, H, W)` logits. Worth +0.225 mIoU over part 1 — the biggest single jump of the week — and the boundary is still the worst of the three classes. `--pretrained` reuses part 2's autoencoder weights: a wash with all 3,680 labels, worth +0.040 mIoU with only 500, which is the honest case for self-supervised pretraining.
- [part4_unet.py](week4_unet/part4_unet.py) — `torch.cat([up, skip], dim=1)`, three times. one run trains the identical network twice, once with the skips zeroed, at *exactly* the same parameter count, so the +0.033 mIoU is provably the skips and not the capacity — and the skipless U-Net, which is the larger model, scores *below* part 3's plain encoder-decoder.
- [part5_unet_visualize.py](week4_unet/part5_unet_visualize.py) — what the skips actually carry: the feature maps handed to the decoder, each model's wrong pixels in red, and accuracy against distance to the nearest boundary — the gain is +0.048 within 1px of an edge and exactly zero 16px away, which is the mechanism rather than the claim.
- [compare_parts.py](week4_unet/compare_parts.py) — the summary slide: every model the week produced, on the same test images, with mean IoU and border-class IoU beside each row. Loads the saved checkpoints rather than retraining.
- [slides/build_slides.py](week4_unet/slides/build_slides.py) — builds the session deck, `slides/week4_unet.pptx`. Every metric on a slide is measured at build time by loading `checkpoints/*.pt`, every tensor shape in the architecture diagrams comes from a real forward pass, and the training curves are read from `results/training_curves.json` — so the deck cannot drift away from the code.
- Every run logs per-epoch curves to trackio: `trackio show --project "week4-unet"`. The instruments are mean IoU and border-class IoU rather than accuracy; `WEEK4_NO_TRACKIO=1` switches the logging off.
- Every training script (parts 1–4, including 2b) takes `--eval`: skip training, load `checkpoints/<run>.pt` with the same flags you trained with, and print the same table. Loading is strict, so a checkpoint from an older version of a model fails loudly instead of running half-random.

## Setup

Create and activate a conda environment (Python 3.12):

```bash
conda create -n tutorial_ml python=3.12 -y
conda activate tutorial_ml
```

Confirm the environment's Python and pip are the ones being used:

```bash
which python
which pip
```

Both paths should point inside the `tutorial_ml` environment (e.g.
`.../envs/tutorial_ml/bin/python`). If they don't, re-run `conda activate tutorial_ml`.

Install the dependencies:

```bash
pip install -r week1_introduction/requirements.txt
pip install -r week2_mlp/requirements.txt
pip install -r week3_cnn/requirements.txt
pip install -r week4_unet/requirements.txt
```

## Running

Each file is a standalone script. For example:

```bash
python week1_introduction/part1_manual_gradient.py
python week2_mlp/part1_mlp_regression.py
```

## Completing the TODOs (students)

The `main` branch holds the **completed** scripts. To work through the
exercises yourself, switch to that week's `week<#>_todo` branch, where the key
lines are blanked out and marked with `# TODO` comments for you to fill in.
Replace `<#>` with the week number — e.g. `week2_todo` for Week 2:

```bash
git checkout week2_todo
```

Fill in the TODOs in that week's scripts (e.g. `week2_mlp/`), then run them to
check your work:

```bash
python week2_mlp/part1_mlp_regression.py
```

When you want to compare against the reference solution, switch back to `main`
(commit or stash your changes first so they aren't lost):

```bash
git checkout main
```

To see the answer for a single file without leaving your branch, use (again
substituting the week number for `<#>`):

```bash
git diff week2_todo main -- week2_mlp/part1_mlp_regression.py
```
