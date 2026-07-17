# tutorial_ml

A hands-on PyTorch tutorial that builds up from gradient descent by hand to a
multi-layer perceptron (MLP) and then a convolutional neural network (CNN).
Each script is self-contained and meant to be run and read top-to-bottom.

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
- [part4_filter_visualization.py](week3_cnn/part4_filter_visualization.py) — visualize the first-layer conv filters the CNN *learned*, echoing Week 2's PCA verification.

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
