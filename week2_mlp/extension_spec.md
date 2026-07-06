# Week 2 Extension: From `make_moons` MLP Classification to MNIST

## Context

This extends the existing Week 2 lesson files:
- `demo_linear_limits.py` — shows a linear model failing to fit a sine wave (regression).
- `part1_mlp_regression.py` — adds ReLU + hidden layer to fit the sine wave.
- `part2_mlp_classification.py` — same MLP idea applied to 2D classification on `make_moons`, with a decision-boundary plot.

The goal now is to bridge from **2D toy classification** (`part2`) to **MNIST digit classification (784-dim input, 10 classes)**, while keeping the exact same narrative style: minimal code changes, same training loop, same loss function pattern, just a change in data/architecture.

**Narrative order (important — this drives the file structure and comments):**

1. **First** show the MLP working on MNIST (`part3_mlp_mnist.py`), making the point that despite the input going from 2 numbers (moons) to 784 numbers (MNIST), the *underlying problem is identical*: find a boundary/hyperplane in a multi-dimensional space that separates the classes. Architecture and training loop are structurally unchanged from `part2_mlp_classification.py`.
2. **Then** run an experiment removing the ReLU (`part3b_mnist_remove_relu.py` — the "surprise" file) — collapsing the two Linear layers into an effectively linear model — and observe that accuracy barely drops (~90-92% vs ~97-98%). This is a deliberate cliffhanger: contrast with `demo_linear_limits.py`, where removing non-linearity was catastrophic on the sine wave, but here it barely matters.
3. **Finally** explain *why* via PCA (`part3c_mnist_pca_verify.py`, or a PCA section appended to the same script) — showing that MNIST digit classes are already nearly linearly separable in raw 784-dim pixel space, and that the hidden-layer representation (from the ReLU version) separates them even more cleanly. This validates the experiment visually.

Please follow the existing code style closely (docstring at top explaining the "big idea", `main()` function, comments that teach rather than just describe, `if __name__ == "__main__":` block).

Please create the following files (can be split as below, or combined into fewer files if that reads better — but keep the narrative order above intact via section headers/comments).

---

## File 1: `part3_mlp_mnist.py`

**Purpose:** Direct sequel to `part2_mlp_classification.py`. Same recipe (Linear → ReLU → Linear, same training loop, same `CrossEntropyLoss`), just applied to MNIST. The point of this file is: **the problem is structurally identical to part2** — just find a boundary in a higher-dimensional space. Do NOT yet discuss linear separability here — save that surprise for File 2.

**Requirements:**

1. **Data loading**: Load MNIST via `torchvision.datasets.MNIST` (download to `./data`, `train=True`/`train=False` splits, `transforms.ToTensor()`). Flatten each 28x28 image to a 784-dim vector (`.view(-1, 784)`). Use `DataLoader` with a reasonable batch size (e.g. 128). To keep runtime reasonable on CPU, it's fine to subsample the training set (e.g. 10,000-20,000 images) — note this tradeoff in a comment.

2. **Model**:
   ```python
   mlp = nn.Sequential(
       nn.Linear(784, 128),
       nn.ReLU(),
       nn.Linear(128, 10),
   )
   ```
   Comment should explicitly say: "same shape as part2_mlp_classification.py's `clf`, just wider (784 in, 128 hidden, 10 classes out). The problem is identical: find a boundary in a multi-dimensional space that separates the classes — the space just has 784 dimensions instead of 2."

3. **Training loop**: same 4-step pattern as `part2` (`optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`), iterate over `DataLoader` batches, track average loss per epoch in a `losses` list. Train ~5-10 epochs (enough to reach ~97%+ test accuracy).

4. **Evaluation**: compute test accuracy the same way as `part2` (`argmax` + mean of correct predictions), print it, e.g.:
   `print(f"MLP (with ReLU) test accuracy: {acc:.2%}")`

5. **Loss curve plot**: reuse `part2`'s `plot_results`-style pattern — a simple training loss plot is enough here (no decision boundary plot yet, since 784D can't be drawn directly — flag this explicitly in a comment/docstring as a teaser: "we'll come back to *seeing* this space later, using PCA").

6. Save the trained model's state (or keep the script structured so File 3's PCA code can reuse this trained model — either via a shared import, or simply retrain inside File 3; keep it simple and follow whatever is easiest to run standalone per file, consistent with how `part1`/`part2`/`demo` are each independently runnable).

---

## File 2: `part3b_mnist_remove_relu.py`

**Purpose:** The cliffhanger experiment. Remove the ReLU, collapse to an effectively linear model, and show accuracy barely drops. This sets up the "why" that PCA will answer in File 3.

**Requirements:**

1. Same data loading as File 1.

2. **Model — no ReLU**:
   ```python
   linear_stack = nn.Sequential(
       nn.Linear(784, 128),
       nn.Linear(128, 10),   # <- ReLU removed
   )
   ```
   Comment must explicitly teach: "Two stacked Linear layers with no activation between them is mathematically still just one linear function (W2(W1x + b1) + b2 = W_combined x + b_combined). So this model has the same *effective* capacity as a single `nn.Linear(784, 10)` — despite having two layers on paper."

3. Also optionally include (for a clean single-layer comparison) a plain `nn.Linear(784, 10)` baseline in the same file or as a clearly separated function, to make the "it's still just linear" point very concrete.

4. **Training loop**: identical 4-step pattern, same epochs/hyperparameters as File 1 for a fair comparison.

5. **Evaluation and framing**: print test accuracy, and explicitly print/comment the contrast:
   ```python
   print(f"Linear model (ReLU removed) test accuracy: {acc:.2%}")
   # Compare: part3_mlp_mnist.py (with ReLU) got ~97-98%.
   # This model, with the non-linearity removed, still gets ~90-92%.
   # Contrast with demo_linear_limits.py: removing non-linearity there was
   # catastrophic (a straight line cannot fit a sine wave at all).
   # Here it barely matters. Why? -> see part3c (PCA analysis).
   ```

6. Keep this file short and focused — it's an experiment + a printed result + a teaching comment, not a heavy visualization file (visualization is File 3's job).

---

## File 3: `part3c_mnist_pca_verify.py`

**Purpose:** Explain *why* removing ReLU barely hurt: MNIST classes are already nearly linearly separable in raw pixel space. Verify this visually with PCA, and show the ReLU/hidden-layer representation separates classes even more cleanly.

**Requirements:**

1. **Concept intro (in docstring/comments)**: 
   - Definition of linearly separable: a dataset is (approximately) linearly separable if a hyperplane `Wx + b` can (nearly) correctly separate the classes just by score ranking.
   - Explain intuitively why higher-dimensional spaces make linear separability easier to achieve: 784 free parameters (weights) give far more flexibility to carve up the space than 2 free parameters (a 2D line's slope + intercept), unlike `make_moons`, whose interleaved shape defeats any straight line regardless of how many dimensions... (i.e., contrast MNIST's near-linear-separability with moons' genuine non-linear structure).

2. **Data**: Load MNIST test set (or reuse a subset, e.g. 2000 samples for speed), flatten to 784-dim.

3. **Models needed**:
   - Either retrain quickly inside this script, or (preferred if simple) import/reuse the trained MLP from File 1 (with ReLU) to extract hidden-layer activations. Keep it simple — retraining a small MLP for a couple epochs inside this script is fine if it keeps each file independently runnable, consistent with the existing files' style.

4. **PCA visualization — two side-by-side scatter plots** using `sklearn.decomposition.PCA(n_components=2)`:
   - **Left subplot**: PCA projection of the *raw flattened pixels* (784-dim → 2D) on the sample, points colored by true digit label (`cmap="tab10"`). Title: "原始像素空間 (PCA)" / "Raw pixel space (PCA)".
   - **Right subplot**: PCA projection of the *hidden layer activations* — run the same sample through the trained MLP's `mlp[:2]` (i.e., `Linear(784,128)` + `ReLU()`, stopping before the final classification layer) under `torch.no_grad()`, get 128-dim hidden representations, then PCA those to 2D. Same coloring. Title: "Hidden layer 表示法 (PCA)" / "Hidden layer representation (PCA)".
   - Add a shared legend/colorbar mapping colors to digit labels 0-9.
   - `fig.suptitle`: something tying back to part2's decision boundary plot, e.g. "Same idea as the moons decision boundary — just projected down to 2D since we can't draw 784 dimensions directly."

5. **Key teaching comment near the plotting code**: 
   - "In part2, we could literally draw the decision boundary because the input was 2D. MNIST is 784-dim, so we can't draw that directly — but PCA lets us compress it down to 2D just to *look* at how well-separated the classes are."
   - "Notice the left plot already shows fairly separated clusters, even though it's just raw pixels with no learning involved — this is the visual evidence for why the ReLU-free model in part3b still got ~90%+ accuracy: the classes were already close to linearly separable."
   - "The right plot (hidden layer) shows tighter, more separated clusters — this is where the extra ~6-8 percentage points from part3_mlp_mnist.py's ReLU come from: the non-linearity cleans up the remaining overlap (e.g. between 4/9 or 3/5/8)."

6. **Style consistency**: top-of-file docstring stating the big idea, `def main():` entry point, `if __name__ == "__main__(): main()"` at bottom, comments that teach.

---

## Summary of the full narrative (for your reference when writing docstrings/comments across all 3 files)

1. **File 1** — MLP on MNIST works great (~97-98%). Same architecture pattern as part2's moons classifier, just higher-dimensional input. The problem is identical: cut a boundary through a multi-dimensional space.
2. **File 2** — Surprise: remove the ReLU (collapse to a linear model) and accuracy barely drops (~90-92%). Contrast sharply with `demo_linear_limits.py`, where removing non-linearity was catastrophic on the sine wave.
3. **File 3** — Explain why: MNIST digit classes are already nearly linearly separable in raw 784-dim pixel space (high-dimensional spaces make this much easier to achieve than in 2D, unlike moons). Verify visually with PCA: raw pixel space already shows decent class separation; the ReLU-based hidden layer representation shows even tighter, cleaner separation — accounting for the remaining accuracy gap.

Please implement all files with runnable code, and keep matplotlib usage consistent with the existing files (simple `plt.show()`, no unnecessary styling). Keep total runtime per file reasonable on CPU (a few minutes at most).
