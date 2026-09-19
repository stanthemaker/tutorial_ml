# Week 5 — Diffusion (DDPM) (plan)

Week 4 ended with a U-Net that maps a photo to a *label map*. Week 5 keeps
that network and changes what goes in and what comes out: **noisy photo in,
noise out**. Iterate that denoiser from pure noise and it becomes a
generator.

The only new component in the week is the **timestep input**. Everything else
is a head swap and a loss swap (the same kind of change as week 4 part 3) or
a loop wrapped around a network that already exists.

| the problem | the concept | where |
|---|---|---|
| can the U-Net remove noise at all? | same `UNet`, head `Conv1x1 → 3`, `MSELoss`, input `x + σε` | part1 |
| at large σ one denoising step returns a blur | MSE regresses to the **mean** of every plausible image | part1 |
| one network for every noise level | noise schedule, `q_sample`, timestep embedding added in `DoubleConv` | part2 |
| predict the clean image, or the noise? | x₀- vs ε-prediction, same network, one flag | part2 |
| a denoiser is not yet a generator | DDPM ancestral sampling: start at pure noise, take 1000 small steps | part3 |
| is it generating, or copying the training set? | nearest-neighbour check against the training images | part3 |
| 1000 steps are slow | DDIM: same checkpoint, no retraining, fewer steps | part4 |
| "draw me a cat" | class conditioning + classifier-free guidance | part5 |

Planned session: parts 1–3. Part 4 is short enough to fit if time allows.
Part 5 and part 6 (visualization) are take-home. **Confirm this split against
measured training times** (see step 0).

Dataset for the whole week: **Oxford-IIIT Pet at 48×48**, downsampled from
week 4's 96×96 uint8 cache (`F.interpolate(..., mode="area")`, cached as
`pet_48_{split}.pt`). It's the same photos as week 4, so the students already
know what a good sample should look like. 48 = 2³·6, so week 4's three
poolings still work: 48 → 24 → 12 → 6. There are no masks this week. Part 5
uses the species label (`target_types="binary-category"`, cat / dog), which
torchvision 0.27 ships.

## Step 0 — feasibility spike (before writing any part)

A generator trained on 3,680 small photos on a laptop may produce blobs. That
would sink the week, so check it first:

1. Write the smallest end-to-end version: `q_sample`, the time-conditioned
   U-Net, the ε-MSE loop and the ancestral sampler, as a throwaway script.
2. Train on trainval at 48×48 with hflip. Time s/epoch on the M1 Pro and look
   at 64 samples after 10 / 30 / 100 epochs.
3. Pass: a student would call most samples "a cat" or "a dog" within a
   training budget that fits the session (or can be precomputed and loaded
   from `checkpoints/`).
4. If it fails, try these in order, measuring each:
   - base width 32 → 64
   - add the test split to the training pool (generation needs no labels)
   - GroupNorm in place of BatchNorm (see the BatchNorm note below)
   - 32×32
   - as a last resort, fall back to MNIST, graded by week 3's `part2.pt`.

The spike also decides T (1000 is the default) and the schedule (linear β,
as in the DDPM paper, or cosine). Keep linear unless the spike shows it
wasting the high-t steps at this resolution.

## Measuring sample quality

FID needs Inception weights and tens of thousands of samples, so we build our
own instruments and state their limits in the docstrings:

* **FD-ours**: the Fréchet distance between Gaussians fitted to features of
  real test images and features of generated images. The feature extractor
  is **week 4's trained U-Net encoder** (`part4_unet_skips_on.pt`, bottleneck
  output, global-average-pooled, input upsampled 48 → 96). That network has
  seen thousands of pet photos, so its features know what a pet looks like.
  Before any model is scored, the metric must sort the sanity cases correctly:
  - real train vs real test (floor)
  - blurred real images
  - part1's one-shot denoise
  - pure noise (ceiling)
  
  If it doesn't, change the metric before using it. Numbers are not
  comparable to published FID, and the docstrings will say so.
* **Nearest neighbour**: for each sample, the closest training image in pixel
  space, shown side by side. With only 3,680 training images, memorization is
  a real risk and worth showing either way. Report the ratio of sample→train
  distance to test→train distance.
* **Cat/dog judge** (part 5 only): a small CNN from week 3, trained on real
  48×48 Pet images, used to check that "generate a cat" yields cats. Its test
  accuracy on real images is the ceiling, and goes in the table.

## Files

### `common.py`
Reuses week 4's `get_device`, `count_params`, `save_history`, `tracked` and
`DoubleConv` (imported, not copied, if the import path stays clean). New:
`load_pet48(split, with_species=False)`, `NoiseSchedule` (β, α, ᾱ buffers
plus `q_sample(x0, t, eps)`), `timestep_embedding(t, dim)` (sinusoid), the
feature extractor and `frechet_distance`, `nearest_train`, `show_grid`.
Images scaled to [-1, 1], not [0, 1]. The noise is zero-mean, so the data
should be too. Say this out loud in the docstring, because it's the first
thing that breaks when a student copies week 4's loader.

### `part1_unet_denoiser.py`
The diff from week 4 part 4, shown literally in the docstring:

    head: Conv1x1(base → 3 classes)  →  Conv1x1(base → 3 channels)
    loss: CrossEntropyLoss(logits, mask)  →  MSELoss(pred, x0)
    input: photo  →  photo + σ·ε

Train one U-Net per σ ∈ {0.1, 0.3, 1.0} (predicting x₀, the target week 4's
autoencoder already knew). PSNR per σ. At σ = 0.1 it's a denoiser. At σ = 1.0
the output is a smooth brown pet-shaped average. That average is the lesson:
the MSE-optimal answer to "what's under this noise?" is the *mean* of every
image that could be under it. At high noise that means a blur, and no amount
of capacity changes it. So what's needed is a different procedure, not a
bigger network.

### `part2_timestep_conditioning.py`
One network for all 1000 noise levels. Two ideas, each as a small diff:

* **the schedule**: `x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε`. Show a row of one photo at
  t = 0, 100, …, 1000.
* **the timestep input**: `timestep_embedding(t) → Linear → SiLU → Linear`,
  then in `DoubleConv` after the first conv:
  `h = h + self.t_proj(temb)[:, :, None, None]`. That one line is the new
  component of the week.

Loss: the DDPM "simple" loss, `MSE(ε̂, ε)` at uniformly random t.
Ablations, as flags on the same script:
* `--no-time` zeroes the embedding, with an identical parameter count (the
  same idea as week 4's `use_skips=False`). A denoiser can partly estimate the
  noise level from the image itself, so this gap may turn out small. Report
  whatever the run shows, per-t.
* `--predict x0|eps` uses the same network, with the target and the formula
  that recovers x̂₀ swapped. Plot loss per t for both.

Plot: test MSE against t. Saves the checkpoint parts 3–6 load.

### `part3_sampling.py`
No training. Loads part2's checkpoint. The DDPM ancestral sampler, about 15
lines, with the formula from the paper next to each line:

    x = randn
    for t = T..1:  x = (x − β_t/√(1−ᾱ_t)·ε̂(x,t)) / √α_t + σ_t·z

The week's central figure: **the same starting noise**, denoised in one
shot (part1's answer: the blur) and in 1000 small steps (a pet). Each small
step only has to guess the mean at a noise level where the mean is still
sharp. The remaining randomness comes from `σ_t·z`, re-injected at every
step. Then: FD-ours for both, with real-vs-real as the reference, and the
nearest-neighbour grid.

### `part4_ddim.py`
Same checkpoint again. The DDIM update (deterministic, η = 0) on a sub-sequence
of timesteps. Sweep 1000 / 250 / 100 / 50 / 20 / 10 steps: FD-ours and
wall-clock per sample. Bonus, because DDIM is deterministic: interpolate
between two noise vectors (slerp) and watch one pet morph into another.

### `part5_guidance.py` (take-home)
Condition on species. Diff from part2: add a label embedding (3 entries:
cat, dog, *none*) to the time embedding. Drop the label to *none* with
probability 0.1 during training. Sample with
`ε̂ = ε̂(x,∅) + w·(ε̂(x,c) − ε̂(x,∅))`. Sweep w ∈ {0, 1, 3, 5, 8}:
- judge accuracy ("asked for a cat, got a cat")
- FD-ours (diversity/quality)

Expected tradeoff: higher w makes them more on-label and less varied. The run
decides whether that shows up here.

### `part6_diffusion_visualize.py` (take-home)
What the sampler is doing:
- the trajectory x_t at 10 checkpoints
- the network's current x̂₀ guess at each (the blur sharpening as t falls,
  which ties back to part1)
- which frequencies get decided when: low-pass vs high-pass energy of x̂₀
  against t.

### `compare_parts.py`
One figure with rows from the same seed: one-shot denoise (part1), DDPM 1000,
DDIM 50, DDIM 10, guided cat, guided dog. FD-ours beside each row. Loads
checkpoints and doesn't retrain.

### `slides/build_slides.py`
Builds `slides/week5_diffusion.pptx` per `slides/genslide_guide.txt` (copied
from week 4). Every metric is measured at build time from `checkpoints/*.pt`
and `results/*.json`. Every shape comes from a real forward pass.

## Open technical questions (the spike or the runs decide)

* **BatchNorm vs GroupNorm.** Week 4's `DoubleConv` uses BatchNorm, and a
  diffusion batch mixes all noise levels, so batch statistics blend t = 5
  with t = 995. Keep BatchNorm if it trains. If it doesn't, the swap becomes
  a measured, same-parameter-count ablation with its own paragraph, not a
  silent change.
* **EMA of weights.** It's standard in DDPM and usually matters a lot for
  sample quality. It's five lines. Include it only if the spike shows a
  visible difference, and then show that difference.
* **Where the time embedding enters.** It enters every `DoubleConv`, encoder
  and decoder. Try the bottleneck only in the spike, as a sanity check, not
  as a part.

## Layout (same as week 4)
```
week5_diffusion/
  part1..part6, common.py, compare_parts.py, plan.md
  checkpoints/   trained weights (*.pt)
  results/       figures, training_curves.json, measured.json
  slides/        build_slides.py, week5_diffusion.pptx, genslide_guide.txt
  data -> ../data
```

## Conventions (unchanged from week 4)
* long narrative module docstring with a literal before/after diff against
  the previous part, and a MEASURED RESULTS table (M1 Pro, wall-clock
  included). **Every number is copied from a real run.** When a run
  contradicts the narrative, rewrite the narrative.
* no strawman models. The one-shot denoiser in part3 is part1's real,
  trained model answering the natural question "why not just denoise once?"
* argparse flags for every ablation. Scripts run bare with sane defaults.
* `WEEK5_SAVE_FIGS=1` writes figures to `results/`. trackio project
  `week5-diffusion`, logging `train/loss`, per-t-bucket test MSE, and
  FD-ours every N epochs. `WEEK5_NO_TRACKIO=1` turns it off.
* `week5_todo` branch blanks: `q_sample`, `timestep_embedding`, the
  `t_proj` line in `DoubleConv`, one DDPM sampling step, and the CFG line.

## Dropped / out of scope
* VAEs and latent diffusion: out of scope for 48×48 pixel-space models.
  One slide mentions that Stable Diffusion runs this same loop in an
  autoencoder's latent space (week 4 part 2a's bottleneck, as it happens).
* Flow matching: not in the main path. At most one closing slide showing it
  as the same network with a different `q_sample` and target.
* Attention blocks in the U-Net: not unless the spike shows the plain
  conv U-Net can't produce coherent samples at 48×48.
