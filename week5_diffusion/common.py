"""Week 5 - shared plumbing for both parts.

Everything here is the same no matter which network predicts the noise:
the noise schedule, the timestep encoding, loading MNIST, the two samplers,
and device selection. Part 1 (U-Net) and Part 2 (transformer) only differ in
the network, so that is the only thing they define.

WHAT EVERY MODEL THIS WEEK DOES
-------------------------------
    input : a noisy digit x_t   (1 x 32 x 32)
            the noise level t   (an integer, 0 = almost clean, 999 = pure noise)
            the digit label y   (0-9, or 10 = "no label", see GUIDANCE below)
    output: the noise it thinks was added   (1 x 32 x 32)

To generate, start from pure noise and repeatedly ask "what is the noise
here?", remove a little of it, and go one level down. After all the levels,
a digit is left.

GUIDANCE
--------
During training the label is replaced by 10 ("no label") 10% of the time, so
the same network also learns to denoise without being told the digit. At
sampling time we run it both ways and push away from the unlabelled answer:
    eps = eps_nolabel + w * (eps_label - eps_nolabel)
w = 1 is the plain labelled model. w > 1 makes the digit follow the label more
strongly, at the cost of less variety.
"""

import math
import os

import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import make_grid

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")      # the repo's shared data/ folder
CKPT_DIR = os.path.join(HERE, "checkpoints")
RESULTS = os.path.join(HERE, "results")

IMAGE_SIZE = 32          # MNIST is 28 x 28; padded to 32 so it halves cleanly 3 times
T = 1000                 # number of noise levels
NUM_CLASSES = 10
NULL_LABEL = 10          # the 11th label: "no label given"


def weights_path(name):
    """checkpoints/<name>.pt -- the one place weights live."""
    return os.path.join(CKPT_DIR, f"{name}.pt")


def results_dir(name):
    """results/<name>/ -- one folder per part, so runs do not overwrite each other."""
    path = os.path.join(RESULTS, name)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------- noise schedule

class NoiseSchedule:
    """How much noise is at each level t. The DDPM paper's linear schedule.

        beta_t      : noise added at step t        (1e-4 rising to 0.02)
        alpha_t     : 1 - beta_t                   (signal kept at step t)
        alpha_bar_t : alpha_1 * ... * alpha_t      (signal left after t steps)

    alpha_bar goes from ~1 (the clean digit) to ~0 (pure noise). It lets us
    jump straight to any level in one line, with no loop:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    """

    def __init__(self, T=T, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t, noise):
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


def timestep_embedding(t, dim):
    """Turn the integer t into `dim` numbers: sines and cosines at many speeds.

    A single number 0..999 is a poor input for a network. Sines at different
    frequencies give each t a distinct pattern where nearby t look similar.
    Same idea as the position encoding in a Transformer.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


# ---------------------------------------------------------------- data

def load_mnist(device):
    """All 60,000 training digits as one tensor on the device.

    Pixels 0..255 -> [-1, 1] (so the background is -1, the same range as the
    noise). 28 x 28 is padded with background to 32 x 32, so the U-Net can
    halve it three times: 32 -> 16 -> 8 -> 4.
    """
    ds = datasets.MNIST(root=DATA, train=True, download=True)
    x = ds.data.float().div(127.5).sub(1).unsqueeze(1)                # N x 1 x 28 x 28
    pad = (IMAGE_SIZE - 28) // 2
    x = F.pad(x, (pad, pad, pad, pad), value=-1.0)                    # N x 1 x 32 x 32
    return x.to(device), ds.targets.to(device)


def to_pixels(x):
    return (x + 1) / 2                                                # [-1, 1] -> [0, 1]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------- sampling

def randn(shape, generator, device):
    """Noise drawn on the CPU (with an optional seeded generator), then moved.
    Same seed -> same noise -> same digits, on any device."""
    return torch.randn(shape, generator=generator).to(device)


@torch.no_grad()
def predict_noise(model, x, t, y, guidance):
    """The network's noise guess, with guidance (see module docstring)."""
    if guidance == 1.0:
        return model(x, t, y)
    null = torch.full_like(y, NULL_LABEL)
    eps_label, eps_nolabel = model(torch.cat([x, x]), torch.cat([t, t]),
                                   torch.cat([y, null])).chunk(2)
    return eps_nolabel + guidance * (eps_label - eps_nolabel)


@torch.no_grad()
def ddpm_sample(model, sched, labels, guidance=3.0, generator=None, trajectory=None):
    """DDPM: 1000 steps, fresh random noise added back at every step.

        x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps)
                  + sqrt(beta_t) * z
    """
    n, device = len(labels), labels.device
    x = randn((n, 1, IMAGE_SIZE, IMAGE_SIZE), generator, device)
    for t in reversed(range(sched.T)):
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        eps = predict_noise(model, x, tt, labels, guidance)
        a, ab, b = sched.alphas[t], sched.alpha_bars[t], sched.betas[t]
        x = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
        if t > 0:
            z = randn(x.shape, generator, device)
            x = x + b.sqrt() * z
        if trajectory is not None:
            trajectory.append((t, x.clamp(-1, 1).cpu()))
    return x.clamp(-1, 1)


@torch.no_grad()
def ddim_sample(model, sched, labels, steps=50, guidance=3.0, generator=None, trajectory=None):
    """DDIM: same trained model, visits only `steps` of the 1000 levels and
    adds no fresh noise. Each step:
        1. guess the clean digit:  x0 = (x_t - sqrt(1 - ab_t) * eps) / sqrt(ab_t)
        2. clamp x0 to [-1, 1], the valid pixel range, and recompute the
           noise that goes with the clamped x0:
                                   eps = (x_t - sqrt(ab_t) * x0) / sqrt(1 - ab_t)
        3. jump to the next level: x   = sqrt(ab_next) * x0 + sqrt(1 - ab_next) * eps
    Step 2's recompute matters with guidance: guidance exaggerates eps, so the
    raw x0 guess is far out of range at high noise. Clamping x0 but jumping
    with the old eps mixes two answers that disagree; measured on Part 1's
    model at w = 3 this turned every sample into scribbles.
    """
    n, device = len(labels), labels.device
    x = randn((n, 1, IMAGE_SIZE, IMAGE_SIZE), generator, device)
    ts = torch.linspace(sched.T - 1, 0, steps).long().tolist()
    for i, t in enumerate(ts):
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        eps = predict_noise(model, x, tt, labels, guidance)
        ab = sched.alpha_bars[t]
        ab_next = sched.alpha_bars[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
        x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
        if trajectory is not None:
            trajectory.append((t, x.clamp(-1, 1).cpu()))
    return x.clamp(-1, 1)


def sample_grid(model, sched, device, per_digit=8, seed=0):
    """Digits 0..9 (one per row), DDIM 50 steps, fixed starting noise."""
    labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(per_digit)
    gen = torch.Generator().manual_seed(seed)
    x = ddim_sample(model, sched, labels, steps=50, guidance=3.0, generator=gen)
    return make_grid(to_pixels(x).cpu(), nrow=per_digit, padding=2, pad_value=1.0)
