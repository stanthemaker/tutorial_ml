"""Week 5 Part 1 - A label-conditioned diffusion model on MNIST, with a U-Net.

    python part1_unet_diffusion.py        # 20 epochs, ~96 s each on an M1 Pro (~32 min)
    trackio show --project week5-diffusion

ONE TRAINING STEP
-----------------
    x0    = a batch of real digits, pixels in [-1, 1]
    t     = a random noise level per image, 0..999
    noise = random Gaussian noise, same shape as x0
    x_t   = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    loss  = mean( (model(x_t, t, label) - noise)^2 )

That is the whole algorithm. The model never sees a full 1000-step chain
during training: each image gets one random level, and the formula above
creates that level directly.

THE NETWORK
-----------
Week 4's U-Net: encoder 32 -> 16 -> 8 -> 4, decoder back to 32, skip
connections (torch.cat) between matching sizes. Two changes:

  * 1 channel in and 1 channel out (a grayscale image, not RGB -> class scores).
  * Inside every block, one extra line adds a vector that encodes t and y:
        h = h + self.emb(emb)[:, :, None, None]
    The same number is added at every pixel of a channel: the noise level and
    the label describe the whole image, not one pixel.

No attention. Attention lets every pixel look at every other pixel. On a
32 x 32 digit the 4 x 4 bottleneck already sees the whole image through plain
convolutions, so there is nothing far away left to connect.

TWO EXTRAS, both a few lines
----------------------------
  * Label dropout: 10% of labels are replaced by "no label" (10). This is
    what makes guidance possible at sampling time (see common.py).
  * EMA: a second copy of the weights that is a slowly moving average of the
    trained weights. It changes less from step to step, so its samples are
    cleaner. The checkpoint stores the EMA copy; that is what sample.py uses.

WHAT GETS LOGGED TO TRACKIO
---------------------------
  * loss every 50 steps and averaged per epoch
  * after every epoch: a grid of generated digits, rows 0..9, always from the
    same starting noise -- so you can watch the same noise turn into digits
    as training goes on.

OUTPUT
------
  checkpoints/part1_unet.pt              overwritten after every epoch
  results/part1_unet/train_log.json      per-epoch loss and seconds (for the slides)
  results/part1_unet/epoch_XX.png        the per-epoch sample grid

LOAD THE TRAINED MODEL AND GENERATE
-----------------------------------
The checkpoint is a dict: "model" (the EMA weights), "config" (this script's
arguments, so the network can be rebuilt at the right size) and "epoch".

From the command line, sample.py does all of the below:
    python sample.py --sampler ddpm --guidance 1      # the plain 1000-step loop, digits 0-9 x 8
    python sample.py                                  # 50-step DDIM with guidance w = 3 (faster)
    python sample.py --digits 777 --n 16 --trajectory

In Python:
    import torch
    from common import NoiseSchedule, ddpm_sample, get_device, weights_path
    from part1_unet_diffusion import UNet

    device = get_device()
    ckpt = torch.load(weights_path("part1_unet"), map_location=device)
    model = UNet(base=ckpt["config"]["base"]).to(device).eval()
    model.load_state_dict(ckpt["model"])

    labels = torch.tensor([7, 7, 3], device=device)          # the digits you want
    x = ddpm_sample(model, NoiseSchedule(device=device), labels, guidance=1.0,
                    generator=torch.Generator().manual_seed(0))
    # x: 3 x 1 x 32 x 32, pixels in [-1, 1]. (x + 1) / 2 gives [0, 1] for
    # torchvision.utils.save_image. Same seed -> same digits.
"""

import argparse
import copy
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import trackio
from torchvision.utils import save_image

from common import (NULL_LABEL, NUM_CLASSES, NoiseSchedule, count_params, get_device,
                    load_mnist, results_dir, sample_grid, timestep_embedding, weights_path)

NAME = "part1_unet"


class Block(nn.Module):
    """Week 4's DoubleConv (conv 3x3 -> norm -> ReLU, twice), plus one line
    that adds the t-and-label vector between the two convs.

    GroupNorm instead of BatchNorm: it normalises each image on its own, so
    an image's output does not depend on which other images share its batch.
    """

    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.ReLU())
        self.emb = nn.Linear(emb_dim, out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.ReLU())

    def forward(self, x, emb):
        h = self.conv1(x)
        h = h + self.emb(emb)[:, :, None, None]    # <- the only new line vs week 4
        return self.conv2(h)


class UNet(nn.Module):
    """
        input x_t                  1 x 32 x 32
        enc1  Block(1 -> 48)      48 x 32 x 32  ──────────────────┐
              MaxPool(2)                                          │
        enc2  Block(48 -> 96)     96 x 16 x 16  ────────┐         │
              MaxPool(2)                                │         │
        enc3  Block(96 -> 192)   192 x  8 x  8  ──┐     │         │
              MaxPool(2)                          │     │         │
        bott  Block(192 -> 384)  384 x  4 x  4    │     │         │
              ConvT(384 -> 192)  192 x  8 x  8    │     │         │
        dec3  Block(384 -> 192)  <- cat ──────────┘     │         │
              ConvT(192 -> 96)    96 x 16 x 16          │         │
        dec2  Block(192 -> 96)   <- cat ────────────────┘         │
              ConvT(96 -> 48)     48 x 32 x 32                    │
        dec1  Block(96 -> 48)    <- cat ──────────────────────────┘
        head  Conv1x1(48 -> 1)     1 x 32 x 32   predicted noise

        t -> sines/cosines -> MLP ─┐
                                   + -> emb (192 numbers) -> into every Block
        y -> lookup table ─────────┘
    """

    def __init__(self, base=48):
        super().__init__()
        self.base = base
        emb_dim = base * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
        self.label_emb = nn.Embedding(NUM_CLASSES + 1, emb_dim)   # 10 digits + "no label"

        self.pool = nn.MaxPool2d(2)
        self.enc1 = Block(1, base, emb_dim)
        self.enc2 = Block(base, base * 2, emb_dim)
        self.enc3 = Block(base * 2, base * 4, emb_dim)
        self.bottleneck = Block(base * 4, base * 8, emb_dim)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = Block(base * 8, base * 4, emb_dim)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = Block(base * 4, base * 2, emb_dim)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = Block(base * 2, base, emb_dim)

        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x, t, y):
        emb = self.time_mlp(timestep_embedding(t, self.base)) + self.label_emb(y)

        s1 = self.enc1(x, emb)
        s2 = self.enc2(self.pool(s1), emb)
        s3 = self.enc3(self.pool(s2), emb)
        b = self.bottleneck(self.pool(s3), emb)

        d3 = self.dec3(torch.cat([self.up3(b), s3], dim=1), emb)
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1), emb)
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1), emb)
        return self.head(d1)


def training_args(description, **network_args):
    """The training recipe, shared by both parts, plus each part's network sizes."""
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--label-drop", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project", default="week5-diffusion")
    for flag, (default, help_text) in network_args.items():
        p.add_argument(f"--{flag}", type=int, default=default, help=help_text)
    return p.parse_args()


def train(model, name, args, device):
    """Train `model` to predict noise. Part 2 calls this with its transformer."""
    out = results_dir(name)
    images, digits = load_mnist(device)
    sched = NoiseSchedule(device=device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    n = len(images)
    steps_per_epoch = n // args.batch_size
    print(f"{name}  device {device}  images {n:,}  parameters {count_params(model):,}  "
          f"steps/epoch {steps_per_epoch}")

    trackio.init(project=args.project, name=name, config={**vars(args), "params": count_params(model)})
    log = {"epoch": [], "loss": [], "seconds": [], "params": count_params(model)}
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total = time.time(), 0.0
        order = torch.randperm(n, device=device)

        for i in range(steps_per_epoch):
            idx = order[i * args.batch_size:(i + 1) * args.batch_size]
            x0, y = images[idx], digits[idx]

            # label dropout: sometimes train without the label
            drop = torch.rand(len(y), device=device) < args.label_drop
            y = torch.where(drop, torch.full_like(y, NULL_LABEL), y)

            t = torch.randint(0, sched.T, (len(x0),), device=device)
            noise = torch.randn_like(x0)
            x_t = sched.add_noise(x0, t, noise)
            loss = F.mse_loss(model(x_t, t, y), noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # EMA: ema = decay * ema + (1 - decay) * model.
            # Early on the decay is smaller, so the average is not dominated
            # by the random starting weights.
            decay = min(args.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                for e, w in zip(ema.parameters(), model.parameters()):
                    e.lerp_(w, 1 - decay)

            step += 1
            total += loss.item()
            if step % 50 == 0:
                trackio.log({"loss": loss.item()}, step=step)

        seconds = time.time() - start
        epoch_loss = total / steps_per_epoch
        grid = sample_grid(ema, sched, device)
        save_image(grid, os.path.join(out, f"epoch_{epoch:02d}.png"))
        trackio.log({"epoch": epoch, "epoch_loss": epoch_loss, "epoch_seconds": seconds,
                     "samples": trackio.Image((grid.permute(1, 2, 0).numpy() * 255).astype("uint8"),
                                              caption=f"epoch {epoch}: rows are digits 0-9")},
                    step=step)

        torch.save({"model": ema.state_dict(), "config": vars(args), "epoch": epoch},
                   weights_path(name))
        log["epoch"].append(epoch); log["loss"].append(epoch_loss); log["seconds"].append(seconds)
        with open(os.path.join(out, "train_log.json"), "w") as f:
            json.dump(log, f, indent=1)
        print(f"epoch {epoch:3d}  loss {epoch_loss:.4f}  {seconds:.1f}s")

    trackio.finish()
    print(f"saved {weights_path(name)}")


def main():
    args = training_args(__doc__, base=(48, "channels in the first block"))
    torch.manual_seed(args.seed)
    device = get_device()
    train(UNet(base=args.base).to(device), NAME, args, device)


if __name__ == "__main__":
    main()
