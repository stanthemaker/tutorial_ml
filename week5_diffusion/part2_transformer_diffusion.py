"""Week 5 Part 2 - The same diffusion model, with a transformer instead of a U-Net.

    python part2_transformer_diffusion.py
    trackio show --project week5-diffusion

WHAT CHANGES FROM PART 1
------------------------
Only the network. The noise schedule, the loss, the label dropout, the EMA,
the training loop (imported from Part 1) and the samplers are identical, and
the network has the same inputs (x_t, t, y) and the same output (the noise).

THE NETWORK: DiT (a diffusion transformer)
------------------------------------------
    input x_t                        1 x 32 x 32
    patchify  Conv(2x2, stride 2)  176 x 16 x 16  -> 256 tokens of 176 numbers
              + position table       (learned: 256 x 176, "where is this patch")
    8 x TransformerBlock           256 x 176      <- e into every block
    norm, Linear(176 -> 4)         256 x 4        each token -> its 2 x 2 pixels
    unpatchify                       1 x 32 x 32   predicted noise

    t -> sines/cosines -> MLP ─┐
                               + -> e (176 numbers)   built exactly as in Part 1
    y -> lookup table ─────────┘

A "token" is one 2 x 2 patch, described by 176 numbers. No pooling, no skip
connections, no size changes: every block works on all 256 patches at full
detail, and attention connects any two patches in one step.

The sizes (176 numbers per token, 8 blocks) are chosen so the parameter count
matches Part 1's U-Net: 4,654,852 here vs 4,586,449.

OUTPUT
------
  checkpoints/part2_transformer.pt              overwritten after every epoch
  results/part2_transformer/train_log.json      per-epoch loss and seconds
  results/part2_transformer/epoch_XX.png        the per-epoch sample grid

LOAD THE TRAINED MODEL AND GENERATE
-----------------------------------
Exactly as in Part 1; only the class and the sizes stored in the config differ.

From the command line:
    python sample.py --run part2_transformer --sampler ddpm --guidance 1
    python sample.py --run part2_transformer          # 50-step DDIM with guidance w = 3

In Python:
    import torch
    from common import NoiseSchedule, ddpm_sample, get_device, weights_path
    from part2_transformer_diffusion import DiT

    device = get_device()
    ckpt = torch.load(weights_path("part2_transformer"), map_location=device)
    c = ckpt["config"]
    model = DiT(dim=c["dim"], depth=c["depth"], heads=c["heads"], patch=c["patch"])
    model = model.to(device).eval()
    model.load_state_dict(ckpt["model"])

    labels = torch.tensor([7, 7, 3], device=device)          # the digits you want
    x = ddpm_sample(model, NoiseSchedule(device=device), labels, guidance=1.0,
                    generator=torch.Generator().manual_seed(0))
    # x: 3 x 1 x 32 x 32, pixels in [-1, 1]. (x + 1) / 2 gives [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import IMAGE_SIZE, NUM_CLASSES, get_device, timestep_embedding
from part1_unet_diffusion import train, training_args

NAME = "part2_transformer"


def modulate(h, shift, scale):
    """Per-channel scale and shift, chosen by the t-and-label vector."""
    return h * (1 + scale[:, None, :]) + shift[:, None, :]


class TransformerBlock(nn.Module):
    """Attention (every patch looks at every patch), then an MLP (each patch
    on its own). Each part is added back onto its input (a residual).

    How t and y get in -- "adaLN-Zero", the DiT paper's recipe:
    the U-Net Block only ADDS e. Here e is turned into six vectors per block:
    a shift and a scale for the normalised input of each part, and a gate
    that multiplies each part's output. The gates start at 0, so every block
    starts as "do nothing" and training switches it on gradually.
    """

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def attention(self, h):
        B, N, D = h.shape
        q, k, v = self.qkv(h).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)            # B x heads x N x D/heads
        return self.proj(out.transpose(1, 2).reshape(B, N, D))

    def forward(self, h, emb):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(emb).chunk(6, dim=1)
        h = h + gate1[:, None, :] * self.attention(modulate(self.norm1(h), shift1, scale1))
        h = h + gate2[:, None, :] * self.mlp(modulate(self.norm2(h), shift2, scale2))
        return h


class DiT(nn.Module):
    """See the module docstring for the shapes."""

    def __init__(self, dim=176, depth=8, heads=4, patch=2):
        super().__init__()
        self.dim, self.patch = dim, patch
        grid = IMAGE_SIZE // patch
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.label_emb = nn.Embedding(NUM_CLASSES + 1, dim)       # 10 digits + "no label"

        self.patchify = nn.Conv2d(1, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.randn(1, grid * grid, dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.head = nn.Linear(dim, patch * patch)
        for layer in (self.ada[1], self.head):                   # start by predicting 0
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, t, y):
        emb = self.time_mlp(timestep_embedding(t, self.dim)) + self.label_emb(y)

        h = self.patchify(x).flatten(2).transpose(1, 2) + self.pos   # B x 256 x dim
        for block in self.blocks:
            h = block(h, emb)
        shift, scale = self.ada(emb).chunk(2, dim=1)
        h = self.head(modulate(self.norm(h), shift, scale))          # B x 256 x (p*p)

        B, g, p = len(x), IMAGE_SIZE // self.patch, self.patch
        h = h.view(B, g, g, p, p).permute(0, 1, 3, 2, 4)             # row, pixel-row, col, pixel-col
        return h.reshape(B, 1, IMAGE_SIZE, IMAGE_SIZE)


def main():
    args = training_args(__doc__, dim=(176, "numbers per token"), depth=(8, "transformer blocks"),
                         heads=(4, "attention heads"), patch=(2, "patch size in pixels"))
    torch.manual_seed(args.seed)
    device = get_device()
    model = DiT(dim=args.dim, depth=args.depth, heads=args.heads, patch=args.patch).to(device)
    train(model, NAME, args, device)


if __name__ == "__main__":
    main()
