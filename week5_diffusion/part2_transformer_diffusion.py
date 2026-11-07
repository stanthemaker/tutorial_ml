"""Week 5 Part 2 - The same diffusion model, with a transformer instead of a U-Net.

    python part2_transformer_diffusion.py --check   # test the code in seconds: no data, no training
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

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import IMAGE_SIZE, NUM_CLASSES, get_device, timestep_embedding
from part1_unet_diffusion import train, training_args

NAME = "part2_transformer"


def scale_and_shift(x, shift, scale):
    """x * (1 + scale) + shift: the scale and shift that e picks, one per channel.
    [:, None, :] makes them B x 1 x 176, so all 256 tokens get the same ones."""
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class TransformerBlock(nn.Module):
    """One transformer block, line by line as on the slides:

        h -> LN -> attention -> + -> LN -> MLP -> +
        '-------------------------'  '------------'   residual: the input is added back

    How t and y get in -- "adaLN-Zero", the DiT paper's recipe. The U-Net Block
    only ADDS e. Here e is turned into six vectors of 176 numbers per block:
      shift, scale  right after each LN: LN(h) * (1 + scale) + shift.
                    A plain LayerNorm learns one fixed scale and shift; here e
                    picks them, so they change with t and the label ("adaptive LN").
      gate          multiplies each half's output before it is added back.
                    It starts at 0 ("-Zero"), so every block starts as "do
                    nothing" and training switches it on gradually.
    """

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        # first half: LN -> attention -> +
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)  # no scale/shift of its own: e gives them
        self.qkv = nn.Linear(dim, 3 * dim)                                  # W_Q, W_K, W_V side by side
        self.proj = nn.Linear(dim, dim)                                     # mixes the heads' outputs
        # second half: LN -> MLP -> +
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        # e -> shift, scale, gate for each half
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)                                  # all six start at 0
        nn.init.zeros_(self.ada[1].bias)

    def attention(self, h):
        """softmax(Q K^T / sqrt(d)) V, run by 4 heads side by side.
        Shapes: B images (128 per training step, 2 in --check); N = 256 tokens
        of D = 176 numbers."""
        B, N, D = h.shape                                     # (B, 256, 176)
        q, k, v = self.qkv(h).chunk(3, dim=-1)                # Q = hW_Q, K = hW_K, V = hW_V: each (B, 256, 176)
        q, k, v = [m.view(B, N, self.heads, D // self.heads).transpose(1, 2)
                   for m in (q, k, v)]                        # cut into 4 heads of 44: each (B, 4, 256, 44)
        out = F.scaled_dot_product_attention(q, k, v)         # softmax(QK^T/sqrt(44)) V, per head: (B, 4, 256, 44)
        out = out.transpose(1, 2).reshape(B, N, D)            # the 4 heads side by side again: (B, 256, 176)
        return self.proj(out)                                 # Linear 176 -> 176: mixes the heads

    def forward(self, h, emb):
        # e -> six vectors of 176 numbers for this block, each (B, 176)
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(emb).chunk(6, dim=1)

        # first half: LN -> attention -> +
        a = self.norm1(h)                        # LN
        a = scale_and_shift(a, shift1, scale1)   # e's scale and shift
        a = self.attention(a)                    # attention
        h = h + gate1[:, None, :] * a            # +: added back onto the input, after e's gate

        # second half: LN -> MLP -> +
        m = self.norm2(h)                        # LN
        m = scale_and_shift(m, shift2, scale2)   # e's scale and shift
        m = self.mlp(m)                          # MLP: 176 -> 704 -> 176, each token on its own
        h = h + gate2[:, None, :] * m            # +: added back onto the input, after e's gate
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

        # the last step: LN with e's scale and shift (no gate), then Linear 176 -> 4
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
        h = self.head(scale_and_shift(self.norm(h), shift, scale))   # B x 256 x (p*p)

        B, g, p = len(x), IMAGE_SIZE // self.patch, self.patch
        h = h.view(B, g, g, p, p).permute(0, 1, 3, 2, 4)             # row, pixel-row, col, pixel-col
        return h.reshape(B, 1, IMAGE_SIZE, IMAGE_SIZE)


# ------------------------------------------------------------------ --check

def check():
    """python part2_transformer_diffusion.py --check

    Tests the transformer code in a few seconds on the CPU, on 2 made-up images:
    no data, no training. Anything you print() inside the model shows up here a
    few times, instead of thousands of times during training."""
    torch.manual_seed(0)
    B, D, heads = 2, 176, 4
    N = (IMAGE_SIZE // 2) ** 2                                  # 256 tokens
    block = TransformerBlock(D, heads)
    nn.init.normal_(block.ada[1].weight, std=0.1)               # a new block has every gate at 0, which
    nn.init.normal_(block.ada[1].bias, std=0.1)                 # would hide mistakes: give it real ones
    h, emb = torch.randn(B, N, D), torch.randn(B, D)

    # The reference: PyTorch's own multi-head attention, given this block's weights.
    torch_attention = nn.MultiheadAttention(D, heads, batch_first=True)
    with torch.no_grad():
        torch_attention.in_proj_weight.copy_(block.qkv.weight)
        torch_attention.in_proj_bias.copy_(block.qkv.bias)
        torch_attention.out_proj.weight.copy_(block.proj.weight)
        torch_attention.out_proj.bias.copy_(block.proj.bias)

    def mark(ok, what):
        print(f"   {'ok   ' if ok else 'WRONG'}  {what}")
        return ok

    def same(got, want):
        return got.shape == want.shape and torch.allclose(got, want, atol=1e-5)

    def run(step, *args):
        """step(*args), or None (with the reason printed) if it fails or is still a TODO."""
        try:
            with torch.no_grad():
                out = step(*args)
        except Exception as err:
            print(f"   not working yet: {type(err).__name__}: {err}")
            return None
        if not isinstance(out, torch.Tensor):
            print(f"   not done yet: it returns {out!r} instead of a tensor")
            return None
        return out

    print("1. scale_and_shift: x * (1 + scale) + shift, the same for every token  (slide 25)")
    x, shift, scale = torch.randn(B, N, D), torch.randn(B, D), torch.randn(B, D)
    out = run(scale_and_shift, x, shift, scale)
    ok_scale = out is not None and mark(out.shape == (B, N, D),
                                        f"output: {tuple(out.shape)}, should be {(B, N, D)}")
    ok_scale = ok_scale and mark(same(out, x * (1 + scale[:, None]) + shift[:, None]), "right numbers")

    print(f"2. attention, on {B} images of {N} tokens x {D} numbers  (slide 24)")
    seen = []                                                   # q's shape on its way into the formula
    sdpa = F.scaled_dot_product_attention

    def spy(q, k, v, *args, **kwargs):
        seen.append(tuple(q.shape))
        return sdpa(q, k, v, *args, **kwargs)

    F.scaled_dot_product_attention = spy
    try:
        out = run(block.attention, h)
    finally:
        F.scaled_dot_product_attention = sdpa
    ok_attention = out is not None
    if ok_attention:
        if seen:
            ok_attention &= mark(seen[0] == (B, heads, N, D // heads),
                                 f"q going into F.scaled_dot_product_attention: {seen[0]}, "
                                 f"should be {(B, heads, N, D // heads)}")
        ok_attention &= mark(out.shape == (B, N, D), f"output: {tuple(out.shape)}, should be {(B, N, D)}")
        with torch.no_grad():
            ok_attention &= mark(same(out, torch_attention(h, h, h, need_weights=False)[0]),
                                 "same numbers as PyTorch's nn.MultiheadAttention")
    if not (ok_scale and ok_attention):
        print("Fix the steps above first: the block and the whole model use them.")
        return

    print("3. the whole block: LN -> attention -> +, LN -> MLP -> +, with e's scale, shift, gate  (slide 25)")
    got = run(block, h, emb)
    if got is None:
        return
    with torch.no_grad():                                       # the same block, written out in full
        s1, c1, g1, s2, c2, g2 = block.ada(emb).chunk(6, dim=1)
        a = block.norm1(h) * (1 + c1[:, None]) + s1[:, None]
        want = h + g1[:, None] * torch_attention(a, a, a, need_weights=False)[0]
        m = block.norm2(want) * (1 + c2[:, None]) + s2[:, None]
        want = want + g2[:, None] * block.mlp(m)
    ok = mark(got.shape == (B, N, D), f"output: {tuple(got.shape)}, should be {(B, N, D)}")
    if not (ok and mark(same(got, want), "same numbers as the block on slide 25")):
        return

    print("4. the whole model, step by step")
    model = DiT(dim=D, heads=heads)
    steps = {"cut into patches (Conv 2x2)": model.patchify, "after each block": model.blocks[0],
             "Linear 176 -> 4 per token": model.head}
    shapes = {}

    def note_shape(name):
        def hook(_layer, _inputs, output):                     # returns None: leaves the output alone
            shapes.setdefault(name, tuple(output.shape))
        return hook

    for name, layer in steps.items():
        layer.register_forward_hook(note_shape(name))
    x = torch.randn(B, 1, IMAGE_SIZE, IMAGE_SIZE)
    out = run(model, x, torch.tensor([10, 900]), torch.tensor([3, 7]))
    if out is None:
        return
    print(f"   input x_t: {tuple(x.shape)}")
    for name, shape in shapes.items():
        print(f"   {name}: {shape}")
    if mark(out.shape == x.shape, f"predicted noise: {tuple(out.shape)}, same shape as x_t"):
        print("All good. Train with:  python part2_transformer_diffusion.py")


def main():
    if "--check" in sys.argv[1:]:
        return check()
    args = training_args(__doc__, dim=(176, "numbers per token"), depth=(8, "transformer blocks"),
                         heads=(4, "attention heads"), patch=(2, "patch size in pixels"))
    torch.manual_seed(args.seed)
    device = get_device()
    model = DiT(dim=args.dim, depth=args.depth, heads=args.heads, patch=args.patch).to(device)
    train(model, NAME, args, device)


if __name__ == "__main__":
    main()
