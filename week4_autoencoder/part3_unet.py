"""Week 4 - Part 3: U-Net. Stop throwing the encoder's detail away.

The demo left us with one observation and one question.

Observation: the CNN autoencoder denoises Pet photos into smooth paint. Shape
is right, texture is gone.

Question: on the way down, the demo's encoder computed a 32 x 48 x 48 feature
map that still had the whiskers in it, then used it once and discarded it. Why
does the decoder not get to see it?

(The U-Net below goes one better. Because its first block is a stride-1
DoubleConv rather than a stride-2 conv, its finest feature map is a full
32 x 96 x 96 -- full input resolution, never downsampled at all.)

A U-Net is the two-line answer. Keep every encoder feature map, and when the
decoder gets back to that resolution, staple the saved map onto the decoder's
own features before continuing:

        x = self.up3(bottleneck)              # 12x12 -> 24x24
        x = torch.cat([x, skip3], dim=1)      # <-- the whole idea

Now the decoder has two sources at every resolution:
  * from below, upsampled: coarse but semantically informed ("this is a cat,
    lying down, facing left")
  * from the side, via the skip: fine-grained but naive ("there is a strong
    edge at exactly this pixel")
and its job shrinks from "hallucinate detail" to "decide which of the sharp
edges I was handed are real, given what I know about the scene". That is a
much easier job, and it is why the whiskers come back.

Three things that trip people up, in the order they will trip you:

  1. cat, not add. `torch.cat([up, skip], dim=1)` stacks along the CHANNEL
     axis: 64 channels + 64 channels = 128 channels, both preserved
     separately. A ResNet-style `up + skip` would sum them into 64 channels,
     mixing the two sources irreversibly. Concatenation keeps them distinct
     and lets the next conv learn how to weigh them. (Both designs are used in
     practice; U-Net concatenates.)

  2. Because of (1), every decoder block's INPUT channel count is doubled.
     dec2 takes 128 channels in, not 64. This is the single most common bug in
     a hand-written U-Net, and the error message -- a channel mismatch inside
     Conv2d -- points at the conv, not at the cat that caused it.

  3. Sizes have to line up exactly, or the cat fails. `padding=1` on every
     3x3 conv means resolution only ever changes at the explicit pool and
     upsample steps, so level k of the encoder and level k of the decoder are
     guaranteed to match. The "U" shape is not decoration -- the symmetry IS
     the alignment guarantee.

To prove the improvement comes from the skips and not from the extra
parameters, main() runs the ablation: the identical network with the
concatenations zeroed out. Same depth, same channel widths, same epochs, same
12x12 bottleneck, and -- as we print -- exactly the same parameter count.

Spoiler for the table it produces: the skipless U-Net, at roughly 7x the size
of the plain CNN autoencoder from the demo, does not reliably beat it. Below
the bottleneck, extra capacity cannot buy back resolution that was already
discarded. Only a path around the bottleneck can.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from common import (get_device, count_params, load_pet, make_loaders,
                    show_grid, psnr, add_noise, sample_batch, save_weights,
                    DoubleConv)
from demo_why_skip_connections import (IMAGE_SIZE, NOISE_SIGMA, build_cnn_ae,
                                       train_denoiser)


class UNet(nn.Module):
    """A U-Net for 96x96 images: three down steps, three up steps.

    The depth is not arbitrary -- it is chosen to match
    demo_why_skip_connections.py exactly. That CNN autoencoder downsampled
    three times, to a 12x12 bottleneck, and lost the whiskers there. This
    network squeezes through the SAME 12x12 bottleneck. If the detail comes
    back, it cannot be because we gave the model an easier squeeze; the only
    structural difference is the skips.

    (The original paper went four levels deep on 572x572 microscope images.
    Same structure, we just stop earlier because our images are smaller.)

        enc1  DoubleConv(3 -> 32)     32 x 96 x 96  ─────────────────────┐
              MaxPool(2)                                                 │ skip1
        enc2  DoubleConv(32 -> 64)    64 x 48 x 48  ───────────┐         │
              MaxPool(2)                                       │ skip2   │
        enc3  DoubleConv(64 -> 128)  128 x 24 x 24  ──┐        │         │
              MaxPool(2)                              │ skip3  │         │
        bott  DoubleConv(128 -> 256) 256 x 12 x 12    │        │         │
              ConvT(256 -> 128)      128 x 24 x 24    │        │         │
        dec3  DoubleConv(256 -> 128) <- cat(128,128) ─┘        │         │
              ConvT(128 -> 64)        64 x 48 x 48             │         │
        dec2  DoubleConv(128 -> 64)  <- cat(64, 64) ───────────┘         │
              ConvT(64 -> 32)         32 x 96 x 96                       │
        dec1  DoubleConv(64 -> 32)   <- cat(32, 32) ─────────────────────┘
        head  Conv1x1(32 -> 3) -> Sigmoid

    Read the three decoder DoubleConv input counts again: 256, 128, 64 -- each
    is twice what the ConvTranspose above it produced. That doubling is point
    (2) in the module docstring, and it is the bug you will write.

    `use_skips=False` builds the same graph with the concatenated skip half
    zeroed out. That keeps the channel counts (and therefore the parameter
    count) identical, so the ablation compares skips against no skips rather
    than a big model against a small one.
    """

    def __init__(self, in_ch=3, out_ch=3, base=32, use_skips=True):
        super().__init__()
        self.use_skips = use_skips
        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(in_ch, base)                # 96x96
        self.enc2 = DoubleConv(base, base * 2)             # 48x48
        self.enc3 = DoubleConv(base * 2, base * 4)         # 24x24
        self.bottleneck = DoubleConv(base * 4, base * 8)   # 12x12

        # Each ConvTranspose halves the channel count; the cat then doubles it
        # straight back, which is why every dec block takes 2x what up gives.
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)   # base*8 = 128 up + 128 skip
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)   # base*4 =  64 up +  64 skip
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)       # base*2 =  32 up +  32 skip

        # A 1x1 conv is just a per-pixel linear layer across channels: it maps
        # 32 feature values at each pixel down to 3 colour values, with no
        # spatial mixing at all. Sigmoid because the answer is pixels in [0,1].
        self.head = nn.Sequential(nn.Conv2d(base, out_ch, kernel_size=1), nn.Sigmoid())

    def _join(self, up, skip):
        """cat along channels -- or cat with zeros when running the ablation."""
        if not self.use_skips:
            skip = torch.zeros_like(skip)
        return torch.cat([up, skip], dim=1)   # dim=1 is the channel axis

    def forward(self, x):
        # ---- encoder: keep every feature map on the way down ----
        s1 = self.enc1(x)                    #  32 x 96 x 96  <- finest detail
        s2 = self.enc2(self.pool(s1))        #  64 x 48 x 48
        s3 = self.enc3(self.pool(s2))        # 128 x 24 x 24
        b = self.bottleneck(self.pool(s3))   # 256 x 12 x 12  <- same as the demo

        # ---- decoder: upsample, re-attach the matching encoder map ----
        d3 = self.dec3(self._join(self.up3(b), s3))   # 128 x 24 x 24
        d2 = self.dec2(self._join(self.up2(d3), s2))  #  64 x 48 x 48
        d1 = self.dec1(self._join(self.up1(d2), s1))  #  32 x 96 x 96
        return self.head(d1)


def describe_shapes(model, device):
    """Print the actual tensor shapes on one forward pass.

    Cheap, and it turns the ASCII diagram above into something verified rather
    than believed. If you modify the architecture, run this first.
    """
    hooks, log = [], []
    for name in ["enc1", "enc2", "enc3", "bottleneck",
                 "up3", "dec3", "up2", "dec2", "up1", "dec1", "head"]:
        module = getattr(model, name)
        hooks.append(module.register_forward_hook(
            lambda m, i, o, n=name: log.append(
                (n, i[0].shape[1], tuple(o.shape[1:])))))
    with torch.no_grad():
        model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device))
    for h in hooks:
        h.remove()

    print(f"U-Net shapes for one 3 x {IMAGE_SIZE} x {IMAGE_SIZE} input:")
    print(f"  {'layer':<11} {'in_ch':>6}  ->  {'output (C, H, W)':<16}")
    for name, in_ch, shape in log:
        note = ""
        if name.startswith("dec"):
            # in_ch here is what came out of torch.cat: upsampled + skip
            note = f"  <- cat({in_ch // 2} up + {in_ch // 2} skip) = {in_ch} in"
        print(f"  {name:<11} {in_ch:>6}  ->  {str(shape):<16}{note}")
    print()


def main():
    torch.manual_seed(0)
    device = get_device()
    print(f"using device: {device}\n")

    train_ds = load_pet("trainval", image_size=IMAGE_SIZE, n_subset=2000)
    test_ds = load_pet("test", image_size=IMAGE_SIZE, n_subset=400)
    train_loader, test_loader = make_loaders(train_ds, test_ds, batch_size=32)

    describe_shapes(UNet().to(device), device)

    # Everything below trains on the SAME task, data, epochs and optimiser as
    # demo_why_skip_connections.py. Only the model changes.
    epochs = 15
    corrupt = lambda x: add_noise(x, NOISE_SIGMA)
    results, rows, titles = {}, [], []

    # Built lazily so the seed reset below applies to each model's init as
    # well as to the batch shuffling -- otherwise the comparison would be
    # partly a comparison of random draws.
    # The third entry is the filename stem: three models train here, so the
    # checkpoints have to say which is which.
    contenders = [
        ("CNN AE (demo)", build_cnn_ae, "part3_cnn_ae"),
        ("U-Net, skips OFF", lambda: UNet(use_skips=False), "part3_unet_skips_off"),
        ("U-Net, skips ON", lambda: UNet(use_skips=True), "part3_unet_skips_on"),
    ]

    for label, build, stem in contenders:
        torch.manual_seed(0)          # same init draw for every contender
        model = build()
        print(f"=== {label} -- {count_params(model):,} parameters ===")
        train_denoiser(model, train_loader, device, epochs=epochs, tag="")
        save_weights(model, stem)
        clean, noisy, recon = sample_batch(model, test_loader, corrupt=corrupt, n=6)
        results[label] = (psnr(recon, clean), count_params(model))
        if not rows:
            rows += [clean, noisy]
            titles += ["clean", "noisy"]
        rows.append(recon)
        titles.append(label)
        print(f"  -> PSNR {results[label][0]:.2f} dB\n")

    print("=" * 62)
    print(f"{'model':<20} {'params':>12} {'PSNR (dB)':>11}")
    print("-" * 62)
    print(f"{'noisy input':<20} {'-':>12} {psnr(rows[1], rows[0]):>11.2f}")
    for label, (score, params) in results.items():
        print(f"{label:<20} {params:>12,} {score:>11.2f}")
    print("=" * 62)
    print("\nTwo readings of this table, and the second is the important one.")
    print("\n1. 'skips ON' vs 'skips OFF': same depth, same channel widths, the")
    print("   same parameter count, the same 12x12 bottleneck. The ONLY")
    print("   difference is whether the encoder's high-resolution feature maps")
    print("   reach the decoder, and it is worth roughly 2-3 dB.")
    print("\n2. 'skips OFF' vs 'CNN AE': the U-Net without its skips is around")
    print("   7x larger than the plain CNN autoencoder and does not reliably")
    print("   beat it. Capacity is not the bottleneck here -- resolution is.")
    print("   Piling on parameters below a 12x12 waist cannot recover detail")
    print("   that was discarded on the way down; only a path around the waist")
    print("   can. That is what a skip connection is.")

    show_grid(rows, titles, n=6,
              suptitle="Denoising Oxford-IIIT Pet: the skip connections are the difference")

    # Zoom in on one image -- at thumbnail size, 'sharper' is arguable.
    box, idx = 40, 0
    c = (IMAGE_SIZE - box) // 2
    fig, axes = plt.subplots(1, len(rows), figsize=(3.0 * len(rows), 3.4))
    for ax, batch, title in zip(axes, rows, titles):
        ax.imshow(batch[idx, :, c:c + box, c:c + box].permute(1, 2, 0).clamp(0, 1))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("Centre crop: where the whiskers went")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
