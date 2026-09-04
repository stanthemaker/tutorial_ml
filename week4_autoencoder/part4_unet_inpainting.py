"""Week 4 - Part 4a: Inpainting. Same U-Net, different corruption.

The point of this file is how little of it is new. Compared to part3_unet.py:

    corrupt = lambda x: add_noise(x, 0.25)        becomes
    corrupt = lambda x: mask_center(x)[0]

and that is essentially the whole diff. The model, the loop, the loss, the
metric, the plots are untouched. Once you have "image in, image out" plus
skip connections, the *task* is just a choice of how to corrupt the input.

It earns its own file anyway, because it stresses the opposite half of the
network from Part 3:

  * Denoising is mostly local. Every output pixel has a noisy version of
    itself in the input; the model averages away the noise. The skips do most
    of the work.
  * Inpainting has a hole. Inside the mask there is NO information at all --
    the skip connections carry mid-grey and nothing else. The only way to
    fill it is to use context from outside the hole, which means the
    bottleneck (the part that sees the whole image at once) has to do the
    thinking, and the skips have to blend the result seamlessly at the seam.

So this is the experiment where "U-Net = bottleneck AND skips" stops being a
slogan. Expect the ablation to come out roughly TIED on hole-only MSE -- which
is the honest and useful result. Part 3 showed skips are worth 2-3 dB when the
detail still exists somewhere in the input; here it does not exist anywhere, so
they have nothing to restore and the bottleneck has to carry the task alone.

If you only ever run Part 3, it is easy to walk away believing "skip
connections improve any image model". This file is the counterexample that
keeps the claim precise.

One honest caveat on the results: MSE-trained inpainting produces blurry,
over-smoothed fills, because when the model is uncertain about the texture the
lowest-MSE answer is the average of all plausible textures. Sharp, plausible
fills need a different objective (adversarial or perceptual losses). That is
out of scope here -- but it is why the fill looks smoother than the
surroundings, and it is not a bug in the architecture.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from common import (get_device, count_params, load_pet, make_loaders,
                    show_grid, psnr, mask_center, sample_batch, save_weights)
from part3_unet import UNet

IMAGE_SIZE = 96
HOLE_FRAC = 0.35        # side length of the hole, as a fraction of the image


def train_inpainter(model, loader, device, epochs=20, lr=1e-3):
    """Identical to train_denoiser, with masking swapped in for noise.

    Note we score the loss over the WHOLE image, not just the hole. Two
    reasons: it keeps the loop identical to Part 3, and it stops the model
    from drifting outside the hole (where it should just copy the input
    through the skips). We report the hole-only error separately, since that
    is the part anyone actually cares about.
    """
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for clean, _ in loader:
            clean = clean.to(device)
            masked, _ = mask_center(clean, HOLE_FRAC)
            recon = model(masked)
            loss = loss_fn(recon, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
        print(f"  epoch {epoch + 1}/{epochs}  MSE {losses[-1]:.5f}")
    return losses


@torch.no_grad()
def hole_mse(model, loader, device):
    """MSE measured only inside the hole -- the number that matters.

    Whole-image MSE is misleading here: ~88% of the pixels are handed to the
    model unchanged, so a model that simply copies its input already scores
    well. Masking the metric strips that freebie away.
    """
    model.eval()
    total, count = 0.0, 0
    for clean, _ in loader:
        clean = clean.to(device)
        masked, mask = mask_center(clean, HOLE_FRAC)
        recon = model(masked)
        sq = ((recon - clean) ** 2) * mask       # mask broadcasts over channels
        total += sq.sum().item()
        count += mask.sum().item() * clean.size(1)
    return total / count


def composite(masked, recon, mask):
    """Keep the real pixels, take only the hole from the model.

    This is what you would actually ship: the model has no business rewriting
    pixels that were never missing. It also makes the seam obvious, which is
    the interesting failure mode.
    """
    return masked * (1 - mask) + recon * mask


def main():
    torch.manual_seed(0)
    device = get_device()
    print(f"using device: {device}\n")

    train_ds = load_pet("trainval", image_size=IMAGE_SIZE, n_subset=2000)
    test_ds = load_pet("test", image_size=IMAGE_SIZE, n_subset=400)
    train_loader, test_loader = make_loaders(train_ds, test_ds, batch_size=32)

    corrupt = lambda x: mask_center(x, HOLE_FRAC)[0]
    rows, titles, results = [], [], {}

    for label, build, stem in [
        ("U-Net, skips OFF", lambda: UNet(use_skips=False), "part4_inpaint_skips_off"),
        ("U-Net, skips ON", lambda: UNet(use_skips=True), "part4_inpaint_skips_on"),
    ]:
        torch.manual_seed(0)
        model = build()
        print(f"=== {label} -- {count_params(model):,} parameters ===")
        train_inpainter(model, train_loader, device, epochs=20)
        save_weights(model, stem)

        clean, masked, recon = sample_batch(model, test_loader, corrupt=corrupt, n=6)
        _, mask = mask_center(clean, HOLE_FRAC)
        filled = composite(masked, recon, mask)

        results[label] = (hole_mse(model, test_loader, device), psnr(filled, clean))
        if not rows:
            rows += [clean, masked]
            titles += ["original", "input (hole)"]
        rows.append(filled)
        titles.append(label)
        print(f"  -> hole-only MSE {results[label][0]:.5f}, "
              f"full-image PSNR {results[label][1]:.2f} dB\n")

    print("=" * 58)
    print(f"{'model':<20} {'hole MSE':>12} {'PSNR (dB)':>11}")
    print("-" * 58)
    for label, (hmse, p) in results.items():
        print(f"{label:<20} {hmse:>12.5f} {p:>11.2f}")
    print("=" * 58)
    print("\nUnlike Part 3, the two rows here score about the same on the hole,")
    print("and 'skips ON' may even come out marginally worse. That is the")
    print("expected result, not a broken experiment, and it is the reason this")
    print("file exists:")
    print("\n  Inside the hole there is nothing for a skip to carry. The skip")
    print("  connections faithfully deliver a patch of flat grey, so both")
    print("  models must fill the gap from the bottleneck alone -- and their")
    print("  bottlenecks are identical. A tool only helps where it applies.")
    print("\nSo do not read Part 3 as 'skip connections make everything better'.")
    print("Read it as 'skip connections restore detail that still exists in the")
    print("input'. Denoising has a noisy copy of every output pixel; inpainting")
    print("does not. Different task, different half of the network doing the")
    print("work.")
    print("\nWhere the skips DO show up here is the seam: check whether the fill")
    print("meets the surrounding pixels smoothly or sits in the image like a")
    print("patch. That is a boundary effect the hole-only MSE cannot see, which")
    print("is a good reminder that one number rarely settles an image question.")

    show_grid(rows, titles, n=6,
              suptitle=f"Inpainting a {int(HOLE_FRAC * 100)}%-wide hole "
                       f"(model output composited into the hole only)")

    # Zoom on the hole and its border, where the seam lives.
    box = int(IMAGE_SIZE * HOLE_FRAC) + 16
    c = (IMAGE_SIZE - box) // 2
    fig, axes = plt.subplots(1, len(rows), figsize=(3.0 * len(rows), 3.4))
    for ax, batch, title in zip(axes, rows, titles):
        ax.imshow(batch[0, :, c:c + box, c:c + box].permute(1, 2, 0).clamp(0, 1))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("Zoom on the hole and its border: check the seam, not just the fill")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
