"""Week 4 - demo: where the CNN autoencoder runs out of road.

Parts 1 and 2 gave the model an easy exam: copy your input. Even a mediocre
autoencoder scores well there, because the answer is sitting right in front of
it. Here we change the task so it cannot be:

    input  = a photograph with Gaussian noise added
    target = the clean photograph

This is *denoising*. It is the first task this week where input != target, and
it is the first time the autoencoder is genuinely useful rather than an
elaborate identity function.

We also change the data. MNIST would let the model off the hook -- a digit is
a few thick strokes on a flat background, so "blurry" and "correct" look
almost the same. Oxford-IIIT Pet is real photographs: fur, whiskers, grass,
eyes. All of that is high-frequency detail, and high-frequency detail is
exactly what a bottleneck cannot afford to keep.

Run this and look at the third row. What you should see:

  * The pose, the silhouette, the rough colours: all correct.
  * The fur, the whiskers, the eyes: gone. Smoothed into paint.

The model did not fail at "understanding the image" -- it clearly knows where
the animal is. It failed at *reproducing detail*, and the reason is
structural. Follow one whisker through the network:

    96x96 -> 48x48 -> 24x24 -> 12x12    every step throws pixels away
    12x12 -> 24x24 -> 48x48 -> 96x96    every step has to invent them back

By the time the signal reaches the 12x12 bottleneck, the whisker is at most a
fraction of one activation. The decoder is being asked to reconstruct
something that is no longer in its input. The blur is not a training problem;
more epochs will not fix it.

Which leads to the question Part 3 answers:

    The encoder's FIRST conv produced a 32 x 48 x 48 feature map, and at half
    resolution the whiskers were still in it. We computed that map, used it
    once on the way down, and then threw it away.
    Why not hand it to the decoder directly?
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from common import (get_device, count_params, load_pet, make_loaders,
                    show_grid, psnr, add_noise, sample_batch, save_weights,
                    DATA_ROOT)

IMAGE_SIZE = 96
NOISE_SIGMA = 0.25


def build_cnn_ae(in_ch=3, base=32):
    """Part 2's architecture, one level deeper because the images are bigger.

    Shape walk-through (3 x 96 x 96 in):

        Conv(3  ->  32, s2)    32 x 48 x 48
        Conv(32 ->  64, s2)    64 x 24 x 24
        Conv(64 -> 128, s2)   128 x 12 x 12   <- bottleneck: everything fits here
        ConvT(128 -> 64, s2)   64 x 24 x 24
        ConvT(64  -> 32, s2)   32 x 48 x 48
        ConvT(32  ->  3, s2)    3 x 96 x 96
        Sigmoid

    Note this bottleneck is not even small in raw numbers -- 128*12*12 = 18432
    values against 3*96*96 = 27648 input values, barely a 1.5x squeeze. The
    problem is not the *count*, it is the *resolution*: whatever the encoder
    kept, it kept at 12x12, and there is no 96x96 information left anywhere in
    the decoder's input. That distinction is the whole point of the demo.
    """
    encoder = nn.Sequential(
        nn.Conv2d(in_ch, base, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1), nn.ReLU(),
    )
    decoder = nn.Sequential(
        nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1), nn.ReLU(),
        nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1), nn.ReLU(),
        nn.ConvTranspose2d(base, in_ch, 4, stride=2, padding=1),
        nn.Sigmoid(),
    )
    return nn.Sequential(encoder, decoder)


def train_denoiser(model, loader, device, epochs=15, lr=1e-3, tag=""):
    """Same four steps as always. The only new line is the corruption.

    Notice where add_noise sits: INSIDE the batch loop, not in the dataset.
    That means every epoch sees a different noise draw of the same photo, so
    the model cannot memorise "this exact noise pattern belongs to this exact
    image" -- it has to learn what noise looks like in general.
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
            noisy = add_noise(clean, NOISE_SIGMA)   # <- input
            recon = model(noisy)
            loss = loss_fn(recon, clean)            # <- target is the CLEAN image
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
        print(f"  {tag}epoch {epoch + 1}/{epochs}  MSE {losses[-1]:.5f}")
    return losses


def plot_detail_zoom(clean, noisy, recon, idx=0, box=32):
    """Crop the centre of one image so the lost detail is unmistakable.

    At 96x96 in a small figure the blur is easy to talk yourself out of.
    Zoomed in, it is not.
    """
    c = (clean.size(-1) - box) // 2
    crop = lambda t: t[idx, :, c:c + box, c:c + box].permute(1, 2, 0).clamp(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6))
    for ax, img, title in zip(axes,
                              [clean, noisy, recon],
                              ["clean (target)", f"noisy (sigma={NOISE_SIGMA})",
                               "CNN AE output"]):
        ax.imshow(crop(img))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Centre crop: the noise is gone, but so is the texture")
    fig.tight_layout()


def main():
    torch.manual_seed(0)
    device = get_device()
    print(f"using device: {device}")
    print(f"Oxford-IIIT Pet (~800MB on first run) lives in {DATA_ROOT}\n")

    train_ds = load_pet("trainval", image_size=IMAGE_SIZE, n_subset=2000)
    test_ds = load_pet("test", image_size=IMAGE_SIZE, n_subset=400)
    train_loader, test_loader = make_loaders(train_ds, test_ds, batch_size=32)
    print(f"train {len(train_ds)} images, test {len(test_ds)} images, "
          f"{IMAGE_SIZE}x{IMAGE_SIZE} RGB")

    model = build_cnn_ae()
    print(f"CNN autoencoder parameters: {count_params(model):,}\n")
    losses = train_denoiser(model, train_loader, device, epochs=15)
    save_weights(model, "demo_why_skip_connections")

    corrupt = lambda x: add_noise(x, NOISE_SIGMA)
    clean, noisy, recon = sample_batch(model, test_loader, corrupt=corrupt, n=6)

    # Two numbers worth putting side by side: how bad the noisy input is, and
    # how much of that the model actually recovered.
    print(f"\nnoisy input vs clean : PSNR {psnr(noisy, clean):5.2f} dB")
    print(f"CNN AE output vs clean: PSNR {psnr(recon, clean):5.2f} dB")
    print("The model IS removing noise -- the score goes up. Now look at the")
    print("pictures and ask what it removed along with the noise.")

    show_grid([clean, noisy, recon],
              ["clean", "noisy", "CNN AE"], n=6,
              suptitle="CNN autoencoder denoising: shape survives, detail does not")

    plot_detail_zoom(clean, noisy, recon, idx=0)

    plt.figure()
    plt.plot(losses, marker="o", color="tab:red")
    plt.title("CNN AE denoising loss (it converges -- that is not the problem)")
    plt.xlabel("epoch")
    plt.ylabel("MSE vs clean image")
    plt.tight_layout()

    print("\n" + "-" * 70)
    print("The encoder's first conv produced a 32 x 48 x 48 feature map, and")
    print("at half resolution the whiskers were still in it. We downsampled")
    print("twice more and asked the decoder to reinvent them from a 12 x 12")
    print("summary. Every one of those intermediate maps was computed, used")
    print("once, and discarded.")
    print("Part 3: stop discarding them. Hand them to the decoder.")
    print("-" * 70)

    plt.show()


if __name__ == "__main__":
    main()
