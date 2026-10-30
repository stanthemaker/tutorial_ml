"""Week 5 - DDPM vs DDIM on a trained checkpoint: speed and quality, from the same noise.

    python compare_samplers.py                         # Part 1's U-Net, 100 digits per class
    python compare_samplers.py --run part2_transformer
    python compare_samplers.py --n 20                  # quicker, noisier numbers
    python compare_samplers.py --ddim-steps 1 5 50     # choose the DDIM step counts

Runs on a Windows/Linux GPU (CUDA), a Mac GPU (MPS) or the CPU -- whatever
common.get_device() finds -- and writes only under this folder.

Every setting starts from the SAME noise (seed 0) and asks for the same digits,
n of each, with no guidance (w = 1), so the sampler is the only difference.
For each setting:

  seconds   wall-clock time on this machine, GPU synchronised before stopping the clock
  agrees    how often a separately trained MNIST classifier (the "judge") reads the
            digit we asked for. A blurry average 7 can still look like a 7, so this
            alone does not catch blur:
  FD        Frechet distance between the generated digits and real test digits,
            measured on the judge's 128 internal features. 0 = same distribution.
            Blur, artefacts AND lack of variety all raise it. The "real digits" row
            (n per class taken from the training set) is the floor for this n.

The judge is a small CNN trained once on the 60,000 training digits (about a
minute) and kept in checkpoints/mnist_judge.pt; its test accuracy is printed and saved.

Output (results/<run>/):
  sampler_compare.json   every number above, plus the machine it was measured on
  sampler_compare.png    one row per setting (same order as the JSON), digits 0-9:
                         the first sample of each digit
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from common import (IMAGE_SIZE, NoiseSchedule, ddim_sample, ddpm_sample, device_name, get_device,
                    load_mnist, results_dir, sync, to_pixels, warm_up, weights_path)
from sample import load_model


# ---------------------------------------------------------------- the judge

class Judge(nn.Module):
    """A plain MNIST CNN. Returns (logits, 128 features); FD uses the features."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 32 x 16 x 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 64 x 8 x 8
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU())
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        f = self.body(x)
        return self.head(f), f


@torch.no_grad()
def judge_all(judge, x, batch=1000):
    logits, feats = zip(*(judge(x[i:i + batch]) for i in range(0, len(x), batch)))
    return torch.cat(logits), torch.cat(feats)


def get_judge(device, epochs=3):
    """Load checkpoints/mnist_judge.pt, or train it once and save it."""
    path = weights_path("mnist_judge")
    judge = Judge().to(device)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        judge.load_state_dict(ckpt["model"])
        return judge.eval(), ckpt["test_acc"]

    print("training the judge (a small MNIST classifier), once ...")
    torch.manual_seed(0)
    x, y = load_mnist(device)
    opt = torch.optim.Adam(judge.parameters(), lr=1e-3)
    for epoch in range(epochs):
        order = torch.randperm(len(x), device=device)
        for i in range(0, len(x), 128):
            idx = order[i:i + 128]
            loss = F.cross_entropy(judge(x[idx])[0], y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    judge.eval()
    xt, yt = load_mnist(device, train=False)
    acc = (judge_all(judge, xt)[0].argmax(1) == yt).float().mean().item()
    torch.save({"model": judge.state_dict(), "test_acc": acc}, path)
    print(f"judge test accuracy {acc:.2%}  ->  {path}")
    return judge, acc


def frechet(a, b):
    """Frechet distance between two feature sets, each modelled as a Gaussian:
        |mu_a - mu_b|^2 + Tr(S_a + S_b - 2 (S_a S_b)^(1/2))
    Computed in float64 on the CPU (eigh is not available on every GPU backend)."""
    a, b = a.double().cpu(), b.double().cpu()
    mu_a, mu_b = a.mean(0), b.mean(0)
    s_a, s_b = torch.cov(a.T), torch.cov(b.T)
    w, v = torch.linalg.eigh(s_a)
    root_a = (v * w.clamp(min=0).sqrt()) @ v.T
    # Tr((S_a S_b)^(1/2)) = Tr((root_a S_b root_a)^(1/2)), and that matrix is symmetric
    cross = torch.linalg.eigvalsh(root_a @ s_b @ root_a).clamp(min=0).sqrt().sum()
    return ((mu_a - mu_b) ** 2).sum().item() + (s_a.trace() + s_b.trace() - 2 * cross).item()


# ---------------------------------------------------------------- sampling

def generate(model, sched, labels, sampler, steps, device, seed, batch):
    """All digits for one setting, in chunks. Chunk k always starts from the noise of
    seed + k, so every setting starts from exactly the same noise."""
    out = []
    sync(device)
    start = time.time()
    for k, i in enumerate(range(0, len(labels), batch)):
        gen = torch.Generator().manual_seed(seed + k)
        y = labels[i:i + batch]
        if sampler == "ddim":
            out.append(ddim_sample(model, sched, y, steps=steps, guidance=1.0, generator=gen))
        else:
            out.append(ddpm_sample(model, sched, y, guidance=1.0, generator=gen))
    sync(device)
    return torch.cat(out), time.time() - start


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", choices=["part1_unet", "part2_transformer"], default="part1_unet")
    p.add_argument("--n", type=int, default=100, help="digits per class, per setting")
    # 1000 is there on purpose: DDIM visiting every level, like DDPM, isolates the one
    # remaining difference -- the fresh noise.
    p.add_argument("--ddim-steps", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100, 1000])
    p.add_argument("--batch", type=int, default=500, help="digits per network call")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = get_device()
    model, ckpt = load_model(args.run, device)
    sched = NoiseSchedule(device=device)
    judge, judge_acc = get_judge(device)
    where = device_name(device)
    print(f"{args.run} (trained {ckpt['epoch']} epochs) on {where}; judge test accuracy {judge_acc:.2%}")

    labels = torch.arange(10, device=device).repeat_interleave(args.n)
    real_test, _ = load_mnist(device, train=False)
    _, real_feats = judge_all(judge, real_test)

    # floor: n real training digits per class, judged the same way
    train_x, train_y = load_mnist(device)
    g = torch.Generator().manual_seed(args.seed)
    real_pick = torch.cat([(train_y == d).nonzero().squeeze(1)[torch.randperm(int((train_y == d).sum()),
                                                                                generator=g)[:args.n].to(device)]
                           for d in range(10)])
    real = train_x[real_pick]

    settings = [("ddim", s) for s in args.ddim_steps] + [("ddpm", sched.T)]
    warm_up(model, device, min(args.batch, len(labels)))
    rows, firsts = [], []
    for sampler, steps in settings:
        x, seconds = generate(model, sched, labels, sampler, steps, device, args.seed, args.batch)
        logits, feats = judge_all(judge, x)
        r = {"sampler": sampler, "steps": steps, "network_calls": steps, "seconds": round(seconds, 2),
             "agrees": round((logits.argmax(1) == labels).float().mean().item(), 4),
             "fd": round(frechet(feats, real_feats), 2)}
        rows.append(r)
        firsts.append(x[::args.n])
        print(f"{sampler.upper():4s} {steps:5d} steps  {seconds:7.1f} s  agrees {r['agrees']:.1%}  FD {r['fd']:.2f}")

    logits, feats = judge_all(judge, real)
    floor = {"sampler": "real", "steps": 0, "network_calls": 0, "seconds": 0.0,
             "agrees": round((logits.argmax(1) == labels).float().mean().item(), 4),
             "fd": round(frechet(feats, real_feats), 2)}
    rows.append(floor)
    firsts.append(real[::args.n])
    print(f"real digits             agrees {floor['agrees']:.1%}  FD {floor['fd']:.2f}  (the floor for n = {args.n})")

    out = results_dir(args.run)
    grid = torch.cat(firsts).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
    save_image(to_pixels(grid), os.path.join(out, "sampler_compare.png"), nrow=10, padding=2, pad_value=1.0)
    with open(os.path.join(out, "sampler_compare.json"), "w") as f:
        json.dump({"device": where, "run": args.run, "epochs_trained": ckpt["epoch"], "n_per_digit": args.n,
                   "digits_per_setting": len(labels), "guidance": 1.0, "seed": args.seed,
                   "judge_test_acc": round(judge_acc, 4), "rows": rows}, f, indent=1)
    print(f"wrote {os.path.join(out, 'sampler_compare.json')} and sampler_compare.png")


if __name__ == "__main__":
    main()
