"""Week 5 - Generate digits from a trained checkpoint (Part 1 or Part 2).

    python sample.py                                  # Part 1's U-Net, DDIM 50 steps, digits 0-9, 8 each
    python sample.py --run part2_transformer          # Part 2's transformer
    python sample.py --sampler ddpm                   # the original 1000-step sampler
    python sample.py --digits 777 --n 16              # sixteen of each of 7, 7, 7
    python sample.py --guidance 1                     # no guidance: the plain labelled model
    python sample.py --trajectory                     # also save noise -> digit, step by step

Output (in results/<run>/):
  samples_<sampler>_w<guidance>.png     one row per requested digit
  trajectory_<sampler>.png              (with --trajectory) one digit per row,
                                        left = pure noise, right = finished digit
  sample_log.json                       seconds taken per run, for the slides
"""

import argparse
import json
import os
import time

import torch
from torchvision.utils import save_image

from common import (NoiseSchedule, ddim_sample, ddpm_sample, get_device, results_dir,
                    weights_path)
from part1_unet_diffusion import UNet
from part2_transformer_diffusion import DiT


def load_model(name, device):
    """Rebuild the network from the sizes stored in the checkpoint, then load the weights."""
    ckpt = torch.load(weights_path(name), map_location=device)
    c = ckpt["config"]
    if "dim" in c:
        model = DiT(dim=c["dim"], depth=c["depth"], heads=c["heads"], patch=c["patch"])
    else:
        model = UNet(base=c["base"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def sync(device):
    """Wait for the GPU to finish, so the timing is real."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", choices=["part1_unet", "part2_transformer"], default="part1_unet")
    p.add_argument("--digits", default="0123456789", help="which digits, one row each")
    p.add_argument("--n", type=int, default=8, help="samples per row")
    p.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    p.add_argument("--steps", type=int, default=50, help="DDIM steps (DDPM always uses 1000)")
    p.add_argument("--guidance", type=float, default=3.0, help="1 = no guidance")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trajectory", action="store_true")
    args = p.parse_args()

    device = get_device()
    model, ckpt = load_model(args.run, device)
    sched = NoiseSchedule(device=device)
    print(f"loaded {weights_path(args.run)}  (trained {ckpt['epoch']} epochs)")

    out_dir = results_dir(args.run)
    labels = torch.tensor([int(d) for d in args.digits], device=device).repeat_interleave(args.n)
    gen = torch.Generator().manual_seed(args.seed)
    traj = [] if args.trajectory else None

    sync(device)
    start = time.time()
    if args.sampler == "ddim":
        x = ddim_sample(model, sched, labels, steps=args.steps, guidance=args.guidance,
                        generator=gen, trajectory=traj)
        calls = args.steps
    else:
        x = ddpm_sample(model, sched, labels, guidance=args.guidance, generator=gen, trajectory=traj)
        calls = sched.T
    sync(device)
    seconds = time.time() - start

    name = f"{args.sampler}_w{args.guidance:g}"
    out = os.path.join(out_dir, f"samples_{name}.png")
    save_image((x + 1) / 2, out, nrow=args.n, padding=2, pad_value=1.0)
    print(f"{len(labels)} digits, {calls} network calls, {seconds:.1f}s  ->  {out}")
    print(f"rows: {' '.join(args.digits)}")

    if traj is not None:
        # 10 evenly spaced moments of the chain, first sample of each row
        picks = torch.linspace(0, len(traj) - 1, 10).long().tolist()
        first = torch.arange(0, len(labels), args.n)
        cols = [traj[i][1][first] for i in picks]                    # 10 x (rows, 1, 32, 32)
        strip = torch.stack(cols, dim=1).flatten(0, 1)               # row-major: digit, then step
        tout = os.path.join(out_dir, f"trajectory_{args.sampler}.png")
        save_image((strip + 1) / 2, tout, nrow=len(picks), padding=2, pad_value=1.0)
        print(f"trajectory (left = noise, right = digit)  ->  {tout}")

    log_path = os.path.join(out_dir, "sample_log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    log[f"{args.sampler}_{calls}steps_w{args.guidance:g}"] = {
        "sampler": args.sampler, "network_calls": calls, "guidance": args.guidance,
        "digits": len(labels), "seconds": round(seconds, 2)}
    with open(log_path, "w") as f:
        json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
