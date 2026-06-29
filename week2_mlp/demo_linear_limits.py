"""Week 2 - Demo: the limit of a linear model.

We build a curvy dataset (a sine wave) and try to fit it with last week's
nn.Linear. It fails on purpose: a straight line can never hug a wave. This
"failure" is the motivation for adding a hidden layer + activation (the MLP).
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def main():
    torch.manual_seed(0)
    X = torch.linspace(-3, 3, 200).reshape(-1, 1)
    y = torch.sin(X) + 0.1 * torch.randn_like(X)  # a curvy relationship

    # TODO: fit this curvy data with a *linear* model and watch it fail.
    #   1. linear  = nn.Linear(_, _)
    #      opt     = torch.optim.SGD(linear.parameters(), lr=0.05)
    #      loss_fn = nn.MSELoss()
    #   2. Train for ~300 epochs with the standard 4-step loop.
    #   3. scatter the data and overlay linear(X).detach():
    #      it can only draw a straight line, so it cannot cover the wave.
    raise NotImplementedError("Fit the sine data with nn.Linear (it should fail).")


if __name__ == "__main__":
    main()
