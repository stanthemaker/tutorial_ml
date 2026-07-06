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

    linear = nn.Linear(1, 1)
    opt = torch.optim.SGD(linear.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    for epoch in range(300):
        loss = loss_fn(linear(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    plt.scatter(X, y, s=8, alpha=0.4)
    plt.plot(X, linear(X).detach(), color="red")  # only ever a straight line
    plt.title("A linear model cannot fit a sine wave")
    plt.show()


if __name__ == "__main__":
    main()
