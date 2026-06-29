"""Week 1 - Part 3: The idiomatic PyTorch way (nn.Linear + optimizer).

This is THE TEMPLATE. Every later lesson (including the MLP next week) reuses
this exact four-step loop; only the `model = ...` line changes.

    forward  -> y_pred = model(X)
    loss     -> loss = loss_fn(y_pred, y)
    backward -> optimizer.zero_grad(); loss.backward()
    step     -> optimizer.step()
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def main():
    torch.manual_seed(0)
    X = torch.linspace(-3, 3, 100).reshape(-1, 1)
    y = 2 * X + 1 + 0.5 * torch.randn_like(X)  # true: w=2, b=1, plus noise

    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    losses = []
    for epoch in range(200):
        y_pred = model(X)            # forward
        loss = loss_fn(y_pred, y)    # loss
        optimizer.zero_grad()        # reset grads
        loss.backward()              # backward
        optimizer.step()             # step
        losses.append(loss.item())

    print(f"weight = {model.weight.item():.4f}, bias = {model.bias.item():.4f}"
          "  (target: 2 and 1)")

    plt.scatter(X, y, s=8, alpha=0.5)
    plt.plot(X, model(X).detach(), color="red")
    plt.title("nn.Linear fit")
    plt.show()


if __name__ == "__main__":
    main()
