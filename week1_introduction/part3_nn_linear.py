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

    # TODO: implement the standard PyTorch training loop.
    #   1. model     = nn.(1, 1)
    #      loss_fn   = nn.()
    #      optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    #   2. Loop for ~200 epochs:
    #        - forward:  y_pred = model(X)
    #        - loss:     loss = loss_fn(y_pred, y)
    #        - optimizer.zero_grad()
    #        - autograd
    #        - step
    #        - record loss.item()
    #   3. print model.weight.item(), model.bias.item() (close to 2 and 1)
    #   4. scatter the data and overlay the fitted line: model(X).detach()
    
    print(f"w = {w.item():.4f}, b = {b.item():.4f}  (target: 2 and 1)")

    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss curve")
    plt.show()

    plt.scatter(X, y, s=8, alpha=0.5)
    plt.plot(X, model(y).detach(), color="red")
    plt.title("Linear fit")
    plt.show()

if __name__ == "__main__":
    main()
