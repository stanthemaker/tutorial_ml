"""Week 1 - Part 1: Pure manual gradient descent.

Goal: feel gradient descent by hand. We compute the gradients ourselves
(no autograd) and update the parameters with the rule:  param -= lr * grad.

The training loop skeleton is always the same:
    forward  -> compute prediction
    loss     -> measure error (MSE)
    backward -> compute gradients
    step     -> update parameters
"""

import torch
import matplotlib.pyplot as plt


def main():
    # Fake data with a known answer (true relation: y = 2x + 1) so we can
    # check whether the model learned the right thing.
    torch.manual_seed(0)
    X = torch.linspace(-3, 3, 100).reshape(-1, 1)
    y = 2 * X + 1 + 0.5 * torch.randn_like(X)  # true: w=2, b=1, plus noise

    w = torch.tensor(0.0)
    b = torch.tensor(0.0)
    lr = 0.05
    losses = []

    for epoch in range(200):
        y_pred = w * X + b                       # forward
        loss = ((y_pred - y) ** 2).mean()        # loss (MSE)
        grad_w = (2 * (y_pred - y) * X).mean()   # gradient by hand
        grad_b = (2 * (y_pred - y)).mean()
        w -= lr * grad_w                         # step
        b -= lr * grad_b
        losses.append(loss.item())

    print(f"w = {w.item():.4f}, b = {b.item():.4f}  (target: 2 and 1)")

    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Manual gradient descent - loss curve")
    plt.show()

    plt.scatter(X, y, s=8, alpha=0.5)
    plt.plot(X, (w * X + b).detach(), color="red")
    plt.title("Linear fit")
    plt.show()


if __name__ == "__main__":
    main()
