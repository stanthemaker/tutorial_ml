"""Week 1 - Part 2: Same loop, but let autograd compute the gradients.

Only the "compute gradients by hand" step changes: we replace the manual
gradient formulas with a single call to loss.backward(). Everything else in
the training loop stays exactly the same.

Two new ideas:
    requires_grad=True  -> tell PyTorch to track this tensor for gradients
    grad.zero_()        -> gradients accumulate, so reset them each step
"""

import torch
import matplotlib.pyplot as plt


def main():
    torch.manual_seed(0)
    X = torch.linspace(-3, 3, 100).reshape(-1, 1)
    y = 2 * X + 1 + 0.5 * torch.randn_like(X)  # true: w=2, b=1, plus noise

    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    lr = 0.05
    losses = []

    for epoch in range(200):
        y_pred = w * X + b                   # forward
        loss = ((y_pred - y) ** 2).mean()    # loss (MSE)
        loss.backward()                      # backward: autograd fills .grad
        with torch.no_grad():                # step (don't track this update)
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()                   # reset grads (they accumulate)
            b.grad.zero_()
        losses.append(loss.item())

    print(f"w = {w.item():.4f}, b = {b.item():.4f}  (target: 2 and 1)")

    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Autograd gradient descent - loss curve")
    plt.show()


if __name__ == "__main__":
    main()
