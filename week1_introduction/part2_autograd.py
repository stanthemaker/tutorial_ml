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

    # TODO: implement gradient descent with autograd.
    #   1. Create w, b with requires_grad=True, and a learning rate.
    #   2. Loop for ~200 epochs:
    #        - forward:  y_pred = w * X + b
    #        - loss:     MSE = mean((y_pred - y) ** 2)
    #        - backward:    # autograd fills w.grad, b.grad
    #        - step inside `with torch.no_grad():`
    #              w -= lr * w.grad ; b -= lr * b.grad
    #        - reset gradients:
    #        - record loss.item()
    #   3. print w, b (should be close to 2 and 1)
    #   4. plot the loss curve
    raise NotImplementedError("Implement autograd-based gradient descent here.")


if __name__ == "__main__":
    main()
