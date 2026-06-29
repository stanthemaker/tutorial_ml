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

    # TODO: implement manual gradient descent.
    #   1. Initialise parameters w, b (as plain tensors) and a learning rate.
    #   2. Loop for ~200 epochs:
    #        - forward:  y_pred = w * X + b
    #        - loss:     MSE 
    #        - gradient: grad_w 
    #                    grad_b 
    #        - step:     w -=  ; b -= 
    #        - record loss.item() into a list
    #   3. print w, b (should be close to 2 and 1)
    #   4. plot the loss curve (epoch vs loss)
    raise NotImplementedError("Implement manual gradient descent here.")


if __name__ == "__main__":
    main()
