"""Week 2 - Part 1: Fit the curve with an MLP (regression).

Same sine data as the demo, same training loop as last week. The ONLY thing
that changes is the model definition: we stack Linear -> ReLU -> Linear with
nn.Sequential. The ReLU (a non-linear activation) is what lets the model bend
the straight line into a curve.

Try afterwards:
    - change the 32 hidden units to 4 (coarser) or 128 (smoother)
    - delete the nn.ReLU() line and watch the model collapse back to a line
      (no activation => an MLP is just linear regression again)
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def main():
    torch.manual_seed(0)
    X = torch.linspace(-3, 3, 200).reshape(-1, 1)
    y = torch.sin(X) + 0.1 * torch.randn_like(X)  # a curvy relationship
    #TODO 1 define model with 32 neurons in hidden layer
    mlp = nn.Sequential(
        
    )
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(1000):       # same 4-step loop as week 1
        # get y_pred
        # get loss
        # clear grad in optimizer
        # backpropagate
        # update weights
        

    plt.scatter(X, y, s=8, alpha=0.4)
    plt.plot(X, mlp(X).detach(), color="red")  # now hugs the sine wave
    plt.title("MLP regression on a sine wave")
    plt.show()


if __name__ == "__main__":
    main()
