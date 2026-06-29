"""Week 2 - Part 2: The same MLP, now doing classification.

Big idea: the training loop is identical. To switch from regression to
classification we only change the *data* and the *loss function*:
    - data : two interleaving half-moons (make_moons) with integer labels
    - loss : nn.CrossEntropyLoss (the classification version of MSE)

The network outputs 2 numbers (one score per class); argmax picks the winner.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


def main():
    torch.manual_seed(0)
    X_np, y_np = make_moons(n_samples=300, noise=0.2, random_state=0)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)

    # TODO: build an MLP classifier and train it.
    #   1. clf = nn.Sequential(
    #          nn.Linear(2, _),   # 2 input features
    #          nn.ReLU(),
    #          nn.Linear(_, 2),   # 2 classes -> 2 outputs
    #      )
    #      loss_fn   = nn.CrossEntropyLoss()
    #      optimizer = torch.optim.Adam(clf.parameters(), lr=0.01)
    #   2. Train for ~500 epochs with the SAME 4-step loop:
    #        logits = clf(X) ; loss = loss_fn(logits, y) ; ... backward/step
    #   3. accuracy: pred = clf(X).argmax(dim=1); acc = (pred == y).float().mean()
    #      print the accuracy.
    raise NotImplementedError("Build and train the MLP classifier here.")


if __name__ == "__main__":
    main()
