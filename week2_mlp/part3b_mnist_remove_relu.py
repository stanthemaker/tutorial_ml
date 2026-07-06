"""Week 2 - Part 3b: The surprise -- remove the ReLU from the MNIST MLP.

Big idea: in demo_linear_limits.py, taking away the non-linearity was
*catastrophic* -- a straight line simply cannot fit a sine wave. You'd expect
the same disaster here. Let's test it: take part3's MLP and delete the ReLU.

    linear_stack = nn.Sequential(
        nn.Linear(784, 128),
        nn.Linear(128, 10),   # <- ReLU removed
    )

Two stacked Linear layers with no activation between them is mathematically
still just ONE linear function:

    W2 (W1 x + b1) + b2  =  (W2 W1) x + (W2 b1 + b2)  =  W_combined x + b_combined

So this model has the same *effective* capacity as a single nn.Linear(784, 10)
-- despite having two layers on paper. It is a linear classifier.

The surprise: on MNIST this barely hurts (~90-92% vs ~97-98% with ReLU).
Removing the non-linearity was fatal on the sine wave but almost free here.
Why? -> see part3c_mnist_pca_verify.py.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# reuse the exact data loading + eval from part3 so the comparison is fair
from part3_mlp_mnist import load_mnist, evaluate


def train_and_score(model, train_loader, test_loader, epochs=15):
    """Same 4-step training loop and hyperparameters as part3."""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            x = images.view(-1, 784)
            logits = model(x)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return evaluate(model, test_loader)


def main():
    torch.manual_seed(0)

    train_ds = load_mnist(train=True, n_subset=20000)  # same subset as part3
    test_ds = load_mnist(train=False)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    # Two Linear layers, no ReLU -> effectively a single linear map.
    linear_stack = nn.Sequential(
        nn.Linear(784, 128),
        nn.Linear(128, 10),  # <- ReLU removed
    )
    acc_stack = train_and_score(linear_stack, train_loader, test_loader)

    # A plain single Linear(784, 10) makes the "it's still just linear" point
    # concrete: same math, one layer, roughly the same accuracy as the stack.
    single_linear = nn.Sequential(nn.Linear(784, 10))
    acc_single = train_and_score(single_linear, train_loader, test_loader)

    print(f"Linear model (ReLU removed) test accuracy: {acc_stack:.2%}")
    print(f"Plain nn.Linear(784, 10) test accuracy:    {acc_single:.2%}")
    # Compare: part3_mlp_mnist.py (with ReLU) got ~97-98%.
    # These models, with the non-linearity removed, still get ~90-92%.
    # Contrast with demo_linear_limits.py: removing non-linearity there was
    # catastrophic (a straight line cannot fit a sine wave at all).
    # Here it barely matters. Why? -> see part3c_mnist_pca_verify.py (PCA).


if __name__ == "__main__":
    main()
