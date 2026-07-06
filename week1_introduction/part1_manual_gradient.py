import torch
import matplotlib.pyplot as plt


def main():
    torch.manual_seed(0)
    X = torch.linspace(-3, 3, 100).reshape(-1, 1)
    y = 2 * X + 1 + 0.5 * torch.randn_like(X)

    w = torch.tensor(0.0)
    b = torch.tensor(0.0)
    lr = 0.05
    losses = []

    for epoch in range(200):
        y_pred = w * X + b  # forward
        loss = ((y_pred - y) ** 2).mean()  # loss (MSE)
        grad_w = (2 * (y_pred - y) * X).mean()
        grad_b = (2 * (y_pred - y)).mean()
        w -= lr * grad_w  # step
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
