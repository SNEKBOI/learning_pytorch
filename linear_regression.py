import torch
import matplotlib.pyplot as plt
import numpy as np

dtype = torch.float


x = np.arange(0, 9)

y = (x + 2 * np.random.random((1,9)) - 1).reshape(9)

x = torch.from_numpy(x)
y = torch.from_numpy(y)

x = x.type(torch.FloatTensor)
y = y.type(torch.FloatTensor)

print(f"x and y shape: {x.shape} {y.shape}")



m = torch.tensor([0], dtype=dtype, requires_grad=True)
b = torch.tensor([0], dtype=dtype, requires_grad=True)


lr = 1e-3
for t in range(5001):
    y_pred = m * x + b

    loss = (y_pred - y).pow(2).sum()
    if t % 500 == 0:
        print(t, loss.item())
    loss.backward()

    with torch.no_grad():
        m -= lr * m.grad
        b -= lr * b.grad

        m.grad.zero_()
        b.grad.zero_()

print(f"m: {m}")
print(f"b: {b}")

hors = [0, 15]
vers = [b, m * 15]

plt.axis('equal')
plt.scatter(x, y)
plt.plot(hors, vers)
plt.show()
