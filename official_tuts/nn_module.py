import torch

bs, ins, hids, outs = 64, 1000, 100, 10

x = torch.randn(bs, ins)
y = torch.randn(bs, outs)

model = torch.nn.Sequential(
    torch.nn.Linear(ins, hids),
    torch.nn.ReLU(),
    torch.nn.Linear(hids, outs),
)

loss_fn = torch.nn.MSELoss(size_average=True)

lr = 1e-4

for t in range(501):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    if t % 50 == 0:
        print(t, loss.item())

    model.zero_grad()

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad

