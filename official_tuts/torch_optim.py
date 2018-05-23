import torch

bs, ins, hids, outs = 64, 1000, 100, 10

x = torch.randn(bs, ins)
y = torch.randn(bs, outs)

model = torch.nn.Sequential(
    torch.nn.Linear(ins, hids),
    torch.nn.ReLU(),
    torch.nn.Linear(hids, outs),
)

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(501):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    
    if t % 50 == 0:
        print(t, loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
