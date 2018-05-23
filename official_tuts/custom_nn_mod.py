import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, ins, hids, outs):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(ins, hids)
        self.linear2 = torch.nn.Linear(hids, outs)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

bs, ins, hids, outs = 64, 1000, 100, 10

x = torch.randn(bs, ins)
y = torch.randn(bs, outs)

model = TwoLayerNet(ins, hids, outs)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(501):
    y_pred = model(x)

    loss = criterion(y_pred, y)

    if t % 50 == 0:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

