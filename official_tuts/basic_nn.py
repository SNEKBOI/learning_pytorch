import torch

dtype = torch.float

bs, ins, hids, outs = 64, 1000, 100, 10

x = torch.randn(bs, ins, dtype=dtype)
y = torch.randn(bs, outs, dtype=dtype)

w1 = torch.randn(ins, hids, dtype=dtype, requires_grad=True)
w2 = torch.randn(hids, outs, dtype=dtype, requires_grad=True)

lr = 1e-6
for i in range(501):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    
    if i % 50 == 0: 
        print(i, loss.item())

    loss.backward()

    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()

