import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 128), out_dim=64):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DeepONet(nn.Module):
    """Multi-output DeepONet.

    The branch network produces one coefficient vector per physical field;
    the trunk network supplies the spatial basis functions.
    """
    def __init__(self, branch_dim=6, trunk_dim=1, width=64, out_dim=3):
        super().__init__()
        self.branch = MLP(branch_dim, out_dim=width * out_dim)
        self.trunk = MLP(trunk_dim, out_dim=width)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.width = width
        self.out_dim = out_dim

    def forward(self, branch, x):
        # b: [B, out_dim, width]
        b = self.branch(branch).view(-1, self.out_dim, self.width)
        # t: [N, width]
        t = self.trunk(x)
        # y: [B, N, out_dim]
        y = torch.einsum("bmw,nw->bnm", b, t)
        return y + self.bias
