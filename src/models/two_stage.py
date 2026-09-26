import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(64,64), out_dim=32):
        super().__init__()
        layers=[]; d=in_dim
        for h in hidden:
            layers += [nn.Linear(d,h), nn.Tanh()]; d=h
        layers.append(nn.Linear(d,out_dim)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class Trunk(nn.Module):
    def __init__(self,width=48): super().__init__(); self.net=MLP(1,out_dim=width)
    def forward(self,x): return self.net(x)

class Branch(nn.Module):
    def __init__(self,branch_dim=6,width=48,out_dim=3): super().__init__(); self.net=MLP(branch_dim,out_dim=width*out_dim); self.width=width; self.out_dim=out_dim
    def forward(self,b): return self.net(b).view(-1,self.out_dim,self.width)

class VanillaDeepONet(nn.Module):
    def __init__(self,branch_dim=6,width=48,out_dim=3):
        super().__init__(); self.branch=Branch(branch_dim,width,out_dim); self.trunk=Trunk(width); self.bias=nn.Parameter(torch.zeros(out_dim)); self.width=width
    def forward(self,b,x): return torch.einsum('bmw,nw->bnm',self.branch(b),self.trunk(x))+self.bias
