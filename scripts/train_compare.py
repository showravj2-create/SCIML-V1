from pathlib import Path
import numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from src.models.two_stage import VanillaDeepONet, Trunk, Branch

torch.manual_seed(7); np.random.seed(7)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
Path('results').mkdir(exist_ok=True)

def load(path):
 d=np.load(path); return d['branch'].astype('float32'),d['target'].astype('float32'),d['x'].astype('float32')
tr_b,tr_y,x=load('data/train.npz'); te_b,te_y,_=load('data/test.npz'); ex_b,ex_y,_=load('data/extrapolation.npz')
bm=tr_b.mean(0); bs=tr_b.std(0)+1e-6; ym=tr_y.mean((0,1)); ys=tr_y.std((0,1))+1e-6
norm=lambda b,y: ((b-bm)/bs,(y-ym)/ys)
tr_b,tr_y=norm(tr_b,tr_y); te_b,te_y=norm(te_b,te_y); ex_b,ex_y=norm(ex_b,ex_y)
B=torch.tensor(tr_b).to(DEVICE); Y=torch.tensor(tr_y).to(DEVICE); X=torch.tensor(x).unsqueeze(-1).to(DEVICE)
BT=torch.tensor(te_b).to(DEVICE); YT=torch.tensor(te_y).to(DEVICE); BX=torch.tensor(ex_b).to(DEVICE); YX=torch.tensor(ex_y).to(DEVICE)
loss=nn.MSELoss()

def rel(pred,true): return (torch.linalg.norm(pred-true)/torch.linalg.norm(true)).item()

# Vanilla DeepONet
van=VanillaDeepONet().to(DEVICE); opt=torch.optim.Adam(van.parameters(),lr=2e-3)
van_hist=[]
for ep in range(300):
 opt.zero_grad(); p=van(B,X); l=loss(p,Y); l.backward(); opt.step(); van_hist.append(l.item())

# Stage 1: train trunk + sample-specific coefficient matrices A_i
trunk=Trunk().to(DEVICE); width=trunk.net.net[-1].out_features
A=nn.Parameter(torch.randn(B.shape[0],3,width,device=DEVICE)*0.05)
opt=torch.optim.Adam(list(trunk.parameters())+[A],lr=2e-3); s1=[]
for ep in range(350):
 opt.zero_grad(); phi=trunk(X); pred=torch.einsum('bnw,bmw->bnm',phi.expand(B.shape[0],-1,-1),A)
 l=loss(pred,Y); l.backward(); opt.step(); s1.append(l.item())
# QR orthonormalization of learned trunk basis
with torch.no_grad():
 phi=trunk(X); Q,R=torch.linalg.qr(phi,mode='reduced')
 # Since phi=Q R, phi @ A = Q @ (R @ A)
 C=torch.einsum('wk,bmk->bmw',R,A)
# Stage 2: learn coefficient operator branch -> C
branch=Branch().to(DEVICE); opt=torch.optim.Adam(branch.parameters(),lr=2e-3); s2=[]
for ep in range(350):
 opt.zero_grad(); predC=branch(B); l=loss(predC,C); l.backward(); opt.step(); s2.append(l.item())
with torch.no_grad():
 q=Q
 pte=torch.einsum('nw,bmw->bnm',q,branch(BT)); pex=torch.einsum('nw,bmw->bnm',q,branch(BX))
 # convert normalized predictions back to physical units
 pv=van(BT,X); px=van(BX,X)

print('DEVICE',DEVICE)
print('Vanilla test rel L2',rel(pv,YT),'extrapolation',rel(px,YX))
print('Two-stage test rel L2',rel(pte,YT),'extrapolation',rel(pex,YX))

# plots
fig,ax=plt.subplots(2,1,figsize=(7,6)); ax[0].plot(van_hist,label='Vanilla'); ax[0].set_title('Vanilla DeepONet training'); ax[0].set_yscale('log'); ax[0].legend(); ax[1].plot(s1,label='Stage 1'); ax[1].plot(s2,label='Stage 2'); ax[1].set_yscale('log'); ax[1].legend(); fig.tight_layout(); fig.savefig('results/training_comparison.png',dpi=180); plt.close(fig)
# first test example, physical units
idx=0
true=te_y[idx]*ys+ym
v= pv[idx].cpu().numpy()*ys+ym; t=pte[idx].cpu().numpy()*ys+ym
fig,ax=plt.subplots(3,1,figsize=(8,8),sharex=True)
for j,name in enumerate(['density','velocity','pressure']):
 ax[j].plot(x,true[:,j],label='numerical'); ax[j].plot(x,v[:,j],'--',label='vanilla'); ax[j].plot(x,t[:,j],':',label='two-stage'); ax[j].set_ylabel(name); ax[j].legend()
ax[-1].set_xlabel('x'); fig.tight_layout(); fig.savefig('results/vanilla_vs_two_stage.png',dpi=180); plt.close(fig)
np.savez('results/metrics.npz', vanilla_test=rel(pv,YT), vanilla_extrapolation=rel(px,YX), two_stage_test=rel(pte,YT), two_stage_extrapolation=rel(pex,YX))
