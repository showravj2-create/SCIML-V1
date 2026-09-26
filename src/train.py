from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from src.models.deeponet import DeepONet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    data = np.load("data/train.npz")
    branch = torch.tensor(data["branch"], dtype=torch.float32)
    target = torch.tensor(data["target"], dtype=torch.float32)
    x = torch.tensor(data["x"], dtype=torch.float32).unsqueeze(-1)

    # Normalize branch variables and outputs for stable training.
    bx_mean, bx_std = branch.mean(0), branch.std(0).clamp_min(1e-6)
    y_mean, y_std = target.mean((0,1)), target.std((0,1)).clamp_min(1e-6)
    branch_n = (branch-bx_mean)/bx_std
    target_n = (target-y_mean)/y_std

    ds = TensorDataset(branch_n, target_n)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    model = DeepONet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(300):
        model.train()
        total = 0.0
        for b, y in dl:
            b, y = b.to(DEVICE), y.to(DEVICE)
            pred = model(b, x.to(DEVICE))
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()*len(b)
        if epoch % 25 == 0:
            print(f"epoch={epoch:03d} loss={total/len(ds):.6e}")

    Path("results").mkdir(exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "bx_mean": bx_mean, "bx_std": bx_std,
        "y_mean": y_mean, "y_std": y_std,
    }, "results/deeponet.pt")
    print("Saved results/deeponet.pt")

if __name__ == "__main__":
    main()
