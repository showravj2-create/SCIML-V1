from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.models.deeponet import DeepONet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_file(path, ckpt, name):
    d = np.load(path)
    c = torch.load(ckpt, map_location=DEVICE)
    model = DeepONet().to(DEVICE)
    model.load_state_dict(c["model"])
    model.eval()

    branch = torch.tensor((d["branch"]-c["bx_mean"].numpy())/c["bx_std"].numpy(), dtype=torch.float32).to(DEVICE)
    x = torch.tensor(d["x"], dtype=torch.float32).unsqueeze(-1).to(DEVICE)

    with torch.no_grad():
        pred_n = model(branch, x).cpu().numpy()
    pred = pred_n*c["y_std"].numpy()+c["y_mean"].numpy()
    true = d["target"]

    rel = np.linalg.norm(pred-true)/np.linalg.norm(true)
    print(f"{name} relative L2 error: {rel:.6f}")

    i = 0
    fig, ax = plt.subplots(3,1,figsize=(8,8),sharex=True)
    labels = ["density", "velocity", "pressure"]
    for j in range(3):
        ax[j].plot(d["x"], true[i,:,j], label="numerical")
        ax[j].plot(d["x"], pred[i,:,j], "--", label="DeepONet")
        ax[j].set_ylabel(labels[j])
        ax[j].legend()
    ax[-1].set_xlabel("x")
    fig.tight_layout()
    Path("results").mkdir(exist_ok=True)
    fig.savefig(f"results/{name}_example.png", dpi=180)
    plt.close(fig)

def main():
    evaluate_file("data/test.npz","results/deeponet.pt","test")
    evaluate_file("data/extrapolation.npz","results/deeponet.pt","extrapolation")

if __name__ == "__main__":
    main()
