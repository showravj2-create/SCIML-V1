import argparse
from pathlib import Path
import numpy as np
from src.physics.euler1d import solve_riemann

def sample_state(rng, extreme=False):
    rho = rng.uniform(0.6, 1.8)
    u = rng.uniform(-0.5, 0.5)
    if extreme:
        p = 10 ** rng.uniform(0.0, 1.3)
    else:
        p = 10 ** rng.uniform(-0.1, 0.6)
    return rho, u, p

def make_split(n, seed, extreme=False):
    rng = np.random.default_rng(seed)
    X, Y, params = [], [], []
    for _ in range(n):
        L = sample_state(rng, extreme=extreme)
        R = sample_state(rng, extreme=extreme)
        x, rho, u, p = solve_riemann(L, R, nx=64, final_time=0.08)
        X.append(np.array(L+R, dtype=np.float32))
        Y.append(np.stack([rho,u,p], axis=-1).astype(np.float32))
        params.append(np.array(L+R, dtype=np.float32))
    return np.stack(X), np.stack(Y), np.stack(params)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--n_test", type=int, default=120)
    ap.add_argument("--n_extrap", type=int, default=120)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    Xtr, Ytr, _ = make_split(args.n_train, 1, extreme=False)
    Xte, Yte, _ = make_split(args.n_test, 2, extreme=False)
    Xex, Yex, _ = make_split(args.n_extrap, 3, extreme=True)

    x = solve_riemann((1,0,1),(0.8,0,0.8), nx=64, final_time=0.08)[0].astype(np.float32)

    np.savez_compressed(out/"train.npz", branch=Xtr, target=Ytr, x=x)
    np.savez_compressed(out/"test.npz", branch=Xte, target=Yte, x=x)
    np.savez_compressed(out/"extrapolation.npz", branch=Xex, target=Yex, x=x)
    print("Saved datasets to", out)

if __name__ == "__main__":
    main()
