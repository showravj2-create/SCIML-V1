# Research analysis: Ahmad Peyvan

## 1. Research fingerprint

Peyvan's current research combines:
- scientific machine learning
- neural operators
- high-order CFD
- compressible/high-speed flows
- physics-aware AI
- geometry-dependent surrogate modeling
- reduced-order models and digital twins

His current Vanderbilt research page explicitly emphasizes high-order CFD and physics-aware ML for high-speed and reacting flows.

## 2. Papers that matter most for this project

### RiemannONets (2024)
Core idea: learn Riemann-problem solution operators with DeepONet, including a two-stage training procedure that creates an interpretable/orthonormal trunk basis. The paper targets pressure ratios up to 1e10 and compares DeepONet variants with a parameter-conditioned U-Net.

**What we borrow:** operator-learning formulation, discontinuous solutions, basis interpretation, two-stage training.

### Transformers as neural operators (2025)
This work studies attention-based operator learning for differential equations with finite regularity and discontinuous solutions.

**What we borrow later:** Transformer baseline and robustness on rough/shock-like solutions.

### Fusion-DeepONet (2026)
The most important current paper for the Vanderbilt position. It maps geometry/flow parameters to high-speed flow fields, compares DeepONet/FNO/U-Net/MeshGraphNet, works with scarce data and irregular grids, and adds derivative-enhanced loss for heat-flux prediction.

**What we borrow later:** scarce-data experimental design, geometry conditioning, irregular-grid thinking, derivative-aware objectives.

## 3. Project progression

Milestone A — 1D Euler/Riemann:
- numerical ground truth
- vanilla DeepONet
- two-stage/orthonormal DeepONet
- error metrics
- interpolation vs extrapolation

Milestone B — stronger operator comparison:
- FNO
- parameter-conditioned U-Net
- Transformer operator

Milestone C — Peyvan-style extension:
- geometry-dependent 2D PDE
- irregular query points
- geometry encoder
- derivative-enhanced loss

Milestone D — research result:
- ablation study
- limited-data curves
- error vs shock strength
- basis-function/SVD analysis
- reproducibility package

## 4. What would make this PhD-level

The goal is NOT to merely copy a paper. A strong portfolio contribution should answer a new question such as:

> How does operator architecture and basis regularization affect generalization to unseen shock strengths when training data are scarce?

That gives us a controlled research question with measurable hypotheses.

## 5. Code-level reproduction insight

The public RiemannONet entry point explicitly trains the trunk first, then the branch, then predicts. The trunk code learns a spatial basis and sample-specific coefficient matrix. The branch stage loads the best trunk checkpoint, performs QR or SVD on the learned trunk basis, transforms the coefficients, and trains the branch network against those transformed coefficients.

Our PyTorch implementation follows that computational structure without copying the original code: Stage 1 learns the spatial trunk basis plus sample coefficients; QR orthogonalizes the basis; Stage 2 learns the coefficient map from initial-condition parameters to the orthogonal basis coefficients.

## 6. First controlled experiment

The first run used 40 training Riemann cases, 10 interpolation test cases and 10 harder extrapolation cases. The numerical solver used 64 spatial points to keep the experiment CPU-friendly.

Observed relative L2 errors:
- Vanilla DeepONet: 0.195 test, 0.638 extrapolation.
- Two-stage DeepONet: 0.248 test, 0.745 extrapolation.

These numbers are **not** presented as evidence against the two-stage method. They are a baseline experiment with a tiny dataset, a low-resolution finite-volume solver, and a simplified architecture. The important research result is that we now have a reproducible baseline and can investigate why the two-stage model underperforms in this regime.

Next experiments should test width, training-set size, activation, orthogonalization choice (QR vs SVD), output normalization, and pressure-ratio extrapolation separately.
