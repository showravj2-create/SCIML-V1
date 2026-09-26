# Peyvan-Inspired Scientific ML: Data-Efficient Neural Operators for Compressible Flow

A research-oriented project inspired by Ahmad Peyvan's work on **RiemannONets**, **neural operators**, and data-efficient surrogates for high-speed flow.

## Research question

Can a neural operator learn the mapping

**initial condition → full PDE solution**

for 1D compressible Euler/Riemann problems, and can a structured two-stage DeepONet improve accuracy and interpretability over a vanilla DeepONet?

This is intentionally smaller than Peyvan's hypersonic CFD problems, but it targets the same methodological ideas:

- operator learning rather than ordinary pointwise regression
- discontinuous/shock-containing PDE solutions
- data-efficient surrogate modeling
- interpretable learned basis functions
- comparison against numerical ground truth

## Why this is a strong PhD portfolio project

Peyvan's research emphasizes neural operators for difficult compressible-flow solutions and, in Fusion-DeepONet, geometry-dependent surrogate modeling with scarce data. His RiemannONets paper studies extreme Riemann problems and a two-stage DeepONet whose trunk basis is orthonormalized before training the branch network.

This project starts with the mathematically cleanest version of that problem, then leaves a clear path toward:
1. harder pressure ratios,
2. parameter extrapolation,
3. Transformer neural operators,
4. geometry-dependent 2D problems,
5. derivative/physics-aware losses.

## Repository

```text
peyvan-inspired-sciml/
├── README.md
├── requirements.txt
├── scripts/
│   └── generate_data.py
├── src/
│   ├── physics/
│   │   └── euler1d.py
│   ├── models/
│   │   └── deeponet.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── 01_end_to_end_walkthrough.ipynb
└── results/
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/generate_data.py --n_train 600 --n_test 120
python -m src.train
python -m src.evaluate
```

## Core experiment

Each sample is a Riemann problem with random left/right states. The numerical solver generates a full space-time-free snapshot at a fixed final time.

The DeepONet receives:

- **branch input:** sampled initial condition/state parameters
- **trunk input:** spatial coordinate x

and predicts:

- density ρ(x)
- velocity u(x)
- pressure p(x)

The important test is not only interpolation. We also reserve harder pressure-ratio cases for an extrapolation test.

## Scientific honesty

This is **not** a reproduction of the full RiemannONets paper. It is a student-built, Peyvan-aligned implementation designed to demonstrate understanding and independent research ability.

The next upgrade should be a faithful reproduction on the paper's public code/data, followed by an original extension.
