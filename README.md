<div align="center">
  <h1>🌌 Quark Optim </h1>
  
  <p><strong>Institutional-Grade PyTorch-Accelerated Firefly Portfolio Optimization Engine</strong></p>

  <p>
    <a href="https://github.com/Anagatam/Quark/actions/workflows/ci.yml">
      <img src="https://github.com/Anagatam/Quark/actions/workflows/ci.yml/badge.svg" alt="CI Status">
    </a>
    <a href="https://pypi.org/project/quark-optim/">
      <img src="https://img.shields.io/pypi/v/quark-optim.svg" alt="PyPI Version">
    </a>
    <a href="https://opensource.org/licenses/Apache-2.0">
      <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
    </a>
    <a href="https://www.python.org/downloads/">
      <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python Version">
    </a>
    <a href="https://pytorch.org/">
      <img src="https://img.shields.io/badge/PyTorch-Accelerated-ee4c2c.svg" alt="PyTorch">
    </a>
  </p>
</div>

---

## ⚡ Non-Convex Portfolio Mastery

Traditional solvers (like SciPy's SLSQP) consistently fail or become trapped in local minima when confronted with realistic institutional constraints:
1. **Strict Cardinality ($K \le 8$)**: Selecting a small exact number of assets from a massive 500+ universe.
2. **Absolute Bounds**: e.g., $w_i \ge 5\%$, $w_i \le 25\%$.
3. **Coherent Risk Penalties**: Non-differentiable $CVaR_{95}$ and Max Drawdown functions.

**Quark** completely bypasses mathematical limitations by simulating highly dimensional swarms of biological "Fireflies" utilizing Heavy-Tailed **Mantegna's Levy Flights**. These entities jump aggressively across the non-convex objective space simultaneously, ensuring convergence on the *true* Global Optimum.

---

## 🏗️ Deep Architectural Features

### 1. The `SwarmTensor` GPU Accelerator
Quark's core evolutionary loops are fully vectorized into multidimensional `torch.Tensor` architectures. When deployed to CUDA/Apple Metal hardware, thousands of Firefly position permutations are calculated simultaneously, slashing NP-Hard search times from hours down to milliseconds.

### 2. Deep Denoising Autoencoder
Instead of relying on fragile historical sample covariances, Quark utilizes PyTorch deep learning to map empirical returns through a non-linear Latent Space Bottleneck. The engine distills idiosyncratic variance and recovers a perfectly denoised, structurally stable `cov_matrix_`.

### 3. Institutional Mathematics ($Luminescence$)
The `luminescence` module implements peer-reviewed quantitative mathematics designed for absolute preservation of Tier-1 capital:
- **Random Matrix Theory (RMT)**: Eigenvalue clipping utilizing Marchenko-Pastur distribution bounds ensures the covariance matrix is strictly Positive Definite (PD), completely preventing degenerate edge-case allocations.
- **Black-Litterman Synthesis**: Harmonically blends Deep Learning expected returns with Market-Implied global equilibrium priors to prevent portfolio implosions during regime shifts.
- **Conditional Value-at-Risk ($CVaR$)**: Optimizes strictly against the 5% worst-case left-tail distribution.

---

## 🚀 Quickstart Guide

### Installation
You can easily install Quark directly from PyPI or build from source using the provided `Makefile`.

```bash
pip install quark-optim

# Or clone for development:
git clone https://github.com/Anagatam/Quark.git
cd Quark
make install
```

### The Beautiful API (`.illuminate(X)`)

We meticulously designed the `MasterQuark` facade to align perfectly with the Scikit-Learn estimator syntax, while maintaining strict Biological Naming Conventions. Feed empirical market prices matrices into the model, and watch the Swarm **Illuminate**.

```python
import pandas as pd
import yfinance as yf
from quark.facade import MasterQuark

# 1. Provide a massive multi-market universe (E.g. US & NIFTY 50)
universe = ["NVDA", "MSFT", "LLY", "RELIANCE.NS", "HDFCBANK.NS"]
prices_df = yf.download(universe, start="2021-01-01", end="2024-01-01")['Close']

# 2. Instantiate the Metaheuristic Optimizer
model = MasterQuark(
    objective_type='composite', # Balances Alpha, Volatility, CVaR, & Max Drawdown
    max_assets=3,               # Strict non-convex Cardinality Constraint (Must purely select 3)
    lower_bound=0.05,           # Minimum 5% Allocation per selected stock
    upper_bound=0.40,           # Maximum 40% Allocation 
    num_fireflies=100,          # Size of the Biological Swarm
    max_iterations=200          # Evolutionary Generations
)

# 3. Absorb Market Physics and Optimize
model.illuminate(prices_df)

# 4. View Institutional Metrics
print(model.metrics_['optimal_weights'])
print(f"Projected Annualized Return: {model.metrics_['annualized_return']:.2%}")
```

---

## 🔬 Directory Structure
```plaintext
quark/
├── base.py                   # Abstract Base Classes (BaseObjective, BaseConstraint)
├── core/
│   ├── luminescence.py       # Core Math (CVaR, Drawdown, RMT, Black-Litterman)
│   └── objectives.py         # JIT/CPU-accelerated biological objective functions
├── data/
│   └── processor.py          # MAD Winsorization & Data cleansing
├── deep/
│   └── autoencoder.py        # PyTorch Risk Autoencoder (Denoising Covariance)
└── optimizer/
    ├── constraints.py        # Deterministic Feasible Space Reparation mappings
    ├── quark.py             # Highly optimized CPU/Numba inner loops
    └── swarmtensor.py        # Full CUDA/PyTorch multidimensional global evaluation
```

---

<div align="center">
  <br/>
  <p>Built with ❤️ by <b>Anagatm Technologies</b></p>
</div>
