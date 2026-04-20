```markdown
# JAX MLI-SF-PSD: Spacetime Fluctuations in Laser Interferometers

**A high-performance JAX implementation** for computing the power spectral density (PSD) of spacetime fluctuation (SF) signals in Michelson laser interferometers.

This repository reproduces **Figures 1, 2, and 4** from the paper:

> **"Signatures of correlation of spacetime fluctuations in laser interferometers"**  
> B. Sharmila, S. M. Vermeulen, A. Datta  
> *Nature Communications* (2026) — DOI: [10.1038/s41467-025-67313-3](https://doi.org/10.1038/s41467-025-67313-3)

---

## ✨ Features

- **Fully JAX-native**: `jit`, `vmap`, and `lax.scan` for GPU/TPU acceleration
- **Three correlation classes** (a, b, c) as defined in the paper
- **Analytical solution** for class (a) (factorised)
- **Rigorous Fabry-Pérot cavity response** (multi-bounce enhancement)
- **Monte Carlo sampling** of optical path difference realisations
- **Property-based tests** (Hypothesis) for physical correctness
- **Exact reproduction** of all paper figures
- Runs in **< 1 second** on GPU for typical frequency grids

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mli-sf-psd.git
cd mli-sf-psd

# Install dependencies
pip install -r requirements.txt

# (Optional) Install in editable mode
pip install -e .
```

### Requirements
- Python ≥ 3.10
- JAX (with GPU/TPU support recommended)
- matplotlib, numpy, hypothesis

---

## Quick Start

```python
from src.plotting import plot_fig1, plot_fig2, plot_fig4

# Reproduce all paper figures
plot_fig1()
plot_fig2()
plot_fig4()
```

This will generate `fig1.png`, `fig2.png`, and `fig4.png` matching the published paper.

---

## Usage Examples

### Compute PSDs

```python
import jax.numpy as jnp
from src.psd_calculator import psd_no_cavity, psd_with_cavity, scaled_psd_c
from src.correlation import rho_exponential_spatial

f = jnp.logspace(5, 9, 400)
L = 3.0          # arm length [m]
ell_r = 0.03     # correlation length [m]

# No cavity
S = psd_no_cavity(f, rho_exponential_spatial, L, ell_r)

# With LIGO-like arm cavity
S_cav = psd_with_cavity(f, rho_exponential_spatial, L, ell_r)

# Scaled dimensionless PSD (as shown in paper)
S_scaled = scaled_psd_c(f, L, ell_r, S)
```

### Monte Carlo Sampling

```python
from src.monte_carlo import sample_opd_ensemble, empirical_psd_from_ensemble

t, opd_ensemble = sample_opd_ensemble(
    rho_exponential_spatial, L=3.0, ell_r=0.03, n_samples=256
)

f_emp, psd_emp = empirical_psd_from_ensemble(t, opd_ensemble)
```

---

## Project Structure

```
mli-sf-psd/
├── src/
│   ├── constants.py          # Physical constants & defaults
│   ├── correlation.py        # ρ functions for all three classes
│   ├── mli_geometry.py       # Arm kinematics & cavity response
│   ├── psd_calculator.py     # Core PSD computation engine
│   ├── monte_carlo.py        # Gaussian process sampling
│   └── plotting.py           # Paper figure reproduction
├── examples/
│   └── reproduce_paper_figures.py
├── tests/
│   └── test_psd.py           # Property-based tests
├── README.md
└── requirements.txt
```

---

## Reproducing the Paper

| Script                  | Reproduces          | Description                              |
|-------------------------|---------------------|------------------------------------------|
| `plot_fig1()`           | Figure 1            | Scaled PSDs for all correlation classes (no cavity) |
| `plot_fig2()`           | Figure 2            | Cavity enhancement: QUEST vs LIGO        |
| `plot_fig4()`           | Figure 4            | κ-dependence for class (c)               |

---

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{sharmila2026signatures,
  title   = {Signatures of correlation of spacetime fluctuations in laser interferometers},
  author  = {Sharmila, B. and Vermeulen, Sander M. and Datta, Animesh},
  journal = {Nature Communications},
  year    = {2026},
  doi     = {10.1038/s41467-025-67313-3}
}
```

And optionally link this repository.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Built with [JAX](https://github.com/google/jax)
- Inspired by and reproducing results from the Sharmila et al. (2026) paper
- Thanks to the authors for open science and detailed supplementary information

---

**Made with ❤️ for fundamental physics and reproducible science.**

```

---

