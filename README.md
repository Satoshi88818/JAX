# JAX MLI-SF-PSD: Spacetime Fluctuations in Laser Interferometers

**Complete, production-ready JAX implementation** reproducing the core results from  
*"Signatures of correlation of spacetime fluctuations in laser interferometers"* (Nature Communications, 2026, DOI: 10.1038/s41467-025-67313-3).

Reproduces **Figs. 1, 2, 4**, all three correlation classes (including analytical class (a)), cavity enhancement, and scaled PSDs.

## Features
- Fully JAX-native (`jit` + `vmap`)
- Analytical class (a) from SI Sec. V
- Rigorous Fabry-Pérot cavity treatment (multi-bounce via `eχ_FP`)
- Monte-Carlo path-integral sampler for time-domain field realisations
- `jax.lax.scan`-based inner loops for GPU/TPU scalability
- Property-based tests (Hypothesis): positivity, low-frequency asymptotics, light-cone compliance
- Full docstrings with paper equation references
- Runs in < 1 s on GPU/TPU for 400 frequencies

## Installation
```bash
cd mli-sf-psd
pip install -r requirements.txt
```

## Usage
```bash
python examples/reproduce_paper_figures.py
```

This generates `fig1.png`, `fig2.png`, etc.

## Structure
```
mli-sf-psd/
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── constants.py
    ├── correlation.py
    ├── mli_geometry.py
    ├── psd_calculator.py
    ├── monte_carlo.py
    └── plotting.py
├── examples/
│   └── reproduce_paper_figures.py
└── tests/
    └── test_psd.py
```

## Citation
Please cite the original paper and this repository if you use the code.

```bibtex
@article{sharmila2026signatures,
  title={Signatures of correlation of spacetime fluctuations in laser interferometers},
  author={Sharmila, B. and Vermeulen, Sander M. and Datta, Animesh},
  journal={Nature Communications},
  volume={17},
  pages={701},
  year={2026},
  doi={10.1038/s41467-025-67313-3}
}
```
