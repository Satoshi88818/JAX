"""
Physical and interferometer constants used throughout the MLI-SF-PSD package.

All values are in SI units unless otherwise noted. Dimensionless ratios such as
mirror reflectivities R_M are plain floats in [0, 1].
"""

import jax.numpy as jnp

# ── Physical constants ────────────────────────────────────────────────────────

C_LIGHT: float = 3.0e8
"""Speed of light in vacuum [m/s]."""

# ── Default interferometer parameters ─────────────────────────────────────────

DEFAULT_L: float = 3.0
"""Default arm length [m], representative of the QUEST table-top MLI (paper Sec. III)."""

DEFAULT_ELL_R: float = 0.03
"""Default correlation length ℓ_r [m], giving κ = ℓ_r / L ≈ 0.01 for QUEST."""

DEFAULT_GAMMA_S: float = 1.0
"""Default dimensionless SF strength Γ_S (Eq. 8).  Set to 1.0 so that
output PSDs are in units of Γ_S·L³/c (class a) or Γ_S·ℓ_r·L²/c (classes b, c)."""

DEFAULT_R_M: float = 0.986
"""Default power reflectivity R_M of the input mirror for a Fabry-Pérot arm
cavity, representative of LIGO (T_M = 1 - R_M ≈ 0.014, paper Eq. 23)."""
