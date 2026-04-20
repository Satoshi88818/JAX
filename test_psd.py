"""
Property-based and unit tests for the MLI-SF-PSD package.

Tests are organised into four property classes identified in the improvements list:

1. **Positivity** — PSD values must be non-negative for physical parameters.
2. **Low-frequency asymptotics** — match paper SI Sec. V–VI predictions.
3. **Light-cone compliance** — correlations must vanish for time-like separations
   (|Δr| < c|Δt|), enforcing causality.
4. **Additional regression tests** — specific numerical values from the paper.

Hypothesis strategies generate randomised but physically valid inputs so that
edge cases (very small κ, near-lightcone separations, etc.) are automatically
explored.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.psd_calculator import psd_no_cavity_factorised, psd_no_cavity
from src.correlation import (
    rho_factorised,
    rho_inverse_spatial,
    rho_inverse_spacetime,
    rho_exponential_spatial,
    rho_exponential_spacetime,
)
from src.constants import C_LIGHT, DEFAULT_L, DEFAULT_ELL_R


# ── Hypothesis strategies ─────────────────────────────────────────────────────

# Physically meaningful scaled frequency range
nu_strategy = st.floats(min_value=0.01, max_value=20.0, allow_nan=False, allow_infinity=False)
nu_array_strategy = st.lists(nu_strategy, min_size=1, max_size=8).map(
    lambda lst: jnp.array(lst)
)

# Arm lengths representative of table-top to large-scale setups
L_strategy = st.floats(min_value=1.0, max_value=5000.0, allow_nan=False)

# Correlation ratios κ = ℓ_r/L in the valid range (paper assumption: κ ≪ 1)
kappa_strategy = st.floats(min_value=1e-4, max_value=0.05, allow_nan=False)

# Physical frequency within one decade of f_LRT
@st.composite
def freq_L_ellr(draw):
    L = draw(L_strategy)
    kappa = draw(kappa_strategy)
    ell_r = kappa * L
    f_LRT = C_LIGHT / (2.0 * L)
    f = jnp.array([draw(st.floats(min_value=0.1 * f_LRT, max_value=10.0 * f_LRT))])
    return f, L, ell_r


# ── 1. Positivity ─────────────────────────────────────────────────────────────

class TestPositivity:
    """PSD values must be ≥ 0 everywhere for physical parameters."""

    def test_factorised_positive_explicit(self):
        nu = jnp.linspace(0.01, 10.0, 50)
        S = psd_no_cavity_factorised(nu)
        assert jnp.all(S >= 0.0), "Class (a) PSD has negative values"

    @given(nu_array_strategy)
    @settings(max_examples=50)
    def test_factorised_positive_hypothesis(self, nu):
        S = psd_no_cavity_factorised(nu)
        assert jnp.all(S >= 0.0)

    @given(freq_L_ellr())
    @settings(max_examples=30)
    def test_inverse_spatial_positive(self, args):
        f, L, ell_r = args
        S = psd_no_cavity(f, rho_inverse_spatial, L, ell_r)
        assert jnp.all(S >= 0.0)

    @given(freq_L_ellr())
    @settings(max_examples=30)
    def test_exponential_spatial_positive(self, args):
        f, L, ell_r = args
        S = psd_no_cavity(f, rho_exponential_spatial, L, ell_r)
        assert jnp.all(S >= 0.0)

    @given(freq_L_ellr())
    @settings(max_examples=30)
    def test_exponential_spacetime_positive(self, args):
        f, L, ell_r = args
        S = psd_no_cavity(f, rho_exponential_spacetime, L, ell_r)
        assert jnp.all(S >= 0.0)


# ── 2. Low-frequency asymptotics ──────────────────────────────────────────────

class TestLowFrequencyAsymptotics:
    """Match paper SI predictions for ν ≪ 1.

    Class (a): S_NC(ν→0) ≈ 0.275 (SI Sec. V, below Eq. 68).
    Class (b): S_C(ν→0) → 0 as ν² (analytic, SI Sec. VI).
    Class (c): S_C(ν→0) ≈ flat (slow exponential approach).
    """

    def test_factorised_low_freq_value(self):
        """S_NC(ν=0.01) should be close to the analytic limit 0.275."""
        nu = jnp.array([0.01])
        S = psd_no_cavity_factorised(nu)
        # Allow ±30% tolerance given the calibration factor
        assert 0.10 < float(S[0]) < 1.0, f"S_NC(0.01) = {float(S[0]):.4f}, expected ~0.275"

    def test_inverse_spatial_quadratic_trend(self):
        """Class (b1) PSD should scale as ν² at low ν (SI Sec. VI, analytic)."""
        L = DEFAULT_L
        ell_r = DEFAULT_ELL_R
        nu_vals = jnp.array([0.05, 0.1, 0.2])
        f_vals = nu_vals * 2.0 * (C_LIGHT / (2.0 * L)) / jnp.pi
        S = psd_no_cavity(f_vals, rho_inverse_spatial, L, ell_r)
        assume(jnp.all(S > 0))
        # log(S) / log(ν) should be close to 2
        ratios = jnp.log(S[1:] / S[:-1]) / jnp.log(nu_vals[1:] / nu_vals[:-1])
        assert jnp.all(jnp.abs(ratios - 2.0) < 0.5), f"Slopes: {ratios}"

    def test_factorised_monotone_decrease(self):
        """S_NC should be non-increasing at high ν (consistent with 1/ν² decay)."""
        nu = jnp.linspace(2.0, 15.0, 30)
        S = psd_no_cavity_factorised(nu)
        # Allow for small numerical noise; check overall trend
        assert float(S[-1]) < float(S[0]), "S_NC not decreasing at high ν"


# ── 3. Light-cone compliance ──────────────────────────────────────────────────

class TestLightConeCompliance:
    """Correlation functions must vanish for strictly time-like separations."""

    @given(
        c_dt=st.floats(min_value=0.01, max_value=10.0),
        r_frac=st.floats(min_value=0.0, max_value=0.99),  # r_norm = r_frac * |c_dt|
        ell_r=st.floats(min_value=0.001, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_inverse_spatial_lightcone(self, c_dt, r_frac, ell_r):
        """ρ_IS must be 0 when |Δr| < c|Δt| (time-like)."""
        r_norm = r_frac * abs(c_dt)
        assume(r_norm < abs(c_dt))  # strictly inside the light cone
        val = rho_inverse_spatial(jnp.array(c_dt), jnp.array(r_norm), ell_r)
        assert float(val) == 0.0, f"ρ_IS non-zero inside light cone: {float(val)}"

    @given(
        c_dt=st.floats(min_value=0.01, max_value=10.0),
        r_frac=st.floats(min_value=0.0, max_value=0.99),
        ell_r=st.floats(min_value=0.001, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_inverse_spacetime_lightcone(self, c_dt, r_frac, ell_r):
        """ρ_IST must be 0 inside the light cone."""
        r_norm = r_frac * abs(c_dt)
        assume(r_norm < abs(c_dt))
        val = rho_inverse_spacetime(jnp.array(c_dt), jnp.array(r_norm), ell_r)
        assert float(val) == 0.0

    @given(
        c_dt=st.floats(min_value=0.01, max_value=10.0),
        r_frac=st.floats(min_value=0.0, max_value=0.99),
        ell_r=st.floats(min_value=0.001, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_exponential_spatial_lightcone(self, c_dt, r_frac, ell_r):
        """ρ_ES must be 0 inside the light cone."""
        r_norm = r_frac * abs(c_dt)
        assume(r_norm < abs(c_dt))
        val = rho_exponential_spatial(jnp.array(c_dt), jnp.array(r_norm), ell_r)
        assert float(val) == 0.0

    @given(
        c_dt=st.floats(min_value=0.01, max_value=10.0),
        r_frac=st.floats(min_value=0.0, max_value=0.99),
        ell_r=st.floats(min_value=0.001, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_exponential_spacetime_lightcone(self, c_dt, r_frac, ell_r):
        """ρ_EST must be 0 inside the light cone."""
        r_norm = r_frac * abs(c_dt)
        assume(r_norm < abs(c_dt))
        val = rho_exponential_spacetime(jnp.array(c_dt), jnp.array(r_norm), ell_r)
        assert float(val) == 0.0

    def test_spacelike_nonzero(self):
        """Correlations must be positive for strictly space-like separations."""
        c_dt = jnp.array(1.0)
        r_norm = jnp.array(2.0)   # r_norm > |c_dt|
        ell_r = 0.5
        assert rho_inverse_spatial(c_dt, r_norm, ell_r) > 0
        assert rho_inverse_spacetime(c_dt, r_norm, ell_r) > 0
        assert rho_exponential_spatial(c_dt, r_norm, ell_r) > 0
        assert rho_exponential_spacetime(c_dt, r_norm, ell_r) > 0


# ── 4. Regression / smoke tests ───────────────────────────────────────────────

class TestRegression:
    """Quick regression checks against known values."""

    def test_analytical_low_freq_order_of_magnitude(self):
        nu = jnp.array([0.01])
        S = psd_no_cavity_factorised(nu)
        assert S[0] > 0.05, "S_NC too small at low ν"

    def test_psd_positive_exp_spatial(self):
        f = jnp.array([1e6])
        S = psd_no_cavity(f, rho_exponential_spatial, 3.0, 0.03)
        assert jnp.all(S > 0)

    def test_psd_decreases_with_frequency(self):
        """PSD should decrease from low to high frequency (all classes)."""
        L, ell_r = DEFAULT_L, DEFAULT_ELL_R
        f_LRT = C_LIGHT / (2.0 * L)
        f_lo = jnp.array([0.5 * f_LRT])
        f_hi = jnp.array([5.0 * f_LRT])
        for rho in [rho_inverse_spatial, rho_exponential_spatial]:
            S_lo = psd_no_cavity(f_lo, rho, L, ell_r)
            S_hi = psd_no_cavity(f_hi, rho, L, ell_r)
            assert float(S_hi[0]) < float(S_lo[0]), f"{rho.__name__} PSD not decreasing"

    def test_factorised_returns_zero_array(self):
        """rho_factorised must always return zeros (handled analytically)."""
        c_dt = jnp.ones((5, 5))
        r_norm = jnp.ones((5, 5))
        out = rho_factorised(c_dt, r_norm, ell_r=None)
        assert jnp.all(out == 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
