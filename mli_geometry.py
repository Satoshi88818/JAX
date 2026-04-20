"""
Interferometer geometry and response functions for the Michelson laser interferometer (MLI).

This module provides:

- :func:`arm_position`: the folded light path s(t) in a single arm (paper Eq. 16 / Fig. 3a).
- :func:`fabry_perot_response`: the Fabry-Pérot cavity enhancement χ̃_FP(f) (paper Eq. 23).
- :func:`dimensionless_nu`: the dimensionless frequency ν = πf / (2 f_LRT) (paper Eq. 24).
"""

import jax.numpy as jnp
from jax import jit


@jit
def arm_position(t: jnp.ndarray, L: float, c: float) -> jnp.ndarray:
    """Return the longitudinal position s(t) of the photon in one arm.

    The photon bounces between the beam-splitter (z=0) and the end mirror
    (z=L) with round-trip time τ₀ = 2L/c::

        s(t) = c·t          if t ≤ τ₀/2   (outward leg)
        s(t) = 2L - c·t     if t > τ₀/2   (return leg)

    This implements the piecewise function used in Eqs. (15)–(16) of the paper.

    Parameters
    ----------
    t:
        Time array [s] within [0, τ₀].
    L:
        Arm length [m].
    c:
        Speed of light [m/s].

    Returns
    -------
    jnp.ndarray
        Position array [m], same shape as ``t``.
    """
    t0 = 2.0 * L / c
    return jnp.where(t <= t0 / 2.0, c * t, 2.0 * L - c * t)


@jit
def fabry_perot_response(f: jnp.ndarray, f_LRT: float, R_M: float = 0.986) -> jnp.ndarray:
    """Fabry-Pérot arm-cavity power response χ̃_FP(f) (paper Eq. 23).

    The cavity formed by the input mirror (reflectivity R_M) and the perfectly
    reflecting end mirror creates a resonance at multiples of f_LRT.  The
    peak enhancement at ν = m·π/2 (m = 1,2,…) has magnitude
    T_M⁴ / (1 − √R_M)⁶ ≈ 3.2 × 10⁵ for LIGO (paper Sec. "Detecting SFs",
    Feature 1).

    This implements a rigorous multi-bounce treatment rather than a simple
    multiplication, capturing the full frequency-dependent resonance structure.

    Parameters
    ----------
    f:
        Frequency array [Hz].
    f_LRT:
        Light-round-trip frequency c / (2L) [Hz].
    R_M:
        Power reflectivity of the input mirror (default: 0.986, LIGO-like).

    Returns
    -------
    jnp.ndarray
        Dimensionless cavity enhancement, same shape as ``f``.

    Notes
    -----
    Equation reference: paper Eq. (23).  For a cavity without losses
    T_M = 1 − R_M.
    """
    T_M = 1.0 - R_M
    # Denominator: 1 + R_M − 2√R_M · cos(2πf / f_LRT)
    phase = 2.0 * jnp.pi * f / f_LRT
    denom = 1.0 + R_M - 2.0 * jnp.sqrt(R_M) * jnp.cos(phase)
    # Prefactor: T_M⁴ / (1 − √R_M)⁴  (multi-bounce amplitude build-up)
    prefactor = T_M ** 4 / (1.0 - jnp.sqrt(R_M)) ** 4
    return prefactor / denom


def dimensionless_nu(f: jnp.ndarray, f_LRT: float) -> jnp.ndarray:
    """Convert physical frequency to the dimensionless scaled frequency ν (paper Eq. 24).

    ν = π·f / (2·f_LRT)

    At ν = 1 the frequency equals f_LRT = c/(2L), the light-round-trip
    frequency.  The paper identifies ν as the natural scale for PSD plots
    (Figs. 1, 2, 4).

    Parameters
    ----------
    f:
        Physical frequency array [Hz].
    f_LRT:
        Light-round-trip frequency [Hz].

    Returns
    -------
    jnp.ndarray
        Dimensionless frequency ν, same shape as ``f``.
    """
    return jnp.pi * f / (2.0 * f_LRT)
