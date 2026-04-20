"""
Two-point correlation functions ρ(c·Δt, Δr) for spacetime fluctuations.

Each function implements one of the three correlation classes introduced in
the paper (Eqs. 10–14):

  (a)  Factorised / delta-in-time       → :func:`rho_factorised`   (Eq. 10)
  (b1) Inverse spatial separation       → :func:`rho_inverse_spatial`   (Eq. 11)
  (b2) Inverse spacetime separation     → :func:`rho_inverse_spacetime`  (Eq. 12)
  (c1) Exponential spatial separation   → :func:`rho_exponential_spatial`  (Eq. 13)
  (c2) Exponential spacetime separation → :func:`rho_exponential_spacetime` (Eq. 14)

All functions share the same signature::

    rho(c_dt, r_norm, ell_r) -> jnp.ndarray

where ``c_dt = c·(t1 + Δτ - t2)`` and ``r_norm = |s(t1) - s(t2)|`` or the
corresponding cross-arm quantity, each broadcastable JAX arrays.

Light-cone compliance is enforced via the Heaviside step
``Θ(|Δr| - c|Δt|)`` (paper Eqs. 11–14): correlations vanish for time-like
separations.
"""

import jax.numpy as jnp
from jax import jit


# ── Numerical stability floor ──────────────────────────────────────────────────
_EPS = 1e-12


@jit
def rho_factorised(c_dt: jnp.ndarray, r_norm: jnp.ndarray, ell_r=None) -> jnp.ndarray:
    """Class (a): factorised / delta-in-time correlation function ρ_F (Eq. 10).

    The temporal part is a Dirac delta δ(Δt/c), so this class is handled
    analytically in :func:`psd_calculator.psd_no_cavity_factorised` (SI Sec. V).
    Returning zeros here signals to the PSD calculator to switch to the
    analytical branch.

    Parameters
    ----------
    c_dt:
        ``c · (t1 + Δτ - t2)`` [m], broadcastable array.
    r_norm:
        Spatial separation ``|Δr|`` [m], broadcastable array.
    ell_r:
        Ignored for class (a); kept for a uniform function signature.

    Returns
    -------
    jnp.ndarray
        Zero array with the shape of ``jnp.broadcast_arrays(c_dt, r_norm)[0]``.
    """
    return jnp.zeros_like(c_dt + r_norm)


@jit
def rho_inverse_spatial(c_dt: jnp.ndarray, r_norm: jnp.ndarray, ell_r: float) -> jnp.ndarray:
    """Class (b1): inverse spatial-separation correlation ρ_IS (Eq. 11).

    ρ_IS(c·Δt, Δr) = (ℓ_r / |Δr|) · Θ(|Δr| - c|Δt|)

    The Heaviside factor enforces light-cone causality: only space-like pairs
    contribute (|Δr| ≥ c|Δt|).  This form arises in models where SFs satisfy
    the wave equation in 3+1 D (paper refs. 16, 17, 35).

    Parameters
    ----------
    c_dt:
        ``c · (t1 + Δτ - t2)`` [m].
    r_norm:
        Spatial separation ``|Δr|`` [m].
    ell_r:
        Correlation length ℓ_r [m].

    Returns
    -------
    jnp.ndarray
        Dimensionless correlation value.
    """
    theta = jnp.where(r_norm >= jnp.abs(c_dt), 1.0, 0.0)
    safe_r = jnp.where(r_norm > _EPS, r_norm, _EPS)
    return (ell_r / safe_r) * theta


@jit
def rho_inverse_spacetime(c_dt: jnp.ndarray, r_norm: jnp.ndarray, ell_r: float) -> jnp.ndarray:
    """Class (b2): inverse spacetime-separation correlation ρ_IST (Eq. 12).

    ρ_IST(c·Δt, Δr) = ℓ_r · Θ(|Δr| - c|Δt|) / sqrt(|Δr|² - c²Δt²)

    Generalisation of (b1) using the Lorentz-invariant spacelike interval.
    The sign convention keeps ρ real-valued (paper below Eq. 12).

    Parameters
    ----------
    c_dt:
        ``c · (t1 + Δτ - t2)`` [m].
    r_norm:
        Spatial separation ``|Δr|`` [m].
    ell_r:
        Correlation length ℓ_r [m].

    Returns
    -------
    jnp.ndarray
        Dimensionless correlation value.
    """
    arg = r_norm ** 2 - c_dt ** 2
    theta = jnp.where(arg >= 0.0, 1.0, 0.0)
    safe_denom = jnp.sqrt(jnp.where(arg > _EPS, arg, _EPS))
    return (ell_r * theta) / safe_denom


@jit
def rho_exponential_spatial(c_dt: jnp.ndarray, r_norm: jnp.ndarray, ell_r: float) -> jnp.ndarray:
    """Class (c1): exponential spatial-separation correlation ρ_ES (Eq. 13).

    ρ_ES(c·Δt, Δr) = exp(-|Δr| / ℓ_r) · Θ(|Δr| - c|Δt|)

    Motivated by models with short-range quantum entanglement (paper ref. 15)
    or mesoscopic gravity (ref. 11).  SC(ν) depends on κ = ℓ_r/L (Fig. 1c,
    Fig. 4 of the paper).

    Parameters
    ----------
    c_dt:
        ``c · (t1 + Δτ - t2)`` [m].
    r_norm:
        Spatial separation ``|Δr|`` [m].
    ell_r:
        Correlation length ℓ_r [m].

    Returns
    -------
    jnp.ndarray
        Dimensionless correlation value.
    """
    theta = jnp.where(r_norm >= jnp.abs(c_dt), 1.0, 0.0)
    return jnp.exp(-r_norm / ell_r) * theta


@jit
def rho_exponential_spacetime(c_dt: jnp.ndarray, r_norm: jnp.ndarray, ell_r: float) -> jnp.ndarray:
    """Class (c2): exponential spacetime-separation correlation ρ_EST (Eq. 14).

    ρ_EST(c·Δt, Δr) = exp(-sqrt(|Δr|² - c²Δt²) / ℓ_r) · Θ(|Δr| - c|Δt|)

    Uses the Lorentz-invariant spacelike interval in the exponent.

    Parameters
    ----------
    c_dt:
        ``c · (t1 + Δτ - t2)`` [m].
    r_norm:
        Spatial separation ``|Δr|`` [m].
    ell_r:
        Correlation length ℓ_r [m].

    Returns
    -------
    jnp.ndarray
        Dimensionless correlation value.
    """
    arg = r_norm ** 2 - c_dt ** 2
    theta = jnp.where(arg >= 0.0, 1.0, 0.0)
    safe_arg = jnp.where(arg > _EPS, arg, _EPS)
    return jnp.exp(-jnp.sqrt(safe_arg) / ell_r) * theta
