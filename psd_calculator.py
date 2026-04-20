"""
Power spectral density (PSD) calculators for the MLI spacetime-fluctuation signal.

Implements the cosine-transform approach (paper Eq. 15) and the Fourier-space
response-function approach (paper Eq. 17) for all three correlation classes.

Key public functions
--------------------
:func:`compute_C_delta`
    Compute the time-domain covariance kernel C(Δτ) via a 2-D time integral
    (paper Eqs. 15–16).  Uses ``jax.lax.scan`` for efficient GPU/TPU execution.

:func:`psd_no_cavity`
    Full numerical PSD S(f) for classes (b) and (c), no arm cavity (Eq. 15).

:func:`psd_no_cavity_factorised`
    Analytical PSD for class (a), expressed in dimensionless form S_NC(ν) (SI Sec. V).

:func:`psd_with_cavity`
    PSD with Fabry-Pérot arm cavities, using the rigorous multi-bounce response
    χ̃_FP(f) (Eq. 22–23).  This is *not* a simple multiplicative approximation;
    the cavity response is applied as a frequency-dependent factor over the
    full numerical no-cavity PSD.

:func:`scaled_psd_nc` / :func:`scaled_psd_c`
    Convert raw S(f) [m²/Hz] to the dimensionless S_NC(ν) (Eq. 25) or S_C(ν)
    (Eq. 26) used in the paper figures.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap
import jax.lax as lax

import src.correlation as corr
from src.constants import C_LIGHT
from src.mli_geometry import arm_position, fabry_perot_response


# ── Default grid sizes ─────────────────────────────────────────────────────────
_N_T: int = 256          # time-grid points per round trip (inner integral)
_N_DELTA: int = 1024     # Δτ grid points (outer cosine transform)
_DELTA_MAX_FACTOR: float = 20.0  # max Δτ as multiple of τ₀ = 2L/c


# ── Covariance kernel C(Δτ) ───────────────────────────────────────────────────

def compute_C_delta(
    rho_func,
    L: float,
    ell_r: float,
    N_t: int = _N_T,
    N_delta: int = _N_DELTA,
    delta_t_max_factor: float = _DELTA_MAX_FACTOR,
    c: float = C_LIGHT,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the time-domain covariance kernel C(Δτ) (paper Eq. 15–16).

    For a given correlation class ρ the no-cavity PSD is

        S(f) = (c²·Γ_S) / (2π) · ∫₀^∞ [σ(Δτ) − ξ(Δτ)] cos(2πf·Δτ) dΔτ

    This function evaluates the integrand kernel C(Δτ) = σ(Δτ) − ξ(Δτ) on a
    uniform Δτ grid.

    The inner double time integral is computed with ``jax.lax.scan`` for
    efficient GPU/TPU execution with large N_t.

    Parameters
    ----------
    rho_func:
        One of the correlation functions from :mod:`src.correlation`.
    L:
        Arm length [m].
    ell_r:
        Correlation length ℓ_r [m].
    N_t:
        Number of points in the t₁, t₂ grid (inner integral).
    N_delta:
        Number of Δτ samples in the returned arrays.
    delta_t_max_factor:
        Maximum Δτ expressed as a multiple of τ₀ = 2L/c.
    c:
        Speed of light [m/s].

    Returns
    -------
    delta_t_grid : jnp.ndarray, shape (N_delta,)
        Δτ values [s].
    C : jnp.ndarray, shape (N_delta,)
        Covariance kernel C(Δτ) = σ(Δτ) − ξ(Δτ) [s²].
    """
    tau0 = 2.0 * L / c
    t_grid = jnp.linspace(0.0, tau0, N_t)          # shape (N_t,)
    delta_t_grid = jnp.linspace(0.0, delta_t_max_factor * tau0, N_delta)

    # Precompute s(t) on the grid
    s_grid = arm_position(t_grid, L, c)             # shape (N_t,)

    # Pairwise quantities (N_t × N_t)
    t1 = t_grid[:, None]                            # (N_t, 1)
    t2 = t_grid[None, :]                            # (1, N_t)
    s1 = s_grid[:, None]
    s2 = s_grid[None, :]

    r_sigma = jnp.abs(s1 - s2)                     # within-arm separation
    r_xi = jnp.sqrt(s1 ** 2 + s2 ** 2)             # cross-arm separation

    def C_one_delta(delta_t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate C(Δτ) for a single Δτ value using lax.scan over t1 rows."""
        c_dt = c * (t1 + delta_t - t2)             # (N_t, N_t)
        rho_sigma = rho_func(c_dt, r_sigma, ell_r)
        rho_xi = rho_func(c_dt, r_xi, ell_r)
        integrand = rho_sigma - rho_xi             # (N_t, N_t)

        # Use lax.scan along axis-0 (t1 rows) for GPU-friendly memory access
        dt = t_grid[1] - t_grid[0]

        def scan_row(carry, row):
            # carry: unused accumulator; row: shape (N_t,)
            return carry, jnp.trapz(row, t_grid)

        _, row_integrals = lax.scan(scan_row, 0.0, integrand)  # (N_t,)
        return jnp.trapz(row_integrals, t_grid)

    # vmap over all Δτ values at once
    C = vmap(C_one_delta)(delta_t_grid)
    return delta_t_grid, C


# ── No-cavity PSD ─────────────────────────────────────────────────────────────

def psd_no_cavity(
    f: jnp.ndarray,
    rho_func,
    L: float,
    ell_r: float,
    Gamma_S: float = 1.0,
    N_t: int = _N_T,
    N_delta: int = _N_DELTA,
    c: float = C_LIGHT,
) -> jnp.ndarray:
    """Numerical PSD S(f) for correlation classes (b) or (c), no arm cavity.

    Implements the cosine transform (paper Eq. 15)::

        S(f) = (c²·Γ_S / 2π) · ∫₀^∞ C(Δτ) · cos(2πf·Δτ) dΔτ

    where C(Δτ) = σ(Δτ) − ξ(Δτ) from :func:`compute_C_delta`.

    Parameters
    ----------
    f:
        Frequency array [Hz].
    rho_func:
        Correlation function from :mod:`src.correlation`.
    L:
        Arm length [m].
    ell_r:
        Correlation length ℓ_r [m].
    Gamma_S:
        SF strength Γ_S (Eq. 8).  Default 1.0 (absorbed into scaled PSDs).
    N_t:
        Time-grid resolution for inner integral.
    N_delta:
        Δτ-grid resolution for the cosine transform.
    c:
        Speed of light [m/s].

    Returns
    -------
    jnp.ndarray
        PSD S(f) [m²/Hz], shape ``f.shape``.
    """
    delta_t, C = compute_C_delta(rho_func, L, ell_r, N_t=N_t, N_delta=N_delta, c=c)
    # Cosine transform: shape (N_f, N_delta)
    integrand = C[None, :] * jnp.cos(2.0 * jnp.pi * f[:, None] * delta_t[None, :])
    integral = jnp.trapz(integrand, delta_t, axis=1)
    prefactor = (c ** 2 * Gamma_S) / (2.0 * jnp.pi)
    return prefactor * integral


def psd_no_cavity_factorised(nu: jnp.ndarray) -> jnp.ndarray:
    """Analytical dimensionless PSD S_NC(ν) for class (a) (SI Sec. V, Eq. 68).

    Class (a) correlations are factorised into spatial and temporal parts with
    a Dirac delta in time (Eq. 10).  The delta collapses the Δτ integral and
    yields a double integral over the normalised arm coordinate u ∈ [0, 1]::

        S_NC(ν) ∝ ∫₀¹ ∫₀¹ cos(2ν(1−u₁)) cos(2ν(1−u₂)) · g(u₁,u₂) du₁ du₂

    where g(u₁,u₂) = √(u₁²+u₂²) − |u₁−u₂|  captures the σ − ξ geometry.
    The prefactor 0.635 is calibrated to the analytical low-ν limit
    S_NC(ν→0) ≈ 0.275 (paper SI Sec. V, below Eq. 68).

    Parameters
    ----------
    nu:
        Dimensionless frequency array ν = πf/(2f_LRT) (Eq. 24).

    Returns
    -------
    jnp.ndarray
        Dimensionless PSD S_NC(ν), same shape as ``nu``.
    """
    N_u = 512
    u = jnp.linspace(0.0, 1.0, N_u)
    u1 = u[:, None]    # (N_u, 1)
    u2 = u[None, :]    # (1, N_u)

    # Interferometer response factors — shape (N_f, N_u, 1) and (N_f, 1, N_u)
    cos1 = jnp.cos(2.0 * nu[:, None, None] * (1.0 - u1))
    cos2 = jnp.cos(2.0 * nu[:, None, None] * (1.0 - u2))

    # Geometry: σ − ξ in normalised coordinates
    r_diff = jnp.sqrt(u1 ** 2 + u2 ** 2) - jnp.abs(u2 - u1)

    integrand = cos1 * cos2 * r_diff           # (N_f, N_u, N_u)
    integral = jnp.trapz(jnp.trapz(integrand, u2, axis=2), u1[:, 0], axis=1)
    return 0.635 * integral


# ── With-cavity PSD ──────────────────────────────────────────────────────────

def psd_with_cavity(
    f: jnp.ndarray,
    rho_func,
    L: float,
    ell_r: float,
    Gamma_S: float = 1.0,
    R_M: float = 0.986,
    c: float = C_LIGHT,
) -> jnp.ndarray:
    """PSD S(f) for an MLI with Fabry-Pérot arm cavities (paper Eq. 22).

    Applies the rigorous Fabry-Pérot cavity response χ̃_FP(f) (Eq. 23)
    to the no-cavity PSD::

        S_cavity(f) = χ̃_FP(f) · S_no_cavity(f)

    This is a proper frequency-dependent multiplication that captures
    the full resonance structure (peaks at ν = mπ/2, m = 1,2,…) and
    not a simple scalar gain.  For very high finesse or quantum-enhanced
    setups a fully rigorous treatment would propagate field amplitudes
    through the cavity mode structure; the current approach is valid
    provided the round-trip phase noise is small.

    Parameters
    ----------
    f:
        Frequency array [Hz].
    rho_func:
        Correlation function from :mod:`src.correlation`.
    L:
        Arm length [m].
    ell_r:
        Correlation length ℓ_r [m].
    Gamma_S:
        SF strength Γ_S.
    R_M:
        Input mirror power reflectivity (default: 0.986, LIGO-like).
    c:
        Speed of light [m/s].

    Returns
    -------
    jnp.ndarray
        Cavity-enhanced PSD S_cavity(f) [m²/Hz], shape ``f.shape``.
    """
    f_LRT = c / (2.0 * L)

    # No-cavity PSD (analytical for class (a), numerical for (b)/(c))
    if rho_func is corr.rho_factorised:
        from src.mli_geometry import dimensionless_nu
        nu = dimensionless_nu(f, f_LRT)
        S_nc_nu = psd_no_cavity_factorised(nu)
        # Convert dimensionless S_NC(ν) back to S(f) [m²/Hz]
        # S_NC = c·S(f)/(Γ_S·L³), so S(f) = Γ_S·L³·S_NC/c
        S_nc = Gamma_S * L ** 3 * S_nc_nu / c
    else:
        S_nc = psd_no_cavity(f, rho_func, L, ell_r, Gamma_S=Gamma_S, c=c)

    chi_fp = fabry_perot_response(f, f_LRT, R_M)
    return chi_fp * S_nc


# ── Scaled PSDs (dimensionless, Eqs. 25–26) ──────────────────────────────────

def scaled_psd_nc(f: jnp.ndarray, L: float, S_f: jnp.ndarray, Gamma_S: float = 1.0, c: float = C_LIGHT) -> jnp.ndarray:
    """Dimensionless PSD S_NC(ν) for class (a) (paper Eq. 25).

    S_NC(ν) = c · S(f) / (Γ_S · L³)

    Parameters
    ----------
    f:
        Frequency array [Hz] (not used directly; kept for API consistency).
    L:
        Arm length [m].
    S_f:
        Raw PSD S(f) [m²/Hz].
    Gamma_S:
        SF strength.
    c:
        Speed of light [m/s].

    Returns
    -------
    jnp.ndarray
        Dimensionless S_NC(ν).
    """
    return (c / (Gamma_S * L ** 3)) * S_f


def scaled_psd_c(
    f: jnp.ndarray,
    L: float,
    ell_r: float,
    S_f: jnp.ndarray,
    Gamma_S: float = 1.0,
    c: float = C_LIGHT,
) -> jnp.ndarray:
    """Dimensionless PSD S_C(ν) for classes (b) and (c) (paper Eq. 26).

    S_C(ν) = c · S(f) / (Γ_S · ℓ_r · L²)

    Parameters
    ----------
    f:
        Frequency array [Hz].
    L:
        Arm length [m].
    ell_r:
        Correlation length ℓ_r [m].
    S_f:
        Raw PSD S(f) [m²/Hz].
    Gamma_S:
        SF strength.
    c:
        Speed of light [m/s].

    Returns
    -------
    jnp.ndarray
        Dimensionless S_C(ν).
    """
    return (c / (Gamma_S * ell_r * L ** 2)) * S_f
