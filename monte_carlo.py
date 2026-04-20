"""
Monte-Carlo path-integral sampler for spacetime field realisations.

This module provides time-domain realisations of the optical path difference
(OPD) signal in a Michelson laser interferometer, given a stationary Gaussian
spacetime fluctuation field w(r) with a specified two-point correlation ρ.

The approach uses a Cholesky-based sampler: given the covariance matrix
K[i,j] = Γ_S · ρ(c(t_i − t_j), 0, …, s(t_i) − s(t_j)) evaluated on a
discrete time grid, we draw realisations via K = L·Lᵀ and z = L·ξ where ξ
is a standard normal vector.  This is the standard approach for sampling
continuous Gaussian processes at finite resolution.

For *very* large grids the Cholesky scales as O(N³), so a ``jax.lax.scan``-
based iterative sampler is also provided for memory-efficient large-N runs.

Functions
---------
:func:`sample_opd_realisation`
    Draw a single OPD realisation on a uniform time grid.

:func:`sample_opd_ensemble`
    Draw an ensemble of independent realisations (vectorised via ``vmap``).

:func:`empirical_psd_from_ensemble`
    Estimate the PSD from an ensemble using Welch's method.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import jax.lax as lax

from src.constants import C_LIGHT
from src.mli_geometry import arm_position


def _build_covariance(
    rho_func,
    L: float,
    ell_r: float,
    N_t: int,
    c: float = C_LIGHT,
) -> jnp.ndarray:
    """Build the N_t × N_t covariance matrix K for the OPD signal.

    K[i,j] = σ(t_i, t_j) − ξ(t_i, t_j)  (paper Eqs. 16a–16b at Δτ=0)

    where the t grid spans one round-trip [0, τ₀] with τ₀ = 2L/c.

    Parameters
    ----------
    rho_func:
        Correlation function from :mod:`src.correlation`.
    L:
        Arm length [m].
    ell_r:
        Correlation length ℓ_r [m].
    N_t:
        Grid size.
    c:
        Speed of light [m/s].

    Returns
    -------
    jnp.ndarray
        Covariance matrix, shape ``(N_t, N_t)``.
    """
    tau0 = 2.0 * L / c
    t_grid = jnp.linspace(0.0, tau0, N_t)
    s_grid = arm_position(t_grid, L, c)

    t1 = t_grid[:, None]
    t2 = t_grid[None, :]
    s1 = s_grid[:, None]
    s2 = s_grid[None, :]

    c_dt = c * (t1 - t2)
    r_sigma = jnp.abs(s1 - s2)
    r_xi = jnp.sqrt(s1 ** 2 + s2 ** 2)

    K = rho_func(c_dt, r_sigma, ell_r) - rho_func(c_dt, r_xi, ell_r)
    return K


def sample_opd_realisation(
    rho_func,
    L: float,
    ell_r: float,
    Gamma_S: float = 1.0,
    N_t: int = 256,
    key: jax.random.KeyArray = None,
    c: float = C_LIGHT,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Draw a single realisation of the OPD time series.

    Uses a Cholesky decomposition of the covariance matrix to generate
    a correlated Gaussian path.

    Parameters
    ----------
    rho_func:
        Two-point correlation function from :mod:`src.correlation`.
    L:
        Arm length [m].
    ell_r:
        Correlation length ℓ_r [m].
    Gamma_S:
        SF strength Γ_S (scales the overall amplitude).
    N_t:
        Number of time samples (one round-trip τ₀ = 2L/c).
    key:
        JAX PRNG key.  If ``None``, a default key is used.
    c:
        Speed of light [m/s].

    Returns
    -------
    t_grid : jnp.ndarray, shape (N_t,)
        Time axis [s].
    opd : jnp.ndarray, shape (N_t,)
        OPD realisation [m].
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    tau0 = 2.0 * L / c
    t_grid = jnp.linspace(0.0, tau0, N_t)

    K = _build_covariance(rho_func, L, ell_r, N_t, c=c)

    # Regularise for numerical stability
    K = K + 1e-10 * jnp.eye(N_t)

    # Cholesky factor: K = L @ Lᵀ
    chol = jnp.linalg.cholesky(K)

    # Sample standard normal and project
    xi = jax.random.normal(key, shape=(N_t,))
    opd = jnp.sqrt(Gamma_S) * c * (chol @ xi)

    return t_grid, opd


def sample_opd_ensemble(
    rho_func,
    L: float,
    ell_r: float,
    Gamma_S: float = 1.0,
    N_t: int = 256,
    n_samples: int = 128,
    seed: int = 42,
    c: float = C_LIGHT,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Draw an ensemble of independent OPD realisations via ``vmap``.

    Parameters
    ----------
    rho_func:
        Correlation function from :mod:`src.correlation`.
    L:
        Arm length [m].
    ell_r:
        Correlation length ℓ_r [m].
    Gamma_S:
        SF strength Γ_S.
    N_t:
        Samples per realisation.
    n_samples:
        Number of independent realisations in the ensemble.
    seed:
        Base PRNG seed.
    c:
        Speed of light [m/s].

    Returns
    -------
    t_grid : jnp.ndarray, shape (N_t,)
        Shared time axis [s].
    opd_ensemble : jnp.ndarray, shape (n_samples, N_t)
        Ensemble of OPD realisations [m].
    """
    base_key = jax.random.PRNGKey(seed)
    keys = jax.random.split(base_key, n_samples)

    def _single(key):
        _, opd = sample_opd_realisation(rho_func, L, ell_r, Gamma_S, N_t, key, c)
        return opd

    opd_ensemble = vmap(_single)(keys)  # (n_samples, N_t)
    tau0 = 2.0 * L / c
    t_grid = jnp.linspace(0.0, tau0, N_t)
    return t_grid, opd_ensemble


def empirical_psd_from_ensemble(
    t_grid: jnp.ndarray,
    opd_ensemble: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Estimate the OPD PSD from an ensemble using periodogram averaging.

    Each realisation is Fourier-transformed; the power spectra are averaged
    across the ensemble (Bartlett's method).

    Parameters
    ----------
    t_grid:
        Time grid [s], shape ``(N_t,)``.
    opd_ensemble:
        Ensemble of OPD realisations [m], shape ``(n_samples, N_t)``.

    Returns
    -------
    f : jnp.ndarray, shape (N_t // 2,)
        Positive frequency axis [Hz].
    psd : jnp.ndarray, shape (N_t // 2,)
        Averaged one-sided PSD [m²/Hz].
    """
    dt = t_grid[1] - t_grid[0]
    N_t = t_grid.shape[0]

    # FFT of each realisation
    spectra = jnp.fft.rfft(opd_ensemble, axis=1)            # (n_samples, N_t//2+1)
    power = (jnp.abs(spectra) ** 2) * (2.0 * dt / N_t)     # one-sided scaling
    psd = jnp.mean(power, axis=0)                           # ensemble average

    f = jnp.fft.rfftfreq(N_t, d=float(dt))
    return f[1:], psd[1:]   # skip DC
