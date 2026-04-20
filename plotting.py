"""
Plotting utilities to reproduce Figures 1, 2, and 4 of the paper.

Each function mirrors one paper figure:

- :func:`plot_fig1`  — Fig. 1: scaled PSD S_C(ν) for classes (b) and (c), no cavity.
- :func:`plot_fig2`  — Fig. 2: cavity-enhanced PSD for QUEST vs LIGO.
- :func:`plot_fig4`  — Fig. 4: κ-dependence of class (c) PSD with SNC = κ·SC scaling.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.psd_calculator import (
    psd_no_cavity,
    psd_no_cavity_factorised,
    psd_with_cavity,
    scaled_psd_c,
    scaled_psd_nc,
)
from src.correlation import (
    rho_inverse_spatial,
    rho_inverse_spacetime,
    rho_exponential_spatial,
    rho_exponential_spacetime,
)
from src.constants import DEFAULT_L, DEFAULT_ELL_R, C_LIGHT, DEFAULT_R_M
from src.mli_geometry import dimensionless_nu


def plot_fig1(save: bool = True, filename: str = "fig1.png") -> None:
    """Reproduce Fig. 1: scaled PSDs for all correlation classes, no arm cavity.

    Plots S_C(ν) = c·S(f) / (Γ_S·ℓ_r·L²) (Eq. 26) for the inverse (b) and
    exponential (c) classes at κ = 0.01.  S_NC(ν) for class (a) is also shown.

    Parameters
    ----------
    save:
        If True, save the figure to ``filename``.
    filename:
        Output file path.
    """
    L = DEFAULT_L
    ell_r = DEFAULT_ELL_R
    f_grid = jnp.logspace(5, 9, 400)
    nu = dimensionless_nu(f_grid, C_LIGHT / (2.0 * L))

    S_a = psd_no_cavity_factorised(nu)
    S_b1 = psd_no_cavity(f_grid, rho_inverse_spatial, L, ell_r)
    S_b2 = psd_no_cavity(f_grid, rho_inverse_spacetime, L, ell_r)
    S_c1 = psd_no_cavity(f_grid, rho_exponential_spatial, L, ell_r)
    S_c2 = psd_no_cavity(f_grid, rho_exponential_spacetime, L, ell_r)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False)

    # (a) Factorised
    axes[0].loglog(nu, S_a, lw=2.2, color='tab:purple')
    axes[0].axvline(1.0, color='k', ls='--', label='ν=1')
    axes[0].set_title('(a) Factorised ρ_F')
    axes[0].set_xlabel('ν')
    axes[0].set_ylabel('S_NC(ν)')
    axes[0].grid(True, which='both', alpha=0.3)
    axes[0].legend()

    # (b) Inverse
    axes[1].loglog(nu, scaled_psd_c(f_grid, L, ell_r, S_b1), label='ρ_IS (b1)', lw=2.2, color='tab:blue')
    axes[1].loglog(nu, scaled_psd_c(f_grid, L, ell_r, S_b2), label='ρ_IST (b2)', lw=2.2, color='tab:orange')
    axes[1].axvline(1.0, color='k', ls='--')
    axes[1].set_title('(b) Inverse ρ_Im')
    axes[1].set_xlabel('ν')
    axes[1].set_ylabel('S_C(ν)')
    axes[1].legend()
    axes[1].grid(True, which='both', alpha=0.3)

    # (c) Exponential
    axes[2].loglog(nu, scaled_psd_c(f_grid, L, ell_r, S_c1), label='ρ_ES (c1)', lw=2.2, color='tab:red')
    axes[2].loglog(nu, scaled_psd_c(f_grid, L, ell_r, S_c2), label='ρ_EST (c2)', lw=2.2, color='tab:cyan')
    axes[2].axvline(1.0, color='k', ls='--')
    axes[2].set_title('(c) Exponential ρ_Em')
    axes[2].set_xlabel('ν')
    axes[2].set_ylabel('S_C(ν)')
    axes[2].legend()
    axes[2].grid(True, which='both', alpha=0.3)

    fig.suptitle('Fig. 1 — MLI output for three correlation classes (JAX)', fontsize=13)
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
    plt.show()


def plot_fig2(save: bool = True, filename: str = "fig2.png") -> None:
    """Reproduce Fig. 2: cavity-enhanced PSD for QUEST vs LIGO.

    Shows S_C(ν) for inverse ρ_IS and exponential ρ_ES correlation functions
    with arm cavities (LIGO, L=4000 m) vs without (QUEST, L=3 m).

    Parameters
    ----------
    save:
        If True, save the figure to ``filename``.
    filename:
        Output file path.
    """
    ell_r = DEFAULT_ELL_R
    f_QUEST = jnp.logspace(5, 9, 400)
    f_LIGO = jnp.logspace(1, 5, 400)

    L_QUEST = 3.0
    L_LIGO = 4000.0

    nu_QUEST = dimensionless_nu(f_QUEST, C_LIGHT / (2.0 * L_QUEST))
    nu_LIGO = dimensionless_nu(f_LIGO, C_LIGHT / (2.0 * L_LIGO))

    fig, axes = plt.subplots(2, 1, figsize=(9, 10))

    for ax, rho_func, title, color_ligo, color_quest in [
        (axes[0], rho_inverse_spatial, 'Inverse ρ_IS', 'tab:red', 'tab:blue'),
        (axes[1], rho_exponential_spatial, 'Exponential ρ_ES', 'tab:orange', 'tab:green'),
    ]:
        S_LIGO = psd_with_cavity(f_LIGO, rho_func, L_LIGO, ell_r, R_M=DEFAULT_R_M)
        S_QUEST = psd_with_cavity(f_QUEST, rho_func, L_QUEST, ell_r, R_M=DEFAULT_R_M)
        S_nc_LIGO = psd_no_cavity(f_LIGO, rho_func, L_LIGO, ell_r)

        ax.loglog(nu_LIGO, scaled_psd_c(f_LIGO, L_LIGO, ell_r, S_LIGO),
                  label='LIGO (cavity)', lw=2.2, color=color_ligo)
        ax.loglog(nu_QUEST, scaled_psd_c(f_QUEST, L_QUEST, ell_r, S_QUEST),
                  label='QUEST (cavity)', lw=2.2, color=color_quest)
        ax.loglog(nu_LIGO, scaled_psd_c(f_LIGO, L_LIGO, ell_r, S_nc_LIGO),
                  label='No cavity (L=4000m)', lw=1.5, ls='--', color='k')
        ax.axvline(1.0, color='grey', ls=':')
        ax.set_title(title)
        ax.set_xlabel('ν')
        ax.set_ylabel('S_C(ν)')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)

    fig.suptitle('Fig. 2 — MLI output with arm cavities (JAX)', fontsize=13)
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
    plt.show()


def plot_fig4(save: bool = True, filename: str = "fig4.png") -> None:
    """Reproduce Fig. 4: κ-dependence of class (c) PSD, SNC = κ·SC.

    Shows S_NC(ν) = κ·S_C(ν) vs ν for exponential ρ_ES at several κ values,
    demonstrating the universal 1/ν² high-frequency tail (paper Sec.
    "Distinguishing correlation functions").

    Parameters
    ----------
    save:
        If True, save to ``filename``.
    filename:
        Output file path.
    """
    L = DEFAULT_L
    f_LRT = C_LIGHT / (2.0 * L)
    f_grid = jnp.logspace(5, 9, 400)
    nu = dimensionless_nu(f_grid, f_LRT)

    kappa_vals = [0.025, 0.01, 0.005, 0.0025]
    styles = ['-', '--', '-.', ':']
    colors = ['tab:red', 'tab:green', 'tab:brown', 'tab:pink']

    plt.figure(figsize=(9, 6))
    for kappa, ls, color in zip(kappa_vals, styles, colors):
        ell_r = kappa * L
        S = psd_no_cavity(f_grid, rho_exponential_spatial, L, ell_r)
        S_nc = kappa * scaled_psd_c(f_grid, L, ell_r, S)
        plt.loglog(nu, S_nc, ls=ls, lw=2.2, color=color, label=f'κ = {kappa}')

    plt.axvline(1.0, color='k', ls='--', label='ν=1')
    # Indicate 1/ν² reference slope
    nu_ref = jnp.logspace(0, 2, 50)
    plt.loglog(nu_ref, 1e-3 / nu_ref ** 2, 'k:', lw=1.2, label='∝ 1/ν²')
    plt.xlabel('Scaled frequency ν')
    plt.ylabel('S_NC(ν) = κ · S_C(ν)')
    plt.title('Fig. 4 — κ-dependence of class (c) (JAX)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
    plt.show()
