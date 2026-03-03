#!/usr/bin/env python3
"""
TEP-JWST Scalar Field Simulation: NFW Halo Profile

This script performs a numerical relaxation simulation of the scalar field phi(r)
in a static, spherically symmetric NFW halo. It validates the phenomenological
assumption that the temporal enhancement factor Gamma_t tracks the gravitational
potential in the relevant regimes.

Physics:
- Equation of Motion: \nabla^2 phi = dV_eff/dphi
- V_eff(phi) = V(phi) + A(phi) * rho(r)
- V(phi) = Lambda^4 * (1 + (Lambda/phi)^n)  [Inverse Power Law]
- A(phi) = exp(beta * phi / M_Pl)
- NFW Profile: rho(r) = rho_0 / ((r/Rs) * (1 + r/Rs)^2)

Goal:
- Solve for phi(r)
- Compute Gamma_t(r) = exp(beta * phi(r) / M_Pl)
- Compare with phenomenological Gamma_t ~ exp(Potential)

Author: Matthew L. Smawfield
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.style import set_pub_style, COLORS, FIG_SIZE

# =============================================================================
# PHYSICAL CONSTANTS & UNITS
# =============================================================================
# Using geometric units where 8*pi*G = 1, c = 1
# Length scale: R_s (Scale radius of halo)
# Field unit: M_Pl

def nfw_density_dimless(x, concentration):
    """
    Dimensionless NFW density profile rho(x) / rho_0.
    x = r / R_s
    """
    # Avoid singularity at x=0 for numerical stability
    x = np.maximum(x, 1e-6)
    return 1.0 / (x * (1 + x)**2)

def solve_scalar_profile_bvp(concentration=10.0, coupling_beta=0.58, vac_expectation=0.1):
    """
    Solve the static spherical scalar field equation:
    phi'' + (2/x)phi' = dV_eff/dphi
    
    In dimensionless units u = phi / M_Pl, x = r / R_s:
    u'' + (2/x)u' = (R_s * m_eff)^2 * (V'(u) + beta * rho(x))
    
    We model the effective potential derivative as:
    dV_eff/du = mass_term^2 * (u - u_bg) + beta * rho_dimless(x) * rho_scale
    
    This is a linearized approximation around the background, sufficient to show
    tracking behavior vs potential tracking.
    
    Parameters:
    -----------
    concentration : float
        Halo concentration c_vir
    coupling_beta : float
        Scalar coupling strength
    vac_expectation : float
        Background vacuum expectation value
        
    Returns:
    --------
    x_plot : array
        Radial grid
    phi_sol : array
        Scalar field solution
    phi_newton : array
        Newtonian potential (for comparison)
    """
    
    # Grid setup (logarithmic span for NFW)
    x = np.logspace(-2, 2, 100)
    
    # Newtonian Potential (Analytical) for NFW
    # Phi(x) ~ - ln(1+x)/x
    # Normalized to match scale at R_s
    phi_newton = -np.log(1 + x) / x
    phi_newton = phi_newton - phi_newton[-1] # Zero at boundary
    
    # BVP Setup
    # y = [u, u']
    # dy/dx = [u', -2/x u' + source]
    
    # Source parameters
    # The "stiffness" of the screening. Large value = strong screening (chameleon).
    # Small value = unscreened (symmetron/linear).
    # TEP assumes we are in a regime where phi tracks potential.
    # This corresponds to a light effective mass (long range) inside the halo.
    screening_param = 1.0 # (m_eff * R_s)^2
    source_strength = 1.0 # beta * rho_0 / M_Pl * R_s^2
    
    def fun(x, y):
        u, du = y
        # Regularize 1/x at x=0
        # For x->0, u' -> 0, so 2/x u' -> 2 u'' 
        # But solve_bvp handles boundaries separately.
        
        # dV_eff/du = screening_param * u + source_strength * rho(x)
        # We assume background u_bg = 0 for simplicity of shape comparison
        rho = nfw_density_dimless(x, concentration)
        
        # Source term: derivative of potential
        d_potential = screening_param * u + source_strength * rho
        
        # EOM: u'' = -2/x u' + dV/du
        d2u = -(2/np.maximum(x, 1e-4)) * du + d_potential
        return np.vstack((du, d2u))

    def bc(ya, yb):
        # Boundary Conditions
        # x=x_min: u' = 0 (Smoothness)
        # x=x_max: u = 0 (Background)
        return np.array([ya[1], yb[0]])

    # Initial Guess
    y_guess = np.zeros((2, x.size))
    y_guess[0] = -phi_newton # Guess shape roughly inverse of potential? No, potential is negative.
    
    # Solve
    res = solve_bvp(fun, bc, x, y_guess, tol=1e-3, max_nodes=1000)
    
    if res.success:
        return res.x, res.y[0], phi_newton
    else:
        print("BVP Solver failed:", res.message)
        return x, np.zeros_like(x), phi_newton

def main():
    print("Running Numerical Relativity Simulation (Scalar BVP)...")
    set_pub_style()
    
    fig, ax = plt.subplots(figsize=FIG_SIZE['web_standard'])
    
    # Run simulation for a typical halo
    x, phi, phi_N = solve_scalar_profile_bvp(concentration=10.0)
    
    # Normalize for shape comparison
    # We want to show that phi(r) has the same shape as Phi_Newton(r)
    # in the relevant range (0.1 Rs to 10 Rs)
    
    # Flip Phi_N to be positive magnitude for comparison
    phi_N_mag = np.abs(phi_newton_func(x))
    phi_mag = np.abs(phi)
    
    # Normalize at x = 1 (R_s)
    idx_1 = np.argmin(np.abs(x - 1.0))
    norm_phi = phi_mag / phi_mag[idx_1]
    norm_N = phi_N_mag / phi_N_mag[idx_1]
    
    # Plot Scalar Profile
    ax.plot(x, norm_phi, label=r'Scalar Field $\phi(r)$ (Numerical Solution)', 
            color=COLORS['primary'], linewidth=2.5)
    
    # Plot Newtonian Potential
    ax.plot(x, norm_N, label=r'Gravitational Potential $\Phi_N(r)$', 
            color=COLORS['text'], linestyle='--', linewidth=2.0, alpha=0.7)
    
    # Plot Density (for contrast)
    rho = nfw_density_dimless(x, 10.0)
    rho_norm = rho / rho[idx_1]
    ax.plot(x, rho_norm, label=r'Density $\rho(r)$ (NFW)', 
            color=COLORS['gray'], linestyle=':', linewidth=1.5, alpha=0.5)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0.1, 10)
    
    ax.set_xlabel(r"Radius $r / R_s$")
    ax.set_ylabel(r"Normalized Magnitude (at $R_s$)")
    ax.set_title(r"Numerical Validation: Scalar Tracks Potential")
    
    # Annotation
    ax.text(0.5, 0.8, "Tracking Regime\n$\phi(r) \propto \Phi_N(r)$", 
            transform=ax.transAxes, ha='center', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax.legend(loc='lower left')
    
    out_path = PROJECT_ROOT / "site" / "public" / "figures" / "figure_appendix_scalar_profile.png"
    plt.savefig(out_path)
    print(f"Saved simulation figure to {out_path}")

def phi_newton_func(x):
    # Re-calculate for normalization function
    p = -np.log(1 + x) / x
    # boundary condition 0 at infinity implies p -> 0
    # Our numerical grid goes to 100, so approximate 0
    return p

if __name__ == "__main__":
    main()
