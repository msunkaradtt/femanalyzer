import numpy as np
import skfem as sk
from skfem.helpers import grad, transpose, trace
from .mechanical_solver import elastic_stiffness
from .thermal_solver import solve_thermal_steady_state

# This new linear form calculates the "load" from thermal expansion
@sk.LinearForm
def thermal_expansion_load(v, w):
    """
    Linear form for the thermo-elastic load.
    This is the integral of: epsilon(v) : D : epsilon_thermal
    """
    eps_v = 0.5 * (grad(v) + transpose(grad(v)))
    
    # Thermal strain tensor (isotropic)
    # epsilon_thermal = alpha * (T - T_ref) * I
    # We only need its trace: tr(eps_th) = 3 * alpha * (T - T_ref)
    delta_T = w.T - w.T_ref
    tr_eps_th = 3 * w.alpha * delta_T
    
    # D : epsilon_thermal = (lambda * tr(eps_th)) * I + (2 * mu * eps_th)
    # This simplifies to: (3 * lambda + 2 * mu) * alpha * delta_T * I
    
    # The term we need is trace(eps_v) * (3 * lambda + 2 * mu) * alpha * delta_T
    return trace(eps_v) * (3 * w.lam + 2 * w.mu) * w.alpha * delta_T

def solve_thermo_mechanical(mesh, material,
                            temp_boundary_fn,
                            fixed_boundary_fn,
                            T_ref):
    """Solves a coupled thermo-mechanical problem."""
    print("Running Thermo-Mechanical Simulation...")
    
    # --- Step 1: Define Structural Basis (P2) ---
    mech_element = sk.ElementVectorH1(sk.ElementTetP2())
    mech_basis = sk.Basis(mesh, mech_element)
    p2_quadrature = mech_basis.quadrature

    # --- Step 2: Solve for Temperature Field ---
    thermal_basis, T_field = solve_thermal_steady_state(
        mesh, material, temp_boundary_fn, quadrature=p2_quadrature
    )
    
    # --- Step 3: Solve for Structural Deformation ---
    A = sk.asm(elastic_stiffness, mech_basis, mu=material.mu, lam=material.lam)

    T_interpolated = thermal_basis.interpolate(T_field)
    
    b = sk.asm(thermal_expansion_load,
               mech_basis,
               T=T_interpolated,
               T_ref=T_ref,
               alpha=material.coefficient_thermal_expansion,
               mu=material.mu,
               lam=material.lam)

    # Apply mechanical constraints (fixed bottom)
    bottom_dofs = mech_basis.get_dofs(fixed_boundary_fn)
    A, b = sk.enforce(A, b, D=bottom_dofs)

    # Solve for displacement u
    u = sk.solve(A, b)
    
    return mech_basis, u