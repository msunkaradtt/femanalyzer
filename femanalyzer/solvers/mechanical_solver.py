import numpy as np
import skfem as sk
from skfem.helpers import grad, transpose, dot

# This is the physics from your fem_pde_solver.py
@sk.BilinearForm
def elastic_stiffness(u, v, w):
    """Bilinear form for linear elasticity (stiffness matrix)."""
    mu = w.mu
    lam = w.lam
    eps_u = 0.5 * (grad(u) + transpose(grad(u)))
    eps_v = 0.5 * (grad(v) + transpose(grad(v)))
    return 2 * mu * np.einsum("ij...,ij...->...", eps_u, eps_v) + \
           lam * np.einsum("ii...,jj...->...", eps_u, eps_v)

@sk.LinearForm
def pressure_load(v, w):
    """Linear form for a uniform pressure load."""
    traction_vector = np.array([0.0, 0.0, -abs(w.pressure)])
    return dot(traction_vector, v)

def solve_pressure_load(mesh, material, pressure,
                        fixed_boundary_fn,
                        pressure_boundary_fn):
    """
    Solves a mechanical problem with a top pressure load.
    
    Boundaries are defined by user-provided functions (lambdas).
    
    Args:
        fixed_boundary_fn: A function (lambda x: ...) that returns True
                           for nodes on the fixed boundary.
        pressure_boundary_fn: A function (lambda x: ...) that returns True
                              for facets on the pressure boundary.
    """
    print("Running Mechanical (Pressure) Simulation...")
    
    element = sk.ElementVectorH1(sk.ElementTetP2())
    basis = sk.Basis(mesh, element)

    # Define boundaries using the provided functions
    bottom_dofs = basis.get_dofs(fixed_boundary_fn)
    top_facets = mesh.facets_satisfying(pressure_boundary_fn)
    
    # Assemble stiffness matrix
    A = sk.asm(elastic_stiffness, basis, mu=material.mu, lam=material.lam)

    # Assemble load vector
    facet_basis = sk.FacetBasis(mesh, element, facets=top_facets)
    b = sk.asm(pressure_load, facet_basis, pressure=pressure)

    # Enforce boundary conditions (fixed bottom)
    A, b = sk.enforce(A, b, D=bottom_dofs)

    # Solve
    u = sk.solve(A, b)
    
    return basis, u