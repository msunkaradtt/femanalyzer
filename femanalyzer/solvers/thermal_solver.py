import skfem as sk
from skfem.helpers import grad, dot
import numpy as np

# Physics for Heat Transfer
@sk.BilinearForm
def conductivity(u, v, w):
    """Bilinear form for steady-state heat equation (conductivity)."""
    return w.k * dot(grad(u), grad(v))

def solve_thermal_steady_state(mesh, material,
                               temp_boundary_fn,
                               quadrature=None):
    """
    Solves for the temperature field T(x,y,z).
    
    Args:
        temp_boundary_fn: A function (lambda x: ...) that returns
                          a temperature (float) for nodes on a boundary,
                          or None for internal nodes.
    """
    print("Running Thermal (Steady-State) Simulation...")
    
    element = sk.ElementTetP1()

    if quadrature is None:
        basis = sk.Basis(mesh, element)
    else:
        print("  (Using provided P2 quadrature for thermal solve)")
        basis = sk.Basis(mesh, element, quadrature=quadrature)
        
    A = sk.asm(conductivity, basis, k=material.thermal_conductivity)

    # Find all Dirichlet DOFs and their values
    dirichlet_dofs = []
    dirichlet_values = []
    
    for dof in basis.get_dofs().flatten():
        # Get the (x,y,z) coordinate for this DOF
        coord = basis.doflocs[:, dof]
        temp = temp_boundary_fn(coord)
        
        if temp is not None:
            dirichlet_dofs.append(dof)
            dirichlet_values.append(temp)

    if not dirichlet_dofs:
        raise ValueError("No temperature boundaries defined. 'temp_boundary_fn' found no nodes.")
        
    dirichlet_dofs = np.array(dirichlet_dofs, dtype=int)
    
    # Create the 'b' vector and pre-set boundary values
    b = np.zeros(basis.N)
    b[dirichlet_dofs] = dirichlet_values

    # Enforce all Dirichlet conditions
    A, b = sk.enforce(A, b, D=dirichlet_dofs)
    
    T = sk.solve(A, b)
    
    return basis, T