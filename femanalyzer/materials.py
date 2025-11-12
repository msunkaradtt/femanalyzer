import types

def get_AlSi10Mg():
    """Returns a SimpleNamespace containing AlSi10Mg properties."""
    mat = types.SimpleNamespace()
    
    # Mechanical properties
    mat.density = 2.67e3          # kg/m^3
    mat.elastic_modulus = 73.5e9  # Pa
    mat.poisson_ratio = 0.33
    
    # Thermal properties
    mat.thermal_conductivity = 140            # W/m·K
    mat.coefficient_thermal_expansion = 21e-6 # 1/K
    
    # Derived properties for FEM
    mat.mu = mat.elastic_modulus / (2 * (1 + mat.poisson_ratio))
    mat.lam = (mat.elastic_modulus * mat.poisson_ratio) / \
              ((1 + mat.poisson_ratio) * (1 - 2 * mat.poisson_ratio))
              
    return mat