from .solvers import mechanical_solver, thermo_mechanical_solver

def run_simulation(mesh, material, config):
    """
    Main entry point for running a simulation.
    
    Args:
        mesh: The skfem.Mesh object.
        material: The material properties object.
        config (dict): A dictionary defining the simulation.
        
    Example Configs:
    
    config_pressure = {
        'type': 'pressure',
        'pressure': 5e6,
        'fixed_boundary_fn': lambda x: x[2] < 0.01,
        'pressure_boundary_fn': lambda x: x[2] > 0.99
    }
    
    config_thermo_mech = {
        'type': 'thermo-mechanical',
        'T_ref': 300.0,
        'fixed_boundary_fn': lambda x: x[2] < 0.01,
        'temp_boundary_fn': lambda x: 600. if x[2] > 0.99 else (300. if x[2] < 0.01 else None)
    }
    """
    sim_type = config.get('type')
    
    if sim_type == 'pressure':
        return mechanical_solver.solve_pressure_load(
            mesh, material,
            pressure=config['pressure'],
            fixed_boundary_fn=config['fixed_boundary_fn'],
            pressure_boundary_fn=config['pressure_boundary_fn']
        )
    
    elif sim_type == 'thermo-mechanical':
        return thermo_mechanical_solver.solve_thermo_mechanical(
            mesh, material, 
            temp_boundary_fn=config['temp_boundary_fn'],
            fixed_boundary_fn=config['fixed_boundary_fn'],
            T_ref=config['T_ref']
        )
        
    else:
        raise ValueError(f"Unknown simulation type: '{sim_type}'")