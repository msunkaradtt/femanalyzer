import numpy as np
import pyvista as pv
import skfem as sk
import copy

# Import from our own package
from .api import run_simulation

def _scale_config(config: dict, factor: float) -> dict:
    """Helper to scale loads in a simulation config."""
    new_config = copy.deepcopy(config)
    
    if new_config['type'] == 'pressure':
        new_config['pressure'] *= factor
    
    elif new_config['type'] == 'thermo-mechanical':
        T_ref = new_config['T_ref']
        
        # 1. Save a reference to the *original* function
        original_boundary_fn = new_config['temp_boundary_fn'] 

        # 2. This is the fix: A factory function that "bakes"
        #    the value of 'factor' into the 'scale_factor' argument.
        def create_scaled_fn(scale_factor=factor):
            
            def scaled_thermal_boundary(x):
                # 3. Call the original function
                original_temp = original_boundary_fn(x) 
                
                if original_temp is not None:
                    # 4. Use the "baked-in" scale_factor
                    return T_ref + (original_temp - T_ref) * scale_factor
                return None
            
            # 5. Return the new function
            return scaled_thermal_boundary
            
        # 6. Overwrite the config with the function returned
        #    by the factory. This function has the correct factor.
        new_config['temp_boundary_fn'] = create_scaled_fn()
        
    return new_config

def solve_time_series(
    mesh: sk.Mesh,
    material,
    sim_config: dict,
    num_steps: int = 30,
    output_file: str = "simulation_results.npz"
):
    """
    Runs a quasi-static (ramped load) simulation and saves all
    displacement results to a compressed .npz file.
    
    Args:
        num_steps: How many steps to solve between 0 and full load.
        output_file: The .npz file to save.
    """
    print(f"Solving time series for '{output_file}'...")
    print(f"Total steps: {num_steps}")

    N_p1_nodes = mesh.p.shape[1]
    displacements_list = [] # A list to hold each frame's (3, N) array
    
    # --- Run Time-Step Loop ---
    load_factors = np.linspace(0.0, 1.0, num_steps)
    
    for i, factor in enumerate(load_factors):
        print(f"  Computing step {i+1}/{num_steps} (Load: {factor*100:.1f}%)")
        
        if factor == 0.0:
            u_at_vertices = np.zeros_like(mesh.p)
        else:
            current_config = _scale_config(sim_config, factor)
            basis, u = run_simulation(mesh, material, current_config)
            
            # Extract P1 displacements (3, N)
            u_at_vertices = np.stack([
                u[0 : 3 * N_p1_nodes : 3],
                u[1 : 3 * N_p1_nodes : 3],
                u[2 : 3 * N_p1_nodes : 3]
            ])
            
        displacements_list.append(u_at_vertices)

    # --- Save Results to File ---
    # Stack the list of (3, N) arrays into one big (S, 3, N) array
    all_displacements = np.array(displacements_list)
    
    # Save everything needed for rendering into a single,
    # compressed .npz file (far more efficient than JSON)
    np.savez_compressed(
        output_file,
        displacements=all_displacements, # (steps, 3, N_nodes)
        mesh_points=mesh.p,             # (3, N_nodes)
        mesh_tets=mesh.t                # (4, N_elements)
    )
    
    print(f"\nSimulation results saved to: '{output_file}'")


def create_video_from_file(
    results_file: str,
    output_filename: str = "simulation.mp4",
    total_time: float = 6.0
):
    """
    Reads a simulation .npz file and renders it as an MP4 video.
    """
    print(f"Loading results from '{results_file}'...")
    
    # Load the entire results database
    try:
        data = np.load(results_file)
        displacements = data['displacements'] # (steps, 3, N_nodes)
        mesh_points = data['mesh_points']     # (3, N_nodes)
        mesh_tets = data['mesh_tets']         # (4, N_elements)
    except Exception as e:
        print(f"Error: Failed to load results file '{results_file}'. {e}")
        return

    num_frames = displacements.shape[0]
    framerate = num_frames / total_time
    
    print(f"Creating video '{output_filename}' ({num_frames} frames, {total_time}s)...")

    # --- Setup PyVista Plotter ---
    plotter = pv.Plotter(off_screen=True)
    plotter.open_movie(output_filename, framerate=framerate)
    
    # Create the base UnstructuredGrid from the mesh data
    # This is the *original, undeformed* mesh
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: mesh_tets.T}, mesh_points.T)
    wireframe = grid.copy() # A copy for the wireframe
    
    # Find sensible scalar bar limits based on the *final* frame
    final_scalars = displacements[-1, 2, :] # Final Z-displacement
    smin, smax = np.min(final_scalars), np.max(final_scalars)
    
    # --- Render Loop ---
    for i in range(num_frames):
        print(f"  Rendering frame {i+1}/{num_frames}...")
        
        # Get the (3, N) displacement array for this frame
        u_at_vertices = displacements[i]
        
        # Update the grid with this frame's vector data
        grid["Displacement"] = u_at_vertices.T
        deformed_grid = grid.warp_by_vector("Displacement")
        
        # Get scalar field (Z-displacement) for coloring
        scalars = u_at_vertices[2, :]
        
        # Render the frame
        plotter.clear()
        plotter.add_mesh(
            deformed_grid,
            cmap='viridis',
            scalars=scalars,
            scalar_bar_args={'title': "Z Displacement (m)"},
            clim=[smin, smax] # Lock the colorbar
        )
        plotter.add_mesh(wireframe, style='wireframe', color='black', opacity=0.2)
        plotter.add_axes()
        plotter.view_isometric()
        
        plotter.write_frame() # Write this frame to the video

    # Clean up
    plotter.close()
    print(f"\nVideo saved successfully: '{output_filename}'")