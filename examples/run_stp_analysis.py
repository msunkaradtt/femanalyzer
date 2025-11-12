import numpy as np
import femanalyzer as fem
import skfem as sk
import matplotlib.pyplot as plt

def get_surface_node_indices(mesh: sk.Mesh, surface_fn) -> np.ndarray:
    """
    Extracts (N,) array of node indices on a surface defined by a
    skfem facet function (like 'pressure_boundary').
    """
    print("Extracting surface node indices using facet function...")
    try:
        # Use the facet function to find all facets on the boundary
        facet_indices = mesh.facets_satisfying(surface_fn)
        
        if facet_indices.size == 0:
            print("Warning: 'get_surface_node_indices' found 0 facets matching the function.")
            return np.array([], dtype=int) # Return empty int array

        # Get all unique node indices from these facets
        our_facet_nodes = mesh.facets[:, facet_indices]
        node_indices = np.unique(our_facet_nodes)
        
        print(f"Found {node_indices.size} nodes on the target surface.")
        return node_indices

    except Exception as e:
         print(f"Warning: Could not find nodes for roughness calculation. Error: {e}")
         return np.array([], dtype=int) # Return empty int array


def run_analysis(
    sim_type, 
    original_mesh, 
    material, 
    sim_config,
    top_surface_node_indices # <-- This is the key change
):
    """
    A helper function to run a full before-and-after analysis
    on a pre-defined set of nodes.
    """
    print(f"\n==============================================")
    print(f"  STARTING ANALYSIS: {sim_type.upper()}")
    print(f"==============================================")
    
    # 1. Get "Before" roughness (using original mesh and indices)
    print("\n--- Measuring 'Before' State ---")
    points_before = original_mesh.p[:, top_surface_node_indices].T # Get coords
    results_before = fem.analyze_surface_roughness(points_before, detrend_method='plane')

    # 2. Run Simulation
    basis, u = fem.run_simulation(original_mesh, material, sim_config)
    
    # 3. Extract P1 displacements
    N_p1_nodes = original_mesh.p.shape[1] 
    u_x_p1 = u[0 : 3 * N_p1_nodes : 3]
    u_y_p1 = u[1 : 3 * N_p1_nodes : 3]
    u_z_p1 = u[2 : 3 * N_p1_nodes : 3]
    u_at_vertices = np.stack([u_x_p1, u_y_p1, u_z_p1]) # Shape (3, N)
    
    # 4. Create new deformed mesh
    new_node_coords = original_mesh.p + u_at_vertices
    deformed_mesh = sk.MeshTet(new_node_coords, original_mesh.t)
    
    # 5. Get "After" roughness (using deformed mesh and THE SAME indices)
    print("\n--- Measuring 'After' State ---")
    points_after = deformed_mesh.p[:, top_surface_node_indices].T # Get coords
    results_after = fem.analyze_surface_roughness(points_after, detrend_method='plane')

    # 6. Print Comparison
    print("\n--- Results Summary ---")
    sq_change = results_after['Sq'] - results_before['Sq']
    print(f"Sq (RMS Roughness) Before: {results_before['Sq']:.6e} m")
    print(f"Sq (RMS Roughness) After:  {results_after['Sq']:.6e} m")
    print(f"Change in Sq (After - Before): {sq_change:.6e} m")

    # 7. VISUALIZE RESULTS
    print("\n--- Generating Visualizations ---")
    
    # Plot the deformed 3D surface
    fem.plot_surface_roughness(points_after, title=f"Deformed Surface ({sim_type})")
    
    # Plot the full 3D FEM result (this opens an interactive PyVista window)
    fem.plot_fem_result(
        original_mesh, 
        u_at_vertices, 
        scalar_field=u_at_vertices[2, :], # Color by Z-displacement
        field_name="Z Displacement (m)"
    )
    print(f"==============================================\n")


def main():
    # --- 1. Setup ---
    print("Initializing...")
    material = fem.get_AlSi10Mg()
    
    # --- 2. Import and Mesh STEP File ---
    try:
        mesh, bbox = fem.create_mesh_from_stp("sample_3.step", mesh_size=0.05)
    except FileNotFoundError:
        print("ERROR: 'sample_1.stp' not found.")
        print("Please place the STEP file in the 'examples' folder.")
        return
    except Exception as e:
        print(f"ERROR: Failed to mesh STEP file. Is Gmsh installed correctly?")
        print(f"Details: {e}")
        return
        
    original_mesh = mesh.copy()

    fem.plot_mesh(original_mesh, title="Initial Meshed STEP File")

    # --- 3. Define Generic Boundary Conditions ---
    tol = (bbox['zmax'] - bbox['zmin']) * 0.1   # 10% tolerance
    
    def fixed_boundary(x):
        return x[2] < bbox['zmin'] + tol

    def pressure_boundary(x):
        # Facet function checks the midpoint (x,y,z) of the facet
        return x[2] > bbox['zmax'] - tol
        
    def thermal_boundary(x):
        # Node function checks one (x,y,z) point
        if x[2] > bbox['zmax'] - tol:
            return 600.0  # Hot top
        if x[2] < bbox['zmin'] + tol:
            return 300.0  # Cool bottom
        return None # Internal node

    # --- 4. Find Surface Nodes ONCE ---
    print("\n--- Finding analysis surface nodes ---")
    # Find the nodes on the "top" surface (as defined by pressure_boundary)
    top_surface_node_indices = get_surface_node_indices(original_mesh, pressure_boundary)

    if top_surface_node_indices.size == 0:
        print("ERROR: Could not find any nodes on the top surface.")
        print("Adjust the 'tol' or 'pressure_boundary' function.")
        return

    # Plot initial state
    points_before = original_mesh.p[:, top_surface_node_indices].T
    fem.plot_surface_roughness(points_before, title="Initial STEP File Surface")

    # --- 5. Run Pressure Analysis ---
    config_pressure = {
        'type': 'pressure',
        'pressure': 5e6,
        'fixed_boundary_fn': fixed_boundary,
        'pressure_boundary_fn': pressure_boundary
    }
    run_analysis(
        "pressure", 
        original_mesh, 
        material, 
        config_pressure, 
        top_surface_node_indices # Pass indices
    )
    
    # --- 6. Run Thermo-Mechanical Analysis ---
    config_thermo = {
        'type': 'thermo-mechanical',
        'T_ref': 300.0,
        'fixed_boundary_fn': fixed_boundary,
        'temp_boundary_fn': thermal_boundary
    }
    run_analysis(
        "thermo-mechanical", 
        original_mesh, 
        material, 
        config_thermo, 
        top_surface_node_indices # Pass same indices
    )

    # --- 7. Show Matplotlib plots ---
    print("Displaying all generated Matplotlib plots...")
    plt.show()

if __name__ == "__main__":
    main()