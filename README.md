# `femanalyzer`

A Python package for analyzing the effect of Finite Element (FEA) simulations on 3D surface roughness.

## Overview

`femanalyzer` provides a complete, end-to-end workflow for computational analysis. It allows you to import a 3D CAD file, generate a simulation-ready volume mesh, measure the initial surface roughness, apply physical loads (like pressure or heat), and then re-measure the roughness on the deformed part to quantify the change.

This tool is designed to bridge the gap between CAD, FEA, and 3D surface metrology.

## Core Features

  * **CAD Import:** Load complex 3D geometries from `.stp` (STEP) files.
  * **Meshing:** Automatically generate 3D tetrahedral (volume) meshes suitable for FEA.
  * **FEA Solvers:**
      * **Mechanical:** Solves 3D linear elasticity problems for deformation under pressure.
      * **Thermo-mechanical:** Solves coupled steady-state thermal and structural problems to find deformation from thermal stress.
  * **Roughness Analysis:**
      * Implements ISO 25178 standards for areal (3D) surface roughness.
      * Calculates **Sa** (Arithmetical Mean Height), **Sq** (RMS Height), and **Sz** (Max Height).
      * Includes `plane` detrending (form removal) via PCA for accurate results on non-flat surfaces.
  * **Visualization:**
      * Interactive 3D visualization of meshes and FEM results (displacement fields) using **PyVista**.
      * 3D scatter plots of rough surfaces using **Matplotlib**.

-----

## Installation & How to Run Locally

Follow these steps to set up and run the package on your local machine.

### 1\. Prerequisites (Dependencies)

This package requires several external libraries. You must also have the **Gmsh** meshing software installed on your system and available in your system's PATH.

You can install all Python dependencies using `pip`. The main libraries are:

  * `numpy`
  * `skfem`
  * `pygmsh`
  * `trimesh`
  * `scikit-learn`
  * `pyvista`
  * `matplotlib`
  * `python-cascaded` (A key runtime dependency for `trimesh` to read STEP files)

### 2\. Setup

1.  **Clone the repository** (or place the `femanalyzer` source code in a project folder). Your folder structure should look like this:

    ```
    femanalyzer_project/
    ├── pyproject.toml
    ├── femanalyzer/
    │   ├── __init__.py
    │   ├── api.py
    │   ├── ... (all other .py files)
    └── examples/
    ```

2.  **Create a Virtual Environment** (Recommended):

    ```bash
    cd femanalyzer_project
    python -m venv venv

    # On Windows
    .\venv\Scripts\activate

    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the Package:** Install `femanalyzer` and its core dependencies.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Runtime Dependencies:** `trimesh`'s STEP file importer requires `python-cascaded`. Install it separately:

    ```bash
    pip install python-cascaded
    ```

### 3\. Example Workflow (How to Run)

This example shows the complete workflow: import a STEP file, measure "before" roughness, run both pressure and thermal simulations, and measure "after" roughness for each.

1.  **Place your STEP file** (e.g., `sample_1.stp`) inside the `examples/` folder.

2.  **Create the script `examples/run_local_analysis.py`** and paste the following code into it.

    ```python
    import numpy as np
    import femanalyzer as fem
    import skfem as sk
    import matplotlib.pyplot as plt

    def get_surface_node_indices(mesh: sk.Mesh, surface_fn) -> np.ndarray:
        """
        Extracts (N,) array of node indices on a surface defined by a
        skfem facet function.
        """
        print("Extracting surface node indices using facet function...")
        try:
            facet_indices = mesh.facets_satisfying(surface_fn)
            if facet_indices.size == 0:
                print("Warning: 'get_surface_node_indices' found 0 facets.")
                return np.array([], dtype=int)

            # 'mesh.facets' holds the (3, N_facets) connectivity array
            our_facet_nodes = mesh.facets[:, facet_indices]
            node_indices = np.unique(our_facet_nodes)
            
            print(f"Found {node_indices.size} nodes on the target surface.")
            return node_indices

        except Exception as e:
            print(f"Warning: Could not find nodes for roughness calculation. Error: {e}")
            return np.array([], dtype=int)


    def run_analysis(
        sim_type, 
        original_mesh, 
        material, 
        sim_config,
        top_surface_node_indices
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
        u_at_vertices = np.stack([
            u[0 : 3 * N_p1_nodes : 3],
            u[1 : 3 * N_p1_nodes : 3],
            u[2 : 3 * N_p1_nodes : 3]
        ])
        
        # 4. Create new deformed mesh
        new_node_coords = original_mesh.p + u_at_vertices
        deformed_mesh = sk.MeshTet(new_node_coords, original_mesh.t)
        
        # 5. Get "After" roughness (using deformed mesh and THE SAME indices)
        print("\n--- Measuring 'After' State ---")
        points_after = deformed_mesh.p[:, top_surface_node_indices].T
        results_after = fem.analyze_surface_roughness(points_after, detrend_method='plane')

        # 6. Print Comparison
        print("\n--- Results Summary ---")
        sq_change = results_after['Sq'] - results_before['Sq']
        print(f"Sq (RMS Roughness) Before: {results_before['Sq']:.6e} m")
        print(f"Sq (RMS Roughness) After:  {results_after['Sq']:.6e} m")
        print(f"Change in Sq (After - Before): {sq_change:.6e} m")

        # 7. VISUALIZE RESULTS
        print("\n--- Generating Visualizations ---")
        fem.plot_surface_roughness(points_after, title=f"Deformed Surface ({sim_type})")
        
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
            mesh, bbox = fem.create_mesh_from_stp("sample_1.stp", mesh_size=0.05)
        except Exception as e:
            print(f"ERROR: Failed to mesh STEP file. Is Gmsh installed correctly?")
            print(f"Details: {e}")
            return
            
        original_mesh = mesh.copy()

        # --- 3. Plot Initial Mesh ---
        # (Close the PyVista window to continue the script)
        fem.plot_mesh(original_mesh, title="Initial Meshed STEP File")

        # --- 4. Define Generic Boundary Conditions ---
        # Use the bounding_box to define boundaries
        tol = (bbox['zmax'] - bbox['zmin']) * 0.1   # 10% tolerance
        
        def fixed_boundary(x):
            # Node function: fix if in bottom 10%
            return x[2] < bbox['zmin'] + tol

        def pressure_boundary(x):
            # Facet function: apply pressure if in top 10%
            return x[2] > bbox['zmax'] - tol
            
        def thermal_boundary(x):
            # Node function: set temperature based on position
            if x[2] > bbox['zmax'] - tol:
                return 600.0  # Hot top
            if x[2] < bbox['zmin'] + tol:
                return 300.0  # Cool bottom
            return None # Internal node

        # --- 5. Find Surface Nodes to Analyze ---
        print("\n--- Finding analysis surface nodes ---")
        top_surface_node_indices = get_surface_node_indices(original_mesh, pressure_boundary)

        if top_surface_node_indices.size == 0:
            print("ERROR: Could not find any nodes on the top surface.")
            return

        points_before = original_mesh.p[:, top_surface_node_indices].T
        fem.plot_surface_roughness(points_before, title="Initial STEP File Surface")

        # --- 6. Run Pressure Analysis ---
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
            top_surface_node_indices
        )
        
        # --- 7. Run Thermo-Mechanical Analysis ---
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
            top_surface_node_indices
        )

        # --- 8. Show Matplotlib plots ---
        print("Displaying all generated Matplotlib plots...")
        plt.show()

    if __name__ == "__main__":
        main()
    ```

3.  **Run the script** from your terminal (make sure your virtual environment is active):

    ```bash
    # Make sure you are in the project's root directory
    python examples/run_local_analysis.py
    ```

    This will:

    1.  Open a PyVista window showing the initial mesh.
    2.  After you close it, it will run the pressure simulation, save a plot, and open a PyVista window of the result.
    3.  After you close it, it will run the thermal simulation, save a plot, and open a final PyVista window.
    4.  Finally, it will open all `matplotlib` surface plots.