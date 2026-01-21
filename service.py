import os
import tempfile
import numpy as np
import trimesh
import skfem
import gmsh
import math

# Import your existing package modules
import femanalyzer as fem 

def create_volume_mesh_from_glb(glb_path: str, mesh_size: float = 2.0):
    """
    Helper: Converts a GLB surface mesh into a volumetric tetrahedral mesh (skfem).
    Robust version 2: Skips CAD parametrization to handle dirty meshes.
    """
    # 1. Load GLB
    mesh_surf = trimesh.load(glb_path, force='mesh', process=True)
    
    # --- AGGRESSIVE CLEANUP ---
    # a. Merge vertices to close cracks
    mesh_surf.merge_vertices(merge_tex=True, merge_norm=True)
    
    # b. Keep only the largest connected component
    # This removes floating "dust" or disconnected parts which cause topology errors
    if not mesh_surf.is_watertight:
        # Split into components
        components = mesh_surf.split(only_watertight=False)
        if len(components) > 0:
            # Find the one with the most vertices (main body)
            largest_component = max(components, key=lambda m: len(m.vertices))
            mesh_surf = largest_component

    # c. Final watertight check and fill
    if not mesh_surf.is_watertight:
        trimesh.repair.fill_holes(mesh_surf)

    # 2. Export to STL
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tf:
        mesh_surf.export(tf.name)
        stl_path = tf.name

    msh_path = stl_path.replace(".stl", ".msh")

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.clear()
        
        # 3. Merge STL
        gmsh.merge(stl_path)
        
        # 4. Cleanup/Classify
        gmsh.model.mesh.removeDuplicateNodes()
        
        # Key Fix: forReparametrization=False
        # We do NOT want to calculate a CAD parametrization (NURBS) because it fails on dirty meshes.
        # We just want to identify the boundary surfaces.
        gmsh.model.mesh.classifySurfaces(40 * math.pi / 180, True, False)
        
        # 5. Create Volume from Discrete Surfaces
        # We skip createGeometry() and go straight to defining the volume.
        surfaces = gmsh.model.getEntities(2)
        surface_tags = [s[1] for s in surfaces]
        
        if not surface_tags:
             raise ValueError("No surfaces identified. The mesh might be empty or invalid.")

        # Add a Surface Loop (Shell)
        l = gmsh.model.geo.addSurfaceLoop(surface_tags)
        # Add a Volume
        gmsh.model.geo.addVolume([l])
        
        gmsh.model.geo.synchronize()
        
        # 6. Mesh Settings
        gmsh.option.setNumber("Mesh.Algorithm3D", 1) # Delaunay
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.1)
        
        # 7. Generate 3D Mesh
        gmsh.model.mesh.generate(3)
        
        # 8. Write MSH
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(msh_path)
        
        gmsh.finalize()

        # 9. Load into skfem
        if not os.path.exists(msh_path):
             raise RuntimeError("Gmsh failed to generate a .msh file.")
             
        mesh_vol = skfem.Mesh.load(msh_path)
        
        # Validate
        if mesh_vol.p.shape[0] != 3:
            raise ValueError(f"Generated mesh is not 3D (Dim={mesh_vol.p.shape[0]}). Input might not be a closed volume.")

    finally:
        # Cleanup
        if os.path.exists(stl_path): os.remove(stl_path)
        if os.path.exists(msh_path): os.remove(msh_path)

    # 10. Extract Boundary Nodes
    b_facets = mesh_vol.boundary_facets()
    boundary_nodes = np.unique(mesh_vol.facets[:, b_facets])
    
    return mesh_vol, boundary_nodes


def run_physics_analysis(glb_file_path: str):
    """
    Main Service Function: 
    1. Meshes the file
    2. Runs Pressure Analysis
    3. Runs Thermal Analysis
    4. Returns formatted results dict
    """
    
    # --- 1. Generate Mesh ---
    # mesh_size is a heuristic. 
    # NOTE: If your model is in METERS, 1.0 is huge. If in MM, 1.0 is fine.
    # Adjust this if meshing fails or is too coarse.
    mesh_vol, surface_indices = create_volume_mesh_from_glb(glb_file_path, mesh_size=2.0)

    # --- 2. Prepare Geometry for Frontend ---
    b_facets_idx = mesh_vol.boundary_facets()
    boundary_faces = mesh_vol.facets[:, b_facets_idx].T # Shape (N_faces, 3)
    surface_nodes = mesh_vol.p.T # Shape (N_nodes, 3)

    # --- 3. Define Boundaries ---
    # Auto-calculate bounding box
    z_coords = mesh_vol.p[2] # This line previously failed
    zmin, zmax = np.min(z_coords), np.max(z_coords)
    tol = (zmax - zmin) * 0.1
    
    # Boundary Conditions
    fixed_bc = lambda x: x[2] < zmin + tol
    press_bc = lambda x: x[2] > zmax - tol
    
    def thermal_bc(x):
        if x[2] > zmax - tol: return 600.0 # Hot Top
        if x[2] < zmin + tol: return 300.0 # Cool Bottom
        return None

    # Load Material
    material = fem.get_AlSi10Mg()

    # --- 4. Run Pressure Simulation ---
    config_p = {
        'type': 'pressure', 
        'pressure': 5e6, # 5 MPa
        'fixed_boundary_fn': fixed_bc, 
        'pressure_boundary_fn': press_bc
    }
    _, u_p = fem.run_simulation(mesh_vol, material, config_p)
    
    # Reshape u_p (flat) to (3, N)
    u_p_vec = np.stack([u_p[0::3], u_p[1::3], u_p[2::3]])
    max_disp_p = float(np.max(np.linalg.norm(u_p_vec, axis=0)))

    # --- 5. Run Thermal Simulation ---
    config_t = {
        'type': 'thermo-mechanical', 
        'T_ref': 300.0,
        'fixed_boundary_fn': fixed_bc, 
        'temp_boundary_fn': thermal_bc
    }
    _, u_t = fem.run_simulation(mesh_vol, material, config_t)
    
    u_t_vec = np.stack([u_t[0::3], u_t[1::3], u_t[2::3]])
    max_disp_t = float(np.max(np.linalg.norm(u_t_vec, axis=0)))

    # --- 6. Return Structured Data ---
    return {
        "geometry": {
            "vertices": surface_nodes.tolist(),
            "faces": boundary_faces.tolist()
        },
        "pressure": {
            "displacement": u_p_vec.T.tolist(), 
            "max_disp": max_disp_p
        },
        "thermal": {
            "displacement": u_t_vec.T.tolist(),
            "max_disp": max_disp_t
        }
    }