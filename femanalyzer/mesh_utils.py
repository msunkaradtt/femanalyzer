import trimesh
import skfem
import numpy as np
import tempfile
import os
import pygmsh

def create_volume_mesh_from_glb(glb_path: str, mesh_size: float = 2.0):
    """
    Converts a GLB surface mesh into a volumetric tetrahedral mesh (skfem).
    Returns: (skfem.MeshTet, dict_of_boundary_nodes)
    """
    # 1. Load GLB as Surface Mesh
    mesh_surf = trimesh.load(glb_path, force='mesh')
    
    # Ensure it's watertight for volume meshing
    if not mesh_surf.is_watertight:
        mesh_surf.fill_holes()

    # 2. Export to STL for Gmsh
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tf:
        mesh_surf.export(tf.name)
        stl_path = tf.name

    # 3. Use pygmsh to generate Volume Mesh
    try:
        with pygmsh.geo.Geometry() as geom:
            # Import the STL surface
            poly = geom.add_polygon(
                # accessing vertices via trimesh might be complex in pygmsh raw
                # simpler to use geom.add_volume with the STL file if supported,
                # but standard pygmsh often wraps .geo scripts.
                # FALLBACK: We use meshio/gmsh directly via skfem if possible, 
                # but here is a standard Gmsh approach:
                [] 
            )
            # NOTE: Robust STL->Volume in pure code can be tricky. 
            # We will use the standard 'merge' command in Gmsh via pygmsh's raw interface
            # or simply rely on Gmsh's CLI if available. 
            pass
        
        # Simpler approach: Use skfem's interface if available, or subprocess gmsh.
        # For this example, we assume `gmsh` is in PATH as per README.
        
        msh_path = stl_path.replace(".stl", ".msh")
        # Run Gmsh to mesh the STL volume
        # -3: 3D mesh, -format msh2: compatible with older meshio/skfem
        cmd = f"gmsh -3 -format msh2 -clscale {mesh_size} \"{stl_path}\" -o \"{msh_path}\""
        os.system(cmd)
        
        if not os.path.exists(msh_path):
             raise RuntimeError("Gmsh failed to generate a mesh.")

        # 4. Load into skfem
        mesh_vol = skfem.Mesh.load(msh_path)
        
    finally:
        if os.path.exists(stl_path): os.remove(stl_path)
        if 'msh_path' in locals() and os.path.exists(msh_path): os.remove(msh_path)

    # 5. Extract Boundary for Visualization
    # We need to identify which nodes are on the surface to send back to frontend
    # boundary_facets is (3, N_facets)
    boundary_facets = mesh_vol.boundary_facets()
    
    # Get unique node indices on the boundary
    boundary_nodes = np.unique(mesh_vol.facets[:, boundary_facets])
    
    return mesh_vol, boundary_nodes