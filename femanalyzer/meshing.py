import numpy as np
import pygmsh
import pygmsh.occ
import skfem as sk
from skfem.io.meshio import from_meshio
import gmsh

def create_unit_cube_mesh(mesh_size: float = 0.2):
    """
    Build a simple unit cube tetrahedral mesh [0,1]^3.
    """
    with pygmsh.geo.Geometry() as geom:
        geom.add_box(0, 1, 0, 1, 0, 1, mesh_size=mesh_size)
        mesh = geom.generate_mesh(dim=3)

    sk_mesh = from_meshio(mesh)
    
    # Find z_min and z_max for boundary conditions
    z = sk_mesh.p[2, :]
    z_min, z_max = float(np.min(z)), float(np.max(z))
    
    print(f"Mesh generated: {len(sk_mesh.p.T)} nodes, {len(sk_mesh.t.T)} elements")
    return sk_mesh, z_min, z_max

def create_mesh_from_stp(filepath: str, mesh_size: float = 0.2):
    """
    Imports a STEP (.stp) file using the pygmsh OCC kernel
    and generates a 3D tetrahedral mesh.
    
    Returns:
        skfem.Mesh: The scikit-fem mesh object.
        dict: A dictionary containing the mesh bounding box.
    """
    print(f"Importing STEP file via pygmsh.occ: {filepath}...")
    
    with pygmsh.occ.geometry.Geometry() as geom:
        
        try:
            entities = geom.import_shapes(filepath)
        except Exception as e:
            print(f"pygmsh.occ.import_shapes() failed.")
            print(f"Details: {e}")
            raise

        gmsh.model.occ.synchronize()

        volumes = [e for e in entities if e.dim == 3]
        
        if not volumes:
            print("Warning: No 3D solid volumes found in STEP file.")
            surfaces = [e for e in entities if e.dim == 2]
            if surfaces:
                print("Found 2D surfaces. Attempting to create volume from surface loop...")
                surface_loop = geom.add_surface_loop(surfaces)
                volume = geom.add_volume(surface_loop)
                print(f"Created new volume (tag={volume.dim_tag}) from surfaces.")
            else:
                raise ValueError("No 3D volumes or 2D surfaces found in STEP file.")
        else:
            print(f"Found {len(volumes)} 3D solid volume(s) in STEP file.")

        
        # *** ADD THESE TWO FIXES ***
        print("Attempting to heal geometry...")
        gmsh.model.occ.healShapes() # <--- 1. Tell gmsh to fix intersections/gaps
        
        print("Switching 3D mesh algorithm to Delaunay (robust)")
        gmsh.option.setNumber("Mesh.Algorithm3D", 7) # <--- 2. Use Delaunay
        # *** END OF FIXES ***

        print(f"Setting global mesh size to {mesh_size}...")
        geom.characteristic_length_min = mesh_size
        geom.characteristic_length_max = mesh_size

        print(f"Generating 3D tetrahedral mesh (size={mesh_size})...")
        mesh_data = geom.generate_mesh(dim=3)

    print("pygmsh volume meshing successful.")

    sk_mesh = from_meshio(mesh_data)
    
    p = sk_mesh.p
    bounding_box = {
        'xmin': np.min(p[0, :]), 'xmax': np.max(p[0, :]),
        'ymin': np.min(p[1, :]), 'ymax': np.max(p[1, :]),
        'zmin': np.min(p[2, :]), 'zmax': np.max(p[2, :]),
    }
    
    print(f"Mesh generated: {len(sk_mesh.p.T)} nodes, {len(sk_mesh.t.T)} elements")
    print(f"Bounding Box: {bounding_box}")
    
    return sk_mesh, bounding_box