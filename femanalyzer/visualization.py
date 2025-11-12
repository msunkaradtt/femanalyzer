import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import skfem as sk

def plot_surface_roughness(points: np.ndarray, title: str = "Surface Geometry"):
    """
    Creates a 3D scatter plot of surface points (N, 3) using matplotlib.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Use 'viridis' colormap and color points by their Z-height
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=10, alpha=0.6)
    
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, label="Z Height (m)")
    plt.tight_layout()
    # We don't call plt.show() here; the user script will call it.
    print(f"Generated plot: '{title}'")


def plot_fem_result(
    mesh: sk.Mesh, 
    displacement: np.ndarray, 
    scalar_field: np.ndarray = None, 
    field_name: str = "Displacement (Z)"
):
    """
    Visualizes the 3D FEM result using PyVista.
    
    Args:
        mesh: The *original* (undeformed) skfem mesh.
        displacement: The (3, N) displacement vector at the P1 nodes.
        scalar_field: (Optional) A (N,) array of scalars to plot (e.g., Z-displacement).
        field_name: The label for the scalar field.
    """
    print("Launching PyVista 3D visualization...")
    
    # 1. Convert skfem mesh to a PyVista UnstructuredGrid
    #    PyVista expects (N_points, 3) and (N_cells, 4) for tet connectivity
    points = mesh.p.T
    
    # PyVista cell format: {<cell_type>: <connectivity_array>}
    cells_dict = {pv.CellType.TETRA: mesh.t.T}
    grid = pv.UnstructuredGrid(cells_dict, points)

    # 2. Add displacement vectors to the grid
    #    PyVista expects (N_points, 3) vector format, so we transpose.
    grid["Displacement"] = displacement.T

    # 3. Add the scalar field to plot (e.g., Z-displacement)
    if scalar_field is not None:
        grid[field_name] = scalar_field
        scalar_bar_title = field_name
    else:
        # If no scalar is given, plot the displacement magnitude
        grid[field_name] = np.linalg.norm(displacement.T, axis=1)
        scalar_bar_title = "Displacement Magnitude (m)"

    # 4. Create a plotter
    plotter = pv.Plotter(window_size=[800, 600])
    
    # 5. Add the DEFORMED mesh (warped by vectors)
    #    'cmap' is the colormap, 'scalars' points to our field
    plotter.add_mesh(
        grid.warp_by_vector("Displacement"), 
        cmap='viridis',
        scalars=field_name,
        scalar_bar_args={'title': scalar_bar_title}
    )

    # 6. (Optional) Add the original mesh as a black wireframe for comparison
    plotter.add_mesh(grid, style='wireframe', color='black', opacity=0.2)
    
    plotter.add_axes()
    plotter.view_isometric()
    
    # 7. Show the interactive window (this blocks the script)
    plotter.show()

def plot_mesh(mesh: sk.Mesh, title: str = "Mesh Visualization"):
    """
    Renders the 3D skfem.Mesh object in PyVista.
    """
    print(f"Plotting initial mesh: '{title}'...")

    # 1. Convert skfem mesh to a PyVista UnstructuredGrid
    points = mesh.p.T
    cells_dict = {pv.CellType.TETRA: mesh.t.T}
    grid = pv.UnstructuredGrid(cells_dict, points)

    # 2. Create a plotter
    plotter = pv.Plotter(window_size=[800, 600])
    
    # 3. Add the mesh as a wireframe
    plotter.add_mesh(
        grid, 
        style='wireframe', 
        color='black', 
        opacity=0.3,
        label="Full Mesh"
    )
    
    plotter.add_axes()
    plotter.view_isometric()
    plotter.add_text(title, font_size=15)
    
    # 4. Show the interactive window
    plotter.show()