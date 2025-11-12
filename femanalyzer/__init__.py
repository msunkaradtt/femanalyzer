# Export the main, user-facing functions
from .api import run_simulation
from .roughness import analyze_surface_roughness
from .meshing import create_unit_cube_mesh, create_mesh_from_stp
from .materials import get_AlSi10Mg
from .visualization import plot_surface_roughness, plot_fem_result, plot_mesh
from .animation import solve_time_series, create_video_from_file

__version__ = "0.1.0"