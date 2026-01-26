Here is the completely rewritten `README.md` reflecting the new capabilities (Web Service, GLB support, and Animation tools).

```markdown
# `femanalyzer`

A Python package for analyzing the effect of Finite Element (FEA) simulations on 3D surface roughness.

## Overview

`femanalyzer` provides a complete, end-to-end workflow for computational analysis. It allows you to import 3D CAD or mesh files, generate simulation-ready volume meshes, measure initial surface roughness, apply physical loads (pressure or thermal), and measure the roughness on the deformed part to quantify the change.

This tool is designed to bridge the gap between CAD, FEA, and 3D surface metrology. It can be used as a standalone Python library for research or as a **REST API service** for web integrations.

## Core Features

### Physics & Analysis
* **Multi-Format Import:**
    * Load precision CAD geometries from **.stp** (STEP) files.
    * **NEW:** Load web-standard **.glb** (GLTF) files via the service layer.
* **Robust Meshing:**
    * Automatically generate 3D tetrahedral (volume) meshes suitable for FEA.
    * Includes robust "repair-and-mesh" pipelines for dirty GLB meshes (merging vertices, filling holes).
* **FEA Solvers:**
    * **Mechanical:** Solves 3D linear elasticity problems (e.g., pressure loads).
    * **Thermo-mechanical:** Solves coupled steady-state thermal and structural problems to find deformation from thermal stress.
* **Roughness Analysis:**
    * Implements ISO 25178 standards for areal (3D) surface roughness (Sa, Sq, Sz).
    * Includes `plane` detrending (form removal) via PCA for accurate results on non-flat surfaces.

### Visualization & Animation
* **Interactive 3D:** Visualize meshes and displacement fields using **PyVista**.
* **Plots:** Generate 3D scatter plots of rough surfaces using **Matplotlib**.
* **NEW: Animation:**
    * Run quasi-static time-series simulations (ramped loads).
    * Render results into **.mp4** videos to visualize deformation over time.

---

## Installation

### 1. Install Python Dependencies
Clone the repository and install the requirements:

```bash
pip install -r requirements.txt

```

---

## Usage Guide

You can use `femanalyzer` in three modes: as a **Local Library** (scripts), as a **Web Service**, or for **Animation**.

### Mode A: Local Library (STEP Files)

This workflow is best for high-precision analysis using CAD files.

1. **Place your STEP file** (e.g., `sample_1.stp`) inside the `examples/` folder.
2. **Run the analysis script:**
```bash
python examples/run_stp_analysis.py

```


This script will:
1. Mesh the STEP file.
2. Calculate "Before" roughness.
3. Run Pressure and Thermal simulations.
4. Calculate "After" roughness and print the comparison.
5. Display interactive 3D plots.



### Mode B: Web Service (GLB Files)

The package now includes a **FastAPI** application (`main.py`) that acts as a backend for web frontends. It accepts `.glb` files, meshes them on the fly, and returns JSON deformation data.

1. **Start the Server:**
```bash
python main.py

```


The server will start on `http://0.0.0.0:8001`.
2. **API Endpoint:** `POST /analyze_physics`
* **Input:** A `.glb` file upload.
* **Process:**
* Cleans and repairs the mesh (filling holes, merging vertices).
* Generates a volume mesh using `gmsh`.
* Runs both Pressure (5MPa) and Thermal (300K-600K) simulations.


* **Output:** A JSON object containing the geometry (vertices/faces) and displacement vectors for both simulations.



### Mode C: Animation (Time Series)

You can run ramped load simulations and generate videos using `femanalyzer.animation`.

```python
import femanalyzer as fem
from femanalyzer import animation

# 1. Setup
mesh, _ = fem.create_mesh_from_stp("examples/sample_3.step")
material = fem.get_AlSi10Mg()

# 2. Define Config
config = {
    'type': 'pressure',
    'pressure': 5e6, # Max pressure
    'fixed_boundary_fn': lambda x: x[2] < 0.01,
    'pressure_boundary_fn': lambda x: x[2] > 0.99
}

# 3. Solve Time Series (saves to .npz)
animation.solve_time_series(
    mesh, material, config, 
    num_steps=30, 
    output_file="results.npz"
)

# 4. Render Video
animation.create_video_from_file("results.npz", "simulation.mp4")

```

---

## Project Structure

```
femanalyzer_project/
├── femanalyzer/
│   ├── __init__.py
│   ├── api.py           # Simulation entry points
│   ├── animation.py     # Time-series and Video tools
│   ├── service.py       # Backend logic for GLB processing
│   ├── solvers/         # Mechanical & Thermal solver logic
│   └── ... 
├── examples/
│   └── run_stp_analysis.py
├── main.py              # FastAPI Web Server
├── setup.py
└── requirements.txt

```

Happy coding!