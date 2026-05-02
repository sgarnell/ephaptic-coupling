---
name: "Ephaptic Coupling Visualization"
description: "Tools for building 3D visualizations and animations of Fan-Shaped Body synapse layers."
version: 1.0
---

# Goal

Provide a modular codebase for visualizing and animating the Drosophila Fan‑Shaped Body (FB) using the precomputed `.npz` synapse datasets. The system must support:

1. **3D interactive visualization** of all FB layers (FB1–FB9), each with a unique color.
2. **Layer toggling**, camera control, and clean scene composition.
3. **Animation of ephaptic oscillatory synchronization**, including:
   - single‑layer oscillations  
   - multi‑layer phase relationships  
   - traveling and standing waves  
4. **Optional MP4 export** of animations for presentations.

This project uses **PyVista** as the primary 3D engine.

---

# Requirements

The agent should generate code that:

- Uses **PyVista** (`pyvista` + `pyvistaqt`) for 3D rendering.
- Loads synapse data from `.npz` files in `FB_Synapse_Dataset/`.
- Follows a **modular architecture**:
  - `visualization/viewer.py` — 3D scene construction
  - `visualization/animation.py` — wave simulation + animation
  - `visualization/utils.py` — data loading helpers
  - `visualization/colors.py` — color palettes and colormaps
- Avoids monolithic scripts; each module should contain focused, reusable functions.
- Does **not** rewrite existing modules; new functionality should be added as new functions.
- Produces readable, maintainable code with clear function boundaries.

---

# High-Level Architecture

## 1. Data Utilities (`utils.py`)
Functions for:
- loading layer `.npz` files  
- extracting point clouds  
- retrieving bounds, centroids, and metadata  

## 2. Color Utilities (`colors.py`)
- fixed color palette for FB1–FB9  
- colormap for oscillation amplitude  
- optional presynaptic-type color mapping  

## 3. 3D Viewer (`viewer.py`)
- build meshes or point clouds for each layer  
- assign unique colors  
- assemble a full FB scene  
- provide interactive rotation, zoom, and layer toggling  

## 4. Animation Engine (`animation.py`)
- generate oscillatory wave fields  
- update point colors or intensities over time  
- animate one or more layers  
- optional MP4 export using `imageio` + `imageio-ffmpeg`  

---

# Constraints

- Code must run inside the `ephaptic_env` Python environment.
- Use PyVista’s Qt-based interactive window (`Plotter` or `PlotterQt`).
- Avoid Jupyter-specific tools (e.g., `ipyvtklink`).
- Keep functions small, composable, and testable.
- Maintain backward compatibility with existing dataset structure.

---

# Deliverables

The agent should be able to generate:

- A complete `visualization/` folder scaffold.
- Starter implementations for each module.
- Example scripts demonstrating:
  - loading a layer  
  - rendering a 3D scene  
  - running a simple oscillation animation  

The agent should not generate large monolithic files; instead, it should follow the modular architecture described above.

---

# End of SKILL.md
