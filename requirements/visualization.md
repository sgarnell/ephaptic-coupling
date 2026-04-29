# Visualization Requirements

This document defines the technical requirements for the 3D visualization subsystem of the Ephaptic Coupling project. It supplements the high‑level goals in `skills/SKILL.md` and provides module‑level guidance for VS Code AI.

## 1. Module Structure

The visualization system must be implemented inside:

```
visualization/
    viewer.py
    utils.py
    colors.py
```

Each file must contain focused, reusable functions. No monolithic scripts.

## 2. Data Loading (`utils.py`)

Implement helper functions for loading `.npz` files from `FB_Synapse_Dataset/`.

Required functions:

```python
load_layer_npz(layer_number: int) -> dict
load_all_layers() -> dict[int, dict]
get_layer_points(layer_number: int) -> np.ndarray
get_layer_bounds(layer_number: int) -> tuple
```

Constraints:

- Must use `numpy.load(..., allow_pickle=True)`
- Must return clean Python/numpy structures
- Must not hardcode file paths; use relative paths

## 3. Color Utilities (`colors.py`)

Provide a fixed color palette for FB1–FB9.

Requirements:

- Define a dictionary mapping layer index → RGB tuple
- Provide a function:

```python
get_layer_color(layer_number: int) -> tuple
```

- Provide a colormap function for oscillation amplitude:

```python
amplitude_to_color(amplitude: float) -> tuple
```

Colors must be visually distinct and consistent across all modules.

## 4. 3D Viewer (`viewer.py`)

The viewer must use PyVista (`pyvista` + `pyvistaqt`) to create an interactive 3D scene.

Required functions:

```python
build_layer_mesh(points: np.ndarray, color: tuple) -> pv.PolyData
build_scene(layers: list[int]) -> pv.Plotter
show_scene(layers: list[int] = [1,2,3,4,5,6,7,8,9])
```

Viewer requirements:

- Each layer must be rendered in its unique color
- Layers must be toggleable (via key bindings or function arguments)
- The scene must support:
  - rotation
  - zoom
  - camera presets (top, side, oblique)
- Use point clouds or small spheres (no heavy meshing required)

## 5. Performance Constraints

- Avoid rendering millions of points at once; downsample if needed
- Use PyVista’s `Plotter` or `PlotterQt`
- Keep memory usage reasonable

## 6. Example Script

Provide an example script in `examples/viewer_example.py`:

- Load FB1
- Render it in 3D
- Allow rotation and zoom

## 7. Non‑Goals

The visualization module must NOT:

- Perform animation
- Modify synapse data
- Compute wave fields
- Export MP4 files

These belong to the animation subsystem.
