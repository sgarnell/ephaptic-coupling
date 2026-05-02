# Phase 2 — 3D Visualization (Static Viewer)

This document defines the scope, deliverables, and test plan for Phase 2 of the Ephaptic Coupling Visualization project.  
Phase 2 introduces a static 3D viewer using PyVista, built on top of the validated Phase 1 data-loading foundation.

---

# 1. Visualization Package

Phase 2 must use the following packages:

- **pyvista** — core 3D geometry and rendering
- **pyvistaqt** — Qt-based interactive viewer backend
- **vtk** — underlying rendering engine (automatically installed with PyVista)

No other visualization libraries (Plotly, Mayavi, Manim, Matplotlib, etc.) may be used in this phase.

---

# 2. Purpose of Phase 2

Phase 2 focuses exclusively on:

- Building a static 3D viewer for FB layers
- Rendering point clouds for each layer
- Applying per-layer colors
- Providing camera presets
- Allowing layer toggling (show/hide)
- Ensuring the viewer is modular and reusable

No animation or time-varying behavior is included in this phase.

---

# 3. Modules Implemented in Phase 2

Phase 2 implements the following modules:

```
visualization/
    viewer.py
    colors.py   (updated if needed)
```

The viewer must satisfy the requirements defined in:
- `requirements/visualization.md`
- `requirements/colors.md`

---

# 4. Required Functionality

## 4.1 Layer Mesh Construction

Implement:

```python
build_layer_mesh(points: np.ndarray, color: tuple) -> pv.PolyData
```

Requirements:

- Represent each synapse as a point or small sphere
- Use PyVista primitives (`PolyData`, `glyphs`, or `points`)
- Apply the layer’s color from `colors.py`
- Must not modify the underlying data

## 4.2 Scene Construction

Implement:

```python
build_scene(layers: list[int]) -> pv.Plotter
```

Requirements:

- Create a PyVista `Plotter`
- Add each requested layer mesh
- Apply consistent scaling and camera defaults
- Do not show the scene yet (return the plotter)

## 4.3 Viewer Entry Point

Implement:

```python
show_scene(layers: list[int] = [1,2,3,4,5,6,7,8,9])
```

Requirements:

- Build the scene
- Display it interactively
- Allow toggling layers via:
  - key bindings, or
  - function arguments

## 4.4 Camera Presets

Implement:

```python
set_camera_top(plotter)
set_camera_side(plotter)
set_camera_oblique(plotter)
```

Requirements:

- Provide three stable, reproducible camera views
- Use PyVista camera controls
- Do not animate the camera

---

# 5. Phase 2 Tests

Create a new test file:

```
tests/phase2_test.py
```

Tests must include:

## 5.1 Mesh Construction Test

- Ensure `build_layer_mesh` returns a `pv.PolyData`
- Ensure the number of points matches the input
- Ensure the color array is applied correctly

## 5.2 Scene Construction Test

- Ensure `build_scene` returns a `pv.Plotter`
- Ensure the correct number of actors is added
- Ensure no exceptions occur when adding multiple layers

## 5.3 Camera Preset Test

- Ensure each camera preset function modifies the plotter’s camera
- Ensure no exceptions occur

## 5.4 Smoke Test for show_scene

- Call `show_scene(layers=[1])` inside a non-interactive test mode
- Ensure no exceptions occur
- Do not open a window during tests (use PyVista off-screen mode)

---

# 6. Out of Scope for Phase 2

Phase 2 explicitly excludes:

- Animation
- Time-varying color changes
- Wave simulation
- MP4 export
- Camera animation
- Density grids
- Interactive UI widgets

These belong to Phase 3.

---

# 7. Success Criteria

Phase 2 is complete when:

- All viewer functions are implemented
- All tests in `phase2_test.py` pass
- The viewer can display any subset of FB layers
- Camera presets work reliably
- No animation code exists in this phase

Only after Phase 2 passes cleanly should development proceed to Phase 3 (animation engine).

---

# End of Phase 2
