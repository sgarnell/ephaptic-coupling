# Utils Module Requirements

This document defines the requirements for the `visualization/utils.py` module.  
It provides data-loading utilities and helper functions used by both the viewer and animation subsystems.

---

## 1. Purpose

The utils module must provide a clean, stable API for loading and accessing synapse data stored in `.npz` files inside `FB_Synapse_Dataset/`.

It must not perform visualization, animation, or color mapping.

---

## 2. Required Functions

### `load_layer_npz(layer_number: int) -> dict`
- Loads a single `.npz` file for a given FB layer.
- Must construct the filename dynamically (e.g., `FB{layer}_incoming_synapses.npz`).
- Must return a dictionary containing all arrays in the file.

### `load_all_layers() -> dict[int, dict]`
- Loads all layer files FB1–FB9.
- Returns a dictionary mapping layer index → layer data dict.

### `get_layer_points(layer_number: int) -> np.ndarray`
- Returns the `points` array for the given layer.

### `get_layer_bounds(layer_number: int) -> tuple`
- Returns the `bounds` array as a tuple.

---

## 3. Constraints

- Must use `numpy.load(..., allow_pickle=True)`.
- Must not hardcode absolute paths.
- Must raise clear errors if files are missing.
- Must not modify the data.

---

## 4. Optional Helper Functions

These may be implemented if useful:

```python
get_centroid(layer_number: int) -> np.ndarray
get_presynaptic_names(layer_number: int) -> list[str]
get_density_grid(layer_number: int) -> np.ndarray
```

---

## 5. Non‑Goals

The utils module must NOT:

- Render any 3D geometry
- Apply colors or colormaps
- Perform animation
- Export videos
- Modify `.npz` contents

It is strictly a data-access layer.
