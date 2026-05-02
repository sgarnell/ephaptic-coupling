# Synapse Dataset Schema Requirements

This document defines the schema for all `.npz` files in `FB_Synapse_Dataset/`.  
VS Code AI must use this schema when writing visualization or animation code.

---

## 1. Overview

Each `.npz` file contains synapse data for either:

- A single FB layer (FB1–FB9), or
- The global dataset (`FB_incoming_all_synapses.npz`)

All files share the same schema.

---

## 2. Schema Definition

Each `.npz` file contains the following keys:

### `points` : `np.ndarray` (shape: N × 3)
- 3D coordinates of synapses.
- Format: `[x, y, z]` in hemibrain space.

### `pre_names` : `np.ndarray` (shape: N)
- Presynaptic neuron name for each synapse.
- Strings, stored as dtype=object.

### `layer` : `int`
- Layer index (1–9).
- Global file uses `-1`.

### `bounds` : `np.ndarray` (shape: 6)
- `[xmin, xmax, ymin, ymax, zmin, zmax]`.

### `centroid` : `np.ndarray` (shape: 3)
- Mean coordinate of all synapses.

### `neuron_names` : `np.ndarray`
- Unique presynaptic neuron names.

### `counts_per_neuron` : `np.ndarray`
- Synapse count per presynaptic neuron.
- Same order as `neuron_names`.

### `density_grid` : `np.ndarray` (optional)
- 3D voxel grid representing synapse density.

### `density_meta` : `dict` (optional)
- Metadata for interpreting the density grid:
  - voxel size
  - grid shape
  - coordinate transform

---

## 3. Constraints

- All arrays must be loaded with `allow_pickle=True`.
- Missing optional fields must be handled gracefully.
- The schema must be treated as stable and read‑only.

---

## 4. Example Access Pattern

```python
data = np.load("FB5_incoming_synapses.npz", allow_pickle=True)

points = data["points"]
pre = data["pre_names"]
centroid = data["centroid"]
bounds = data["bounds"]
```


# 5. Example Schema (JSON‑like)

The following example illustrates the structure of a typical layer file:

```json
{
  "points": [
    [193.19, 187.63, 141.22],
    [192.88, 188.02, 140.95],
    ...
  ],

  "pre_names": [
    "FB1D_R_2",
    "FB1D_R_2",
    "FB1D_L_1",
    ...
  ],

  "layer": 1,

  "bounds": [
    71.28, 274.95,
    124.77, 287.13,
    23.91, 265.24
  ],

  "centroid": [193.19, 187.63, 141.22],

  "neuron_names": [
    "FB1D_R_2",
    "FB1D_R_1",
    "FB1D_L_1",
    ...
  ],

  "counts_per_neuron": [
    387,
    366,
    365,
    ...
  ],

  "density_grid": null,

  "density_meta": null
}
```

# 6. Example Access Pattern

data = np.load("FB5_incoming_synapses.npz", allow_pickle=True)

points = data["points"]
pre = data["pre_names"]
centroid = data["centroid"]
bounds = data["bounds"]


## 7. Non‑Goals

This document does NOT define:

- How to visualize the data
- How to animate the data
- How to compute new fields

It only defines the structure of the existing dataset.