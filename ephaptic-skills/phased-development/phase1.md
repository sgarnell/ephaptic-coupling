# Phase 1 — Data Loading, Extraction, and Sanity Validation

This document defines the scope, deliverables, and test plan for Phase 1 of the Ephaptic Coupling visualization project.  
Phase 1 establishes the foundation for all later visualization and animation work.

---

# 1. Purpose of Phase 1

Phase 1 focuses exclusively on:

- Loading `.npz` synapse datasets
- Validating the schema
- Extracting core fields (points, bounds, centroid, presynaptic names)
- Implementing basic utility functions
- Running sanity checks to ensure the dataset is correct and complete

No visualization or animation code is included in this phase.

---

# 2. Modules Implemented in Phase 1

Phase 1 implements the following modules:

```
visualization/
    utils.py
tests/
    phase1_test.py
```

The `utils.py` module must satisfy the requirements defined in `requirements/utils.md`.

---

# 3. Required Functionality

## 3.1 Data Loading

Implement:

```python
load_layer_npz(layer_number: int) -> dict
load_all_layers() -> dict[int, dict]
```

These functions must:

- Load `.npz` files from `FB_Synapse_Dataset/`
- Use `numpy.load(..., allow_pickle=True)`
- Return clean Python/numpy structures
- Raise clear errors if files are missing or corrupted

## 3.2 Extraction Helpers

Implement:

```python
get_layer_points(layer_number: int) -> np.ndarray
get_layer_bounds(layer_number: int) -> tuple
get_centroid(layer_number: int) -> np.ndarray
get_presynaptic_names(layer_number: int) -> list[str]
```

These functions must:

- Extract fields from the loaded `.npz` dict
- Validate that required keys exist
- Return well‑typed values

## 3.3 Schema Validation

Implement a function:

```python
validate_schema(layer_data: dict) -> bool
```

This function must check for:

- Required keys: points, pre_names, bounds, centroid
- Correct shapes and dtypes
- Optional keys handled gracefully

---

# 4. Phase 1 Sanity Tests

A `tests/phase1_test.py` file must be created with the following tests:

## 4.1 Load Test

- Load all 9 layers
- Assert that each layer loads without error
- Assert that each layer contains required keys

## 4.2 Shape Test

For each layer:

- `points` must be N×3
- `bounds` must be length 6
- `centroid` must be length 3
- `pre_names` must match length of `points`

## 4.3 Value Test

For each layer:

- Bounds must contain finite numbers
- Centroid must lie within bounds
- Points must lie within bounds

## 4.4 Global Summary Test

Print:

- Total synapses per layer
- Unique presynaptic neuron count
- Bounds and centroid

This ensures the dataset is internally consistent.

---

# 5. Out of Scope for Phase 1

Phase 1 explicitly excludes:

- PyVista visualization
- Layer color mapping
- Animation or wave simulation
- MP4 export
- UI or interactivity
- Scene building

These belong to Phase 2 and Phase 3.

---

# 6. Success Criteria

Phase 1 is complete when:

- All utility functions are implemented
- All schema validation functions pass
- `phase1_test.py` runs without errors
- The dataset is confirmed to be structurally correct
- No missing or malformed fields remain

Only after Phase 1 passes cleanly should development proceed to Phase 2 (3D visualization).

---

# End of Phase 1