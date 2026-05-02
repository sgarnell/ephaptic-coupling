
# Color Module Requirements

This document defines the requirements for the `visualization/colors.py` module.  
It provides color palettes and colormap utilities used by both the viewer and animation subsystems.

---

## 1. Purpose

The colors module must provide:

- A fixed color palette for FB1–FB9
- A function to retrieve a layer’s color
- A function to map oscillation amplitude → RGB color

It must not load data or build geometry.

---

## 2. Layer Color Palette

Define a dictionary:

```python
LAYER_COLORS = {
    1: (R, G, B),
    2: (R, G, B),
    ...
    9: (R, G, B)
}
```

Requirements:

- Colors must be visually distinct.
- Colors must be stable across all modules.
- RGB values must be floats in [0, 1].

---

## 3. Required Functions

### `get_layer_color(layer_number: int) -> tuple`
- Returns the RGB tuple for the given layer.
- Must raise a clear error for invalid layer numbers.

### `amplitude_to_color(amplitude: float) -> tuple`
- Maps a scalar amplitude (0–1) to a color.
- Must support smooth gradients.
- Recommended: use a simple blue→white→red or viridis-like gradient.

---

## 4. Optional Functions

```python
presynaptic_type_to_color(name: str) -> tuple
phase_to_color(phase: float) -> tuple
```

These are optional but may be useful later.

---

## 5. Non‑Goals

The colors module must NOT:

- Load `.npz` files
- Build PyVista meshes
- Perform animation
- Export videos

It is strictly a color utility layer.