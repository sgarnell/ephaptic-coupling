# Phase 4 – Spatiotemporal Ephaptic Propagation

## 1. Purpose

Phase 4 extends the Phase 3 animation engine from a **global temporal pulse** to a **spatiotemporal ephaptic propagation model**. The goal is to visualize how activity patterns travel through FB synaptic layers as waves, rather than having all points in a layer brighten and dim in lockstep.

Phase 4 does **not** change the viewer API or the basic animation pipeline. It adds a new “physics layer” that computes per‑point brightness over time and feeds that into the existing rendering mechanism.

---

## 2. Current baseline (Phase 3 recap)

Phase 3 behavior (see `visualization/animation.py`):

- One global brightness factor per frame:
  

\[
  B(t) = 1 + A \sin(2\pi f t)
  \]


- For each layer:
  - Convert base RGB → HSV
  - Scale the **value** (V) channel by `B(t)`
  - Clamp to \([0, 1]\)
  - Convert back to RGB
  - Apply the same RGB to **all points** in that layer for that frame
- Geometry is fixed; only `mesh.point_data["colors"]` is updated in‑place.
- `generate_frame(plotter, t, config)` is the central hook.

Key properties:

- Temporal modulation only
- No spatial variation
- All points in a layer are synchronized
- All layers share the same time course

---

## 3. Phase 4 goals

Phase 4 introduces **spatial structure** and **propagation**:

1. **Per‑point brightness modulation**
   - Brightness depends on both position \(\mathbf{x}\) and time \(t\):
     

\[
     B(\mathbf{x}, t) = 1 + A \sin(\mathbf{k}\cdot\mathbf{x} - \omega t + \phi)
     \]


   - Different points can be at different phases of the wave.

2. **Traveling waves**
   - Visible wavefronts moving through the point cloud.
   - Direction and wavelength controlled by parameters (e.g. wave vector \(\mathbf{k}\)).

3. **Conduction velocity**
   - Wave speed is configurable (e.g. mm/ms or arbitrary units).
   - Velocity maps to how fast phase propagates across space.

4. **Layer‑to‑layer coupling (optional in initial implementation)**
   - Activity in one layer can influence phase/brightness in another.
   - Simple coupling first (e.g. fixed delay or scaling), more complex models later.

5. **Multiple modes**
   - At minimum:
     - **Global pulse** (Phase 3 behavior, for backward compatibility)
     - **Spatial traveling wave** (new Phase 4 mode)
   - Mode selected via configuration / CLI flag.

---

## 4. Constraints and non‑goals

- **Do not change**:
  - `visualization.viewer` public API
  - `play_animation` signature
  - `export_animation` signature
- **Do not rebuild geometry per frame**:
  - Continue to update only `mesh.point_data["colors"]`.
- **Do not introduce heavy dependencies**:
  - Stay within NumPy + PyVista + standard library.
- **Non‑goal** (for initial Phase 4):
  - Full biophysical ephaptic simulation.
  - Detailed neuron‑level modeling.

Phase 4 is a **visual, phenomenological** model of propagation, not a full simulator.

---

## 5. Proposed extensions

### 5.1. Configuration extensions

Extend `AnimationConfig` (or a Phase 4–specific config) with:

- `mode: str`
  - `"global_pulse"` (Phase 3 behavior)
  - `"traveling_wave"` (Phase 4 behavior)
- `wave_direction: tuple[float, float, float]`
  - Normalized 3D vector for propagation direction.
- `wavelength: float`
  - Spatial period of the wave (same units as point coordinates).
- `velocity: float`
  - Propagation speed along `wave_direction`.
- `phase_offset: float`
  - Global phase shift (optional).
- `layer_coupling: float`
  - Simple scalar controlling how strongly layers influence each other (optional for v1).

### 5.2. Data needed per point

For each point:

- Position \(\mathbf{x} \in \mathbb{R}^3\) (already available via `mesh.points`).
- Base RGB color (already stored per layer).
- Derived scalar:
  - Projected coordinate along wave direction:
    

\[
    s = \mathbf{k}\cdot\mathbf{x}
    \]


  - Or normalized distance along a chosen axis.

These can be computed on the fly or precomputed and cached per mesh.

---

## 6. Phase 4 brightness model

For **traveling wave** mode, a simple model:

1. Define wave vector \(\mathbf{k}\) from direction and wavelength:
   

\[
   \mathbf{k} = \frac{2\pi}{\lambda} \hat{d}
   \]


   where \(\hat{d}\) is the unit wave direction and \(\lambda\) is wavelength.

2. Define angular frequency from velocity:
   

\[
   \omega = \|\mathbf{k}\| \cdot v
   \]



3. For each point at position \(\mathbf{x}\) and time \(t\):
   

\[
   B(\mathbf{x}, t) = 1 + A \sin(\mathbf{k}\cdot\mathbf{x} - \omega t + \phi)
   \]



4. Clamp brightness as in Phase 3 and apply to HSV value channel.

Optional layer coupling (v1 idea):

- Each layer gets a base phase offset or delay.
- Simple model:
  

\[
  B_\ell(\mathbf{x}, t) = 1 + A \sin(\mathbf{k}\cdot\mathbf{x} - \omega (t - \Delta t_\ell) + \phi)
  \]


  where \(\Delta t_\ell\) is a per‑layer delay.

---

## 7. Integration with existing code

### 7.1. Keep `generate_frame` as the central hook

Phase 4 should **not** replace `generate_frame`, but extend its behavior:

- Option A: Add a mode switch inside `generate_frame`:
  - If `config.mode == "global_pulse"` → current Phase 3 logic.
  - If `config.mode == "traveling_wave"` → new per‑point brightness logic.

- Option B: Factor out helpers:
  - `compute_global_brightness(t, config) -> float`
  - `compute_spatial_brightness(points, t, config) -> np.ndarray`
  - `apply_brightness_to_mesh(mesh, base_rgb, brightness_array)`

### 7.2. Performance considerations

- Use vectorized NumPy operations over all points in a mesh.
- Avoid Python loops over individual points.
- Precompute:
  - Wave direction unit vector.
  - Wave vector \(\mathbf{k}\).
  - Projected coordinates `s = k·x` per point, if beneficial.
- Keep memory layout compatible with PyVista expectations.

---

## 8. Testing and validation

Phase 4 should include:

1. **Unit tests** for:
   - Brightness computation given fixed positions and times.
   - Mode switching between global pulse and traveling wave.
   - Clamping behavior.

2. **Visual sanity checks**:
   - Single layer, simple wave direction, obvious wavelength.
   - Verify visible wavefronts move at the expected speed.
   - Compare global pulse vs traveling wave modes.

3. **Regression safety**:
   - Ensure Phase 3 behavior is unchanged when `mode="global_pulse"`.

---

## 9. Implementation roadmap

1. Extend `AnimationConfig` with Phase 4 parameters and `mode`.
2. Add internal helpers for:
   - Wave parameter computation (k, ω).
   - Per‑point brightness computation.
3. Integrate a mode switch into `generate_frame`.
4. Add a simple CLI flag in the Phase 3 manual test script to select mode.
5. Add tests and a small demo configuration for Phase 4.

Phase 4 should remain a **thin, well‑isolated layer** on top of the existing Phase 3 animation engine, not a rewrite.
