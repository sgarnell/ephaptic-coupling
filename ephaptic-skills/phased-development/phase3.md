
# Phase 3 — Animation Engine (Time-Varying Visualization)

This document defines the scope, deliverables, and test plan for Phase 3 of the Ephaptic Coupling Visualization project.  
Phase 3 introduces time-varying animation, color modulation, and MP4 export on top of the static viewer built in Phase 2.

---

# 1. Purpose of Phase 3

Phase 3 focuses on adding **temporal dynamics** to the visualization:

- Time-varying color fields (oscillations, waves, envelopes)
- Frame-by-frame scene updates
- Animation playback inside the PyVista viewer
- MP4 export using PyVista’s movie tools
- A clean, modular animation API

No physics simulation is included in this phase.  
Phase 3 is purely about **visual animation**, not ephaptic modeling.

---

# 2. Modules Implemented in Phase 3

Phase 3 introduces:

```
visualization/
    animation.py
```

Phase 2 modules (`viewer.py`, `colors.py`) must remain unchanged except where explicitly required.

---

# 3. Required Functionality

## 3.1 Animation Data Model

Implement:

```python
class AnimationConfig:
    duration: float
    fps: int
    amplitude: float
    frequency: float
    color_map: str
```

This configuration object defines the animation parameters.

## 3.2 Frame Generation (Revised)

Implement:

    generate_frame(plotter, t: float, config: AnimationConfig)

Requirements:

- Apply a time-varying **brightness modulation** to each point’s color.
- Brightness must follow a sinusoidal envelope:
      brightness(t) = 1 + A * sin(2π f t)
  where A = config.amplitude and f = config.frequency.
- **Hue must remain constant** for all points.
- Only the brightness (value channel) may change over time.
- The underlying geometry and color array shape must remain unchanged.
- The scene must NOT be rebuilt each frame; only color arrays are updated.


## 3.3 Animation Playback (Revised)

Implement:

    play_animation(layers: list[int], config: AnimationConfig)

Requirements:

- Build the scene using Phase 2 viewer functions.
- For each frame:
    - Compute brightness modulation using the sinusoidal envelope.
    - Update per-point brightness while preserving hue.
    - Do NOT modify geometry, camera, or layer structure.
- Display the animation interactively in PyVistaQt.
- Playback must be smooth and deterministic.


## 3.4 MP4 Export

Implement:

```python
export_animation(layers: list[int], config: AnimationConfig, filename: str)
```

Requirements:

- Use PyVista’s movie-writing tools
- Render frames off-screen
- Write MP4 at the requested FPS
- Ensure deterministic output

---

# 4. Phase 3 Tests

Create:

```
tests/phase3_test.py
```

Tests must include:

## 4.1 Config Test
- Ensure `AnimationConfig` stores values correctly
- Ensure defaults behave as expected

## 4.2 Frame Update Test (Revised)

- Build a scene with one layer.
- Capture the initial per-point color array.
- Call generate_frame at two different time points (e.g., t=0 and t=0.1).
- Verify that:
    - Brightness values change between frames.
    - Hue values remain constant.
    - The color array shape is unchanged.
- Ensure no exceptions occur.



## 4.3 Playback Smoke Test
- Call `play_animation` in off-screen mode
- Ensure no exceptions occur
- Do not open a window during tests

## 4.4 MP4 Export Test
- Export a 1-second animation at low FPS
- Ensure the MP4 file is created
- Ensure file size is > 0 bytes
- Remove file after test

---

# 5. Out of Scope for Phase 3

Phase 3 explicitly excludes:

- Physical ephaptic coupling simulation
- Wave propagation models
- Connectivity-based dynamics
- Real-time user interaction widgets
- Shader-based GPU effects

These belong to Phase 4.

---

# 6. Success Criteria

Phase 3 is complete when:

- All animation functions are implemented
- All tests in `phase3_test.py` pass
- Animations play smoothly in the viewer
- MP4 export works reliably
- No physics or simulation code exists in this phase

Only after Phase 3 passes cleanly should development proceed to Phase 4 (ephaptic simulation engine).

---

# End of Phase 3
