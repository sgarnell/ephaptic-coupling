--

# Animation Requirements

This document defines the technical requirements for the ephaptic oscillation animation subsystem. It supplements the high‑level goals in `skills/SKILL.md`.

## 1. Module Structure

The animation system must be implemented inside:

```
visualization/
    animation.py
    colors.py   (shared)
    utils.py    (shared)
```

## 2. Wave Simulation (`animation.py`)

Implement functions to generate oscillatory wave fields over synapse point clouds.

Required functions:

```python
wave_function(t: float, frequency: float, phase: float) -> float
generate_wave_field(points: np.ndarray, t: float, frequency: float, phase_offset: float) -> np.ndarray
apply_wave_to_colors(points: np.ndarray, t: float, frequency: float, phase_offset: float) -> np.ndarray
```

Wave rules:

- Use sinusoidal oscillation: sin(2πft + phase)
- Support per‑layer phase offsets
- Support global phase gradients (optional)

## 3. Animation Engine

Implement a PyVista‑based animation loop.

Required functions:

```python
animate_layer(layer_number: int, duration: float, fps: int)
animate_layers(layer_list: list[int], duration: float, fps: int)
```

Animation requirements:

- Update point colors or intensities each frame
- Use PyVista’s `write_frame()` for MP4 export
- Support real‑time preview in the interactive window
- Support optional MP4 export:

```python
plotter.open_movie("filename.mp4")
...
plotter.close()
```

Dependencies:

- imageio
- imageio-ffmpeg

## 4. Camera and Scene Control

Animations must:

- Maintain a stable camera unless explicitly animated
- Allow optional camera rotation (orbit)
- Allow optional zoom pulsing

## 5. Example Script

Provide an example script in `examples/animation_example.py`:

- Load FB5
- Animate a simple oscillation
- Export a 3‑second MP4

## 6. Performance Constraints

- Avoid recomputing static geometry each frame
- Only update colors or scalar fields
- Use numpy vectorization where possible

## 7. Non‑Goals

The animation module must NOT:

- Load data directly from disk (use utils)
- Define color palettes (use colors)
- Build the 3D viewer scene (use viewer)

