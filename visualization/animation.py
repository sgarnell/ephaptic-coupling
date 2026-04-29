"""
visualization/animation.py

Phase 3 animation engine for the Ephaptic Coupling Visualization project.

Provides sinusoidal brightness-modulated animation of FB synapse layers.
Hue is always preserved; only the HSV value (brightness) channel changes
each frame, so every layer retains its identity colour throughout playback.

Modulation rule
---------------
    brightness(t) = 1 + A * sin(2π f t)

where A = config.amplitude and f = config.frequency.

The factor is applied to the base layer colour's HSV *value* channel and
clamped to [0, 1] before conversion back to RGB.

Requires: pyvista, vtk
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass

import numpy as np
import pyvista as pv

from visualization.colors import get_layer_color
from visualization.utils import get_layer_points
from visualization.viewer import set_camera_oblique


# ---------------------------------------------------------------------------
# Animation configuration
# ---------------------------------------------------------------------------


@dataclass
class AnimationConfig:
    """Parameters that control a brightness-modulated animation.

    Attributes
    ----------
    duration:
        Total animation length in seconds. Default: 3.0.
    fps:
        Frames per second for playback and export. Default: 24.
    amplitude:
        Modulation depth A ∈ [0, 1).  Controls how much brightness
        oscillates around the base value.  At A = 0.5 the brightness
        factor cycles between 0.5× (dark) and 1.5× (bright), clamped to
        [0, 1] in HSV space. Default: 0.5.
    frequency:
        Oscillation frequency in Hz. Default: 1.0.
    color_map:
        Reserved for future hue-mapping extensions. Default: ``"hsv"``.
    """

    duration: float = 3.0
    fps: int = 24
    amplitude: float = 0.5
    frequency: float = 1.0
    color_map: str = "hsv"


def compute_brightness_at_point(
    p: np.ndarray,
    t: float,
    cfg: AnimationConfig,
) -> float:
    """Compute brightness B(x, t) at a point for the traveling-wave model.

    Parameters
    ----------
    p : np.ndarray
        3D point coordinates, shape (3,).
    t : float
        Current simulation time in seconds.
    cfg : AnimationConfig
        Animation configuration with wave parameters.

    Returns
    -------
    float
        Brightness scalar B(x, t) = 1 + A * sin(k·x - ωt + φ).
    """
    d = np.array(cfg.wave_direction, dtype=np.float64)
    d_norm = np.linalg.norm(d)
    if d_norm == 0:
        raise ValueError("wave_direction cannot be the zero vector")
    d_hat = d / d_norm
    k = (2.0 * math.pi / cfg.wavelength) * d_hat
    omega = 2.0 * math.pi * cfg.frequency
    k_dot_x = float(np.dot(k, p))
    brightness = 1.0 + cfg.amplitude * math.sin(
        k_dot_x - omega * t + cfg.phase_offset
    )
    return brightness


# ---------------------------------------------------------------------------
# Internal scene builder (animation-aware)
# ---------------------------------------------------------------------------


def _build_animation_scene(
    layers: list[int],
    off_screen: bool = False,
    opacity: float = 0.25,
    point_size: int = 5,
) -> pv.Plotter:
    """Build a Plotter with per-point RGB scalars ready for animation.

    Unlike :func:`~visualization.viewer.build_scene`, the mapper uses
    ``scalars="colors"`` so that updating ``mesh.point_data["colors"]``
    immediately affects the rendered image on the next render call —
    without rebuilding any geometry.

    The returned plotter carries a private attribute ``_anim_meshes``:
    a ``list[tuple[pv.PolyData, tuple[float, float, float]]]`` with one
    entry per layer (mesh, base_rgb_tuple), used by :func:`generate_frame`.

    Parameters
    ----------
    layers:
        FB layer numbers (1–9) to include.
    off_screen:
        If ``True`` render off-screen (for tests and MP4 export).
    opacity:
        Point opacity in [0, 1]. Default: 0.25.
    point_size:
        Point diameter in pixels. Default: 5.

    Returns
    -------
    pv.Plotter
        Configured plotter with ``_anim_meshes`` attribute set.
    """
    plotter = pv.Plotter(off_screen=off_screen)
    anim_meshes: list[tuple[pv.PolyData, tuple[float, float, float]]] = []

    for layer_num in layers:
        points = get_layer_points(layer_num)
        color = get_layer_color(layer_num)
        n = len(points)

        mesh = pv.PolyData(np.asarray(points, dtype=np.float32))
        mesh.point_data["colors"] = np.tile(
            np.array(color, dtype=np.float32), (n, 1)
        )

        plotter.add_mesh(
            mesh,
            scalars="colors",
            rgb=True,
            point_size=point_size,
            opacity=opacity,
            render_points_as_spheres=False,
            show_scalar_bar=False,
        )
        anim_meshes.append((mesh, color))

    plotter.reset_camera()
    plotter._anim_meshes = anim_meshes  # type: ignore[attr-defined]
    return plotter


# ---------------------------------------------------------------------------
# Frame generation
# ---------------------------------------------------------------------------


def generate_frame(
    plotter: pv.Plotter,
    t: float,
    config: AnimationConfig,
) -> None:
    """Update per-point colours for time *t* using brightness modulation.

    The brightness multiplier is computed as::

        brightness(t) = 1 + config.amplitude * sin(2π * config.frequency * t)

    It is applied to the *value* (V) channel of each layer's base colour in
    HSV space.  The result is clamped to [0, 1] and converted back to RGB.
    Hue (H) and saturation (S) are never modified.

    The geometry and colour-array *shape* are unchanged; only the values in
    ``mesh.point_data["colors"]`` are updated in-place.

    Parameters
    ----------
    plotter:
        A Plotter returned by :func:`_build_animation_scene`.  Must have
        the ``_anim_meshes`` attribute set.
    t:
        Current simulation time in seconds.
    config:
        Animation parameters (amplitude, frequency, …).
    """
    brightness = 1.0 + config.amplitude * math.sin(
        2.0 * math.pi * config.frequency * t
    )

    for mesh, base_rgb in plotter._anim_meshes:  # type: ignore[attr-defined]
        # Convert base RGB → HSV, scale value channel, clamp, convert back.
        h, s, v = colorsys.rgb_to_hsv(*base_rgb)
        v_new = min(1.0, max(0.0, v * brightness))
        r, g, b = colorsys.hsv_to_rgb(h, s, v_new)

        modulated = np.array([r, g, b], dtype=np.float32)
        n = len(mesh.points)
        mesh.point_data["colors"] = np.tile(modulated, (n, 1))
        mesh.Modified()  # notify VTK pipeline that data changed

    plotter.render()


# ---------------------------------------------------------------------------
# Interactive playback
# ---------------------------------------------------------------------------


def play_animation(
    layers: list[int],
    config: AnimationConfig,
    off_screen: bool = False,
) -> None:
    """Play the brightness-modulated animation.

    Parameters
    ----------
    layers:
        FB layer numbers (1–9) to display.
    config:
        Animation parameters.
    off_screen:
        If ``True``, run without opening a window (generates all frames in
        sequence then closes the plotter).  Intended for automated tests.
        Default: ``False`` (interactive window).
    """
    plotter = _build_animation_scene(layers, off_screen=off_screen)
    set_camera_oblique(plotter)

    n_frames = max(1, int(config.duration * config.fps))

    if off_screen:
        # Generate every frame sequentially without a GUI event loop.
        for i in range(n_frames):
            t = i / config.fps
            generate_frame(plotter, t, config)
        plotter.close()
    else:
        # Interactive: drive frame updates via a PyVista timer callback.
        duration_ms = max(1, int(1000 / config.fps))

        def _frame_cb(step: int) -> None:
            t = step / config.fps
            generate_frame(plotter, t, config)

        plotter.add_timer_event(
            max_steps=n_frames,
            duration=duration_ms,
            callback=_frame_cb,
        )
        plotter.show()


# ---------------------------------------------------------------------------
# MP4 export
# ---------------------------------------------------------------------------


def export_animation(
    layers: list[int],
    config: AnimationConfig,
    filename: str,
) -> None:
    """Render and write the animation as an MP4 file.

    Frames are rendered off-screen at ``config.fps`` and written to
    *filename* using PyVista's built-in movie-writing tools (requires
    ``imageio`` and ``ffmpeg`` to be available in the environment).

    Parameters
    ----------
    layers:
        FB layer numbers (1–9) to include.
    config:
        Animation parameters (duration, fps, amplitude, frequency).
    filename:
        Output path for the MP4 file, e.g. ``"output/anim.mp4"``.
        Parent directory must already exist.
    """
    plotter = _build_animation_scene(layers, off_screen=True)
    set_camera_oblique(plotter)

    n_frames = max(1, int(config.duration * config.fps))
    plotter.open_movie(filename, framerate=config.fps)

    for i in range(n_frames):
        t = i / config.fps
        generate_frame(plotter, t, config)
        plotter.write_frame()

    plotter.close()
