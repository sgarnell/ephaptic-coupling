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
    mode:
        Animation mode selector. Default: ``"global_pulse"``.
    wave_direction:
        3D propagation direction for traveling-wave mode. Default:
        ``(1.0, 0.0, 0.0)``.
    wavelength:
        Spatial period of the wave in scene units. Default: ``100.0``.
    velocity:
        Wave propagation speed. Default: ``1.0``.
    phase_offset:
        Global phase offset applied to the waveform. Default: ``0.0``.
    layer_coupling:
        Optional per-layer delay step in seconds. Default: ``0.0``.
    """

    duration: float = 3.0
    fps: int = 24
    amplitude: float = 0.5
    frequency: float = 1.0
    color_map: str = "hsv"
    mode: str = "global_pulse"
    wave_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    wavelength: float = 100.0
    velocity: float = 1.0
    phase_offset: float = 0.0
    layer_coupling: float = 0.0

    def __post_init__(self) -> None:
        allowed_modes = {"global_pulse", "traveling_wave"}
        if self.mode not in allowed_modes:
            raise ValueError(
                f"mode must be one of {sorted(allowed_modes)}, got {self.mode!r}"
            )

        wave_direction = np.asarray(self.wave_direction, dtype=np.float64)
        if wave_direction.shape != (3,):
            raise ValueError(
                "wave_direction must be a 3-vector of the form (x, y, z)"
            )

        if not np.isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("wavelength must be finite and greater than 0")

        if not np.isfinite(self.velocity):
            raise ValueError("velocity must be finite")

        if not np.isfinite(self.phase_offset):
            raise ValueError("phase_offset must be finite")

        if not np.isfinite(self.layer_coupling):
            raise ValueError("layer_coupling must be finite")



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


def _prepare_wave_parameters(config: AnimationConfig) -> tuple[np.ndarray, float]:
    """Compute the shared wave vector and angular frequency for a scene."""
    wave_direction = np.asarray(config.wave_direction, dtype=np.float64)
    direction_norm = np.linalg.norm(wave_direction)
    if direction_norm == 0:
        raise ValueError("wave_direction cannot be the zero vector")

    direction_unit = wave_direction / direction_norm
    wave_k = (2.0 * math.pi / config.wavelength) * direction_unit
    wave_omega = float(np.linalg.norm(wave_k) * config.velocity)
    return wave_k, wave_omega


def _compute_layer_delay(
    layer_num: int,
    reference_layer: int,
    layer_coupling: float,
) -> float:
    """Compute the per-layer delay offset in seconds."""
    return float((layer_num - reference_layer) * layer_coupling)


def _coerce_base_v(base_v: object) -> float:
    """Convert cached base_v values to a Python float."""
    return float(np.asarray(base_v).reshape(()))


def _iter_anim_meshes(
    plotter: pv.Plotter,
) -> list[tuple[pv.PolyData, object]]:
    """Return the legacy animation mesh list, preserving its tuple contract."""
    return list(getattr(plotter, "_anim_meshes", []))


def _build_minimal_anim_records(plotter: pv.Plotter) -> list[dict[str, object]]:
    """Build the record cache needed for the legacy global-pulse path."""
    anim_meshes = _iter_anim_meshes(plotter)
    existing_records = list(getattr(plotter, "_anim_records", []))
    anim_records: list[dict[str, object]] = []

    for index, (mesh, base_rgb) in enumerate(anim_meshes):
        if index < len(existing_records):
            layer_num = int(existing_records[index].get("layer_num", index + 1))
        else:
            layer_num = index + 1

        base_rgb_array = np.asarray(base_rgb, dtype=np.float32)
        base_v = colorsys.rgb_to_hsv(*base_rgb_array)[2]
        projected_s = mesh.point_data.get(
            "projected_s",
            np.zeros(len(mesh.points), dtype=np.float32),
        )
        anim_records.append(
            {
                "mesh": mesh,
                "base_rgb": base_rgb_array,
                "base_v": base_v,
                "layer_num": layer_num,
                "layer_delay": 0.0,
                "projected_s": projected_s,
            }
        )

    return anim_records


def _build_traveling_wave_records(
    plotter: pv.Plotter,
    config: AnimationConfig,
) -> list[dict[str, object]]:
    """Build or refresh the scene cache for traveling-wave rendering."""
    wave_k, wave_omega = _prepare_wave_parameters(config)
    plotter._wave_k = wave_k  # type: ignore[attr-defined]
    plotter._wave_omega = wave_omega  # type: ignore[attr-defined]

    anim_meshes = _iter_anim_meshes(plotter)
    existing_records = list(getattr(plotter, "_anim_records", []))
    reference_layer = 1
    if existing_records:
        reference_layer = int(existing_records[0].get("layer_num", 1))

    anim_records: list[dict[str, object]] = []
    for index, (mesh, base_rgb) in enumerate(anim_meshes):
        if index < len(existing_records):
            layer_num = int(existing_records[index].get("layer_num", index + 1))
        else:
            layer_num = index + 1

        base_rgb_array = np.asarray(base_rgb, dtype=np.float32)
        base_v = colorsys.rgb_to_hsv(*base_rgb_array)[2]
        projected_s = np.asarray(mesh.points @ wave_k, dtype=np.float32)
        mesh.point_data["projected_s"] = projected_s
        anim_records.append(
            {
                "mesh": mesh,
                "base_rgb": base_rgb_array,
                "base_v": base_v,
                "layer_num": layer_num,
                "layer_delay": _compute_layer_delay(
                    layer_num,
                    reference_layer,
                    config.layer_coupling,
                ),
                "projected_s": mesh.point_data["projected_s"],
            }
        )

    plotter._wave_cache_signature = (  # type: ignore[attr-defined]
        config.mode,
        tuple(np.asarray(config.wave_direction, dtype=np.float64).tolist()),
        float(config.wavelength),
        float(config.velocity),
        float(config.layer_coupling),
        len(anim_records),
    )
    return anim_records


def _ensure_animation_cache(
    plotter: pv.Plotter,
    config: AnimationConfig,
) -> None:
    """Ensure the plotter carries the cache shape expected by generate_frame."""
    plotter._wave_mode = config.mode  # type: ignore[attr-defined]

    if config.mode == "global_pulse":
        if not hasattr(plotter, "_anim_records"):
            plotter._anim_records = _build_minimal_anim_records(plotter)  # type: ignore[attr-defined]
        return

    signature = (
        config.mode,
        tuple(np.asarray(config.wave_direction, dtype=np.float64).tolist()),
        float(config.wavelength),
        float(config.velocity),
        float(config.layer_coupling),
        len(_iter_anim_meshes(plotter)),
    )
    if getattr(plotter, "_wave_cache_signature", None) != signature:
        plotter._anim_records = _build_traveling_wave_records(plotter, config)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Internal scene builder (animation-aware)
# ---------------------------------------------------------------------------


def _build_animation_scene(
    layers: list[int],
    off_screen: bool = False,
    opacity: float = 0.25,
    point_size: int = 5,
    config: AnimationConfig | None = None,
) -> pv.Plotter:
    """Build a Plotter with per-point RGB scalars ready for animation.

    Unlike :func:`~visualization.viewer.build_scene`, the mapper uses
    ``scalars="colors"`` so that updating ``mesh.point_data["colors"]``
    immediately affects the rendered image on the next render call —
    without rebuilding any geometry.

        The returned plotter carries private attributes for the current Phase 3
        runtime path plus the cached Phase 4 scene data:

        - ``_anim_meshes``: current ``list[tuple[pv.PolyData, tuple[float, float, float]]]``
            used by :func:`generate_frame`.
        - ``_anim_records``: richer per-mesh cache dictionaries for the future
            traveling-wave path.
        - ``_wave_mode``, ``_wave_k``, ``_wave_omega``: shared wave parameters.

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
    if config is None:
        config = AnimationConfig()

    plotter = pv.Plotter(off_screen=off_screen)
    anim_meshes: list[tuple[pv.PolyData, tuple[float, float, float]]] = []
    anim_records: list[dict[str, object]] = []

    if config.mode == "traveling_wave":
        wave_k, wave_omega = _prepare_wave_parameters(config)
    else:
        wave_k = np.zeros(3, dtype=np.float64)
        wave_omega = 0.0

    plotter._wave_mode = config.mode  # type: ignore[attr-defined]
    plotter._wave_k = wave_k  # type: ignore[attr-defined]
    plotter._wave_omega = wave_omega  # type: ignore[attr-defined]

    reference_layer = layers[0] if layers else 0

    for layer_num in layers:
        points = get_layer_points(layer_num)
        color = get_layer_color(layer_num)
        n = len(points)

        mesh = pv.PolyData(np.asarray(points, dtype=np.float32))
        mesh.point_data["colors"] = np.tile(
            np.array(color, dtype=np.float32), (n, 1)
        )

        projected_s = np.asarray(mesh.points @ wave_k, dtype=np.float32)
        mesh.point_data["projected_s"] = projected_s
        projected_s = mesh.point_data["projected_s"]

        base_v = colorsys.rgb_to_hsv(*color)[2]
        layer_delay = _compute_layer_delay(
            layer_num,
            reference_layer,
            config.layer_coupling,
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
        anim_records.append(
            {
                "mesh": mesh,
                "base_rgb": color,
                "base_v": base_v,
                "layer_num": layer_num,
                "layer_delay": layer_delay,
                "projected_s": projected_s,
            }
        )

    plotter.reset_camera()
    plotter._anim_meshes = anim_meshes  # type: ignore[attr-defined]
    plotter._anim_records = anim_records  # type: ignore[attr-defined]
    return plotter


def _generate_frame_global_pulse(
    plotter: pv.Plotter,
    t: float,
    config: AnimationConfig,
) -> None:
    """Render one frame using the Phase 3 global pulse model."""
    brightness = 1.0 + config.amplitude * np.sin(2.0 * math.pi * config.frequency * t)

    records = getattr(plotter, "_anim_records", None)
    if records is None:
        for mesh, base_rgb in plotter._anim_meshes:  # type: ignore[attr-defined]
            h, s, v = colorsys.rgb_to_hsv(*base_rgb)
            v_new = min(1.0, max(0.0, v * float(brightness)))
            r, g, b = colorsys.hsv_to_rgb(h, s, v_new)
            modulated = np.array([r, g, b], dtype=np.float32)
            n = len(mesh.points)
            mesh.point_data["colors"] = np.tile(modulated, (n, 1))
            mesh.Modified()
    else:
        for record in records:
            mesh = record["mesh"]
            base_rgb = np.asarray(record["base_rgb"], dtype=np.float32)
            base_v = float(record["base_v"])

            v_scaled = base_v * float(brightness)
            v_new = min(1.0, max(0.0, v_scaled))
            scale = v_new / base_v if base_v > 0.0 else 0.0

            new_colour = base_rgb * scale
            n = len(mesh.points)
            mesh.point_data["colors"] = np.tile(new_colour.astype(np.float32), (n, 1))
            mesh.Modified()

    plotter.render()


def _generate_frame_traveling_wave(
    plotter: pv.Plotter,
    t: float,
    config: AnimationConfig,
) -> None:
    """Render one frame using the traveling-wave model."""
    records = getattr(plotter, "_anim_records", None)
    if records is None:
        for mesh, base_rgb in plotter._anim_meshes:  # type: ignore[attr-defined]
            h, s, v = colorsys.rgb_to_hsv(*base_rgb)
            n = len(mesh.points)
            colors = np.empty((n, 3), dtype=np.float32)
            for i, p in enumerate(mesh.points):
                brightness = compute_brightness_at_point(p, t, config)
                v_new = min(1.0, max(0.0, v * brightness))
                r, g, b = colorsys.hsv_to_rgb(h, s, v_new)
                colors[i] = (r, g, b)
            mesh.point_data["colors"] = colors
            mesh.Modified()
        plotter.render()
        return

    omega = float(plotter._wave_omega)  # type: ignore[attr-defined]
    phase_offset = float(config.phase_offset)
    amplitude = float(config.amplitude)

    for record in records:
        mesh = record["mesh"]
        projected_s = np.asarray(record["projected_s"], dtype=np.float32)
        base_rgb = np.asarray(record["base_rgb"], dtype=np.float32)
        base_v = float(record["base_v"])
        layer_delay = float(record["layer_delay"])

        arg = projected_s - (omega * t) + phase_offset + (omega * layer_delay)
        raw = 1.0 + amplitude * np.sin(arg)
        v_scaled = base_v * raw
        v_new = np.clip(v_scaled, 0.0, 1.0)

        if base_v > 0.0:
            scale = v_new / base_v
        else:
            scale = np.zeros_like(v_new)

        new_colours = base_rgb[np.newaxis, :] * scale[:, np.newaxis]
        mesh.point_data["colors"] = new_colours.astype(np.float32, copy=False)
        mesh.Modified()

    plotter.render()


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
    mode = getattr(plotter, "_wave_mode", "traveling_wave")

    if mode == "global_pulse":
        _generate_frame_global_pulse(plotter, t, config)
    elif mode == "traveling_wave":
        _generate_frame_traveling_wave(plotter, t, config)
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")


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
    plotter = _build_animation_scene(layers, off_screen=off_screen, config=config)
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
    plotter = _build_animation_scene(layers, off_screen=True, config=config)
    set_camera_oblique(plotter)

    n_frames = max(1, int(config.duration * config.fps))
    plotter.open_movie(filename, framerate=config.fps)

    for i in range(n_frames):
        t = i / config.fps
        generate_frame(plotter, t, config)
        plotter.write_frame()

    plotter.close()


