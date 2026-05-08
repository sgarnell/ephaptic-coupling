#!/usr/bin/env python3
"""Animate the ExR7-driven ephaptic field using per-synapse weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from ephaptic_coupling.FB_Synapse_Dataset.analysis import (
    compute_spatial_stats,
    generate_exr7_waveform,
    get_synapse_weights,
    prepare_synapse_plotter,
    select_synapses,
)
from ephaptic_coupling.visualization.viewer import set_camera_oblique

try:
    from pyvistaqt import BackgroundPlotter
except Exception:  # pragma: no cover - optional dependency
    BackgroundPlotter = None


SOURCE_COLOR = np.array([1.0, 0.92, 0.10], dtype=np.float32)
TARGET_COLORS = {
    "FB4Y": np.array([0.20, 0.45, 0.95], dtype=np.float32),
    "FB5AB": np.array([0.15, 0.80, 0.85], dtype=np.float32),
}
ACTIVE_COLORS = {
    "ExR7": np.array([1.0, 0.50, 0.10], dtype=np.float32),
    "FB4Y": np.array([0.20, 0.95, 0.20], dtype=np.float32),
    "FB5AB": np.array([0.55, 1.0, 0.10], dtype=np.float32),
}
DEFAULT_PROPAGATION_SPEED_UM_S = 18.0
DEFAULT_FIELD_GAIN = 3.5
DEFAULT_BLOOM_RISE_MS = 18.0
DEFAULT_BLOOM_DECAY_MS = 140.0
DEFAULT_VISUAL_PERIOD_S = 0.75


def _scaled_colors(base_rgb: np.ndarray, active_rgb: np.ndarray, brightness: np.ndarray) -> np.ndarray:
    base_rgb = np.asarray(base_rgb, dtype=np.float32)
    active_rgb = np.asarray(active_rgb, dtype=np.float32)
    brightness = np.asarray(brightness, dtype=np.float32)
    return np.clip(
        base_rgb[np.newaxis, :] * (1.0 - brightness[:, np.newaxis])
        + active_rgb[np.newaxis, :] * brightness[:, np.newaxis],
        0.0,
        1.0,
    )


def _brightness_from_wave(
    weights: np.ndarray,
    wave_value: np.ndarray,
    field_gain: float,
) -> np.ndarray:
    wave_value = np.asarray(wave_value, dtype=np.float32)
    wave_level = np.clip(wave_value, 0.0, 1.0)
    return np.clip(0.05 + field_gain * weights * wave_level, 0.0, 1.0).astype(np.float32)


def _update_mesh_colors(
    mesh: pv.PolyData,
    base_rgb: np.ndarray,
    active_rgb: np.ndarray,
    weights: np.ndarray,
    wave_value: np.ndarray,
    field_gain: float,
) -> None:
    brightness = _brightness_from_wave(weights, wave_value, field_gain)
    mesh.point_data["colors"] = _scaled_colors(base_rgb, active_rgb, brightness).astype(np.float32)
    mesh.Modified()


def _make_mesh(points: np.ndarray, base_rgb: np.ndarray, active_rgb: np.ndarray, weights: np.ndarray) -> pv.PolyData:
    mesh = prepare_synapse_plotter(points, weights=weights)
    mesh.point_data["colors"] = _scaled_colors(
        base_rgb,
        active_rgb,
        np.full(len(points), 0.05, dtype=np.float32),
    ).astype(np.float32)
    return mesh


def _bloom_envelope(
    t: np.ndarray | float,
    amplitude: float,
    frequency: float,
    phase: float,
    rise_ms: float,
    decay_ms: float,
    visual_period_s: float,
) -> np.ndarray:
    t = np.asarray(t, dtype=np.float32)
    if amplitude <= 0.0 or visual_period_s <= 0.0:
        return np.zeros_like(t, dtype=np.float32)

    rise_s = max(rise_ms / 1000.0, 1e-4)
    decay_s = max(decay_ms / 1000.0, rise_s + 1e-4)
    if frequency > 0.0:
        first_event_s = (np.pi / 2.0 - phase) / (2.0 * np.pi * frequency)
    else:
        first_event_s = 0.0
    history_s = max(4.0 * decay_s, 2.0 * visual_period_s)

    t_min = float(np.min(t))
    t_max = float(np.max(t))
    start_index = int(np.floor((t_min - history_s - first_event_s) / visual_period_s))
    end_index = int(np.ceil((t_max - first_event_s) / visual_period_s))
    event_times = first_event_s + np.arange(start_index, end_index + 1, dtype=np.float32) * visual_period_s

    dt = t[..., np.newaxis] - event_times
    valid = dt >= 0.0
    kernel = np.zeros_like(dt, dtype=np.float32)
    kernel[valid] = np.exp(-dt[valid] / decay_s) - np.exp(-dt[valid] / rise_s)

    activity = np.sum(kernel, axis=-1)
    compressed = 1.0 - np.exp(-np.clip(amplitude, 0.0, None) * activity)
    return np.clip(compressed, 0.0, 1.0).astype(np.float32)


def _plot_waveform(duration_s: float, fps: int, frequency_hz: float, amplitude: float, phase: float) -> None:
    sample_count = max(2, int(duration_s * fps))
    t = np.linspace(0.0, duration_s, sample_count, endpoint=False)
    waveform = generate_exr7_waveform(t, amplitude=amplitude, frequency=frequency_hz, phase=phase)

    figure, axis = plt.subplots(figsize=(8, 3))
    axis.plot(t, waveform, color="#cc7a00", linewidth=2)
    axis.set_title("ExR7 Input Waveform")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("I(t)")
    axis.grid(True, alpha=0.25)
    axis.set_xlim(float(t[0]), float(t[-1]) if len(t) > 1 else duration_s)
    figure.tight_layout()
    plt.show()


def _build_scene(
    source_points: np.ndarray,
    target_groups: dict[str, tuple[np.ndarray, np.ndarray]],
    interactive: bool,
    propagation_speed_um_s: float,
):
    if interactive and BackgroundPlotter is not None:
        plotter = BackgroundPlotter()
    else:
        plotter = pv.Plotter(off_screen=not interactive)

    plotter.set_background("black")
    plotter.add_axes()

    source_active = ACTIVE_COLORS["ExR7"]
    source_mesh = _make_mesh(source_points, SOURCE_COLOR, source_active, np.ones(len(source_points), dtype=np.float32))
    source_mesh.point_data["delay_s"] = np.zeros(len(source_points), dtype=np.float32)
    legend_entries: list[tuple[str, tuple[float, float, float]]] = [("ExR7 source", tuple(float(channel) for channel in SOURCE_COLOR))]
    plotter.add_mesh(
        source_mesh,
        scalars="colors",
        rgb=True,
        point_size=5,
        opacity=1.0,
        render_points_as_spheres=True,
        show_scalar_bar=False,
    )

    target_meshes: list[dict[str, object]] = []
    source_tree = KDTree(source_points)
    for region_name, (points, weights) in target_groups.items():
        base_rgb = TARGET_COLORS.get(region_name, np.array([0.8, 0.8, 0.8], dtype=np.float32))
        active_rgb = ACTIVE_COLORS.get(region_name, np.array([1.0, 1.0, 1.0], dtype=np.float32))
        legend_entries.append((region_name, tuple(float(channel) for channel in base_rgb)))
        display_weights = np.sqrt(np.clip(weights, 0.0, 1.0)).astype(np.float32)
        distances, _ = source_tree.query(points, k=1)
        delay_s = np.asarray(distances, dtype=np.float32) / max(propagation_speed_um_s, 1e-6)
        mesh = _make_mesh(points, base_rgb, active_rgb, display_weights)
        mesh.point_data["delay_s"] = delay_s
        plotter.add_mesh(
            mesh,
            scalars="colors",
            rgb=True,
            point_size=4,
            opacity=0.95,
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )
        target_meshes.append(
            {
                "mesh": mesh,
                "base_rgb": base_rgb,
                "active_rgb": active_rgb,
                "weights": display_weights,
                "delay_s": delay_s,
                "region": region_name,
            }
        )

    plotter.reset_camera()
    set_camera_oblique(plotter)
    plotter.add_legend(
        legend_entries,
        bcolor=(0.05, 0.05, 0.05),
        face="circle",
        border=True,
        size=(0.18, 0.16),
    )
    plotter.add_text(
        "Field state shifts color within each neuron type",
        position="upper_left",
        font_size=10,
        color="white",
    )
    scene = {
        "source_mesh": source_mesh,
        "source_active_rgb": source_active,
        "source_delay_s": np.zeros(len(source_points), dtype=np.float32),
        "target_meshes": target_meshes,
    }
    return plotter, scene


def _render_frame(
    plotter: pv.Plotter,
    scene: dict[str, object],
    current_time: float,
    amplitude: float,
    frequency: float,
    phase: float,
    field_gain: float,
    bloom_rise_ms: float,
    bloom_decay_ms: float,
    visual_period_s: float,
) -> None:
    source_mesh = scene["source_mesh"]
    source_active_rgb = scene["source_active_rgb"]

    source_level = float(
        _bloom_envelope(
            current_time,
            amplitude,
            frequency,
            phase,
            bloom_rise_ms,
            bloom_decay_ms,
            visual_period_s,
        ).reshape(())
    )
    source_wave = np.full(
        len(source_mesh.points),
        source_level,
        dtype=np.float32,
    )
    _update_mesh_colors(
        source_mesh,
        SOURCE_COLOR,
        source_active_rgb,
        np.ones(len(source_wave), dtype=np.float32),
        source_wave,
        field_gain,
    )

    for record in scene["target_meshes"]:
        delayed_time = current_time - record["delay_s"]
        local_wave = _bloom_envelope(
            delayed_time,
            amplitude,
            frequency,
            phase,
            bloom_rise_ms,
            bloom_decay_ms,
            visual_period_s,
        )
        _update_mesh_colors(
            record["mesh"],
            record["base_rgb"],
            record["active_rgb"],
            record["weights"],
            local_wave,
            field_gain,
        )

    plotter.render()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate ExR7-driven ephaptic input over FB synapses")
    parser.add_argument("--radius", type=float, default=7.5, help="Ephaptic radius in microns")
    parser.add_argument("--freq", type=float, default=20.0, help="Driving waveform frequency in Hz")
    parser.add_argument("--amplitude", type=float, default=1.0, help="Driving waveform amplitude")
    parser.add_argument("--phase", type=float, default=0.0, help="Driving waveform phase offset in radians")
    parser.add_argument("--duration", type=float, default=15.0, help="Animation duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Animation frame rate")
    parser.add_argument("--exr-region", default="ExR7", help="Incoming neuron name prefix")
    parser.add_argument("--fb-regions", nargs="+", default=["FB4Y", "FB5AB"], help="Receiving neuron name prefixes")
    parser.add_argument("--movie", type=str, default=None, help="Optional MP4 output path")
    parser.add_argument("--show-waveform", action="store_true", help="Display the waveform preview before rendering")
    parser.add_argument("--propagation-speed", type=float, default=DEFAULT_PROPAGATION_SPEED_UM_S, help="Spatial propagation speed in um/s")
    parser.add_argument("--field-gain", type=float, default=DEFAULT_FIELD_GAIN, help="Brightness gain used to emphasize the moving field")
    parser.add_argument("--bloom-rise-ms", type=float, default=DEFAULT_BLOOM_RISE_MS, help="Visual rise time for the rendered bloom response in milliseconds")
    parser.add_argument("--bloom-decay-ms", type=float, default=DEFAULT_BLOOM_DECAY_MS, help="Visual decay time for the rendered bloom response in milliseconds")
    parser.add_argument("--visual-period-s", type=float, default=DEFAULT_VISUAL_PERIOD_S, help="Seconds between visible radial bloom launches")
    parser.add_argument("--show-all-targets", action="store_true", help="Render all FB target synapses instead of only the spatially affected subset")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not (10.0 <= args.freq <= 30.0):
        raise ValueError("--freq must be between 10 and 30 Hz for Phase 6")

    source_points, source_pre_names, source_neuron_names, _ = select_synapses(pre_prefix=args.exr_region)
    source_stats = compute_spatial_stats(source_points)

    target_groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for region_name in args.fb_regions:
        points, _, _, _ = select_synapses(neuron_prefix=region_name)
        weights = get_synapse_weights(source_points, points, radius_um=args.radius)
        if args.show_all_targets:
            display_points = points
            display_weights = weights
        else:
            affected_mask = weights > 0.0
            display_points = points[affected_mask]
            display_weights = weights[affected_mask]

        target_groups[region_name] = (display_points, display_weights)

    print(f"ExR7 source synapses: {source_stats['count']}")
    print(f"Source centroid: {source_stats['centroid']}")
    for region_name in args.fb_regions:
        all_points, _, _, _ = select_synapses(neuron_prefix=region_name)
        all_weights = get_synapse_weights(source_points, all_points, radius_um=args.radius)
        displayed_points, displayed_weights = target_groups[region_name]
        stats = compute_spatial_stats(displayed_points)
        affected_count = int(np.sum(all_weights > 0.0))
        print(
            f"{region_name}: total={len(all_points)}, affected={affected_count}, displayed={stats['count']}, "
            f"mean_weight={float(np.mean(displayed_weights)) if len(displayed_weights) else 0.0:.4f}, "
            f"weight_gradient={float(np.max(displayed_weights) - np.min(displayed_weights)) if len(displayed_weights) else 0.0:.4f}"
        )

    if args.show_waveform:
        _plot_waveform(args.duration, args.fps, args.freq, args.amplitude, args.phase)

    interactive = args.movie is None
    plotter, scene = _build_scene(
        source_points,
        target_groups,
        interactive=interactive,
        propagation_speed_um_s=args.propagation_speed,
    )
    print(f"Model: radial source pulse with distance-delayed FB targets, visual_period={args.visual_period_s:.2f}s")

    n_frames = max(1, int(args.duration * args.fps))

    if args.movie is not None:
        movie_path = Path(args.movie)
        movie_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.open_movie(str(movie_path), framerate=args.fps)

        for frame_index in range(n_frames):
            current_time = frame_index / args.fps
            _render_frame(
                plotter,
                scene,
                current_time,
                args.amplitude,
                args.freq,
                args.phase,
                args.field_gain,
                args.bloom_rise_ms,
                args.bloom_decay_ms,
                args.visual_period_s,
            )
            plotter.write_frame()

        plotter.close()
        print(f"Movie written to {movie_path}")
        return

    if BackgroundPlotter is None:
        raise RuntimeError("Interactive animation requires pyvistaqt to be installed")

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    plotter.app_window.signal_close.connect(app.quit)

    state = {"frame": 0}

    def advance_frame() -> None:
        frame_index = int(state["frame"])
        current_time = (frame_index % n_frames) / args.fps
        _render_frame(
            plotter,
            scene,
            current_time,
            args.amplitude,
            args.freq,
            args.phase,
            args.field_gain,
            args.bloom_rise_ms,
            args.bloom_decay_ms,
            args.visual_period_s,
        )
        state["frame"] = frame_index + 1

    timer = QTimer(plotter)
    timer.timeout.connect(advance_frame)
    timer.start(max(1, int(1000 / args.fps)))
    plotter._animation_timer = timer  # type: ignore[attr-defined]
    plotter._animation_state = state  # type: ignore[attr-defined]
    _render_frame(
        plotter,
        scene,
        0.0,
        args.amplitude,
        args.freq,
        args.phase,
        args.field_gain,
        args.bloom_rise_ms,
        args.bloom_decay_ms,
        args.visual_period_s,
    )
    plotter.show()
    app.exec()


if __name__ == "__main__":
    main()