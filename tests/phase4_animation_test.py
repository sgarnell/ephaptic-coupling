"""Pytest suite for animation.py - Phase 4 traveling-wave modulation."""

from __future__ import annotations

import colorsys
import math

import numpy as np
import pytest
import pyvista as pv

from ephaptic_coupling.visualization.animation import (
    AnimationConfig,
    _build_animation_scene,
    compute_brightness_at_point,
    generate_frame,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockPlotter:
    """Minimal plotter stand-in that holds animation meshes and a fake render."""

    def __init__(self) -> None:
        self._anim_meshes: list[tuple[pv.PolyData, tuple[float, float, float]]] = []

    def render(self) -> None:
        pass  # no-op for tests


def _make_mesh(points: list[tuple[float, float, float]]) -> pv.PolyData:
    """Create a PolyData mesh with given points and placeholder colours."""
    arr = np.array(points, dtype=np.float32)
    mesh = pv.PolyData(arr)
    n = len(arr)
    mesh.point_data["colors"] = np.zeros((n, 3), dtype=np.float32)
    return mesh


def _rgb_rows_to_hsv(colors: np.ndarray) -> np.ndarray:
    """Convert an RGB array to an HSV array row by row."""
    return np.asarray(
        [colorsys.rgb_to_hsv(float(r), float(g), float(b)) for r, g, b in colors],
        dtype=np.float64,
    )


def _assert_hue_and_saturation_preserved(
    mesh: pv.PolyData,
    base_rgb: tuple[float, float, float],
    atol: float = 1e-6,
) -> None:
    """Check that every point keeps the layer hue and saturation."""
    base_h, base_s, _ = colorsys.rgb_to_hsv(*base_rgb)
    hsv = _rgb_rows_to_hsv(np.asarray(mesh.point_data["colors"], dtype=np.float64))
    assert np.allclose(hsv[:, 0], base_h, atol=atol), (
        f"Hue changed away from {base_h:.6f}: {hsv[:, 0][:5]}"
    )
    assert np.allclose(hsv[:, 1], base_s, atol=atol), (
        f"Saturation changed away from {base_s:.6f}: {hsv[:, 1][:5]}"
    )


def _find_phase_shift_pair(projected_s: np.ndarray, delta_s: float) -> tuple[int, int, float]:
    """Find a pair of points whose projected distance matches the target shift."""
    projected = np.asarray(projected_s, dtype=np.float64)
    order = np.argsort(projected)
    sorted_projected = projected[order]
    targets = sorted_projected - delta_s
    insert_positions = np.searchsorted(sorted_projected, targets)

    lower_positions = np.clip(insert_positions - 1, 0, len(sorted_projected) - 1)
    upper_positions = np.clip(insert_positions, 0, len(sorted_projected) - 1)

    lower_diffs = np.abs(sorted_projected[lower_positions] - targets)
    upper_diffs = np.abs(sorted_projected[upper_positions] - targets)
    choose_lower = lower_diffs <= upper_diffs

    source_position = int(np.argmin(np.minimum(lower_diffs, upper_diffs)))
    target_position = int(
        lower_positions[source_position]
        if choose_lower[source_position]
        else upper_positions[source_position]
    )
    error = float(min(lower_diffs[source_position], upper_diffs[source_position]))
    return int(order[source_position]), int(order[target_position]), error


# ---------------------------------------------------------------------------
# Tests for compute_brightness_at_point
# ---------------------------------------------------------------------------


class TestComputeBrightnessAtPoint:
    """Tests for the core mathematical function."""

    def test_different_points_give_different_brightness(self) -> None:
        """Brightness should vary spatially when wave_direction is nonzero."""
        cfg = AnimationConfig(
            amplitude=0.5,
            wavelength=2.0,
            wave_direction=(1.0, 0.0, 0.0),
            frequency=0.0,
            phase_offset=0.0,
        )
        p1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        p2 = np.array([0.5, 0.0, 0.0], dtype=np.float64)
        b1 = compute_brightness_at_point(p1, t=0.0, cfg=cfg)
        b2 = compute_brightness_at_point(p2, t=0.0, cfg=cfg)
        assert b1 != b2, (
            f"Expected different brightness for points {p1} and {p2}, got {b1}, {b2}"
        )

    def test_amplitude_zero_gives_unity(self) -> None:
        """When amplitude=0, brightness should always be exactly 1.0."""
        cfg = AnimationConfig(
            amplitude=0.0,
            wavelength=1.0,
            wave_direction=(1.0, 0.0, 0.0),
            velocity=1.0,
            frequency=1.0,
            phase_offset=0.0,
        )
        p = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        for t in [0.0, 0.1, 0.5, 2.0]:
            b = compute_brightness_at_point(p, t, cfg)
            assert math.isclose(b, 1.0, rel_tol=1e-9), (
                f"At t={t}, expected 1.0 got {b}"
            )

    def test_phase_offset_shifts_brightness(self) -> None:
        """Changing phase_offset should change brightness at a given time."""
        cfg_zero = AnimationConfig(
            amplitude=0.5,
            wavelength=1.0,
            wave_direction=(0.0, 0.0, 1.0),
            velocity=1.0,
            frequency=0.0,
            phase_offset=0.0,
        )
        cfg_halfpi = AnimationConfig(
            amplitude=0.5,
            wavelength=1.0,
            wave_direction=(0.0, 0.0, 1.0),
            velocity=1.0,
            frequency=0.0,
            phase_offset=math.pi / 2,
        )
        p = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        b1 = compute_brightness_at_point(p, t=0.0, cfg=cfg_zero)
        b2 = compute_brightness_at_point(p, t=0.0, cfg=cfg_halfpi)
        # φ=0 → sin(0)=0 → B=1.0; φ=π/2 → sin(π/2)=1 → B=1.5
        assert math.isclose(b1, 1.0, rel_tol=1e-9), (
            f"Expected brightness 1.0, got {b1}"
        )
        assert math.isclose(b2, 1.5, rel_tol=1e-9), (
            f"Expected brightness 1.5, got {b2}"
        )
        assert not math.isclose(b1, b2, rel_tol=1e-9), (
            f"Phase offset should change brightness, got {b1} and {b2}"
        )

    def test_wavelength_affects_k_dot_x(self) -> None:
        """Smaller wavelength → larger k → faster spatial oscillation."""
        cfg_short = AnimationConfig(
            amplitude=0.5,
            wavelength=0.1,
            wave_direction=(1.0, 0.0, 0.0),
            velocity=1.0,
            frequency=0.0,
            phase_offset=0.0,
        )
        cfg_long = AnimationConfig(
            amplitude=0.5,
            wavelength=10.0,
            wave_direction=(1.0, 0.0, 0.0),
            velocity=1.0,
            frequency=0.0,
            phase_offset=0.0,
        )
        dx = 0.01  # small shift that avoids a zero-crossing for λ=0.1
        p0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        p1 = np.array([dx, 0.0, 0.0], dtype=np.float64)
        b_short0 = compute_brightness_at_point(p0, t=0.0, cfg=cfg_short)
        b_short1 = compute_brightness_at_point(p1, t=0.0, cfg=cfg_short)
        b_long0 = compute_brightness_at_point(p0, t=0.0, cfg=cfg_long)
        b_long1 = compute_brightness_at_point(p1, t=0.0, cfg=cfg_long)
        diff_short = abs(b_short0 - b_short1)
        diff_long = abs(b_long0 - b_long1)
        assert diff_short > diff_long, (
            f"Shorter λ should give larger brightness diff, "
            f"got short_diff={diff_short}, long_diff={diff_long}"
        )

    def test_wave_direction_is_normalised(self) -> None:
        """Brightness depends only on direction, not scale of the vector."""
        cfg_scaled = AnimationConfig(
            amplitude=0.5,
            wavelength=1.0,
            wave_direction=(0.0, 0.0, 5.0),  # length 5, same as (0,0,1)
            velocity=1.0,
            frequency=0.0,
            phase_offset=0.0,
        )
        cfg_unit = AnimationConfig(
            amplitude=0.5,
            wavelength=1.0,
            wave_direction=(0.0, 0.0, 1.0),
            velocity=1.0,
            frequency=0.0,
            phase_offset=0.0,
        )
        p = np.array([0.0, 0.0, 0.12], dtype=np.float64)
        b1 = compute_brightness_at_point(p, t=0.0, cfg=cfg_scaled)
        b2 = compute_brightness_at_point(p, t=0.0, cfg=cfg_unit)
        assert math.isclose(b1, b2, rel_tol=1e-9), (
            f"Wave direction should be normalised, got {b1} vs {b2}"
        )

    def test_zero_wave_direction_raises(self) -> None:
        """A zero vector should raise ValueError."""
        cfg = AnimationConfig(
            wave_direction=(0.0, 0.0, 0.0),
        )
        p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        with pytest.raises(ValueError, match="wave_direction cannot be the zero vector"):
            compute_brightness_at_point(p, 0.0, cfg)

    def test_velocity_controls_time_phase(self) -> None:
        """A full propagation period should repeat the brightness."""
        cfg = AnimationConfig(
            amplitude=0.5,
            wavelength=2.0,
            wave_direction=(0.0, 0.0, 1.0),
            velocity=4.0,
            frequency=11.0,
            phase_offset=0.0,
        )
        p = np.array([0.0, 0.0, 0.25], dtype=np.float64)
        T = cfg.wavelength / cfg.velocity
        b1 = compute_brightness_at_point(p, 0.0, cfg)
        b2 = compute_brightness_at_point(p, T, cfg)
        assert math.isclose(b1, b2, rel_tol=1e-9), (
            f"Expected repetition after one propagation period, got {b1} and {b2}"
        )

    def test_frequency_is_ignored(self) -> None:
        """Changing frequency must not alter traveling-wave brightness."""
        cfg_low = AnimationConfig(
            amplitude=0.5,
            wavelength=2.0,
            wave_direction=(1.0, 0.0, 0.0),
            velocity=3.0,
            frequency=0.25,
            phase_offset=0.0,
        )
        cfg_high = AnimationConfig(
            amplitude=0.5,
            wavelength=2.0,
            wave_direction=(1.0, 0.0, 0.0),
            velocity=3.0,
            frequency=99.0,
            phase_offset=0.0,
        )
        p = np.array([0.2, 0.0, 0.0], dtype=np.float64)
        b_low = compute_brightness_at_point(p, 0.4, cfg_low)
        b_high = compute_brightness_at_point(p, 0.4, cfg_high)
        assert math.isclose(b_low, b_high, rel_tol=1e-9), (
            f"Frequency should be ignored, got {b_low} and {b_high}"
        )


# ---------------------------------------------------------------------------
# Tests for generate_frame
# ---------------------------------------------------------------------------


class TestTravelingWaveFramePath:
    """Integration tests for the cached traveling-wave frame path."""

    @pytest.fixture
    def scene_cfg(self):
        config = AnimationConfig(
            mode="traveling_wave",
            duration=1.0,
            fps=10,
            amplitude=0.45,
            frequency=123.0,
            wavelength=50.0,
            velocity=1.0,
            wave_direction=(1.0, 0.0, 0.0),
            phase_offset=0.25,
            layer_coupling=0.0,
        )
        plotter = _build_animation_scene([1, 5, 9], off_screen=True, config=config)
        yield plotter, config
        plotter.close()

    @pytest.fixture
    def static_scene_cfg(self):
        config = AnimationConfig(
            mode="traveling_wave",
            duration=1.0,
            fps=10,
            amplitude=0.45,
            frequency=123.0,
            wavelength=50.0,
            velocity=0.0,
            wave_direction=(1.0, 0.0, 0.0),
            phase_offset=0.25,
            layer_coupling=0.0,
        )
        plotter = _build_animation_scene([1, 5, 9], off_screen=True, config=config)
        yield plotter, config
        plotter.close()

    @pytest.fixture
    def single_layer_scene(self):
        config = AnimationConfig(
            mode="traveling_wave",
            duration=1.0,
            fps=10,
            amplitude=0.45,
            frequency=17.0,
            wavelength=50.0,
            velocity=1.0,
            wave_direction=(1.0, 0.0, 0.0),
            phase_offset=0.25,
            layer_coupling=0.0,
        )
        plotter = _build_animation_scene([5], off_screen=True, config=config)
        yield plotter, config
        plotter.close()

    def test_hue_and_saturation_preserved_across_frames(self, scene_cfg) -> None:
        """Every point must keep its layer hue and saturation over time."""
        plotter, config = scene_cfg
        times = [0.0, 0.08, 0.17, 0.31]

        for t in times:
            generate_frame(plotter, t=t, config=config)
            for mesh, base_rgb in plotter._anim_meshes:
                _assert_hue_and_saturation_preserved(mesh, base_rgb)

    def test_hue_and_saturation_survive_clamping(self) -> None:
        """Clipping the value channel high must still preserve hue and saturation."""
        config = AnimationConfig(
            mode="traveling_wave",
            amplitude=0.95,
            frequency=42.0,
            wavelength=50.0,
            velocity=1.0,
            wave_direction=(1.0, 0.0, 0.0),
            phase_offset=math.pi / 2,
        )
        plotter = _build_animation_scene([5], off_screen=True, config=config)
        try:
            mesh, base_rgb = plotter._anim_meshes[0]
            generate_frame(plotter, t=0.0, config=config)
            assert np.max(mesh.point_data["colors"]) <= 1.0 + 1e-6
            _assert_hue_and_saturation_preserved(mesh, base_rgb, atol=1e-5)
        finally:
            plotter.close()

    def test_frequency_is_ignored_by_traveling_wave_frames(self, scene_cfg) -> None:
        """Changing frequency must not alter traveling-wave frame output."""
        plotter, config = scene_cfg
        mesh_snapshots = []

        generate_frame(plotter, t=0.23, config=config)
        for mesh, _ in plotter._anim_meshes:
            mesh_snapshots.append(np.array(mesh.point_data["colors"], copy=True))

        alt_config = AnimationConfig(
            mode="traveling_wave",
            amplitude=config.amplitude,
            frequency=0.001,
            wavelength=config.wavelength,
            velocity=config.velocity,
            wave_direction=config.wave_direction,
            phase_offset=config.phase_offset,
            layer_coupling=config.layer_coupling,
        )
        generate_frame(plotter, t=0.23, config=alt_config)
        for snapshot, (mesh, _) in zip(mesh_snapshots, plotter._anim_meshes):
            assert np.allclose(snapshot, mesh.point_data["colors"], atol=1e-6)

    def test_velocity_zero_produces_static_pattern(self, static_scene_cfg) -> None:
        """When velocity is zero, the traveling-wave frame must be time invariant."""
        plotter, config = static_scene_cfg

        generate_frame(plotter, t=0.0, config=config)
        first = [np.array(mesh.point_data["colors"], copy=True) for mesh, _ in plotter._anim_meshes]

        generate_frame(plotter, t=0.73, config=config)
        second = [np.array(mesh.point_data["colors"], copy=True) for mesh, _ in plotter._anim_meshes]

        for before, after in zip(first, second):
            assert np.allclose(before, after, atol=1e-6)

    def test_phase_advance_matches_wave_speed(self, single_layer_scene) -> None:
        """Brightness should shift by omega * dt over time."""
        plotter, config = single_layer_scene
        mesh, _ = plotter._anim_meshes[0]
        projected_s = np.asarray(plotter._anim_records[0]["projected_s"], dtype=np.float64)

        dt = 0.008
        delta_s = float(plotter._wave_omega) * dt
        i, j, error = _find_phase_shift_pair(projected_s, delta_s)
        assert error < 1e-4, f"No stable point pair found for delta_s={delta_s}: error={error}"

        generate_frame(plotter, t=0.0, config=config)
        colors_t0 = np.array(mesh.point_data["colors"], copy=True)
        generate_frame(plotter, t=dt, config=config)
        colors_t1 = np.array(mesh.point_data["colors"], copy=True)

        assert np.allclose(colors_t1[i], colors_t0[j], atol=1e-5), (
            f"Expected spatial phase shift to match wave speed, got {colors_t1[i]} vs {colors_t0[j]}"
        )


class TestTravelingWaveNoColorsysFallback:
    """The traveling-wave path should fail fast when its cache is missing."""

    def test_missing_anim_records_raises_runtime_error(self) -> None:
        plotter = _MockPlotter()
        mesh = _make_mesh([(0.0, 0.0, 0.0)])
        plotter._anim_meshes.append((mesh, (1.0, 0.5, 0.2)))
        cfg = AnimationConfig(
            mode="traveling_wave",
            amplitude=0.5,
            wavelength=1.0,
            velocity=1.0,
            wave_direction=(1.0, 0.0, 0.0),
        )

        with pytest.raises(RuntimeError, match="_anim_records"):
            generate_frame(plotter, 0.0, cfg)
