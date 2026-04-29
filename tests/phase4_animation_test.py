"""Pytest suite for animation.py – Phase 4 traveling-wave modulation."""

from __future__ import annotations

import colorsys
import math

import numpy as np
import pytest
import pyvista as pv

from visualization.animation import (
    AnimationConfig,
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


def _make_mesh(
    points: list[tuple[float, float, float]],
    base_rgb: tuple[float, float, float],
) -> pv.PolyData:
    """Create a PolyData mesh with given points and initial placeholder colours."""
    arr = np.array(points, dtype=np.float32)
    mesh = pv.PolyData(arr)
    n = len(arr)
    mesh.point_data["colors"] = np.zeros((n, 3), dtype=np.float32)
    return mesh


def _colors_as_set(mesh: pv.PolyData) -> set[tuple[float, ...]]:
    """Return unique RGB colours in the mesh as a set of tuples for comparison."""
    arr = mesh.point_data["colors"]
    return set(tuple(row) for row in arr)


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
            frequency=0.0,
            phase_offset=0.0,
        )
        cfg_halfpi = AnimationConfig(
            amplitude=0.5,
            wavelength=1.0,
            wave_direction=(0.0, 0.0, 1.0),
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
            frequency=0.0,
            phase_offset=0.0,
        )
        cfg_long = AnimationConfig(
            amplitude=0.5,
            wavelength=10.0,
            wave_direction=(1.0, 0.0, 0.0),
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
            frequency=0.0,
            phase_offset=0.0,
        )
        cfg_unit = AnimationConfig(
            amplitude=0.5,
            wavelength=1.0,
            wave_direction=(0.0, 0.0, 1.0),
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

    def test_time_evolution(self) -> None:
        """At times separated by period T=1/f, same point should have same brightness."""
        cfg = AnimationConfig(
            amplitude=0.5,
            wavelength=1.0,
            wave_direction=(0.0, 0.0, 1.0),
            frequency=2.0,  # period = 0.5 s
            phase_offset=0.0,
        )
        p = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        T = 1.0 / cfg.frequency
        b1 = compute_brightness_at_point(p, 0.0, cfg)
        b2 = compute_brightness_at_point(p, T, cfg)
        assert math.isclose(b1, b2, rel_tol=1e-9), (
            f"Expected periodicity, got {b1} and {b2} at t=0 and t={T}"
        )


# ---------------------------------------------------------------------------
# Tests for generate_frame
# ---------------------------------------------------------------------------


class TestGenerateFrame:
    """Integration tests using real pv.PolyData meshes."""

    BASE_RGB = (1.0, 0.5, 0.2)  # distinct colour, non-zero saturation, hue

    @staticmethod
    def _hsv_from_rgb(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
        return colorsys.rgb_to_hsv(*rgb)

    def test_per_point_colours_are_different(self) -> None:
        """Points at different positions should get different RGB values."""
        plotter = _MockPlotter()
        base_rgb = (0.5, 0.25, 0.1)  # V < 1 so clamping doesn't mask differences
        points = [(0.0, 0.0, 0.0), (0.25, 0.0, 0.0)]
        mesh = _make_mesh(points, base_rgb)
        plotter._anim_meshes.append((mesh, base_rgb))
        cfg = AnimationConfig(
            amplitude=0.5,
            wavelength=1.0,
            wave_direction=(1.0, 0.0, 0.0),
            frequency=0.0,
        )
        generate_frame(plotter, 0.0, cfg)
        colors = _colors_as_set(mesh)
        assert len(colors) == 2, f"Expected 2 distinct colours, got {len(colors)}"

    def test_colours_stay_within_valid_rgb(self) -> None:
        """Generated RGB values must be in [0, 1]."""
        plotter = _MockPlotter()
        points = [(0.0, 0.0, 0.0), (0.2, 0.1, 0.3), (10.0, -5.0, 3.0)]
        mesh = _make_mesh(points, self.BASE_RGB)
        plotter._anim_meshes.append((mesh, self.BASE_RGB))
        cfg = AnimationConfig(
            amplitude=2.0,  # large amplitude to push beyond limits
            wavelength=1.0,
            wave_direction=(1.0, 1.0, 1.0),
            frequency=1.0,
        )
        generate_frame(plotter, 0.0, cfg)
        color_array = mesh.point_data["colors"]
        assert np.all(color_array >= 0.0) and np.all(color_array <= 1.0), (
            "RGB values must be clamped to [0, 1]"
        )

    def test_v_channel_clamping(self) -> None:
        """When brightness > 1/v, v_new must be clamped at 1.0 (no overshoot)."""
        plotter = _MockPlotter()
        # Use a colour whose V > 1 / (1 + amplitude) so clamping engages.
        # B_max = 1 + 0.9 = 1.9, so need v0 > 1/1.9 ≈ 0.526.
        base_rgb = (0.0, 0.6, 0.0)  # HSV: v = 0.6
        h0, s0, v0 = colorsys.rgb_to_hsv(*base_rgb)
        assert v0 > 0.52, f"Test setup requires V > 0.526, got {v0}"
        points = [(0.0, 0.0, 0.0)]
        mesh = _make_mesh(points, base_rgb)
        plotter._anim_meshes.append((mesh, base_rgb))
        cfg = AnimationConfig(
            amplitude=0.9,
            wavelength=100.0,
            wave_direction=(0.0, 0.0, 1.0),
            frequency=0.0,
            phase_offset=math.pi / 2,  # sin(π/2)=1 → B=1.9
        )
        generate_frame(plotter, 0.0, cfg)
        r, g, b = mesh.point_data["colors"][0]
        h_new, s_new, v_new = colorsys.rgb_to_hsv(r, g, b)
        # v_new should be clamped to 1.0
        assert math.isclose(v_new, 1.0, rel_tol=1e-6), (
            f"V channel should clamp at 1.0, got {v_new}"
        )
        # hue and saturation should be preserved
        assert math.isclose(h_new, h0, rel_tol=1e-6), (
            f"Hue changed {h0} -> {h_new}"
        )
        assert math.isclose(s_new, s0, rel_tol=1e-6), (
            f"Saturation changed {s0} -> {s_new}"
        )

    def test_hue_and_saturation_preserved(self) -> None:
        """Vary brightness but check that H and S remain identical to base."""
        plotter = _MockPlotter()
        points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        mesh = _make_mesh(points, self.BASE_RGB)
        plotter._anim_meshes.append((mesh, self.BASE_RGB))
        cfg = AnimationConfig(
            amplitude=0.8,
            wavelength=2.0,
            wave_direction=(0.0, 1.0, 0.0),
            frequency=1.0,
        )
        h_base, s_base, v_base = colorsys.rgb_to_hsv(*self.BASE_RGB)
        generate_frame(plotter, 0.3, cfg)
        for row in mesh.point_data["colors"]:
            h, s, v = colorsys.rgb_to_hsv(*row)
            assert math.isclose(h, h_base, rel_tol=1e-6), f"Hue changed to {h}"
            assert math.isclose(s, s_base, rel_tol=1e-6), f"Saturation changed to {s}"

    def test_multiple_meshes_update_independently(self) -> None:
        """Two meshes with different base colours should both be updated."""
        plotter = _MockPlotter()
        rgb1 = (0.8, 0.2, 0.1)
        rgb2 = (0.1, 0.7, 0.3)
        mesh1 = _make_mesh([(0.0, 0.0, 0.0)], rgb1)
        mesh2 = _make_mesh([(0.0, 0.0, 0.0)], rgb2)
        plotter._anim_meshes.append((mesh1, rgb1))
        plotter._anim_meshes.append((mesh2, rgb2))
        cfg = AnimationConfig(
            amplitude=0.5,
            wavelength=10.0,
            wave_direction=(0.0, 0.0, 1.0),
            frequency=0.0,
        )
        generate_frame(plotter, 0.0, cfg)
        c1 = tuple(mesh1.point_data["colors"][0])
        c2 = tuple(mesh2.point_data["colors"][0])
        assert c1 != c2, f"Different base colours should yield different results: {c1}, {c2}"
        # Each should preserve its own hue
        h1, _, _ = colorsys.rgb_to_hsv(*c1)
        h1_base, _, _ = colorsys.rgb_to_hsv(*rgb1)
        assert math.isclose(h1, h1_base, rel_tol=1e-6)
        h2, _, _ = colorsys.rgb_to_hsv(*c2)
        h2_base, _, _ = colorsys.rgb_to_hsv(*rgb2)
        assert math.isclose(h2, h2_base, rel_tol=1e-6)
