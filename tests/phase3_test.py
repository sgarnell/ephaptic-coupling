"""
tests/phase3_test.py

Automated pytest tests for Phase 3 — Animation Engine.

All tests run fully off-screen; no interactive window is opened.
Phase 1 and Phase 2 tests are not modified or re-run here.

Run from the project root:
    python -m pytest tests/phase3_test.py -v
"""

from __future__ import annotations

import colorsys
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from ephaptic_coupling.visualization.animation import (
    AnimationConfig,
    _build_animation_scene,
    export_animation,
    generate_frame,
    play_animation,
)
from ephaptic_coupling.visualization.colors import get_layer_color


# ---------------------------------------------------------------------------
# 4.1  AnimationConfig tests
# ---------------------------------------------------------------------------


class TestAnimationConfig:
    """Verify AnimationConfig stores values and provides correct defaults."""

    def test_stores_custom_duration(self):
        cfg = AnimationConfig(duration=5.0)
        assert cfg.duration == pytest.approx(5.0)

    def test_stores_custom_fps(self):
        cfg = AnimationConfig(fps=30)
        assert cfg.fps == 30

    def test_stores_custom_amplitude(self):
        cfg = AnimationConfig(amplitude=0.3)
        assert cfg.amplitude == pytest.approx(0.3)

    def test_stores_custom_frequency(self):
        cfg = AnimationConfig(frequency=2.0)
        assert cfg.frequency == pytest.approx(2.0)

    def test_stores_custom_color_map(self):
        cfg = AnimationConfig(color_map="viridis")
        assert cfg.color_map == "viridis"

    def test_default_duration(self):
        assert AnimationConfig().duration == pytest.approx(3.0)

    def test_default_fps(self):
        assert AnimationConfig().fps == 24

    def test_default_amplitude(self):
        assert AnimationConfig().amplitude == pytest.approx(0.5)

    def test_default_frequency(self):
        assert AnimationConfig().frequency == pytest.approx(1.0)

    def test_default_color_map_is_string(self):
        assert isinstance(AnimationConfig().color_map, str)

    def test_stores_all_fields_together(self):
        cfg = AnimationConfig(
            duration=2.5, fps=12, amplitude=0.2, frequency=0.5, color_map="coolwarm"
        )
        assert cfg.duration == pytest.approx(2.5)
        assert cfg.fps == 12
        assert cfg.amplitude == pytest.approx(0.2)
        assert cfg.frequency == pytest.approx(0.5)
        assert cfg.color_map == "coolwarm"


# ---------------------------------------------------------------------------
# 4.2  Frame update tests
# ---------------------------------------------------------------------------


class TestGenerateFrame:
    """Verify brightness modulation, hue preservation, and shape stability."""

    # Shared fixture: off-screen scene with a single layer that has V < 1
    # in HSV so brightness changes are clearly visible.
    # Layer 5 (sky blue) has HSV value ≈ 0.773 — well-suited for both
    # brightening (up to 1.0) and dimming (down to ~0.39 at amplitude=0.5).

    @pytest.fixture
    def scene_cfg(self):
        config = AnimationConfig(
            duration=1.0, fps=10, amplitude=0.5, frequency=1.0
        )
        plotter = _build_animation_scene([5], off_screen=True)
        yield plotter, config
        plotter.close()

    # ---- shape tests -------------------------------------------------------

    def test_color_array_shape_unchanged_after_single_frame(self, scene_cfg):
        plotter, config = scene_cfg
        mesh, _ = plotter._anim_meshes[0]
        initial_shape = mesh.point_data["colors"].shape

        generate_frame(plotter, t=0.0, config=config)

        assert mesh.point_data["colors"].shape == initial_shape

    def test_color_array_shape_unchanged_across_multiple_frames(self, scene_cfg):
        plotter, config = scene_cfg
        mesh, _ = plotter._anim_meshes[0]
        n_points = len(mesh.points)

        for i in range(10):
            generate_frame(plotter, t=i * 0.1, config=config)
            shape = mesh.point_data["colors"].shape
            assert shape == (n_points, 3), (
                f"Shape changed at step {i}: expected ({n_points}, 3), got {shape}"
            )

    def test_color_array_has_three_channels(self, scene_cfg):
        plotter, config = scene_cfg
        mesh, _ = plotter._anim_meshes[0]
        generate_frame(plotter, t=0.5, config=config)
        assert mesh.point_data["colors"].shape[1] == 3

    # ---- brightness tests --------------------------------------------------

    def test_brightness_changes_from_base_to_peak(self, scene_cfg):
        """Colour at t=0 (sin=0, brightness=1) must differ from peak (t=T/4)."""
        plotter, config = scene_cfg
        mesh, _ = plotter._anim_meshes[0]

        # t = 0 → brightness = 1 + A*sin(0) = 1.0  (unchanged from base)
        generate_frame(plotter, t=0.0, config=config)
        colors_t0 = mesh.point_data["colors"][0].copy()

        # t = 1/(4f) → brightness = 1 + A  (peak brightness)
        t_peak = 1.0 / (4.0 * config.frequency)
        generate_frame(plotter, t=t_peak, config=config)
        colors_peak = mesh.point_data["colors"][0].copy()

        assert not np.allclose(colors_t0, colors_peak, atol=1e-4), (
            "Expected colour to change between t=0 and peak, but arrays are identical."
        )

    def test_brightness_changes_from_base_to_trough(self, scene_cfg):
        """Colour at t=0 must differ from trough (t=3T/4)."""
        plotter, config = scene_cfg
        mesh, _ = plotter._anim_meshes[0]

        generate_frame(plotter, t=0.0, config=config)
        colors_t0 = mesh.point_data["colors"][0].copy()

        t_trough = 3.0 / (4.0 * config.frequency)
        generate_frame(plotter, t=t_trough, config=config)
        colors_trough = mesh.point_data["colors"][0].copy()

        assert not np.allclose(colors_t0, colors_trough, atol=1e-4), (
            "Expected colour to dim at trough but got the same values as t=0."
        )

    def test_zero_amplitude_produces_no_change(self):
        """With amplitude=0 the colours must remain identical across frames."""
        config = AnimationConfig(amplitude=0.0, frequency=1.0)
        plotter = _build_animation_scene([1], off_screen=True)
        try:
            mesh, _ = plotter._anim_meshes[0]
            generate_frame(plotter, t=0.0, config=config)
            colors_t0 = mesh.point_data["colors"][0].copy()
            generate_frame(plotter, t=0.25, config=config)
            colors_t1 = mesh.point_data["colors"][0].copy()
            assert np.allclose(colors_t0, colors_t1, atol=1e-5), (
                "amplitude=0 should produce no change in colour."
            )
        finally:
            plotter.close()

    # ---- hue preservation tests --------------------------------------------

    def test_hue_constant_across_frames(self, scene_cfg):
        """Hue (H in HSV) must be identical at every frame."""
        plotter, config = scene_cfg
        mesh, base_rgb = plotter._anim_meshes[0]
        base_h, _, _ = colorsys.rgb_to_hsv(*base_rgb)

        for i, t in enumerate([0.0, 0.1, 0.25, 0.5, 0.75, 1.0]):
            generate_frame(plotter, t=t, config=config)
            rgb = mesh.point_data["colors"][0]
            h, _, _ = colorsys.rgb_to_hsv(
                float(rgb[0]), float(rgb[1]), float(rgb[2])
            )
            assert abs(h - base_h) < 1e-3, (
                f"Hue changed at t={t}: expected {base_h:.6f}, got {h:.6f}"
            )

    def test_hue_constant_multiple_layers(self):
        """Hue must be preserved for every layer independently."""
        config = AnimationConfig(amplitude=0.4, frequency=2.0)
        layers = [1, 3, 5, 7, 9]
        plotter = _build_animation_scene(layers, off_screen=True)
        try:
            for t in [0.0, 0.125, 0.5]:
                generate_frame(plotter, t=t, config=config)
                for (mesh, base_rgb) in plotter._anim_meshes:
                    base_h, _, _ = colorsys.rgb_to_hsv(*base_rgb)
                    rgb = mesh.point_data["colors"][0]
                    h, _, _ = colorsys.rgb_to_hsv(
                        float(rgb[0]), float(rgb[1]), float(rgb[2])
                    )
                    assert abs(h - base_h) < 1e-3, (
                        f"Hue changed for layer with base {base_rgb} at t={t}: "
                        f"expected {base_h:.6f}, got {h:.6f}"
                    )
        finally:
            plotter.close()

    def test_generate_frame_no_exception(self, scene_cfg):
        plotter, config = scene_cfg
        # Should not raise for any t in a full cycle.
        for i in range(config.fps):
            generate_frame(plotter, t=i / config.fps, config=config)


# ---------------------------------------------------------------------------
# 4.3  Playback smoke tests
# ---------------------------------------------------------------------------


class TestPlayAnimation:
    """Verify play_animation runs without exceptions in off-screen mode."""

    def test_single_layer_off_screen(self):
        config = AnimationConfig(duration=0.2, fps=5)
        play_animation([5], config, off_screen=True)

    def test_multiple_layers_off_screen(self):
        config = AnimationConfig(duration=0.1, fps=5)
        play_animation([1, 5, 9], config, off_screen=True)

    def test_all_layers_off_screen(self):
        config = AnimationConfig(duration=0.1, fps=3)
        play_animation(list(range(1, 10)), config, off_screen=True)

    def test_very_short_duration_off_screen(self):
        """A duration shorter than one frame must not crash (n_frames >= 1)."""
        config = AnimationConfig(duration=0.01, fps=24)
        play_animation([1], config, off_screen=True)


# ---------------------------------------------------------------------------
# 4.4  MP4 export tests
# ---------------------------------------------------------------------------


class TestExportAnimation:
    """Verify export_animation creates a non-empty MP4 file."""

    def test_export_creates_file(self, tmp_path):
        config = AnimationConfig(duration=0.5, fps=5, amplitude=0.4, frequency=1.0)
        out = str(tmp_path / "test_anim.mp4")
        export_animation([5], config, out)
        assert os.path.exists(out), "MP4 file was not created."
        assert os.path.getsize(out) > 0, "MP4 file is empty."

    def test_export_multiple_layers(self, tmp_path):
        config = AnimationConfig(duration=0.3, fps=5)
        out = str(tmp_path / "multi_layer.mp4")
        export_animation([1, 5, 9], config, out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_exported_file_is_removable(self, tmp_path):
        """Verify the file handle is released after export (no lock)."""
        config = AnimationConfig(duration=0.2, fps=5)
        out = str(tmp_path / "cleanup_test.mp4")
        export_animation([1], config, out)
        assert os.path.exists(out)
        os.remove(out)
        assert not os.path.exists(out)
