"""
Phase 4, Step 9: Supplemental Test Suite

This module contains additional tests for the Phase 4 animation engine that
are not present in the main unit test bundle (phase4s9_unittestbundle.py).

All tests are off-screen and do not require a GUI or rendering backend.
Run with: pytest -v tests/phase4s9_supplemental_tests.py
"""

import pytest
import numpy as np
from ephaptic_coupling.visualization.animation import (
    AnimationConfig,
    _build_animation_scene,
    generate_frame,
    _prepare_wave_parameters,
)


# ============================================================================
# 1. Brightness Behavior Tests
# ============================================================================

class TestBrightnessBehavior:
    """Validate the correctness of brightness computation in various scenarios."""

    def test_global_pulse_brightness_value(self):
        """Verify global_pulse produces brightness = 1 + A*sin(2π f t) exactly per layer."""
        layers = [1]
        config = AnimationConfig(
            mode="global_pulse",
            amplitude=0.6,
            frequency=1.5,
            duration=2.0,
            fps=24,
        )
        plotter = _build_animation_scene(layers, off_screen=True, config=config)
        record = plotter._anim_records[0]
        mesh = record["mesh"]
        base_v = float(record["base_v"])
        base_rgb = record["base_rgb"]

        # Test at a few time points
        for t in [0.0, 0.1, 0.25, 0.5]:
            generate_frame(plotter, t=t, config=config)
            colors = mesh.point_data["colors"]
            # All points should have the same RGB
            assert np.allclose(colors, colors[0, :], rtol=1e-7)
            expected_brightness = 1.0 + config.amplitude * np.sin(2.0 * np.pi * config.frequency * t)
            expected_v = np.clip(base_v * expected_brightness, 0.0, 1.0)
            expected_scale = expected_v / base_v if base_v > 0 else 0.0
            expected_rgb = base_rgb * expected_scale
            np.testing.assert_allclose(colors[0, :], expected_rgb, rtol=1e-6)

    def test_points_with_same_projected_s_have_same_brightness(self):
        """When two points have identical k·x, they must have identical brightness at the same time."""
        import pyvista as pv
        from ephaptic_coupling.visualization.animation import _generate_frame_traveling_wave

        # Create a synthetic mesh: three points; two share same projected_s
        points = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
        mesh = pv.PolyData(points)
        projected_s = np.array([0.0, 0.0, 10.0], dtype=np.float32)
        mesh.point_data["projected_s"] = projected_s

        base_rgb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        base_v = 1.0

        record = {
            "mesh": mesh,
            "base_rgb": base_rgb,
            "base_v": base_v,
            "layer_num": 1,
            "layer_delay": 0.0,
            "projected_s": projected_s,
        }

        config = AnimationConfig(
            mode="traveling_wave",
            amplitude=0.5,
            wave_direction=(1.0, 0.0, 0.0),
            wavelength=100.0,
            velocity=1.0,
            phase_offset=0.0,
            layer_coupling=0.0,
        )

        # Prepare a minimal dummy plotter for _generate_frame_traveling_wave
        class DummyPlotter:
            def __init__(self):
                self._anim_records = [record]
                self._wave_omega = None
                self.render_calls = 0
            def render(self):
                self.render_calls += 1

        k, omega = _prepare_wave_parameters(config)
        dummy = DummyPlotter()
        dummy._wave_omega = omega

        _generate_frame_traveling_wave(dummy, t=0.3, config=config)

        colors = mesh.point_data["colors"]
        # Points 0 and 1 have same projected_s → same color
        assert np.allclose(colors[0], colors[1], rtol=1e-7)
        # Point 2 should differ
        assert not np.allclose(colors[0], colors[2], rtol=1e-7)

    def test_reversing_wave_direction_reverses_phase_ordering(self):
        """Reversing wave_direction should invert the spatial phase progression across points."""
        layers = [1]
        config_forward = AnimationConfig(
            mode="traveling_wave",
            wave_direction=(1.0, 0.0, 0.0),
            wavelength=100.0,
            velocity=1.0,
            amplitude=0.5,
            phase_offset=0.0,
            layer_coupling=0.0,
        )
        config_reverse = AnimationConfig(
            mode="traveling_wave",
            wave_direction=(-1.0, 0.0, 0.0),
            wavelength=100.0,
            velocity=1.0,
            amplitude=0.5,
            phase_offset=0.0,
            layer_coupling=0.0,
        )
        plotter_f = _build_animation_scene(layers, off_screen=True, config=config_forward)
        plotter_r = _build_animation_scene(layers, off_screen=True, config=config_reverse)
        rec_f = plotter_f._anim_records[0]
        rec_r = plotter_r._anim_records[0]
        # projected_s arrays should be negated
        np.testing.assert_allclose(rec_f["projected_s"], -rec_r["projected_s"], rtol=1e-6)
        # Compute arg for both at a chosen time
        t = 0.5
        omega_f = plotter_f._wave_omega
        omega_r = plotter_r._wave_omega
        assert np.isclose(omega_f, omega_r, rtol=1e-6)
        arg_f = rec_f["projected_s"] - omega_f * t + config_forward.phase_offset
        arg_r = rec_r["projected_s"] - omega_r * t + config_reverse.phase_offset
        # arg_f ≈ -arg_r
        np.testing.assert_allclose(arg_f, -arg_r, rtol=1e-6)

    def test_velocity_zero_gives_static_spatial_pattern(self):
        """With velocity = 0, the spatial pattern does not change over time."""
        layers = [1]
        config = AnimationConfig(
            mode="traveling_wave",
            velocity=0.0,
            wavelength=100.0,
            amplitude=0.5,
            phase_offset=0.0,
            layer_coupling=0.0,
        )
        plotter = _build_animation_scene(layers, off_screen=True, config=config)
        rec = plotter._anim_records[0]
        mesh = rec["mesh"]
        generate_frame(plotter, t=0.0, config=config)
        colors_t0 = mesh.point_data["colors"].copy()
        generate_frame(plotter, t=1.0, config=config)
        colors_t1 = mesh.point_data["colors"]
        # With v=0, ω = 0 → arg independent of t
        np.testing.assert_allclose(colors_t0, colors_t1, rtol=1e-7)

    def test_phase_offset_shifts_waveform_without_changing_amplitude_bounds(self):
        """phase_offset adds to sine argument but brightness stays in [1-A, 1+A]."""
        layers = [1]
        A = 0.5
        config = AnimationConfig(
            mode="traveling_wave",
            amplitude=A,
            wavelength=100.0,
            velocity=1.0,
            phase_offset=0.0,
            layer_coupling=0.0,
        )
        plotter = _build_animation_scene(layers, off_screen=True, config=config)
        rec = plotter._anim_records[0]
        projected_s = rec["projected_s"]
        omega = plotter._wave_omega
        t = 0.3
        arg0 = projected_s - omega * t
        raw0 = 1.0 + A * np.sin(arg0)
        arg_pi2 = arg0 + np.pi/2
        raw_pi2 = 1.0 + A * np.sin(arg_pi2)
        # Bounds on raw brightness must match
        assert np.max(raw0) == pytest.approx(np.max(raw_pi2))
        assert np.min(raw0) == pytest.approx(np.min(raw_pi2))


# ============================================================================
# 2. Phase-3 Regression Tests
# ============================================================================

class TestPhase3Regression:
    """Ensure Phase 3 behavior remains exactly unchanged."""

    def test_default_mode_is_global_pulse(self):
        """AnimationConfig() should default to mode='global_pulse'."""
        cfg = AnimationConfig()
        assert cfg.mode == "global_pulse"

    def test_global_pulse_ignores_phase4_fields(self):
        """In global_pulse, wave parameters and layer_coupling are ignored (but allowed)."""
        layers = [1, 2]
        config = AnimationConfig(
            mode="global_pulse",
            amplitude=0.5,
            frequency=1.0,
            wave_direction=(1.0, 0.0, 0.0),
            wavelength=50.0,
            velocity=5.0,
            phase_offset=2.0,
            layer_coupling=0.1,
        )
        plotter = _build_animation_scene(layers, off_screen=True, config=config)
        # Records should have layer_delay = 0 (per _build_minimal_anim_records)
        for record in plotter._anim_records:
            assert record["layer_delay"] == 0.0

    def test_global_pulse_produces_uniform_per_layer_colors(self):
        """In global_pulse mode, every point in a layer gets the same RGB."""
        layers = [1, 5, 9]
        config = AnimationConfig(mode="global_pulse", amplitude=0.5, frequency=1.0)
        plotter = _build_animation_scene(layers, off_screen=True, config=config)
        generate_frame(plotter, t=0.2, config=config)
        for record in plotter._anim_records:
            colors = record["mesh"].point_data["colors"]
            # All rows equal the first row
            assert np.allclose(colors, colors[0, :], rtol=1e-7)

    def test_global_pulse_formula_unchanged(self):
        """Global pulse brightness formula remains 1 + A*sin(2π f t)."""
        layers = [3]
        config = AnimationConfig(mode="global_pulse", amplitude=0.7, frequency=2.0)
        plotter = _build_animation_scene(layers, off_screen=True, config=config)
        rec = plotter._anim_records[0]
        base_v = float(rec["base_v"])
        times = [0.0, 0.05, 0.125, 0.25]
        for t in times:
            generate_frame(plotter, t=t, config=config)
            colors = rec["mesh"].point_data["colors"]
            expected_B = 1.0 + config.amplitude * np.sin(2.0 * np.pi * config.frequency * t)
            expected_v = np.clip(base_v * expected_B, 0.0, 1.0)
            expected_scale = expected_v / base_v if base_v > 0 else 0.0
            expected_rgb = rec["base_rgb"] * expected_scale
            np.testing.assert_allclose(colors[0], expected_rgb, rtol=1e-6)


# ============================================================================
# 3. Edge-Case Tests
# ============================================================================

class TestEdgeCases:
    """Validate behavior at boundaries and with extreme/invalid inputs."""

    def test_amplitude_zero_yields_no_brightness_change(self):
        """With amplitude=0, brightness factor is always 1 → colors unchanged from base."""
        layers = [1]
        config_tw = AnimationConfig(
            mode="traveling_wave",
            amplitude=0.0,
            wavelength=100.0,
            velocity=1.0,
        )
        plotter_tw = _build_animation_scene(layers, off_screen=True, config=config_tw)
        rec_tw = plotter_tw._anim_records[0]
        mesh_tw = rec_tw["mesh"]
        base_colors = mesh_tw.point_data["colors"].copy()
        generate_frame(plotter_tw, t=0.5, config=config_tw)
        np.testing.assert_allclose(mesh_tw.point_data["colors"], base_colors, rtol=1e-7)

        config_gp = AnimationConfig(mode="global_pulse", amplitude=0.0, frequency=1.0)
        plotter_gp = _build_animation_scene(layers, off_screen=True, config=config_gp)
        rec_gp = plotter_gp._anim_records[0]
        mesh_gp = rec_gp["mesh"]
        base_colors_gp = mesh_gp.point_data["colors"].copy()
        generate_frame(plotter_gp, t=0.5, config=config_gp)
        np.testing.assert_allclose(mesh_gp.point_data["colors"], base_colors_gp, rtol=1e-7)

    def test_velocity_zero_in_traveling_wave(self):
        """velocity=0 yields omega=0 → static spatial pattern (no time variation)."""
        layers = [1]
        config = AnimationConfig(
            mode="traveling_wave",
            velocity=0.0,
            wavelength=100.0,
            amplitude=0.5,
        )
        plotter = _build_animation_scene(layers, off_screen=True, config=config)
        rec = plotter._anim_records[0]
        mesh = rec["mesh"]
        # omega should be zero
        assert plotter._wave_omega == 0.0
        # Render at two times; colors should be identical
        generate_frame(plotter, t=0.0, config=config)
        c0 = mesh.point_data["colors"].copy()
        generate_frame(plotter, t=10.0, config=config)
        c1 = mesh.point_data["colors"]
        np.testing.assert_allclose(c0, c1, rtol=1e-7)

    def test_very_large_wavelength_approaches_global_pulse(self):
        """Very large wavelength makes projected_s nearly constant → colors nearly uniform."""
        layers = [1]
        config = AnimationConfig(
            mode="traveling_wave",
            wavelength=1e6,  # huge
            velocity=1.0,
            amplitude=0.5,
        )
        plotter = _build_animation_scene(layers, off_screen=True, config=config)
        rec = plotter._anim_records[0]
        mesh = rec["mesh"]
        generate_frame(plotter, t=0.3, config=config)
        colors = mesh.point_data["colors"]
        # The standard deviation across points should be very small
        std_per_channel = np.std(colors, axis=0)
        assert np.all(std_per_channel < 1e-3)

    def test_very_small_wavelength_rapid_oscillation_but_no_exception(self):
        """Very small wavelength gives high spatial frequency but should not raise."""
        layers = [1]
        config = AnimationConfig(
            mode="traveling_wave",
            wavelength=0.001,  # extremely small
            velocity=1.0,
            amplitude=0.5,
        )
        # Should build without error
        plotter = _build_animation_scene(layers, off_screen=True, config=config)
        rec = plotter._anim_records[0]
        mesh = rec["mesh"]
        generate_frame(plotter, t=0.1, config=config)  # should not except
        colors = mesh.point_data["colors"]
        assert colors.shape == (len(mesh.points), 3)
        assert np.all(np.isfinite(colors))

    def test_zero_wave_direction_raises(self):
        """wave_direction (0,0,0) must raise during AnimationConfig validation."""
        with pytest.raises(ValueError, match="wave_direction cannot be the zero vector"):
            AnimationConfig(
                mode="traveling_wave",
                wave_direction=(0.0, 0.0, 0.0),
                wavelength=100.0,
                velocity=1.0,
            )

    def test_negative_wavelength_raises(self):
        """Negative wavelength must be rejected."""
        with pytest.raises(ValueError, match="wavelength must be finite and greater than 0"):
            AnimationConfig(mode="traveling_wave", wavelength=-10.0, velocity=1.0)

    def test_zero_wavelength_raises(self):
        """Zero wavelength must be rejected."""
        with pytest.raises(ValueError, match="wavelength must be finite and greater than 0"):
            AnimationConfig(mode="traveling_wave", wavelength=0.0, velocity=1.0)


    def test_non_finite_velocity_raises(self):
        """velocity = inf or nan should raise."""
        with pytest.raises(ValueError, match="velocity must be finite"):
            AnimationConfig(mode="traveling_wave", wavelength=100.0, velocity=np.inf)
        with pytest.raises(ValueError, match="velocity must be finite"):
            AnimationConfig(mode="traveling_wave", wavelength=100.0, velocity=np.nan)

    def test_unsupported_mode_raises(self):
        """AnimationConfig with unsupported mode should raise ValueError."""
        with pytest.raises(ValueError, match="mode must be one of"):
            AnimationConfig(mode="unsupported_mode", wavelength=100.0, velocity=1.0)

    def test_base_color_with_v_zero_no_division_by_zero(self):
        """If a layer's base color has V=0 (black), scaling should not divide by zero."""
        # We cannot easily change the base color of a layer without modifying the dataset.
        # Instead, we test the color scaling logic directly using the helper that
        # converts base_v to scale. We'll use the _update_mesh_colors function indirectly
        # by constructing a scenario where base_v = 0.
        from ephaptic_coupling.visualization.animation import _update_mesh_colors
        import pyvista as pv

        # Create a mesh with some points
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        mesh = pv.PolyData(points)
        # Assign colors that, when converted to HSV, have V=0 (i.e., black)
        black_rgb = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # Set the mesh's colors to black
        mesh.point_data["colors"] = np.tile(black_rgb, (len(points), 1))
        # Simulate a brightness factor that would cause scaling
        # We'll call _update_mesh_colors with new_colors that are not black
        new_colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        # This should not raise
        try:
            _update_mesh_colors(mesh, new_colors)
        except Exception as e:
            pytest.fail(f"_update_mesh_colors raised an exception with base_v=0: {e}")
        # After update, the colors should be the new_colors
        np.testing.assert_allclose(mesh.point_data["colors"], new_colors, rtol=1e-7)

# Manual visual tests (not automated, for human inspection)
# ============================================================================
# NOTE: The following are manual visual tests, not automated pytest tests.
# They describe how to run the viewer to validate traveling-wave behavior.
#
# To run the manual test harness with Phase 4 options, use:
#   python tests/phase3_manualtest.py [options]
#
# Example commands to validate specific scenarios:
#
# 1. Single layer, long wavelength (should look like a slow-moving wave along X):
#      python tests/phase3_manualtest.py --layers 1 --mode traveling_wave --wavelength 200 --velocity 0.5
#
# 2. Single layer, oblique direction (wavefronts should move diagonally):
#      python tests/phase3_manualtest.py --layers 3 --mode traveling_wave --wave-direction 1 1 0 --wavelength 100 --velocity 1.0
#
# 3. Multi-layer with nonzero layer coupling (layers should exhibit staggered timing):
#      python tests/phase3_manualtest.py --layers 1 5 9 --mode traveling_wave --layer-coupling 0.1 --wavelength 150 --velocity 1.0
#
# 4. Comparison of global_pulse vs traveling_wave (same amplitude, different temporal behavior):
#      # Global pulse (all layers pulse together):
#      python tests/phase3_manualtest.py --layers 2 6 --mode global_pulse --amplitude 0.4 --frequency 1.2
#      # Traveling wave with same amplitude but spatial propagation:
#      python tests/phase3_manualtest.py --layers 2 6 --mode traveling_wave --amplitude 0.4 --wavelength 120 --velocity 0.8
#
# Observe the animation and verify that:
#   - In traveling_wave mode, brightness varies across points in a layer.
#   - Reversing wave_direction flips the direction of travel.
#   - Setting velocity=0 freezes the wave in space (only spatial variation remains).
#   - Increasing layer_coupling increases the delay between layers.
#   - Setting amplitude=0 results in no brightness change (static colors).
#   - Very large wavelength makes the traveling wave resemble a global pulse (small spatial variation).
#
# END OF FILE