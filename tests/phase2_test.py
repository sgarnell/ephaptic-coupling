"""
tests/phase2_test.py

Phase 2 test suite — static 3D viewer (PyVista).

Covers:
  - Mesh construction (build_layer_mesh)
  - Scene construction (build_scene)
  - Camera presets (set_camera_top/side/oblique)
  - Smoke test for show_scene in off-screen mode

Run with:
    python -m pytest tests/phase2_test.py -v
"""

import numpy as np
import pytest
import pyvista as pv

from visualization.viewer import (
    build_layer_mesh,
    build_scene,
    show_scene,
    set_camera_top,
    set_camera_side,
    set_camera_oblique,
)
from visualization.colors import get_layer_color, LAYER_COLORS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_points(n: int = 100) -> np.ndarray:
    """Return a float32 (N, 3) array of random points."""
    rng = np.random.default_rng(42)
    return rng.random((n, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# 5.1  Mesh construction tests
# ---------------------------------------------------------------------------

class TestBuildLayerMesh:
    """Mesh construction: build_layer_mesh returns a valid PolyData."""

    def test_returns_polydata(self):
        points = _random_points(100)
        color = get_layer_color(1)
        mesh = build_layer_mesh(points, color)
        assert isinstance(mesh, pv.PolyData)

    def test_point_count_matches_input(self):
        n = 73
        points = _random_points(n)
        color = get_layer_color(3)
        mesh = build_layer_mesh(points, color)
        assert mesh.n_points == n

    def test_color_array_present(self):
        points = _random_points(50)
        color = get_layer_color(5)
        mesh = build_layer_mesh(points, color)
        assert "colors" in mesh.point_data

    def test_color_array_shape(self):
        n = 60
        points = _random_points(n)
        color = get_layer_color(7)
        mesh = build_layer_mesh(points, color)
        colors = mesh.point_data["colors"]
        assert colors.shape == (n, 3)

    def test_color_array_values_correct(self):
        points = _random_points(40)
        color = get_layer_color(2)
        mesh = build_layer_mesh(points, color)
        colors = mesh.point_data["colors"]
        # Every row must equal the input color
        np.testing.assert_allclose(colors, np.tile(color, (40, 1)), atol=1e-5)

    def test_does_not_modify_input(self):
        points = _random_points(30)
        original = points.copy()
        build_layer_mesh(points, get_layer_color(1))
        np.testing.assert_array_equal(points, original)


# ---------------------------------------------------------------------------
# 5.2  Scene construction tests
# ---------------------------------------------------------------------------

class TestBuildScene:
    """Scene construction: build_scene returns a configured Plotter."""

    def test_returns_plotter(self):
        plotter = build_scene([1], off_screen=True)
        assert isinstance(plotter, pv.Plotter)
        plotter.close()

    def test_single_layer_actor_added(self):
        plotter = build_scene([1], off_screen=True)
        n_actors = plotter.renderer.GetActors().GetNumberOfItems()
        assert n_actors >= 1
        plotter.close()

    def test_multiple_layers_actor_count(self):
        layers = [1, 2, 3]
        plotter = build_scene(layers, off_screen=True)
        n_actors = plotter.renderer.GetActors().GetNumberOfItems()
        assert n_actors >= len(layers)
        plotter.close()

    def test_all_layers_no_exception(self):
        plotter = build_scene([1, 2, 3, 4, 5, 6, 7, 8, 9], off_screen=True)
        assert isinstance(plotter, pv.Plotter)
        plotter.close()

    def test_subset_of_layers_no_exception(self):
        for subset in ([1], [4, 5], [1, 5, 9]):
            plotter = build_scene(subset, off_screen=True)
            assert isinstance(plotter, pv.Plotter)
            plotter.close()


# ---------------------------------------------------------------------------
# 5.3  Camera preset tests
# ---------------------------------------------------------------------------

class TestCameraPresets:
    """Camera presets: each function modifies the plotter camera without error."""

    def _make_plotter(self) -> pv.Plotter:
        """Return an off-screen plotter with one layer loaded."""
        return build_scene([1], off_screen=True)

    def test_set_camera_top_no_exception(self):
        plotter = self._make_plotter()
        set_camera_top(plotter)
        plotter.close()

    def test_set_camera_side_no_exception(self):
        plotter = self._make_plotter()
        set_camera_side(plotter)
        plotter.close()

    def test_set_camera_oblique_no_exception(self):
        plotter = self._make_plotter()
        set_camera_oblique(plotter)
        plotter.close()

    def test_camera_presets_produce_distinct_positions(self):
        """Top, side, and oblique presets must yield different camera positions."""
        plotter = self._make_plotter()

        set_camera_top(plotter)
        pos_top = np.array(plotter.camera.position)

        set_camera_side(plotter)
        pos_side = np.array(plotter.camera.position)

        set_camera_oblique(plotter)
        pos_oblique = np.array(plotter.camera.position)

        plotter.close()

        # At least two of the three positions must differ
        all_same = (
            np.allclose(pos_top, pos_side, atol=1e-3)
            and np.allclose(pos_side, pos_oblique, atol=1e-3)
        )
        assert not all_same, (
            "All three camera presets produced identical positions; "
            "presets are not functioning correctly."
        )

    def test_camera_presets_camera_is_not_none(self):
        plotter = self._make_plotter()
        set_camera_top(plotter)
        assert plotter.camera is not None
        set_camera_side(plotter)
        assert plotter.camera is not None
        set_camera_oblique(plotter)
        assert plotter.camera is not None
        plotter.close()


# ---------------------------------------------------------------------------
# 5.4  Smoke test for show_scene
# ---------------------------------------------------------------------------

class TestShowSceneSmoke:
    """Smoke tests: show_scene completes without exceptions in off-screen mode."""

    def test_show_scene_single_layer(self):
        show_scene(layers=[1], off_screen=True)

    def test_show_scene_multiple_layers(self):
        show_scene(layers=[1, 5, 9], off_screen=True)

    def test_show_scene_all_layers(self):
        show_scene(layers=[1, 2, 3, 4, 5, 6, 7, 8, 9], off_screen=True)

    def test_show_scene_default_layers(self):
        """Calling show_scene with no arguments should default to all 9 layers."""
        show_scene(off_screen=True)
