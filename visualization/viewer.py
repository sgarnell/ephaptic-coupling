"""
visualization/viewer.py

3D static viewer for Ephaptic Coupling synapse data.
Renders FB layers as point clouds using PyVista.

Requires:  pyvista, pyvistaqt, vtk
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from visualization.colors import get_layer_color
from visualization.utils import get_layer_points

# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------


def build_layer_mesh(points: np.ndarray, color: tuple) -> pv.PolyData:
    """Build a PyVista PolyData point cloud for a single FB layer.

    Parameters
    ----------
    points:
        Shape (N, 3) array of 3D synapse coordinates.
    color:
        RGB tuple with values in [0, 1], as returned by
        :func:`visualization.colors.get_layer_color`.

    Returns
    -------
    pv.PolyData
        Point cloud mesh. A ``"colors"`` point-data array of shape (N, 3)
        holds the per-point RGB values (same color for all points in the layer).
    """
    mesh = pv.PolyData(np.asarray(points, dtype=np.float32))
    n = len(points)
    colors = np.tile(np.array(color, dtype=np.float32), (n, 1))
    mesh.point_data["colors"] = colors
    return mesh


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------


def build_scene(
    layers: list[int],
    off_screen: bool = False,
    opacity: float = 0.25,
    point_size: int = 5,
    show_axes: bool = True,
    show_bounds: bool = False,
) -> pv.Plotter:
    """Build a PyVista Plotter containing the requested FB layers.

    The plotter is returned *without* being shown; call :func:`show_scene` or
    ``plotter.show()`` to display it.

    Parameters
    ----------
    layers:
        List of layer numbers (1–9) to include in the scene.
    off_screen:
        If True, render off-screen (no window). Useful for testing.
    opacity:
        Point opacity in [0, 1]. Values below 1.0 enable transparency so
        that dense synapse clouds show depth structure. Default: 0.25.
    point_size:
        Rendered point diameter in pixels. Default: 5.
    show_axes:
        If True, add an orientation axes widget in the bottom-left corner.
        The widget rotates with the scene so you always know which way
        X/Y/Z point. Default: True.
    show_bounds:
        If True, draw a labelled bounding box showing the world-space
        coordinate ranges of the loaded layers. Default: False.

    Returns
    -------
    pv.Plotter
        Configured plotter with one actor per requested layer.
    """
    plotter = pv.Plotter(off_screen=off_screen)

    for layer_num in layers:
        points = get_layer_points(layer_num)
        color = get_layer_color(layer_num)
        mesh = build_layer_mesh(points, color)
        mesh.set_active_scalars(None)  # prevent auto-scalar override of color=
        plotter.add_mesh(
            mesh,
            color=color,
            scalars=None,
            point_size=point_size,
            opacity=opacity,
            render_points_as_spheres=False,
        )

    if show_axes:
        plotter.add_axes(interactive=True, line_width=3)

    if show_bounds:
        plotter.show_bounds(
            grid=True,
            location="outer",
            ticks="both",
            font_size=10,
            xtitle="X",
            ytitle="Y",
            ztitle="Z",
        )

    plotter.reset_camera()
    return plotter


# ---------------------------------------------------------------------------
# Viewer entry point
# ---------------------------------------------------------------------------


def show_scene(
    layers: list[int] | None = None,
    off_screen: bool = False,
    opacity: float = 0.25,
    point_size: int = 5,
    show_axes: bool = True,
    show_bounds: bool = False,
) -> None:
    """Build and display the 3D scene for the requested FB layers.

    Parameters
    ----------
    layers:
        Layer numbers (1–9) to display. Defaults to all nine layers.
    off_screen:
        If True, render off-screen instead of opening an interactive window.
        Intended for testing only.
    opacity:
        Point opacity in [0, 1]. Default: 0.25.
    point_size:
        Rendered point diameter in pixels. Default: 5.
    show_axes:
        If True, add orientation axes widget. Default: True.
    show_bounds:
        If True, draw labelled bounding box. Default: False.
    """
    if layers is None:
        layers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    plotter = build_scene(
        layers,
        off_screen=off_screen,
        opacity=opacity,
        point_size=point_size,
        show_axes=show_axes,
        show_bounds=show_bounds,
    )
    plotter.show()


# ---------------------------------------------------------------------------
# Camera presets
# ---------------------------------------------------------------------------


def set_camera_top(plotter: pv.Plotter) -> None:
    """Set the camera to a top-down view (looking down the Z axis).

    Parameters
    ----------
    plotter:
        PyVista Plotter to modify in-place.
    """
    plotter.view_xy()


def set_camera_side(plotter: pv.Plotter) -> None:
    """Set the camera to a side view (looking along the X axis).

    Parameters
    ----------
    plotter:
        PyVista Plotter to modify in-place.
    """
    plotter.view_yz()


def set_camera_oblique(plotter: pv.Plotter) -> None:
    """Set the camera to a calibrated oblique view.

    Camera position and focal point were locked in from the live HUD
    at coordinates that give a clear angled view of all FB layers.

    Parameters
    ----------
    plotter:
        PyVista Plotter to modify in-place.
    """
    plotter.camera.position = (210.0, 715.0, 70.0)
    plotter.camera.focal_point = (160.0, 185.0, 160.0)
    plotter.camera.up = (0.0, 0.0, 1.0)
    plotter.reset_camera_clipping_range()
