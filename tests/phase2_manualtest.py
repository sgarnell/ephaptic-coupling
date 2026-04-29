"""
tests/phase2_manualtest.py

Manual inspection harness for the Phase 2 static 3D viewer.

This is NOT a pytest test.  Run it directly from the project root:

    python tests/phase2_manualtest.py
    python tests/phase2_manualtest.py --layers 1 2 5 --camera oblique
    python tests/phase2_manualtest.py --layers all --camera top --bg white
    python tests/phase2_manualtest.py --layers 3 7 --camera side --point-size 6 --window 1600 1200
    python tests/phase2_manualtest.py --layers all --show-bounds --show-legend --show-camera

Controls inside the viewer window:
    q / Esc     close the window
    mouse drag  rotate the scene
    scroll      zoom
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root (parent of tests/) is on sys.path so that
# `visualization` is importable when the script is run directly:
#   python tests/phase2_manualtest.py ...
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

ALL_LAYERS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
CAMERA_PRESETS = ("top", "side", "oblique")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="phase2_manualtest.py",
        description="Manual inspection viewer for FB synapse layers (Phase 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--layers",
        nargs="+",
        default=["all"],
        metavar="LAYER",
        help=(
            'Layer numbers to display (1–9), or the keyword "all". '
            "Examples: --layers 1 2 5   --layers all"
        ),
    )
    parser.add_argument(
        "--camera",
        choices=CAMERA_PRESETS,
        default="oblique",
        help="Camera preset to apply on startup. Default: oblique.",
    )
    parser.add_argument(
        "--point-size",
        type=int,
        default=5,
        metavar="PX",
        help="Point size in pixels for rendered synapses. Default: 5.",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.25,
        metavar="ALPHA",
        help=(
            "Point opacity in [0, 1]. Values below 1.0 reveal depth structure "
            "in dense clouds. Default: 0.25."
        ),
    )
    parser.add_argument(
        "--bg",
        default="black",
        metavar="COLOR",
        help=(
            "Background color name or hex string (e.g. black, white, #1a1a2e). "
            "Default: black."
        ),
    )
    parser.add_argument(
        "--window",
        nargs=2,
        type=int,
        default=[1280, 900],
        metavar=("WIDTH", "HEIGHT"),
        help="Window size in pixels. Default: 1280 900.",
    )
    parser.add_argument(
        "--camera-pos",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Override camera position in world coordinates. "
            "Read the values from the live HUD and paste them here. "
            "Example: --camera-pos 155.3 727.9 165.4"
        ),
    )
    parser.add_argument(
        "--camera-focus",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Override camera focal point in world coordinates. "
            "Example: --camera-focus 159.6 184.0 158.3"
        ),
    )
    parser.add_argument(
        "--camera-up",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Override camera up vector. Default when --camera-pos is given: 0 0 1. "
            "Example: --camera-up 0 0 1"
        ),
    )
    parser.add_argument(
        "--no-axes",
        action="store_true",
        default=False,
        help="Hide the orientation axes widget (shown by default).",
    )
    parser.add_argument(
        "--show-bounds",
        action="store_true",
        default=False,
        help="Draw a labelled bounding box showing X/Y/Z coordinate ranges.",
    )
    parser.add_argument(
        "--show-legend",
        action="store_true",
        default=False,
        help="Overlay a colour legend mapping each displayed layer to its colour.",
    )
    parser.add_argument(
        "--show-camera",
        action="store_true",
        default=False,
        help="Show a live camera-position readout that updates as you rotate.",
    )
    parser.add_argument(
        "--show-rotation",
        action="store_true",
        default=False,
        help=(
            "Show three on-screen rotation buttons (Slow / Medium / Fast). "
            "Click a button to start turntable auto-rotation; click again to stop."
        ),
    )

    return parser.parse_args(argv)

def resolve_layers(raw: list[str]) -> list[int]:
    """Convert the raw --layers values into a sorted list of integers."""
    if len(raw) == 1 and raw[0].lower() == "all":
        return ALL_LAYERS

    layers: list[int] = []
    for token in raw:
        try:
            n = int(token)
        except ValueError:
            print(f"[error] Invalid layer value: {token!r}. Must be an integer 1–9 or 'all'.")
            sys.exit(1)
        if n not in range(1, 10):
            print(f"[error] Layer {n} is out of range. Must be 1–9.")
            sys.exit(1)
        layers.append(n)

    return sorted(set(layers))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    layers = resolve_layers(args.layers)

    print(f"Layers     : {layers}")
    print(f"Camera     : {args.camera}")
    if args.camera_pos:
        print(f"  pos      : {args.camera_pos}")
    if args.camera_focus:
        print(f"  focus    : {args.camera_focus}")
    if args.camera_up:
        print(f"  up       : {args.camera_up}")
    print(f"Point size : {args.point_size} px")
    print(f"Opacity    : {args.opacity}")
    print(f"Background : {args.bg!r}")
    print(f"Window     : {args.window[0]} × {args.window[1]}")
    print(f"Axes       : {'off' if args.no_axes else 'on'}")
    print(f"Bounds     : {'on' if args.show_bounds else 'off'}")
    print(f"Legend     : {'on' if args.show_legend else 'off'}")
    print(f"Camera HUD : {'on' if args.show_camera else 'off'}")
    print(f"Rotation   : {'on' if args.show_rotation else 'off'}")
    print()

    # ------------------------------------------------------------------
    # Lazy import so that argparse --help works without pyvista installed.
    # ------------------------------------------------------------------
    from visualization.viewer import (
        build_scene,
        set_camera_oblique,
        set_camera_side,
        set_camera_top,
    )

    _CAMERA_FN = {
        "top": set_camera_top,
        "side": set_camera_side,
        "oblique": set_camera_oblique,
    }

    print("Loading synapse data and building scene …")
    plotter = build_scene(
        layers,
        off_screen=False,
        opacity=args.opacity,
        point_size=args.point_size,
        show_axes=not args.no_axes,
        show_bounds=args.show_bounds,
    )

    # Apply window size
    plotter.window_size = args.window

    # Apply background color
    try:
        plotter.set_background(args.bg)
    except Exception as exc:
        print(f"[warning] Could not set background {args.bg!r}: {exc}")

    # Layer colour legend
    if args.show_legend:
        from visualization.colors import LAYER_COLORS
        legend_entries = [[f"FB{n}", LAYER_COLORS[n]] for n in layers]
        plotter.add_legend(
            legend_entries,
            bcolor=(0.05, 0.05, 0.05),
            border=True,
            loc="lower right",
            size=(0.10, 0.022 * len(layers)),
        )

    # Apply camera preset, then optionally override with exact coordinates
    _CAMERA_FN[args.camera](plotter)
    if args.camera_pos is not None:
        plotter.camera.position = tuple(args.camera_pos)
    if args.camera_focus is not None:
        plotter.camera.focal_point = tuple(args.camera_focus)
    if args.camera_up is not None:
        plotter.camera.up = tuple(args.camera_up)
    elif args.camera_pos is not None:
        plotter.camera.up = (0.0, 0.0, 1.0)  # sane default when pos is set
    if args.camera_pos is not None:
        plotter.reset_camera_clipping_range()

    # Live camera-position HUD — updates on every render (drag, zoom, key)
    if args.show_camera:
        # Use fractional viewport position (not a named corner) so the returned
        # actor is a plain vtkTextActor that supports SetInput().
        hud = plotter.add_text(
            "Camera: —",
            position=(0.01, 0.02),
            font_size=9,
            color="white",
            shadow=True,
        )

        def _update_hud(_obj=None, _event=None) -> None:
            pos = plotter.camera.position
            fp  = plotter.camera.focal_point
            hud.SetInput(
                f"Camera  x={pos[0]:7.1f}  y={pos[1]:7.1f}  z={pos[2]:7.1f}\n"
                f"Focus   x={fp[0]:7.1f}   y={fp[1]:7.1f}   z={fp[2]:7.1f}"
            )

        # RenderEvent fires on every frame — live update as you drag/zoom
        plotter.iren.add_observer("RenderEvent", _update_hud)
        _update_hud()  # populate immediately before first interaction

    # Auto-rotation buttons  (Slow / Medium / Fast)
    if args.show_rotation:
        # Degrees of azimuth per timer tick (timer fires every 30 ms ≈ 33 fps).
        _SPEEDS = [
            ("Slow",   0.20, "cyan"),
            ("Medium", 0.60, "orange"),
            ("Fast",   1.50, "tomato"),
        ]
        _rot  = {"step": 0.0, "active_idx": None, "timer_started": False}  # shared mutable state
        _btns = []               # widget refs for mutual-exclusion logic

        def _rotation_tick(_obj=None, _event=None) -> None:
            if _rot["step"] != 0.0:
                plotter.renderer.GetActiveCamera().Azimuth(_rot["step"])
                plotter.reset_camera_clipping_range()
                plotter.render()

        def _ensure_rotation_timer() -> None:
            if _rot["timer_started"]:
                return
            vtk_iren = plotter.iren.interactor
            vtk_iren.AddObserver("TimerEvent", _rotation_tick)
            vtk_iren.CreateRepeatingTimer(30)  # 30 ms ≈ 33 fps
            _rot["timer_started"] = True

        def _make_rot_cb(step_val: float, btn_idx: int):
            def _cb(state: bool) -> None:
                if state:
                    _ensure_rotation_timer()
                    _rot["active_idx"] = btn_idx
                    _rot["step"] = step_val
                    # Uncheck the other two buttons (radio-button behaviour)
                    for i, w in enumerate(_btns):
                        if i != btn_idx:
                            w.GetRepresentation().SetState(0)
                            w.Modified()
                else:
                    # Ignore "off" callbacks caused by turning off non-active buttons.
                    if _rot["active_idx"] == btn_idx:
                        _rot["active_idx"] = None
                        _rot["step"] = 0.0
            return _cb

        win_w, win_h = args.window
        btn_size = 26
        btn_gap = 10
        btn_x = win_w - btn_size - 14  # near right edge

        for i, (label, step, col_on) in enumerate(_SPEEDS):
            btn_y = win_h - 55 - i * (btn_size + btn_gap)
            widget = plotter.add_checkbox_button_widget(
                _make_rot_cb(step, i),
                value=False,
                position=(btn_x, btn_y),
                size=btn_size,
                border_size=3,
                color_on=col_on,
                color_off=(0.5, 0.5, 0.5),
                background_color=(0.08, 0.08, 0.08),
            )
            _btns.append(widget)

            # Use pixel coordinates so labels anchor with the buttons.
            plotter.add_text(
                label,
                position=(btn_x - 74, btn_y + 4),
                font_size=10,
                color="white",
                shadow=True,
            )

        plotter.add_text(
            "Auto-Rotate",
            position=(btn_x - 92, win_h - 28),
            font_size=9,
            color="white",
            shadow=True,
        )

    print("Opening viewer … (close the window or press Q to quit)")
    plotter.show()


if __name__ == "__main__":
    main()
