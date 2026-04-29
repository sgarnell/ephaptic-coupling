"""
tests/phase3_manualtest.py

Manual inspection harness for the Phase 3 animation engine.

This is NOT a pytest test. Run it directly from the project root:

    python tests/phase3_manualtest.py
    python tests/phase3_manualtest.py --layers 1 2 5 --camera top
    python tests/phase3_manualtest.py --layers all --duration 4 --fps 30 --amplitude 0.35 --frequency 1.5
    python tests/phase3_manualtest.py --layers 3 7 --bg white --window 1600 1000

Controls inside the viewer window:
    q / Esc     close the window
    mouse drag  rotate the scene
    scroll      zoom
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from visualization.animation import AnimationConfig, _build_animation_scene, generate_frame
from visualization.viewer import set_camera_oblique, set_camera_side, set_camera_top

ALL_LAYERS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
CAMERA_PRESETS = ("top", "side", "oblique")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="phase3_manualtest.py",
        description="Manual inspection viewer for Phase 3 animation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--layers",
        nargs="+",
        default=["all"],
        metavar="LAYER",
        help=(
            'Layer numbers to display (1-9), or the keyword "all". '
            "Examples: --layers 1 2 5   --layers all"
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        metavar="SEC",
        help="Animation duration in seconds. Default: 3.0.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        metavar="FPS",
        help="Frames per second. Default: 24.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.5,
        metavar="A",
        help="Brightness modulation amplitude. Default: 0.5.",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=1.0,
        metavar="HZ",
        help="Brightness modulation frequency in Hz. Default: 1.0.",
    )
    parser.add_argument(
        "--camera",
        choices=CAMERA_PRESETS,
        default="oblique",
        help="Camera preset to apply on startup. Default: oblique.",
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
        "--opacity",
        type=float,
        default=0.25,
        metavar="ALPHA",
        help="Point opacity in [0, 1]. Default: 0.25.",
    )
    parser.add_argument(
        "--point-size",
        type=int,
        default=5,
        metavar="PX",
        help="Point size in pixels. Default: 5.",
    )

    return parser.parse_args(argv)


def resolve_layers(raw: list[str]) -> list[int]:
    """Convert raw --layers values into a sorted list of unique layer numbers."""
    if len(raw) == 1 and raw[0].lower() == "all":
        return ALL_LAYERS

    layers: list[int] = []
    for token in raw:
        try:
            n = int(token)
        except ValueError:
            print(f"[error] Invalid layer value: {token!r}. Must be an integer 1-9 or 'all'.")
            sys.exit(1)
        if n not in range(1, 10):
            print(f"[error] Layer {n} is out of range. Must be 1-9.")
            sys.exit(1)
        layers.append(n)

    return sorted(set(layers))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    layers = resolve_layers(args.layers)

    if args.duration <= 0:
        print("[error] --duration must be > 0")
        sys.exit(1)
    if args.fps <= 0:
        print("[error] --fps must be > 0")
        sys.exit(1)
    if args.window[0] <= 0 or args.window[1] <= 0:
        print("[error] --window width and height must be > 0")
        sys.exit(1)

    config = AnimationConfig(
        duration=args.duration,
        fps=args.fps,
        amplitude=args.amplitude,
        frequency=args.frequency,
    )

    print(f"Layers     : {layers}")
    print(f"Camera     : {args.camera}")
    print(f"Duration   : {args.duration} s")
    print(f"FPS        : {args.fps}")
    print(f"Amplitude  : {args.amplitude}")
    print(f"Frequency  : {args.frequency} Hz")
    print(f"Background : {args.bg!r}")
    print(f"Window     : {args.window[0]} x {args.window[1]}")
    print(f"Point size : {args.point_size} px")
    print(f"Opacity    : {args.opacity}")
    print()

    print("Loading synapse data and building animation scene ...")
    plotter = _build_animation_scene(
        layers,
        off_screen=False,
        opacity=args.opacity,
        point_size=args.point_size,
    )

    plotter.window_size = args.window

    try:
        plotter.set_background(args.bg)
    except Exception as exc:
        print(f"[warning] Could not set background {args.bg!r}: {exc}")

    camera_fn = {
        "top": set_camera_top,
        "side": set_camera_side,
        "oblique": set_camera_oblique,
    }
    camera_fn[args.camera](plotter)

    n_frames = max(1, int(config.duration * config.fps))
    duration_ms = max(1, int(1000 / config.fps))

    state = {
        "frame": 0,
        "running": True,
        "stopped": False,
        "loop": True,
        "timer_id": None,
        "timer_started": False,
    }

    # Render initial frame and place a status HUD so frame progression is visible.
    generate_frame(plotter, t=0.0, config=config)
    hud = plotter.add_text(
        "",
        position=(0.01, 0.96),
        font_size=10,
        color="white",
        shadow=True,
    )

    def _refresh_hud() -> None:
        t = state["frame"] / config.fps
        if state["stopped"]:
            mode = "STOPPED"
        elif state["running"]:
            mode = "RUNNING"
        else:
            mode = "PAUSED"
        loop_str = "ON" if state["loop"] else "OFF"
        hud.SetInput(
            f"Phase 3 Animation  |  {mode}  |  frame {state['frame']}/{n_frames}  |  t={t:0.2f}s  |  loop={loop_str}"
        )

    _refresh_hud()

    def _ensure_animation_timer() -> None:
        if state["timer_started"]:
            return
        vtk_iren = plotter.iren.interactor
        vtk_iren.AddObserver("TimerEvent", _on_timer)
        state["timer_id"] = vtk_iren.CreateRepeatingTimer(duration_ms)
        state["timer_started"] = True

    def _set_run(_checked: bool) -> None:
        _ensure_animation_timer()
        state["running"] = True
        if state["stopped"] or state["frame"] >= n_frames:
            state["frame"] = 0
            state["stopped"] = False
            generate_frame(plotter, t=0.0, config=config)
        _refresh_hud()

    def _set_pause(_checked: bool) -> None:
        state["running"] = False
        state["stopped"] = False
        _refresh_hud()

    def _set_stop(_checked: bool) -> None:
        state["running"] = False
        state["stopped"] = True
        state["frame"] = 0
        generate_frame(plotter, t=0.0, config=config)
        _refresh_hud()

    def _set_loop(checked: bool) -> None:
        state["loop"] = bool(checked)
        _refresh_hud()

    plotter.add_text("Run", position=(0.01, 0.88), font_size=10, color="white")
    plotter.add_checkbox_button_widget(
        _set_run,
        value=False,
        position=(10, 525),
        size=24,
        color_on="limegreen",
        color_off="dimgray",
    )

    plotter.add_text("Pause", position=(0.01, 0.82), font_size=10, color="white")
    plotter.add_checkbox_button_widget(
        _set_pause,
        value=False,
        position=(10, 485),
        size=24,
        color_on="gold",
        color_off="dimgray",
    )

    plotter.add_text("Stop", position=(0.01, 0.76), font_size=10, color="white")
    plotter.add_checkbox_button_widget(
        _set_stop,
        value=False,
        position=(10, 445),
        size=24,
        color_on="tomato",
        color_off="dimgray",
    )

    plotter.add_text("Loop", position=(0.01, 0.70), font_size=10, color="white")
    plotter.add_checkbox_button_widget(
        _set_loop,
        value=True,
        position=(10, 405),
        size=24,
        color_on="deepskyblue",
        color_off="dimgray",
    )

    def _on_timer(_obj=None, _event=None) -> None:
        if not state["running"]:
            return

        if state["frame"] >= n_frames:
            if state["loop"]:
                state["frame"] = 0
            else:
                state["running"] = False
                _refresh_hud()
                return

        t = state["frame"] / config.fps
        generate_frame(plotter, t, config)
        state["frame"] += 1
        _refresh_hud()
    plotter.show()


if __name__ == "__main__":
    main()
