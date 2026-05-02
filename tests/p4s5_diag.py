import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ephaptic_coupling.visualization.animation import AnimationConfig, _build_animation_scene, generate_frame


def _dump_plotter_state(plotter, label: str) -> None:
    print(f"[{label}] type(plotter)={type(plotter)}")
    print(f"[{label}] type(plotter.iren)={type(getattr(plotter, 'iren', None))}")
    print(
        f"[{label}] type(plotter.render_window)={type(getattr(plotter, 'render_window', None))}"
    )

    actors = getattr(getattr(plotter, "renderer", None), "actors", {})
    if hasattr(actors, "items"):
        actor_items = list(actors.items())
    else:
        actor_items = list(enumerate(list(actors)))

    print(f"[{label}] actor_count={len(actor_items)}")
    for actor_name, actor in actor_items:
        mapper = getattr(actor, "mapper", None)
        print(f"[{label}] actor={actor_name!r}, type(actor)={type(actor)}, type(mapper)={type(mapper)}")
        if mapper is None:
            continue

        array_name = getattr(mapper, "array_name", None)
        scalar_visibility = getattr(mapper, "scalar_visibility", None)
        scalar_mode = mapper.GetScalarModeAsString() if hasattr(mapper, "GetScalarModeAsString") else None
        color_mode = mapper.GetColorModeAsString() if hasattr(mapper, "GetColorModeAsString") else None
        print(f"[{label}] actor={actor_name!r}, mapper.array_name={array_name!r}")
        print(f"[{label}] actor={actor_name!r}, mapper.scalar_visibility={scalar_visibility!r}")
        print(f"[{label}] actor={actor_name!r}, mapper.scalar_mode={scalar_mode!r}")
        print(f"[{label}] actor={actor_name!r}, mapper.color_mode={color_mode!r}")

        input_data = mapper.GetInput() if hasattr(mapper, "GetInput") else None
        point_data = (
            input_data.GetPointData()
            if input_data is not None and hasattr(input_data, "GetPointData")
            else None
        )
        active_scalars = (
            point_data.GetScalars().GetName()
            if point_data is not None and hasattr(point_data, "GetScalars") and point_data.GetScalars() is not None
            else None
        )
        colors_array = (
            point_data.GetArray("colors")
            if point_data is not None and hasattr(point_data, "GetArray")
            else None
        )
        print(
            f"[{label}] actor={actor_name!r}, colors array exists={colors_array is not None}"
        )
        print(f"[{label}] actor={actor_name!r}, active point-data scalars={active_scalars!r}")

config = AnimationConfig(
    mode="traveling_wave",
    velocity=1.0,
    wavelength=50.0,
    amplitude=0.5,
    fps=30,
    duration=5.0,
)

app = QApplication.instance() or QApplication(sys.argv)
plotter = _build_animation_scene([1,5,9], off_screen=False, config=config)
_dump_plotter_state(plotter, "BOOT")

state = {"step": 0}

def cb() -> None:
    step = state["step"]
    print("TIMER FIRED step=", step)
    if step == 0:
        _dump_plotter_state(plotter, "FIRST_TICK")
    generate_frame(plotter, t=step / config.fps, config=config)
    state["step"] = step + 1
    if state["step"] >= 10:
        timer.stop()
        plotter.close()
        app.quit()

timer = QTimer(plotter)
timer.timeout.connect(cb)
timer.start(33)

plotter.show()
app.exec()
