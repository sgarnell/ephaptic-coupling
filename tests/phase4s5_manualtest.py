#!/usr/bin/env python3
"""
Manual visual test harness for Phase-4 traveling-wave animation.

This script allows a human to visually inspect the traveling-wave animation
by constructing an AnimationConfig with user-specified parameters and
calling play_animation in interactive mode.
"""

import pyvista as pv
import argparse
from ephaptic_coupling.visualization import animation as animation_module




pv.OFF_SCREEN = False
pv.set_plot_theme("document") # optional but recommended



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual visual test for Phase-4 traveling-wave animation."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable animation telemetry output (default: off)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="List of FB layer numbers to display (default: 1 5 9)",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=1.0,
        help="Wave propagation speed (default: 1.0)",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=50.0,
        help="Spatial period of the wave in scene units (default: 50.0)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.5,
        help="Modulation depth A in (0, 1) (default: 0.5)",
    )
    parser.add_argument(
        "--phase-offset",
        type=float,
        default=0.0,
        help="Global phase offset applied to the waveform (default: 0.0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for playback (default: 30)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Total animation length in seconds (default: 5.0)",
    )

    args = parser.parse_args()

    animation_module.set_animation_telemetry(args.verbose)

    config = animation_module.AnimationConfig(
        mode="traveling_wave",
        velocity=args.velocity,
        wavelength=args.wavelength,
        amplitude=args.amplitude,
        phase_offset=args.phase_offset,
        fps=args.fps,
        duration=args.duration,
    )

    if args.verbose:
        print("Traveling-wave animation configuration:")
        print(f"  mode: {config.mode}")
        print(f"  layers: {args.layers}")
        print(f"  velocity: {config.velocity}")
        print(f"  wavelength: {config.wavelength}")
        print(f"  amplitude: {config.amplitude}")
        print(f"  phase_offset: {config.phase_offset}")
        print(f"  fps: {config.fps}")
        print(f"  duration: {config.duration}")
        print()

    animation_module.play_animation(args.layers, config, off_screen=False)


if __name__ == "__main__":
    main()
