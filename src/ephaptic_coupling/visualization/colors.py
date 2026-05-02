"""
visualization/colors.py

Color palette and colormap utilities for the Ephaptic Coupling project.
Provides per-layer colors for FB1–FB9 and an amplitude-to-color mapping.

This module must NOT load data, build geometry, or perform animation.
"""

# ---------------------------------------------------------------------------
# Layer color palette
# ---------------------------------------------------------------------------

# Visually distinct RGB colors (floats in [0, 1]) for FB layers 1–9.
# Based on ColorBrewer Set1 + extended palette.
LAYER_COLORS: dict[int, tuple[float, float, float]] = {
    1: (0.894, 0.102, 0.110),   # red
    2: (0.992, 0.553, 0.235),   # orange
    3: (1.000, 0.878, 0.196),   # yellow
    4: (0.302, 0.686, 0.290),   # green
    5: (0.173, 0.627, 0.773),   # sky blue
    6: (0.122, 0.471, 0.706),   # deep blue
    7: (0.596, 0.306, 0.639),   # purple
    8: (0.894, 0.467, 0.761),   # pink
    9: (0.651, 0.337, 0.157),   # brown
}


def get_layer_color(layer_number: int) -> tuple[float, float, float]:
    """Return the RGB color tuple for the given FB layer.

    Parameters
    ----------
    layer_number:
        Integer 1–9 identifying the FB layer.

    Returns
    -------
    tuple[float, float, float]
        RGB values each in [0, 1].

    Raises
    ------
    ValueError
        If layer_number is not in 1–9.
    """
    if layer_number not in LAYER_COLORS:
        raise ValueError(
            f"Invalid layer number {layer_number!r}. Must be an integer from 1 to 9."
        )
    return LAYER_COLORS[layer_number]


def amplitude_to_color(amplitude: float) -> tuple[float, float, float]:
    """Map a scalar amplitude in [0, 1] to an RGB color.

    Uses a blue → white → red gradient:
    - 0.0  → pure blue  (0, 0, 1)
    - 0.5  → white      (1, 1, 1)
    - 1.0  → pure red   (1, 0, 0)

    Parameters
    ----------
    amplitude:
        Scalar value in [0, 1]. Values outside this range are clamped.

    Returns
    -------
    tuple[float, float, float]
        RGB values each in [0, 1].
    """
    amplitude = max(0.0, min(1.0, float(amplitude)))
    if amplitude <= 0.5:
        t = amplitude * 2.0          # 0 → 1 in the blue→white half
        return (t, t, 1.0)
    else:
        t = (amplitude - 0.5) * 2.0  # 0 → 1 in the white→red half
        return (1.0, 1.0 - t, 1.0 - t)
