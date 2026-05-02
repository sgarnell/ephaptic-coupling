"""
visualization/utils.py

Data-loading utilities and extraction helpers for the Ephaptic Coupling project.
Provides a stable, read-only API over the FB_Synapse_Dataset/ .npz files.

No visualization, animation, or color-mapping code belongs here.
"""

import pathlib
import numpy as np

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_DATASET_DIR = pathlib.Path(__file__).parent.parent / "FB_Synapse_Dataset"

_REQUIRED_KEYS = {"points", "pre_names", "bounds", "centroid"}
_OPTIONAL_KEYS = {"layer", "neuron_names", "counts_per_neuron", "density_grid", "density_meta"}


def _layer_path(layer_number: int) -> pathlib.Path:
    """Return the absolute path to the .npz file for the given layer number."""
    return _DATASET_DIR / f"FB{layer_number}_incoming_synapses.npz"


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------


def load_layer_npz(layer_number: int) -> dict:
    """Load a single FB layer .npz file.

    Parameters
    ----------
    layer_number:
        Integer 1–9 identifying the FB layer.

    Returns
    -------
    dict
        All arrays contained in the file, keyed by field name.

    Raises
    ------
    FileNotFoundError
        If the expected .npz file does not exist.
    ValueError
        If layer_number is outside the valid range.
    """
    if layer_number not in range(1, 10):
        raise ValueError(f"layer_number must be between 1 and 9, got {layer_number!r}.")

    path = _layer_path(layer_number)
    if not path.exists():
        raise FileNotFoundError(
            f"Layer file not found: {path}\n"
            f"Ensure FB_Synapse_Dataset/ is present in the project root."
        )

    raw = np.load(path, allow_pickle=True)
    return {key: raw[key] for key in raw.files}


def load_all_layers() -> dict:
    """Load all FB layers (FB1–FB9).

    Returns
    -------
    dict[int, dict]
        Mapping of layer index → layer data dict.
    """
    return {layer: load_layer_npz(layer) for layer in range(1, 10)}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _require_key(layer_data: dict, key: str, layer_number: int | str = "?") -> np.ndarray:
    """Raise KeyError with a helpful message if key is missing from layer_data."""
    if key not in layer_data:
        raise KeyError(
            f"Required field '{key}' is missing from layer {layer_number} data. "
            f"Available keys: {list(layer_data.keys())}"
        )
    return layer_data[key]


def get_layer_points(layer_number: int) -> np.ndarray:
    """Return the N×3 points array for the given layer.

    Parameters
    ----------
    layer_number:
        Integer 1–9.

    Returns
    -------
    np.ndarray
        Shape (N, 3) — 3D synapse coordinates.
    """
    data = load_layer_npz(layer_number)
    points = _require_key(data, "points", layer_number)
    return np.asarray(points, dtype=np.float32)


def get_layer_bounds(layer_number: int) -> tuple:
    """Return the spatial bounds for the given layer as a tuple.

    Returns
    -------
    tuple
        (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    data = load_layer_npz(layer_number)
    bounds = _require_key(data, "bounds", layer_number)
    return tuple(float(v) for v in bounds)


def get_centroid(layer_number: int) -> np.ndarray:
    """Return the centroid (mean coordinate) for the given layer.

    Returns
    -------
    np.ndarray
        Shape (3,) — [x, y, z].
    """
    data = load_layer_npz(layer_number)
    return _require_key(data, "centroid", layer_number)


def get_presynaptic_names(layer_number: int) -> list:
    """Return the list of presynaptic neuron names for each synapse.

    Returns
    -------
    list[str]
        Length N — one name per synapse row.
    """
    data = load_layer_npz(layer_number)
    pre = _require_key(data, "pre_names", layer_number)
    return [str(name) for name in pre]


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_schema(layer_data: dict) -> bool:
    """Validate that layer_data conforms to the expected schema.

    Checks
    ------
    - All required keys are present.
    - ``points`` is 2-D with 3 columns.
    - ``bounds`` has exactly 6 elements and contains finite values.
    - ``centroid`` has exactly 3 elements.
    - ``pre_names`` length matches the number of rows in ``points``.

    Parameters
    ----------
    layer_data:
        Dict as returned by :func:`load_layer_npz`.

    Returns
    -------
    bool
        True if the data is valid.

    Raises
    ------
    ValueError
        With a descriptive message if any validation step fails.
    """
    # --- required keys ---
    missing = _REQUIRED_KEYS - set(layer_data.keys())
    if missing:
        raise ValueError(f"Schema validation failed: missing required keys {missing}.")

    points = layer_data["points"]
    bounds = layer_data["bounds"]
    centroid = layer_data["centroid"]
    pre_names = layer_data["pre_names"]

    # --- points shape ---
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Schema validation failed: 'points' must be N×3, got shape {points.shape}."
        )

    n_synapses = points.shape[0]

    # --- bounds ---
    if len(bounds) != 6:
        raise ValueError(
            f"Schema validation failed: 'bounds' must have 6 elements, got {len(bounds)}."
        )
    if not np.all(np.isfinite(bounds.astype(float))):
        raise ValueError("Schema validation failed: 'bounds' contains non-finite values.")

    # --- centroid ---
    if len(centroid) != 3:
        raise ValueError(
            f"Schema validation failed: 'centroid' must have 3 elements, got {len(centroid)}."
        )

    # --- pre_names length ---
    if len(pre_names) != n_synapses:
        raise ValueError(
            f"Schema validation failed: 'pre_names' length ({len(pre_names)}) "
            f"does not match number of points ({n_synapses})."
        )

    return True
