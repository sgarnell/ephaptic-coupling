"""
tests/phase1_test.py

Phase 1 sanity tests for the Ephaptic Coupling project.
Covers: loading, shape validation, value correctness, and global summary.

Run with:
    pytest tests/phase1_test.py -v
"""

import json
from pathlib import Path

import numpy as np
import pytest

from visualization.utils import (
    load_layer_npz,
    load_all_layers,
    get_layer_points,
    get_layer_bounds,
    get_centroid,
    get_presynaptic_names,
    validate_schema,
)

LAYER_NUMBERS = list(range(1, 10))
REQUIRED_KEYS = {"points", "pre_names", "bounds", "centroid"}


def preview_array(arr, max_rows=5):
    """Return a truncated list-of-lists preview for a numpy array."""
    arr_np = np.asarray(arr)
    preview = arr_np[:max_rows]
    return preview.tolist()


# ---------------------------------------------------------------------------
# 4.1 Load Test
# ---------------------------------------------------------------------------


class TestLoad:
    """Load each layer and confirm required keys are present."""

    @pytest.mark.parametrize("layer", LAYER_NUMBERS)
    def test_layer_loads_without_error(self, layer):
        """Each layer file must load cleanly."""
        data = load_layer_npz(layer)
        assert isinstance(data, dict), f"Layer {layer}: expected dict, got {type(data)}"

    @pytest.mark.parametrize("layer", LAYER_NUMBERS)
    def test_required_keys_present(self, layer):
        """Each layer must contain all required schema keys."""
        data = load_layer_npz(layer)
        missing = REQUIRED_KEYS - set(data.keys())
        assert not missing, f"Layer {layer}: missing keys {missing}"

    def test_load_all_layers_returns_nine(self):
        """load_all_layers() must return exactly 9 entries."""
        all_layers = load_all_layers()
        assert len(all_layers) == 9, f"Expected 9 layers, got {len(all_layers)}"
        assert set(all_layers.keys()) == set(LAYER_NUMBERS)

    @pytest.mark.parametrize("layer", LAYER_NUMBERS)
    def test_schema_validates(self, layer):
        """validate_schema must return True for every layer."""
        data = load_layer_npz(layer)
        assert validate_schema(data) is True, f"Layer {layer}: schema validation failed"


# ---------------------------------------------------------------------------
# 4.2 Shape Test
# ---------------------------------------------------------------------------


class TestShapes:
    """Verify array dimensions match the schema specification."""

    @pytest.mark.parametrize("layer", LAYER_NUMBERS)
    def test_points_shape_is_Nx3(self, layer):
        """points must be exactly N×3."""
        points = get_layer_points(layer)
        assert points.ndim == 2, f"Layer {layer}: points must be 2-D, got ndim={points.ndim}"
        assert points.shape[1] == 3, (
            f"Layer {layer}: points must have 3 columns, got {points.shape[1]}"
        )

    @pytest.mark.parametrize("layer", LAYER_NUMBERS)
    def test_bounds_length_is_6(self, layer):
        """bounds must have exactly 6 elements."""
        bounds = get_layer_bounds(layer)
        assert len(bounds) == 6, f"Layer {layer}: bounds length must be 6, got {len(bounds)}"

    @pytest.mark.parametrize("layer", LAYER_NUMBERS)
    def test_centroid_length_is_3(self, layer):
        """centroid must have exactly 3 elements."""
        centroid = get_centroid(layer)
        assert len(centroid) == 3, (
            f"Layer {layer}: centroid length must be 3, got {len(centroid)}"
        )

    @pytest.mark.parametrize("layer", LAYER_NUMBERS)
    def test_pre_names_length_matches_points(self, layer):
        """pre_names must have the same length as the number of point rows."""
        points = get_layer_points(layer)
        names = get_presynaptic_names(layer)
        assert len(names) == points.shape[0], (
            f"Layer {layer}: pre_names length ({len(names)}) != points rows ({points.shape[0]})"
        )


# ---------------------------------------------------------------------------
# 4.3 Value Test
# ---------------------------------------------------------------------------


class TestValues:
    """Verify data values are consistent and geometrically sensible."""

    @pytest.mark.parametrize("layer", LAYER_NUMBERS)
    def test_bounds_are_finite(self, layer):
        """All bounds values must be finite numbers."""
        bounds = get_layer_bounds(layer)
        assert all(np.isfinite(v) for v in bounds), (
            f"Layer {layer}: bounds contain non-finite values: {bounds}"
        )

    @pytest.mark.parametrize("layer", LAYER_NUMBERS)
    def test_centroid_within_bounds(self, layer):
        """Centroid must lie within the stated bounds."""
        bounds = get_layer_bounds(layer)
        centroid = get_centroid(layer)
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])

        assert xmin <= cx <= xmax, (
            f"Layer {layer}: centroid x={cx:.3f} outside bounds [{xmin:.3f}, {xmax:.3f}]"
        )
        assert ymin <= cy <= ymax, (
            f"Layer {layer}: centroid y={cy:.3f} outside bounds [{ymin:.3f}, {ymax:.3f}]"
        )
        assert zmin <= cz <= zmax, (
            f"Layer {layer}: centroid z={cz:.3f} outside bounds [{zmin:.3f}, {zmax:.3f}]"
        )

    @pytest.mark.parametrize("layer", LAYER_NUMBERS)
    def test_points_within_bounds(self, layer):
        """All point coordinates must lie within the stated bounds."""
        bounds = get_layer_bounds(layer)
        points = get_layer_points(layer)
        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        assert np.all(points[:, 0] >= xmin) and np.all(points[:, 0] <= xmax), (
            f"Layer {layer}: some x-coordinates are outside bounds [{xmin}, {xmax}]"
        )
        assert np.all(points[:, 1] >= ymin) and np.all(points[:, 1] <= ymax), (
            f"Layer {layer}: some y-coordinates are outside bounds [{ymin}, {ymax}]"
        )
        assert np.all(points[:, 2] >= zmin) and np.all(points[:, 2] <= zmax), (
            f"Layer {layer}: some z-coordinates are outside bounds [{zmin}, {zmax}]"
        )


# ---------------------------------------------------------------------------
# 4.4 Global Summary Test
# ---------------------------------------------------------------------------


class TestGlobalSummary:
    """Print a per-layer summary confirming dataset internal consistency."""

    def test_global_summary(self, capsys):
        """Load all layers and print a summary table."""
        all_layers = load_all_layers()
        json_output = {"layers": []}

        print("\n" + "=" * 60)
        print(f"{'Layer':<8} {'Synapses':>10} {'Unique Neurons':>15} {'Centroid':>30}")
        print("-" * 60)

        total_synapses = 0

        for layer_num in LAYER_NUMBERS:
            data = all_layers[layer_num]
            points = data["points"]
            pre_names = data["pre_names"]
            centroid = data["centroid"]
            bounds = data["bounds"]
            neuron_names = data.get("neuron_names", np.unique(pre_names))
            fb_neurons = [str(n) for n in np.asarray(neuron_names) if str(n).startswith(f"FB{layer_num}")]
            fb_neuron_names_preview = fb_neurons[:5]
            if len(fb_neurons) > 5:
                fb_neuron_names_preview = fb_neuron_names_preview + ["..."]

            n_synapses = points.shape[0]
            n_unique = len(np.unique(neuron_names))
            cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])
            total_synapses += n_synapses

            print(
                f"FB{layer_num:<6} {n_synapses:>10,} {n_unique:>15,} "
                f"  ({cx:.1f}, {cy:.1f}, {cz:.1f})"
            )

            print(
                f"{'':8} bounds: x=[{bounds[0]:.1f},{bounds[1]:.1f}] "
                f"y=[{bounds[2]:.1f},{bounds[3]:.1f}] "
                f"z=[{bounds[4]:.1f},{bounds[5]:.1f}]"
            )

            json_output["layers"].append(
                {
                    "layer": int(layer_num),
                    "num_synapses": int(n_synapses),
                    "num_unique_neurons": int(n_unique),
                    "centroid": [float(cx), float(cy), float(cz)],
                    "bounds": [float(v) for v in bounds],
                    "points_preview": preview_array(points, max_rows=5),
                    "pre_names_preview": [str(name) for name in pre_names[:5]],
                    "neuron_names_preview": [str(name) for name in np.asarray(neuron_names)[:5]],
                    "fb_neuron_names_preview": fb_neuron_names_preview,
                    "num_fb_neurons": int(len(fb_neurons)),
                }
            )

        print("-" * 60)
        print(f"{'TOTAL':<8} {total_synapses:>10,}")
        print("=" * 60)

        # Structural assertions
        assert len(all_layers) == 9, "Expected 9 layers in global summary."
        assert total_synapses > 0, "Total synapse count must be > 0."

        output_path = Path(__file__).parent / "phase1_output.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2)
