"""Analysis helpers for the FB synapse datasets.

The module keeps the Phase 5/6 data extraction and per-synapse weighting logic
small and local. It does not introduce a new package layer or animation engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree


_DATASET_PATH = Path(__file__).with_name("FB_incoming_all_synapses.npz")


def load_synapse_dataset() -> np.lib.npyio.NpzFile:
    """Load the archived synapse dataset shipped with the project."""
    return np.load(_DATASET_PATH, allow_pickle=True)


def _as_str_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=str)


def _contains_token(values: np.ndarray, token: str) -> np.ndarray:
    return np.char.find(_as_str_array(values), token) >= 0


def _expanded_neuron_labels(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    """Expand per-neuron labels to one label per synapse row."""
    neuron_names = _as_str_array(archive["neuron_names"])
    counts = np.asarray(archive["counts_per_neuron"], dtype=int)
    expanded = np.repeat(neuron_names, counts)

    points = np.asarray(archive["points"], dtype=np.float32)
    if expanded.shape[0] != points.shape[0]:
        raise ValueError("Expanded neuron labels do not match the synapse point count")

    return expanded


def select_synapses(
    pre_prefix: str | None = None,
    neuron_prefix: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return filtered synapses and the mask used to select them."""
    archive = load_synapse_dataset()
    points = np.asarray(archive["points"], dtype=np.float32)
    pre_names = _as_str_array(archive["pre_names"])
    neuron_names = _expanded_neuron_labels(archive)

    mask = np.ones(points.shape[0], dtype=bool)
    if pre_prefix is not None:
        mask &= _contains_token(pre_names, pre_prefix)
    if neuron_prefix is not None:
        mask &= np.char.startswith(neuron_names, neuron_prefix)

    return points[mask], pre_names[mask], neuron_names[mask], mask


def get_exr7_fb4_points() -> np.ndarray:
    """Return all synapse coordinates whose presynaptic name contains ExR7."""
    points, _, _, _ = select_synapses(pre_prefix="ExR7")
    return points


def get_fb_synapses_by_regions(
    region_prefixes: list[str] | tuple[str, ...] = ["FB4Y", "FB5AB"],
) -> tuple[np.ndarray, np.ndarray]:
    """Return synapses whose receiving neuron name matches one of the prefixes."""
    archive = load_synapse_dataset()
    points = np.asarray(archive["points"], dtype=np.float32)
    neuron_names = _expanded_neuron_labels(archive)
    prefixes = tuple(region_prefixes)
    mask = np.array([any(name.startswith(prefix) for prefix in prefixes) for name in neuron_names])
    return points[mask], neuron_names[mask]


def extract_exr7_fb4y_synapses(target_prefix: str = "FB4Y") -> dict:
    """Extract ExR7 synapses landing on the requested FB target region."""
    points, pre_names, neuron_names, mask = select_synapses(
        pre_prefix="ExR7",
        neuron_prefix=target_prefix,
    )
    stats = compute_spatial_stats(points)
    return {
        "points": points,
        "pre_names": pre_names,
        "neuron_names": neuron_names,
        "mask": mask,
        "summary": stats,
    }


def compute_spatial_stats(points: np.ndarray) -> dict:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return {
            "count": 0,
            "centroid": np.zeros(3, dtype=np.float32),
            "bounds": np.zeros(6, dtype=np.float32),
        }

    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    bounds = np.array(
        [minimum[0], maximum[0], minimum[1], maximum[1], minimum[2], maximum[2]],
        dtype=np.float32,
    )
    return {
        "count": int(points.shape[0]),
        "centroid": np.mean(points, axis=0),
        "bounds": bounds,
    }


def generate_exr7_waveform(
    t: np.ndarray,
    amplitude: float,
    frequency: float,
    phase: float = 0.0,
) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


def prepare_synapse_plotter(points: np.ndarray, weights: np.ndarray | None = None) -> pv.PolyData:
    cloud = pv.PolyData(np.asarray(points, dtype=np.float32))
    if weights is not None:
        cloud.point_data["ephaptic_weight"] = np.asarray(weights, dtype=np.float32)
    return cloud


def get_synapse_weights(
    exr7_points: np.ndarray,
    fb_points: np.ndarray,
    radius_um: float,
) -> np.ndarray:
    """Compute a per-FB-synapse linear proximity weight to the nearest ExR7 synapse."""
    exr7_points = np.asarray(exr7_points, dtype=np.float32)
    fb_points = np.asarray(fb_points, dtype=np.float32)

    weights = np.zeros(fb_points.shape[0], dtype=np.float32)
    if exr7_points.size == 0 or fb_points.size == 0 or radius_um <= 0:
        return weights

    tree = KDTree(exr7_points)
    distances, _ = tree.query(fb_points, k=1, distance_upper_bound=radius_um)
    valid_mask = np.isfinite(distances) & (distances < radius_um)
    weights[valid_mask] = np.clip(1.0 - distances[valid_mask] / radius_um, 0.0, 1.0)
    return weights


def perform_ephaptic_filtering(
    exr7_points: np.ndarray,
    fb_points: np.ndarray,
    radius_um: float = 5.0,
):
    tree = KDTree(fb_points)
    indices = tree.query_ball_point(exr7_points, r=radius_um)
    source_mask = np.array([len(idx) > 0 for idx in indices])
    target_mask = np.zeros(len(fb_points), dtype=bool)
    for match_list in indices:
        target_mask[np.asarray(match_list, dtype=int)] = True
    weights = get_synapse_weights(exr7_points, fb_points, radius_um)
    return exr7_points[source_mask], fb_points[target_mask], weights[target_mask]


def calculate_synaptic_response(
    exr7_points: np.ndarray,
    fb_points: np.ndarray,
    radius_um: float = 5.0,
    amplitude: float = 1.0,
):
    weights = amplitude * get_synapse_weights(exr7_points, fb_points, radius_um=radius_um)
    valid_mask = weights > 0.0
    return weights[valid_mask], valid_mask


def export_coupling_to_json(
    exr7_points: np.ndarray,
    fb_points: np.ndarray,
    output_path: str,
    radius_um: float = 5.0,
    freq: float = 10.0,
    duration_s: float = 1.0,
    fs: int = 1000,
):
    weights = get_synapse_weights(exr7_points, fb_points, radius_um=radius_um)
    total_weight = float(np.sum(weights))
    t = np.linspace(0, duration_s, int(duration_s * fs), endpoint=False)
    influence = total_weight * generate_exr7_waveform(t, amplitude=1.0, frequency=freq)
    payload = {
        "metadata": {"radius_um": radius_um, "frequency_hz": freq},
        "data": {"time": t.tolist(), "influence": influence.tolist()},
    }
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=4)


def aggregate_ephaptic_per_neuron(
    exr7_points: np.ndarray,
    fb_points: np.ndarray,
    fb_neuron_names: np.ndarray,
    radius_um: float,
    freq: float,
    duration_s: float,
    fs: int = 1000,
):
    weights = get_synapse_weights(exr7_points, fb_points, radius_um=radius_um)
    t = np.linspace(0, duration_s, int(duration_s * fs), endpoint=False)
    waveform = generate_exr7_waveform(t, amplitude=1.0, frequency=freq)

    results: dict[str, dict] = {}
    fb_neuron_names = _as_str_array(fb_neuron_names)

    for neuron_name in np.unique(fb_neuron_names):
        neuron_mask = fb_neuron_names == neuron_name
        neuron_weights = weights[neuron_mask]
        influence = float(np.sum(neuron_weights)) * waveform
        results[str(neuron_name)] = {
            "influence": influence.tolist(),
            "synapse_count": int(np.sum(neuron_mask)),
            "weights": neuron_weights.tolist(),
            "weight_sum": float(np.sum(neuron_weights)),
        }

    return results


def export_per_neuron_json(aggregation_dict: dict, output_path: str, metadata: dict):
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump({"metadata": metadata, "neurons": aggregation_dict}, file_handle, indent=4)


def diagnostic_visualization(exr7_all, fb_all, fb_names, fb_influenced):
    """Plot a simple overlay for ExR7, FB4Y, FB5AB, and influenced synapses."""
    import pyvista as pv
    from scipy.spatial import KDTree

    tree = KDTree(fb_all)
    _, idx = tree.query(fb_influenced)
    influ_names = np.asarray(fb_names)[idx]

    fb_names = np.asarray(fb_names).astype(str)
    influ_names = np.asarray(influ_names).astype(str)
    fb4_mask = np.char.startswith(fb_names, "FB4Y")
    fb5_mask = np.char.startswith(fb_names, "FB5AB")
    infl4_mask = np.char.startswith(influ_names, "FB4Y")
    infl5_mask = np.char.startswith(influ_names, "FB5AB")

    plotter = pv.Plotter()
    plotter.set_background("black")

    plotter.add_mesh(
        pv.PolyData(exr7_all),
        color="yellow",
        point_size=2,
        render_points_as_spheres=True,
        label="ExR7",
    )
    plotter.add_mesh(
        pv.PolyData(fb_all[fb4_mask]),
        color="blue",
        point_size=2,
        render_points_as_spheres=True,
        label="FB4Y",
    )
    plotter.add_mesh(
        pv.PolyData(fb_all[fb5_mask]),
        color="cyan",
        point_size=2,
        render_points_as_spheres=True,
        label="FB5AB",
    )
    plotter.add_mesh(
        pv.PolyData(fb_influenced[infl4_mask]),
        color="green",
        point_size=5,
        render_points_as_spheres=True,
        label="Influenced FB4Y",
    )
    plotter.add_mesh(
        pv.PolyData(fb_influenced[infl5_mask]),
        color="lime",
        point_size=5,
        render_points_as_spheres=True,
        label="Influenced FB5AB",
    )

    plotter.add_legend()
    plotter.show()