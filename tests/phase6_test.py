import numpy as np

from ephaptic_coupling.FB_Synapse_Dataset.analysis import (
    extract_exr7_fb4y_synapses,
    generate_exr7_waveform,
    get_synapse_weights,
    select_synapses,
)


def test_get_synapse_weights_linear_decay():
    exr7_points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    fb_points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    weights = get_synapse_weights(exr7_points, fb_points, radius_um=2.0)

    assert weights.shape == (3,)
    np.testing.assert_allclose(weights, [1.0, 0.5, 0.0], rtol=1e-6)


def test_extract_exr7_fb4y_smoke():
    result = extract_exr7_fb4y_synapses("FB4Y")

    assert result["points"].ndim == 2
    assert result["points"].shape[1] == 3
    assert result["summary"]["count"] == result["points"].shape[0]


def test_select_synapses_by_prefixes():
    exr7_points, _, _, _ = select_synapses(pre_prefix="ExR7")
    fb4y_points, _, fb4y_neuron_names, _ = select_synapses(neuron_prefix="FB4Y")

    assert exr7_points.shape[0] > 0
    assert fb4y_points.shape[0] > 0
    assert fb4y_neuron_names.shape[0] == fb4y_points.shape[0]


def test_waveform_parameters_are_respected():
    t = np.array([0.0, 0.25], dtype=np.float32)
    wave = generate_exr7_waveform(t, amplitude=1.0, frequency=1.0, phase=0.0)

    np.testing.assert_allclose(wave, [0.0, 1.0], atol=1e-6)