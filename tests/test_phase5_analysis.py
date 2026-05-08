import numpy as np
import pytest
from src.ephaptic_coupling.FB_Synapse_Dataset.analysis import (
    compute_spatial_stats,
    generate_exr7_waveform,
    prepare_synapse_plotter
)

def test_filtering_logic():
    # Mock data as separate parallel arrays per schema
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    pre_names = np.array(['ExR7', 'InR7', 'ExR7'])
    layer = np.array([4, 4, 3])
    
    # Verify masking logic
    mask = (pre_names == 'ExR7') & (layer == 4)
    filtered = points[mask]
    
    assert len(filtered) == 1
    assert np.allclose(filtered[0], [0, 0, 0])

def test_spatial_stats():
    points = np.array([[0, 0, 0], [2, 2, 2]])
    stats = compute_spatial_stats(points)
    assert stats['count'] == 2
    assert np.allclose(stats['centroid'], [1.0, 1.0, 1.0])

def test_waveform_phase():
    """Verify phase offset by checking t=0 value."""
    t = np.array([0.0])
    # sin(phase)
    wave = generate_exr7_waveform(t, amplitude=1.0, frequency=10.0, phase=np.pi/2)
    assert np.isclose(wave[0], 1.0)

def test_waveform_frequency_zero_crossings():
    """Verify frequency by counting zero-crossings.
    10Hz sine over 1s should have 20 crossings (one up, one down per period)."""
    freq = 10.0
    t = np.linspace(0, 1.0, 1000)
    wave = generate_exr7_waveform(t, amplitude=1.0, frequency=freq)

    # Sign changes indicate zero crossings
    crossings = np.where(np.diff(np.sign(wave)))[0]
    # For frequency F, crossings = 2 * F
    assert len(crossings) == int(2 * freq)

def test_visualization_prep():
    """Verify that PolyData is initialized with the correct number of points."""
    points = np.array([[0, 0, 0], [1, 1, 1]])
    cloud = prepare_synapse_plotter(points)
    # PyVista object initialization check
    assert cloud.n_points == 2

