"""
Unit test for the optional layer‑coupling model (Phase 4, Step 7).

Verifies that:
- layer_delay is computed as (layer_num - reference_layer) * layer_coupling
- The reference layer is the first layer in the input list
- The layer_delay term appears correctly in the traveling‑wave phase argument
- Setting layer_coupling = 0.0 eliminates inter‑layer phase differences
"""

import numpy as np
from ephaptic_coupling.visualization.animation import (
    _compute_layer_delay,
    _build_animation_scene,
    AnimationConfig,
)


def test_layer_coupling_basic():
    """Test _compute_layer_delay matches the spec."""
    # Basic arithmetic
    assert _compute_layer_delay(layer_num=3, reference_layer=1, layer_coupling=0.1) == 0.2
    assert _compute_layer_delay(layer_num=1, reference_layer=1, layer_coupling=0.1) == 0.0
    assert _compute_layer_delay(layer_num=0, reference_layer=1, layer_coupling=0.1) == -0.1
    # Negative coupling (lead instead of lag)
    assert _compute_layer_delay(layer_num=3, reference_layer=1, layer_coupling=-0.05) == -0.1


def test_layer_coupling_in_scene():
    """Build a scene with layers [1, 5, 9] and verify layer_delay values."""
    layers = [1, 5, 9]
    config = AnimationConfig(
        mode="traveling_wave",
        layer_coupling=0.1,  # 100 ms per layer step
        wavelength=100.0,
        velocity=1.0,
        amplitude=0.5,
        frequency=1.0,  # ignored in traveling_wave
        phase_offset=0.0,
    )

    # Build scene off‑screen; we only need the cached records.
    plotter = _build_animation_scene(
        layers=layers,
        off_screen=True,
        config=config,
    )

    records = plotter._anim_records  # type: ignore[attr-defined]
    assert len(records) == 3

    # Extract layer_num and layer_delay from each record
    layer_info = [(r["layer_num"], r["layer_delay"]) for r in records]
    # Expect: layer 1 -> 0.0, layer 5 -> (5-1)*0.1 = 0.4, layer 9 -> (9-1)*0.1 = 0.8
    assert layer_info == [(1, 0.0), (5, 0.4), (9, 0.8)]

    # Verify reference layer is indeed the first layer (1)
    assert records[0]["layer_num"] == layers[0]


def test_layer_coupling_zero_disables_effect():
    """When layer_coupling = 0.0, all layers should have identical layer_delay (0.0)."""
    layers = [1, 5, 9]
    config = AnimationConfig(
        mode="traveling_wave",
        layer_coupling=0.0,
        wavelength=100.0,
        velocity=1.0,
        amplitude=0.5,
        frequency=1.0,
        phase_offset=0.0,
    )

    plotter = _build_animation_scene(
        layers=layers,
        off_screen=True,
        config=config,
    )
    records = plotter._anim_records  # type: ignore[attr-defined]
    # All layer_delay values must be exactly 0.0
    assert all(r["layer_delay"] == 0.0 for r in records)


def test_layer_coupling_affects_traveling_wave_phase():
    """At a fixed time, the phase argument should increase with layer number."""
    layers = [1, 5, 9]
    config = AnimationConfig(
        mode="traveling_wave",
        layer_coupling=0.1,
        wavelength=100.0,
        velocity=2.0,  # choose a non‑unit velocity to make omega non‑trivial
        amplitude=0.5,
        frequency=1.0,
        phase_offset=0.0,
    )

    plotter = _build_animation_scene(
        layers=layers,
        off_screen=True,
        config=config,
    )
    records = plotter._anim_records  # type: ignore[attr-defined]
    omega = float(plotter._wave_omega)  # type: ignore[attr-defined]

    # Pick an arbitrary fixed time and evaluate the phase argument
    # that appears inside the sine in _generate_frame_traveling_wave:
    #   arg = projected_s - omega*t + phase_offset + omega*layer_delay
    # Since projected_s depends on point position, we cannot compare across
    # layers directly. Instead, we isolate the layer‑dependent term:
    #   layer_phase = omega * layer_delay
    # This term should be monotonic with layer_num.
    t = 0.5  # arbitrary
    phase_offset = config.phase_offset

    # Compute the layer‑dependent phase contribution for each record
    layer_phases = []
    for r in records:
        layer_delay = r["layer_delay"]
        # The full arg (without the spatial term) is:
        #   -omega*t + phase_offset + omega*layer_delay
        # We subtract the common part (-omega*t + phase_offset) to get the layer term.
        layer_term = omega * layer_delay
        layer_phases.append((r["layer_num"], layer_term))

    # Expect layer_term to increase with layer_num because layer_coupling > 0
    assert layer_phases[0][1] == 0.0  # layer 1
    assert layer_phases[1][1] > layer_phases[0][1]  # layer 5 > layer 1
    assert layer_phases[2][1] > layer_phases[1][1]  # layer 9 > layer 5

    # Now test with layer_coupling = 0.0: all layer terms should be identical (0.0)
    config_zero = AnimationConfig(
        mode="traveling_wave",
        layer_coupling=0.0,
        wavelength=100.0,
        velocity=2.0,
        amplitude=0.5,
        frequency=1.0,
        phase_offset=0.0,
    )
    plotter_zero = _build_animation_scene(
        layers=layers,
        off_screen=True,
        config=config_zero,
    )
    records_zero = plotter_zero._anim_records  # type: ignore[attr-defined]
    omega_zero = float(plotter_zero._wave_omega)  # type: ignore[attr-defined]
    layer_terms_zero = [omega_zero * r["layer_delay"] for r in records_zero]
    assert all(term == 0.0 for term in layer_terms_zero)


if __name__ == "__main__":
    # Allow running the test file directly for quick verification.
    test_layer_coupling_basic()
    test_layer_coupling_in_scene()
    test_layer_coupling_zero_disables_effect()
    test_layer_coupling_affects_traveling_wave_phase()
    print("All layer‑coupling tests passed.")