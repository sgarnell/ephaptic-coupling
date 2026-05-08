# Phase 6 — Ephaptic Animation & Dynamic Visualization Engine

## Objective
Integrate the per‑neuron ephaptic influence data with the existing 3D animation engine to produce a **biologically realistic, time‑varying visualization** of ephaptic coupling within the Fan‑Shaped Body (FB). This phase focuses on driving synapse brightness over time based on real ephaptic influence values derived in Phase 5.

The goal is to animate:
- **ExR7 synapses** oscillating according to the driving waveform.
- **FB4Y synapses** showing strong ephaptic brightness modulation.
- **FB5AB synapses** showing weak or minimal modulation.
- A coherent spatiotemporal ephaptic field that reflects the true geometry and influence distribution.

## Scope
Phase 6 includes:

### 1. Time‑Driven Brightness Modulation
- Use the ExR7 waveform \(I(t)\) to animate the incoming neuron.
- Use per‑neuron ephaptic influence \(I_k(t)\) to animate each FB neuron.
- Map influence → brightness using biologically plausible scaling (no artificial normalization that hides true differences).

### 2. Per‑Synapse Influence Mapping
- For each FB neuron, distribute its ephaptic influence across its synapse cloud.
- Optionally modulate brightness by per‑synapse weight for finer realism.
- Maintain the true amplitude differences between neuron types (e.g., FB4Y >> FB5AB).

### 3. Animation Engine Integration
- Extend the existing PyVista‑based animation engine.
- At each animation frame:
  - Update ExR7 brightness.
  - Update brightness for all FB synapses based on their neuron’s influence at time \(t\).
  - Render the updated frame.

### 4. Biological Realism
- Preserve waveform frequency and phase.
- Maintain spatial fidelity: synapses closer to ExR7 should appear more strongly modulated.
- Avoid smoothing or averaging that obscures the geometry‑driven differences.

### 5. Output
- A reproducible animation script (`animate_ephaptic_field.py`).
- Optional frame‑by‑frame export for video assembly.
- A dynamic visualization that clearly shows:
  - ExR7 oscillation,
  - strong FB4Y ephaptic response,
  - weak FB5AB response,
  - correct spatial distribution of influence.

## Non‑Goals
Phase 6 does **not** include:
- new ephaptic physics models,
- new decay functions,
- new geometry processing,
- color‑mapping standardization (reserved for Phase 7).

This phase focuses solely on **driving the animation with real ephaptic data**.

## Deliverables
- `animate_ephaptic_field.py` (new script)
- Updates to animation utilities in `analysis.py` or `visualization.py`
- A working animation demonstrating:
  - ExR7 → FB4Y strong coupling,
  - ExR7 → FB5AB weak coupling,
  - accurate spatiotemporal modulation.

## Success Criteria
Phase 6 is complete when:
- The animation visually reflects the true geometry and ephaptic influence.
- Brightness modulation matches the per‑neuron influence curves.
- ExR7, FB4Y, and FB5AB differences are clearly visible.
- The animation can be reproduced with different radii/frequencies via CLI.

# Phase 6 Addendum — Per‑Synapse Brightness Modulation Patch Plan
# (Request for Detailed Engineering Plan Only — No Code Implementation)

This addendum modifies the Phase 6 animation plan to ensure that synapse brightness is computed **per synapse**, using each synapse’s individual ephaptic weight, rather than applying a uniform per‑neuron brightness. This produces a biologically realistic spatial gradient of ephaptic influence.

## Goal
Replace per‑neuron brightness modulation with **per‑synapse brightness modulation**, using the ephaptic weights already computed in Phase 5. This ensures that synapses closer to ExR7 appear more strongly modulated, while distant synapses show weaker modulation, without introducing any artificial sweeping wavefronts.

## Requirements

### 1. Use Per‑Synapse Weights
- During Phase 5, each FB synapse already received a distance‑based ephaptic weight \(W_j\).
- These weights must be preserved and passed into the animation pipeline.
- Brightness at time \(t\) becomes:
  

\[
  B_j(t) = W_j \cdot I(t)
  \]


  where \(I(t)\) is the ExR7 driving waveform.

### 2. JSON Extension (Non‑Breaking)
If needed, extend the JSON export to include a `weights` array for each neuron:
```json
"neurons": {
  "FB4Y_001": {
    "influence": [...],
    "synapse_count": N,
    "weights": [w_1, w_2, ..., w_N]
  }
}

This is optional if weights are already available in memory during animation.

### 3. Animation Engine Update
Modify the animation loop so that:
- Each synapse receives its own brightness value.
- Brightness is updated by multiplying:
  - the synapse’s static weight \(W_j\),
  - the time‑varying waveform value \(I(t)\).

### 4. Spatial Realism
This change ensures:
- No artificial left→right sweeping wave.
- A biologically plausible “halo” of influence around ExR7 synapses.
- Strong FB4Y modulation and weak FB5AB modulation emerge naturally from geometry.

### 5. Implementation Notes
- No new physics model is introduced.
- No changes to the per‑neuron influence logic are required.
- The animation engine simply uses per‑synapse scalars instead of per‑neuron scalars.
- Performance impact is minimal because PyVista supports per‑point scalar updates efficiently.

## Deliverables
- Updated JSON export (optional).
- Updated scalar‑update logic in `animation.py`.
- Updated `animate_ephaptic_field.py` to compute brightness per synapse.

## Success Criteria
- Synapses near ExR7 visibly oscillate with high amplitude.
- Synapses farther away show weaker modulation.
- FB4Y and FB5AB differences emerge naturally from geometry.
- No sweeping wavefront appears unless explicitly added in a future phase