# Phase 6 — Spatiotemporal Per‑Synapse Animation Engine  

This execution plan implements biologically realistic ephaptic animation by transitioning from per‑neuron brightness modulation to **per‑synapse** modulation. Each synapse receives its own ephaptic weight, producing a natural spatial gradient (“halo effect”) instead of a uniform block‑level modulation.

---

## 1. Analysis Module Updates  
**File:** `src/ephaptic_coupling/FB_Synapse_Dataset/analysis.py`

### A. Update `calculate_synaptic_response`
- Ensure the function returns the **full array of per‑synapse weights**.
- Each weight corresponds to a single FB synapse.
- Out‑of‑range synapses receive weight `0.0`.

### B. New Utility: `get_synapse_weights(exr7_points, fb_points, radius_um)`
**Logic:**
- Compute distances from each FB synapse to the nearest ExR7 synapse.
- Apply the linear decay kernel:
  

\[
  W_j = 1 - \frac{d_j}{r}
  \]


- Clamp negative values to zero.

**Output:**  
- A NumPy array of length `len(fb_points)` containing all synapse weights.

### C. JSON Extension (Optional, Non‑Breaking)
Extend `export_per_neuron_json` to include per‑synapse weights:

\`\`\`json
"neurons": {
  "FB4Y_001": {
    "influence": [...],
    "synapse_count": 6123,
    "weights": [0.85, 0.12, 0.05, ...]
  }
}
\`\`\`

This allows the animation script to avoid recomputing geometry.

---

## 2. Animation Engine Updates  
**File:** `scripts/animate_ephaptic_field.py`

### A. Per‑Synapse Initialization
- Load synapse coordinates and their corresponding weights.
- Store weights in:
mesh.point_data["ephaptic_weight"]

Code

### B. Frame‑by‑Frame Brightness Update
At each animation frame \(t\):

1. Compute the ExR7 waveform value:
global_wave = generate_exr7_waveform(t)

Code

2. Broadcast across all synapses:
current_brightness = synapse_weights * global_wave

Code

3. Update PyVista scalars:
plotter.update_scalars(current_brightness, mesh=fb_mesh)

Code

### C. Performance Optimization
- Synapse weights are pushed to the GPU **once**.
- Only the time‑varying multiplier changes each frame.
- No KDTree or distance calculations occur inside the animation loop.

---

## 3. Data Flow Summary

### Preparation
- Load synapse coordinates.
- Load per‑synapse weights (from JSON or recomputed).
- Load waveform parameters (frequency, duration, sampling rate).

### Spatial Indexing
- `perform_ephaptic_filtering` identifies which synapses are within radius.
- `calculate_synaptic_response` returns the full weight array.

### Frame Loop
For each frame:
- Compute waveform value.
- Multiply waveform × weights.
- Update mesh scalars.
- Render.

---

## 4. Diagnostic Reporting
During initialization, print:

- **Max Influence Gradient:**  
`max(weights) - min(weights)`

- **Spatial Mean Influence:**  
`mean(weights)`

These help verify that the spatial gradient is biologically plausible.

---

## 5. Engineering Decisions

### A. Performance
- Storing weights in JSON avoids recomputing geometry.
- PyVista’s `update_scalars` is optimized for per‑point updates.
- The animation loop remains O(1) per frame.

### B. Biological Realism
- Each synapse carries its own static ephaptic weight.
- Brightness reflects true spatial proximity to ExR7.
- No artificial sweeping wavefront is introduced.
- FB4Y vs FB5AB differences emerge naturally from geometry.

---

## 6. Deliverables

- Updated `analysis.py` with per‑synapse weight utilities.
- Updated JSON export (optional).
- Updated `animate_ephaptic_field.py` using per‑synapse brightness.
- Verified diagnostics confirming spatial gradients.
- A biologically realistic ephaptic animation.

---

## 7. Success Criteria

- Synapses near ExR7 oscillate with high amplitude.
- Distant synapses oscillate weakly.
- FB4Y synapses show strong modulation; FB5AB synapses show minimal modulation.
- No sweeping wavefront appears.
- Animation runs smoothly at target framerate.