# Phase 5 — Biologically Inspired Input Signal Modeling

## Overview

Phase 5 introduces biologically grounded input signals into the Fan‑Shaped Body (FB) model.  
The goal is to define, extract, and visualize realistic perturbation sources before attempting any ephaptic propagation modeling or PCI (Perturbation Complexity Index) analysis.

This phase focuses exclusively on **input definition**, not propagation.

---

## 1. Biological Motivation

Recent discussions with Dr. Van Swinderen highlight the importance of modeling realistic sensory‑driven perturbations in the Drosophila central complex.  
In particular, the **ExR7 neuron class** provides a well‑characterized visual input pathway from the Ellipsoid Body (EB) into the Fan‑Shaped Body (FB), specifically targeting **FB4Y**.

Before modeling ephaptic interactions or PCI, we must:

- Identify the exact spatial footprint of ExR7 synapses in FB4Y  
- Define a canonical time‑varying input waveform  
- Validate that this input can be visualized and analyzed independently of downstream propagation

---

## 2. Data Source

All required anatomical data already exists in the project:

- `FB_incoming_all_synapses.npz`  
  Contains:
  - Presynaptic neuron type (e.g., `"ExR7"`)
  - FB layer index
  - 3D synapse coordinates
  - Optional region labels (e.g., Y‑tiles)

This dataset is the foundation for Phase 5.

---

## 3. Phase 5 Goals

### Goal 1 — Extract ExR7 → FB4Y Synapses

Implement a small analysis module that:

1. Loads the synapse dataset  
2. Filters for presynaptic neuron type `"ExR7"`  
3. Restricts to FB layer 4 (and Y‑region if available)  
4. Produces:
   - A NumPy array of synapse coordinates  
   - A PyVista `PolyData` object for visualization  
   - A summary of synapse counts and spatial distribution  

This defines the **spatial footprint** of the biological input.

---

### Goal 2 — Define a Canonical ExR7 Input Waveform

Create a simple, tunable sinusoidal input:



\[
I_{\text{ExR7}}(t) = A \cdot \sin(2\pi f t + \phi)
\]



Where:

- \( f \) ∈ 10–30 Hz (user‑selectable)
- \( A \) = amplitude (default 1.0)
- \( \phi \) = phase offset (default 0)

This waveform represents the **temporal structure** of the ExR7 input.

---

### Goal 3 — Visualize the Input Signal

Provide a small script or notebook that:

- Plots the ExR7 sinusoid over time  
- Shows the ExR7 synapse cloud in 3D  
- Optionally overlays the synapse cloud on the FB4 mesh  

This step ensures the input is:

- Correctly extracted  
- Correctly parameterized  
- Easy to inspect visually  

---

### Goal 4 — Prepare for Later Phases (Not Implemented Here)

Phase 5 explicitly does **not** include:

- Ephaptic propagation  
- Layer coupling dynamics  
- PCI computation  
- Integration with the animation engine  

Those will be addressed in Phase 6+.

Phase 5 ends once we have:

- A validated ExR7 synapse set  
- A validated ExR7 input waveform  
- A clear visualization of both  

---

## 4. Deliverables

- `phase5_exr7_input.py` or a Jupyter notebook  
- 3D visualization of ExR7→FB4Y synapses  
- Time‑series plot of the ExR7 sinusoidal input  
- This `phase5.md` requirements document  

---

## 5. Success Criteria

Phase 5 is complete when:

- ExR7 synapses can be extracted reliably  
- Their spatial distribution is visualized  
- A sinusoidal input waveform is defined and plotted  
- The input signal is ready to be injected into later simulation phases
