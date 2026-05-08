**phase 5 General OveriewOverview based on phase5 skill document**

Phase 5 should stay analysis-only, per phase5.md: load the synapse archive, extract ExR7 to FB4Y, visualize the footprint, define a sinusoidal ExR7 input, and plot it. No changes to the animation engine, traveling-wave code, layer coupling, PCI, or any new public API.

# File Plan

- Add one notebook: phase5_exr7_input.ipynb.
- Keep all analysis logic in that notebook unless a tiny helper is clearly worth extracting.
- If helper code is needed, place it in an existing analysis location such as tests/phase5_test.py for validation only, or keep it as notebook cells.
- Do not add anything under src/.
- Do not add new top-level packages, CLI entry points, or viewer subsystems.

# Patch Structure

1. Title and scope cell.
- State that Phase 5 is analysis-only.
- State the exact goals: load FB_incoming_all_synapses.npz, extract ExR7 to FB4Y synapses, visualize the spatial footprint, define the ExR7 waveform, and plot it.

2. Imports cell.
- Import numpy, matplotlib, and pyvista.
- Import only existing project helpers if they already help with dataset access or plotting.
- Keep the notebook self-contained otherwise.

3. Dataset inspection cell.
- Load FB_incoming_all_synapses.npz with numpy.
- Print the archive keys and sample field shapes.
- Confirm which fields correspond to presynaptic type, layer index, coordinates, and optional Y-region labels.

4. Extraction cell.
- Filter rows where presynaptic type is ExR7.
- Filter those rows to FB layer 4.
- If a Y-region label exists, narrow to FB4Y using that label.
- Return a NumPy coordinate array and a small summary dictionary with counts and bounds.

5. 3D visualization cell.
- Build a PyVista PolyData from the extracted coordinates.
- Render the synapse cloud in 3D.
- If an FB4 mesh is already available through existing project data or helpers, overlay it as a faint reference.
- If not, show the cloud alone.


6. Geometric Filtering Cell — Identify ExR7 Synapses Near FB Layer 4
**Goal:** Replace the old “filter by layer index” with a biologically grounded geometric proximity filter.
**Requirements:**
- Load ExR7 synapse coordinates (already extracted).
- Load FB layer 4 geometry (mesh, point cloud, or voxel representation).
- Build a KD‑tree (or similar spatial index) for FB4Y points.
- For each ExR7 synapse, compute nearest‑neighbor distance to FB4Y.
- Select synapses within a biologically plausible ephaptic radius (e.g., 3–5 µm).
- Return:
  - `exr7_fb4_candidates`: synapses capable of ephaptic influence.
  - `distances`: optional diagnostic array.

**Outputs to print:**
- Total ExR7 synapse count.
- Number of ExR7 synapses near FB4Y.
- Distance threshold used.

7. Waveform Generation Cell — Create Sinusoidal Input for Filtered Synapses
**Goal:** Generate the oscillatory input signal that drives the ephaptic effect.
**Requirements:**
- Define sinusoid:  
  \( I(t) = A \sin(2\pi f t + \phi) \)
- Parameters:
  - Frequency \( f \in [10, 30] \) Hz
  - Amplitude \( A \)
  - Phase \( \phi \) (default 0)
- Generate:
  - Time vector (e.g., 0–500 ms at 1 kHz)
  - Waveform samples
- Plot waveform with:
  - Clean axes
  - Frequency, amplitude, phase labeled

**Outputs to print:**
- Frequency, amplitude, phase
- Duration and sampling rate

8. Ephaptic Visualization + Summary Cell
**Goal:** Visualize ExR7 oscillation and compute the ephaptic field interacting with FB4Y.
**Requirements:**
- Animate ExR7 synapse intensities over time:
  - Color or size modulated by \( I(t) \)
- Overlay FB4Y geometry.
- Compute ephaptic field at FB4Y points using kernel:
  - \( E(x,t) = k \sum_j \frac{I_j(t)}{d(x, x_j)} \)
- Visualize:
  - Field intensity on FB4Y over time
  - Propagation pattern
**Summary outputs:**
- Number of ExR7 synapses influencing FB4Y
- Coordinate bounds and centroid of filtered synapses
- Waveform parameters used
- Ephaptic kernel parameters (k, distance threshold)

9. Per‑Neuron Ephaptic Aggregation

Goal
Replace the global ephaptic influence (single summed waveform) with **per‑neuron** ephaptic influence vectors. Each FB neuron receives a time‑series based solely on the synapses within its ephaptic radius.

Requirements

- Synapse-to-Neuron Mapping**
   - Extend the analysis pipeline to associate each FB4/FB5 synapse with its parent neuron.
   - Use existing metadata in the FB_incoming_all_synapses.npz dataset (each synapse already stores its presynaptic neuron name).
   - Build a dictionary:
     ```
     neuron_to_synapses = {
         "FB4Y_001": [indices...],
         "FB4Y_002": [...],
         "FB5AB_001": [...],
         ...
     }
     ```

- Per‑Synapse Ephaptic Contribution**
   - Reuse the distance‑weighted ephaptic logic from Step 7:
     

\[
     I_j(t) = W_j \cdot \sin(2\pi f t)
     \]


   - Do **not** collapse these contributions globally.

- Per‑Neuron Aggregation**
   - For each neuron \(k\):
     

\[
     I_k(t) = \sum_{j \in S_k} I_j(t)
     \]


   - Produce a dictionary:
     ```
     {
       "FB4Y_001": [... time series ...],
       "FB4Y_002": [...],
       "FB5AB_001": [...],
       ...
     }
     ```

- JSON Export Format**
   - Replace the single global waveform with a structured JSON:
     ```
     {
       "metadata": {
         "radius_um": R,
         "frequency_hz": F,
         "duration_s": D,
         "fs": FS
       },
       "neurons": {
         "FB4Y_001": { "influence": [...], "synapse_count": N1 },
         "FB4Y_002": { "influence": [...], "synapse_count": N2 },
         "FB5AB_001": { "influence": [...], "synapse_count": N3 }
       }
     }
     ```

- Optional:
    - Export ephaptic field as a NumPy array for Phase 6 integration.

- Proposed File Changes
    - A. Modify `analysis.py`

    - New Function: `aggregate_ephaptic_per_neuron(exr7_points, fb_points, fb_neuron_ids, radius_um, freq, duration_s, fs)`**

        Responsibilities:
        - Perform KD‑tree filtering (reuse Step 6).
        - Compute per‑synapse weights (reuse Step 7).
        - Group synapses by neuron ID.
        - Generate per‑neuron time‑series.
        - Return a dictionary suitable for JSON export.

    - New Function: `export_per_neuron_json(aggregation_dict, output_path)`**
        - Writes the structured JSON described above.

    - B. New Script: `scripts/generate_per_neuron_ephaptic_json.py`

    Responsibilities:
    - Hard‑coded test configuration (radius=5µm, freq=10Hz).
    - Loads ExR7 synapses and FB4/FB5 synapses.
    - Calls `aggregate_ephaptic_per_neuron`.
    - Writes `ephaptic_per_neuron.json`.
    - Prints summary:
    - Number of neurons influenced.
    - Per‑neuron synapse counts.
    - Peak amplitude per neuron.

    - Test Case: ExR7 → FB4Y + FB5AB Overlap**

    Purpose:
    - Demonstrate that the ephaptic wave splits correctly across two neurons with different synapse clouds.

    Procedure:
    1. Load ExR7 synapses.
    2. Load FB4Y and FB5AB synapses.
    3. Run per‑neuron aggregation.
    4. Validate:
    - FB4Y receives a large influence (dense innervation).
    - FB5AB receives a smaller influence (sparse innervation).
    - The two waveforms differ in amplitude but share the same frequency.
    - The sum of all per‑neuron waves equals the global wave from Step 8.

    Expected Outcome:
    - A JSON file showing distinct ephaptic waveforms for FB4Y and FB5AB.
    - Console output confirming the split:


- 10. Standardized Color Mapping for Ephaptic Visualization

Define a fixed color palette and mapping rules so that:
- The **incoming neuron** is always shown in a consistent, high‑visibility color.
- Up to **four receiving neuron types** can be visualized simultaneously without visual overload.
- Ephaptically influenced synapses use a **derived color** that clearly indicates “influence” while preserving the identity of the receiving neuron.

- Requirements

    A. **Incoming Neuron Color (Fixed)**
    - Always use **Yellow** for the incoming neuron (e.g., ExR7).
    - This ensures immediate visual recognition of the source of ephaptic influence.

    B. **Receiving Neuron Base Colors (Up to 4 Types)**
    Assign the following fixed colors:
    - FB Neuron Type 1** → Red  
    - FB Neuron Type 2** → Light Red  
    - FB Neuron Type 3** → Blue  
    - FB Neuron Type 4** → Cyan  

    These colors are chosen for:
        - high contrast  
        - minimal perceptual blending  
        - compatibility with PyVista’s default rendering  

    C. **Influenced Synapse Colors (Derived)**
    For each receiving neuron type, the ephaptically influenced synapses use a **derived color**:
    - Red → **Orange**  
    - Light Red → **Light Orange**  
    - Blue → **Green**  
    - Cyan → **Light Green**  

    This ensures:
        - the influenced synapses remain visually tied to their parent neuron  
        - the influence is clearly distinguishable from the base synapse cloud  

    4. **Color Mapping Table (Canonical)**

    | Role                        | Base Color | Influenced Color |
    |-----------------------------|------------|------------------|
    | Incoming neuron (ExR7)      | Yellow     | N/A              |
    | Receiving neuron type 1     | Red        | Orange           |
    | Receiving neuron type 2     | Light Red  | Light Orange     |
    | Receiving neuron type 3     | Blue       | Green            |
    | Receiving neuron type 4     | Cyan       | Light Green      |

    5. **Visualization Logic**
    - All visualization functions must accept a `color_map` dictionary.
    - Default behavior uses the canonical mapping above.
    - The plotter must render:
        - Incoming synapses (yellow)
        - Receiving neuron synapses (base colors)
        - Ephaptically influenced synapses (derived colors)
    - Legend entries must reflect the mapping.

### Proposed File Changes

#### A. Modify `analysis.py`
Add:
- `get_default_color_map()`
- `apply_color_map(neuron_ids, synapse_indices, influence_mask)`

These functions:
- return the canonical color mapping
- assign colors to synapse point clouds based on neuron identity and influence status

#### B. Modify Visualization Functions
All PyVista visualization routines (existing and future) must:
- accept a `color_map` parameter
- use the standardized mapping
- render influenced synapses using derived colors

#### C. New Test Case: Multi‑Neuron Ephaptic Visualization

**Test Case: ExR7 → FB4Y + FB5AB**

Purpose:
- Demonstrate that the color mapping prevents visual overload.
- Show clear separation between:
  - ExR7 synapses (yellow)
  - FB4Y synapses (red)
  - FB4Y influenced synapses (orange)
  - FB5AB synapses (blue)
  - FB5AB influenced synapses (green)

Procedure:
1. Load ExR7, FB4Y, and FB5AB synapses.
2. Run per‑neuron ephaptic aggregation (Step 9).
3. Visualize using the standardized color map.
4. Validate:
   - No color collisions.
   - Influenced synapses are visually distinct.
   - Overlapping regions remain interpretable.

Expected Output:
- A PyVista scene where ExR7 → FB4Y + FB5AB influence is visually separable.
- A legend showing all six colors.

### Constraints
- No new dependencies.
- Color logic must remain centralized and reusable.
- Must integrate with Steps 6–9 without modifying their core logic.

### Deliverable
A detailed patch plan describing:
- new color‑mapping functions
- modifications to visualization routines
- the canonical color table
- the multi‑neuron test case
- integration with per‑neuron ephaptic aggregation


## Minimal Helper Functions

Keep helpers local and small. If they are written at all, they should be limited to these notebook-level functions:

- load_synapse_archive(path)
- extract_exr7_fb4y(archive)
- build_synapse_polydata(points)
- make_exr7_waveform(t, amplitude, frequency, phase)
- summarize_extraction(points)

checks:

- Extraction returns at least one ExR7 to FB4Y synapse when the dataset contains them.
- The extracted array has shape N×3.
- The summary count matches the coordinate array length.
- The waveform function returns the expected values for known inputs.
- The frequency validation rejects values outside 10–30 Hz.


## Plans Outcome for Steps 1-3:

openclaw/phase5_loader.py (new module) containing:

inspect_synapse_archive(path) – loads and inspects .npz archive schema
extract_exr7_fb4y(points, presynaptic_type, target_layer) – filters ExR7→FB4Y synapses
summarize_extraction(points) – computes summary statistics (count, centroid, span, etc.)
tests/phase5_test.py (new test file) containing unit tests for:

Archive inspection (structured and plain array cases)
Extraction filtering (both structured and plain ndarray paths)
Summary statistics (empty and non-empty cases)

## Detailed Outcome for Step 4

The src/ephaptic_coupling/FB_Synapse_Dataset/analysis.py now includes the generate_exr7_waveform helper function, and tests/test_phase5_analysis.py contains the comprehensive test suite verifying waveform amplitude, phase shifting, and frequency consistency via zero-crossing counts.


## Patch Plan: 3D Visualization (Step 5)

File: src/ephaptic_coupling/FB_Synapse_Dataset/analysis.py We will add a helper to prepare the data for visualization. This keeps rendering logic distinct from the compute logic.

File: tests/test_phase5_analysis.py We will add a non-GUI test to ensure the visualization data structure is correctly initialized.

Data Access: Minimal helper function in analysis.py to target the FB_incoming_all_synapses.npz dataset using the schema-defined points, pre_names, and layer keys.
Filtering: Subset extraction for ExR7 neurons in layer 4.
Analysis Helpers: Statistical computation (count, centroid) and sinusoidal waveform generation.
Visualization Prep: PyVista integration for preparing 3D PolyData.
Verification: A dedicated test suite in tests/test_phase5_analysis.py covering filtering logic, spatial statistics, waveform timing/phase, frequency accuracy, and visualization data structure initialization.

