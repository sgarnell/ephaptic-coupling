# Ephaptic Coupling — Fan-Shaped Body Synapse Dataset

This repository contains a complete, layer-resolved dataset of **incoming synapses to the Drosophila Fan-Shaped Body (FB)**, extracted from the Janelia hemibrain connectome.  
The dataset includes:

- A **global synapse file** containing all incoming synapses to the FB  
- **Per-layer synapse files** for FB1–FB9  
- Presynaptic neuron identity for every synapse  
- 3D coordinates (x, y, z) in hemibrain space  
- Bounds, centroids, and optional density grids  
- Scripts for extraction, sanity checking, and visualization

This dataset supports research into **ephaptic coupling**, **wave‑guide modeling**, **FB layer geometry**, and **presynaptic connectivity analysis**.

---

## 📁 Repository Structure

```
FB_Synapse_Dataset/
│
├── FB_incoming_all_synapses.npz      # All incoming synapses to the FB
├── FB1_incoming_synapses.npz         # Layer 1 synapses
├── FB2_incoming_synapses.npz
├── ...
├── FB9_incoming_synapses.npz
│
├── scripts/
│   ├── build_global_psd.py
│   ├── build_layer_files.py
│   ├── sanity_check.py
│   └── visualization_examples.py
│
└── README.md
```

---

## 📦 File Format (`.npz`)

Each `.npz` file contains the following arrays:

| Key                | Description |
|--------------------|-------------|
| `points`           | Nx3 array of synapse coordinates (x, y, z) |
| `pre_names`        | Presynaptic neuron name for each synapse |
| `layer`            | Layer index (1–9) or -1 for global |
| `bounds`           | `[xmin, xmax, ymin, ymax, zmin, zmax]` |
| `centroid`         | Mean coordinate of all synapses |
| `neuron_names`     | Unique presynaptic neuron types |
| `counts_per_neuron`| Synapse count per presynaptic neuron |
| `density_grid`     | Optional 3D voxel grid (if enabled) |
| `density_meta`     | Metadata for density grid |

---

## 🧪 Loading the Data

```python
import numpy as np

data = np.load("FB1_incoming_synapses.npz", allow_pickle=True)

points = data["points"]
pre = data["pre_names"]
centroid = data["centroid"]

print(points.shape)
print(pre[:10])
```

---

## 🧬 How the Dataset Was Generated

The extraction pipeline:

1. Identify all neurons with synapses in the Fan-Shaped Body  
2. Build a **global PSD map** of all incoming synapses  
3. For each FB layer (1–9):  
   - Extract the layer’s point cloud  
   - Match each point to the global PSD  
   - Compute bounds, centroid, and presynaptic histogram  
   - Optionally compute a 3D density grid  
4. Save each layer as a standalone `.npz` file  

The full global dataset contains:

- **2,952,590 synapses**  
- **29,964 unique presynaptic neuron IDs**  
- Extraction time: ~7.25 minutes on a modern workstation  

---

## 📊 Example: Sanity Check Output

```
=== File Summary ===
Total synapses: 36198
Layer index: 1
Bounds: [ 71.28 274.952 124.776 287.136  23.912 265.248]
Centroid: [193.19 187.63 141.22]

--- Top Presynaptic Neuron Types ---
FB1D_R_2                       : 387
FB1D_R_1                       : 366
FB1D_L_1                       : 365
...
```

---

## 🖼 Visualization

You can visualize synapses using PyVista:

```python
import pyvista as pv
import numpy as np

data = np.load("FB5_incoming_synapses.npz", allow_pickle=True)
points = data["points"]

cloud = pv.PolyData(points)
cloud.plot(point_size=3)
```

---

## 📜 License

Choose a license (MIT recommended) and place it in `LICENSE`.

---

## 🤝 Contributions

Pull requests are welcome — especially for:

- Visualization tools  
- Analysis notebooks  
- Connectivity matrices  
- Wave‑guide simulation modules  

---

## 📧 Contact

Maintainer: **Saul Garnell**  
For questions or collaboration, open an issue or reach out directly.
