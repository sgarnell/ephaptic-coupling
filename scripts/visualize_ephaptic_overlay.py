# scripts/visualize_ephaptic_overlay.py
import argparse
import numpy as np
import pyvista as pv
import src.ephaptic_coupling.FB_Synapse_Dataset.analysis

def main():
    # Configuration: Allow command line overrides
    parser = argparse.ArgumentParser(description="Visualize ephaptic segments")
    parser.add_argument("--radius", type=float, default=5.0, help="Ephaptic radius in um")
    parser.add_argument("--regions", nargs='+', default=['FB4Y', 'FB5AB'], help="FB regions to include")
    args = parser.parse_args()

    # 1. Load ExR7 points
    exr7 = src.ephaptic_coupling.FB_Synapse_Dataset.analysis.get_exr7_fb4_points()
    
    # 2. Load FB points and names for the requested regions
    fb_all, fb_names = src.ephaptic_coupling.FB_Synapse_Dataset.analysis.get_fb_synapses_by_regions(args.regions)
    
    # 3. Perform filtering
    candidates, influenced, _ = src.ephaptic_coupling.FB_Synapse_Dataset.analysis.perform_ephaptic_filtering(
        exr7, 
        fb_all, 
        args.radius
    )
    
    # 4. Run Visualization
    print(f"Visualizing ExR7 influence on: {args.regions}")
    print(f"Total FB synapses processed: {len(fb_all)}")
    print(f"Influenced synapses detected: {len(influenced)}")
    
    src.ephaptic_coupling.FB_Synapse_Dataset.analysis.diagnostic_visualization(
        exr7, 
        fb_all, 
        fb_names, 
        influenced
    )

if __name__ == "__main__":
    main()