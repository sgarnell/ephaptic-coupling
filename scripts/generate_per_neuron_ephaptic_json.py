# scripts/generate_per_neuron_ephaptic_json.py
import argparse
import numpy as np
import datetime
from src.ephaptic_coupling.FB_Synapse_Dataset.analysis import (
    get_fb_synapses_by_regions,
    aggregate_ephaptic_per_neuron,
    export_per_neuron_json
)

def main():
    parser = argparse.ArgumentParser(description="Generate flexible spatial ephaptic JSON")
    parser.add_argument("--radius", type=float, default=5.0, help="Ephaptic radius in um")
    parser.add_argument("--freq", type=float, default=10.0, help="Waveform frequency in Hz")
    parser.add_argument("--exr-region", default="ExR7", help="Presynaptic neuron prefix")
    parser.add_argument("--fb-regions", nargs='+', default=['FB4Y', 'FB5AB'], help="FB regions to include")
    args = parser.parse_args()
    
    # 1. Load data
    exr_points, exr_names = get_fb_synapses_by_regions([args.exr_region])
    fb_points, fb_names = get_fb_synapses_by_regions(args.fb_regions)
    
    # 2. Compute aggregation
    metadata = {
        "radius_um": args.radius, 
        "frequency_hz": args.freq,
        "source": args.exr_region,
        "targets": args.fb_regions,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    aggregation = aggregate_ephaptic_per_neuron(
        exr_points, 
        fb_points, 
        fb_names, 
        args.radius, 
        args.freq, 
        duration_s=1.0
    )
    
    # 3. Create unique filename
    # Format: ephaptic_ExR7_FB4Y_FB5AB_5.0um_10.0hz.json
    region_str = "_".join(args.fb_regions)
    output_file = f"ephaptic_{args.exr_region}_{region_str}_{args.radius}um_{args.freq}hz.json"
    
    # 4. Export
    export_per_neuron_json(aggregation, output_file, metadata)
    
    print(f"Aggregation complete for {len(aggregation)} neurons.")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()