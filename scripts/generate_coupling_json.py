import os
from src.ephaptic_coupling.FB_Synapse_Dataset.analysis import (
    get_exr7_fb4_points,
    get_fb4y_points,
    export_coupling_to_json
)

def main():
    exr7 = get_exr7_fb4_points()
    fb4y = get_fb4y_points()
    
    output_file = "ephaptic_coupling_data.json"
    export_coupling_to_json(exr7, fb4y, output_file, radius_um=5.0, freq=10.0)
    
    print(f"Successfully exported simulation data to {output_file}")

if __name__ == "__main__":
    main()