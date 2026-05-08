import numpy as np
from src.ephaptic_coupling.FB_Synapse_Dataset.analysis import (
    get_exr7_fb4_points,
    get_fb4y_points,
    calculate_synaptic_response,
    generate_exr7_waveform
)

def main():
    exr7 = get_exr7_fb4_points()
    fb4y = get_fb4y_points()
    
    radii = [2.0, 5.0, 8.0]
    t = np.array([0.0])  # Sample at t=0
    
    print(f"Analyzing Ephaptic Load (Waveform T=0)")
    for r in radii:
        weights, mask = calculate_synaptic_response(exr7, fb4y, radius_um=r)
        wave = generate_exr7_waveform(t, amplitude=1.0, frequency=10.0)
        total_load = np.sum(weights) * wave[0]
        print(f"Radius {r}um | Candidates: {len(weights)} | Aggregate Weight: {np.sum(weights):.2f} | Load at T=0: {total_load:.2f}")

if __name__ == "__main__":
    main()