import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    input_file = "ephaptic_per_neuron.json"
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    neurons = data['neurons']
    metadata = data['metadata']
    
    # Reconstruct time axis based on metadata since it's not stored in payload
    # Defaulting to 1 second duration as per the generation script
    duration = 1.0 
    # Get influence length from the first neuron in the dict
    first_neuron = next(iter(neurons.values()))
    num_points = len(first_neuron['influence'])
    time = np.linspace(0, duration, num_points)

    plt.figure(figsize=(10, 6))
    
    for nid, record in neurons.items():
        influence = record['influence']
        plt.plot(time, influence, label=nid, alpha=0.5)

    plt.title(f"Per-Neuron Ephaptic Influence Waves ({metadata['frequency_hz']} Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Influence Amplitude")
    plt.grid(True, alpha=0.3)
    
    # Only show legend if there aren't too many neurons to block the view
    if len(neurons) < 15:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()