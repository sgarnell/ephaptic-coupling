# scripts/visualize_exr7_fb4y.py
import pyvista as pv
from src.ephaptic_coupling.FB_Synapse_Dataset.analysis import (
    get_exr7_fb4_points,
    prepare_synapse_plotter
)

def main():
    # Load filtered ExR7 -> FB4Y synapse coordinates
    points = get_exr7_fb4_points()
    
    # Prepare PyVista PolyData for rendering
    cloud = prepare_synapse_plotter(points)
    
    # Visualize using PyVista
    # Render points with a larger size for visibility and enable axes
    plotter = pv.Plotter()
    plotter.add_mesh(cloud, color='cyan', point_size=5.0, render_points_as_spheres=True)
    plotter.add_axes()
    plotter.set_background('black')
    plotter.show()

if __name__ == "__main__":
    main()