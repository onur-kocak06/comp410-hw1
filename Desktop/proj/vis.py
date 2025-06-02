import pyvista as pv
import trimesh
import numpy as np

# Load the mesh using trimesh
mesh = trimesh.load('altmodel.obj', force='mesh')

# Indices of vertices to highlight
highlightvert = [5710, 2965, 480, 5264, 2843,3818]  # Example indices

# Extract coordinates of highlighted vertices
highlight_coords = mesh.vertices[highlightvert]

# Print their coordinates
print("Highlighted Vertex Coordinates:")
for idx, coord in zip(highlightvert, highlight_coords):
    print(f"Index {idx}: {coord}")

# Convert trimesh to PyVista mesh
faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).astype(np.int_)
pv_mesh = pv.PolyData(mesh.vertices, faces)

# Create plotter
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, color='lightgray', opacity=0.5, show_edges=True)

# Add red spheres and index labels at highlighted vertices
for idx, point in zip(highlightvert, highlight_coords):
    sphere = pv.Sphere(radius=0.005 * mesh.scale, center=point)
    plotter.add_mesh(sphere, color='red')
    plotter.add_point_labels([point], [str(idx)], font_size=10, point_color='blue', point_size=5)

# Show plot
plotter.show(title='Highlighted Vertices in altmodel.obj')
