#!/usr/bin/env python3
"""
Example of integrating the PyOpenSCAD viewer with M3dRenderer.
This example shows how to visualize manifold3d objects using the viewer.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path to import pythonopenscad
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import manifold3d as m3d
    from pythonopenscad.m3dapi import M3dRenderer
    from pythonopenscad.viewer.viewer import Model, Viewer
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure manifold3d, PyOpenGL, and PyGLM are installed.")
    print("Try: pip install manifold3d PyOpenGL PyOpenGL-accelerate PyGLM")
    sys.exit(1)

def manifold_to_model(manifold) -> Model:
    """Convert a manifold3d Manifold to a viewer Model."""
    # Get the mesh from the manifold
    mesh = manifold.to_mesh()
    
    # Extract vertex positions and triangle indices
    positions = mesh.vert_properties
    triangles = mesh.tri_verts
    
    tri_indices = triangles.reshape(-1)
    
    # Flatten triangles and use to index positions
    vertex_data = positions[tri_indices]
    
    # Flatten the vertex data to 1D
    flattened_vertex_data = vertex_data.reshape(-1)
    
    # Create a model from the vertex data
    return Model(flattened_vertex_data)


def create_m3d_example_models():
    """Create example 3D models using M3dRenderer."""
    renderer = M3dRenderer()
    
    # Create a cube
    cube = renderer._cube(size=(1.0, 1.0, 1.0), center=True)
    cube_manifold = cube.get_solid_manifold().translate([-2.0, -2.0, 0.0])
    
    # Create a sphere
    sphere = renderer._color_renderer("green")._sphere(radius=0.8, fn=32)
    sphere_manifold = sphere.get_solid_manifold().translate([2.0, -2.0, 0.0])
    
    # Create a cylinder
    cylinder = renderer._color_renderer("deepskyblue")._cylinder(h=1.5, r_base=0.5, r_top=0.5, fn=32, center=True)
    cylinder_manifold = cylinder.get_solid_manifold().translate([-2.0, 2.0, 0.0])
    
    # Create a complex model using CSG operations
    # Create a union of cube and sphere
    cube3 = renderer._color_renderer("darkkhaki")._cube(size=1.2, center=True)
    cube_sphere = renderer._union([cube3, sphere]).get_solid_manifold().translate([2.0, 2.0, 0.0])
    
    # Create a difference of cylinder from a cube
    cube2 = renderer._color_renderer("darkviolet")._cube(size=(1.5, 1.5, 1.5), center=True)
    cylinder2 = renderer._color_renderer("peru")._cylinder(h=3.0, r_base=0.4, r_top=0.4, fn=32, center=True)
    cube_with_hole = renderer._difference([cube2, cylinder2]).get_solid_manifold().translate([0.0, 0.0, 0.0])
    
    # Convert to viewer models with different colors
    models = [
        manifold_to_model(cube_manifold),         # Red cube
        manifold_to_model(sphere_manifold),       # Green sphere
        manifold_to_model(cylinder_manifold),     # Blue cylinder
        manifold_to_model(cube_sphere),           # Yellow cube+sphere
        manifold_to_model(cube_with_hole),        # Cyan cube with hole
    ]
    
    return models

def apply_translations(models, positions):
    """Apply translations to vertex data of models."""
    for i, (model, pos) in enumerate(zip(models, positions)):
        # Get the raw vertex data
        data = model.data
        
        # Apply translation to positions (every 10 values, first 3 values)
        for j in range(0, len(data), model.stride):
            data[j] += pos[0]
            data[j+1] += pos[1]
            data[j+2] += pos[2]
        
        # Update the model's data
        model.data = data
        
        # Recompute the bounding box
        model._compute_bounding_box()

def main():
    # Create M3d models and get their positions
    models= create_m3d_example_models()
    
    # Apply translations to position the models
    #apply_translations(models, positions)
    
    # Create a viewer with all models
    viewer = Viewer(models, title="PyOpenSCAD M3D Viewer Example")
    
    print("Viewer controls:")
    print(viewer.VIEWER_HELP_TEXT)
    
    # Start the main loop
    viewer.run()

if __name__ == "__main__":
    main() 