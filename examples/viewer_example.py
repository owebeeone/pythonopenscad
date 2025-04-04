#!/usr/bin/env python3
"""
Example usage of the PyOpenSCAD viewer module.
This example creates several 3D models and displays them in the viewer.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path to import pythonopenscad
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pythonopenscad.viewer.viewer import Model, Viewer
except ImportError:
    print("Failed to import viewer module. Make sure PyOpenGL and PyGLM are installed.")
    print("Try: pip install PyOpenGL PyOpenGL-accelerate PyGLM")
    sys.exit(1)

def create_cube_data(size=1.0, color=(0.0, 0.7, 0.2, 1.0)):
    """Create vertex data for a cube."""
    s = size / 2
    vertices = [
        # Front face
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
        # Back face
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
    ]
    
    # Define the 6 face normals
    normals = [
        [0, 0, 1],    # front
        [0, 0, -1],   # back
        [0, -1, 0],   # bottom
        [0, 1, 0],    # top
        [-1, 0, 0],   # left
        [1, 0, 0]     # right
    ]
    
    # Define the faces using indices
    faces = [
        [0, 1, 2, 3],  # front
        [4, 7, 6, 5],  # back
        [0, 4, 5, 1],  # bottom
        [3, 2, 6, 7],  # top
        [0, 3, 7, 4],  # left
        [1, 5, 6, 2]   # right
    ]
    
    # Create vertex data for each face
    vertex_data = []
    for face_idx, face in enumerate(faces):
        normal = normals[face_idx]
        
        # Create two triangles for each face
        tri1 = [face[0], face[1], face[2]]
        tri2 = [face[0], face[2], face[3]]
        
        for tri in [tri1, tri2]:
            for vertex_idx in tri:
                # position
                vertex_data.extend(vertices[vertex_idx])
                # color
                vertex_data.extend(color)
                # normal
                vertex_data.extend(normal)
    
    return np.array(vertex_data, dtype=np.float32)

def create_sphere_data(radius=1.0, color=(0.2, 0.2, 0.8, 1.0), segments=20):
    """Create vertex data for a sphere using UV-sphere construction."""
    vertex_data = []
    
    # Generate the vertices
    for i in range(segments + 1):
        theta = i * np.pi / segments
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        for j in range(segments * 2):
            phi = j * 2 * np.pi / (segments * 2)
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            
            # Position
            x = radius * sin_theta * cos_phi
            y = radius * sin_theta * sin_phi
            z = radius * cos_theta
            
            # Normal (normalized position for a sphere)
            nx = sin_theta * cos_phi
            ny = sin_theta * sin_phi
            nz = cos_theta
            
            # Add this vertex
            vertex_data.extend([x, y, z])
            vertex_data.extend(color)
            vertex_data.extend([nx, ny, nz])
    
    # Generate the triangles
    indices = []
    for i in range(segments):
        for j in range(segments * 2):
            next_j = (j + 1) % (segments * 2)
            
            # Get the indices of the four vertices of a quad
            p1 = i * (segments * 2 + 1) + j
            p2 = i * (segments * 2 + 1) + next_j
            p3 = (i + 1) * (segments * 2 + 1) + next_j
            p4 = (i + 1) * (segments * 2 + 1) + j
            
            # Two triangles make a quad - with correct winding order
            # (counter-clockwise when viewed from outside)
            indices.extend([p2, p1, p3])  # First triangle
            indices.extend([p3, p1, p4])  # Second triangle
    
    # Create vertex data from indices
    indexed_vertex_data = []
    vertices_per_point = 10  # 3 position + 4 color + 3 normal
    
    for idx in indices:
        start = idx * vertices_per_point
        end = start + vertices_per_point
        indexed_vertex_data.extend(vertex_data[start:end])
    
    return np.array(indexed_vertex_data, dtype=np.float32)

def create_torus_data(major_radius=1.0, minor_radius=0.3, color=(0.8, 0.2, 0.2, 1.0), segments=20):
    """Create vertex data for a torus."""
    vertex_data = []
    
    # Generate the vertices
    for i in range(segments):
        theta = i * 2 * np.pi / segments
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        for j in range(segments):
            phi = j * 2 * np.pi / segments
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            
            # Position
            x = (major_radius + minor_radius * cos_phi) * cos_theta
            y = (major_radius + minor_radius * cos_phi) * sin_theta
            z = minor_radius * sin_phi
            
            # Normal - pointing outward from the torus surface
            nx = cos_phi * cos_theta
            ny = cos_phi * sin_theta
            nz = sin_phi
            
            # Add this vertex
            vertex_data.extend([x, y, z])
            vertex_data.extend(color)
            vertex_data.extend([nx, ny, nz])
    
    # Generate the triangles
    indices = []
    for i in range(segments):
        next_i = (i + 1) % segments
        for j in range(segments):
            next_j = (j + 1) % segments
            
            # Get the indices of the four vertices of a quad
            p1 = i * segments + j
            p2 = i * segments + next_j
            p3 = next_i * segments + next_j
            p4 = next_i * segments + j
            
            # Two triangles make a quad - with correct winding order
            # (counter-clockwise when viewed from outside)
            indices.extend([p2, p1, p3])  # First triangle
            indices.extend([p3, p1, p4])  # Second triangle
    
    # Create vertex data from indices
    indexed_vertex_data = []
    vertices_per_point = 10  # 3 position + 4 color + 3 normal
    
    for idx in indices:
        start = idx * vertices_per_point
        end = start + vertices_per_point
        indexed_vertex_data.extend(vertex_data[start:end])
    
    return np.array(indexed_vertex_data, dtype=np.float32)

def main():
    
    # Create vertex data for our models
    cube_data = create_cube_data(size=1.0, color=(0.0, 0.7, 0.2, 1.0))
    sphere_data = create_sphere_data(radius=0.8, color=(0.2, 0.2, 0.8, 1.0))
    torus_data = create_torus_data(major_radius=1.5, minor_radius=0.3, color=(0.8, 0.2, 0.2, 1.0))
    
    # Create models
    cube = Model(cube_data)
    sphere = Model(sphere_data)
    torus = Model(torus_data)
    
    # Position the models
    # (In a more complex example, you'd use transformation matrices)
    
    # Create a viewer with all models
    viewer = Viewer([cube, sphere, torus], title="PyOpenSCAD Viewer Example")
    
    print("Viewer controls:")
    print(viewer.VIEWER_HELP_TEXT)
    
    # Start the main loop
    viewer.run()

if __name__ == "__main__":
    main() 