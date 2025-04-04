
import numpy as np
from pythonopenscad.viewer.model import Model


def create_triangle_model(size: float = 1.5) -> Model:
    """Create a simple triangle model for testing."""
    # Create a simple colored triangle with different colors for each vertex
    s = size
    vertex_data = np.array([
        # position (3)     # color (4)           # normal (3)
        -s, -s, 0.0,     1.0, 0.0, 0.0, 1.0,   0.0, 0.0, 1.0,  # Red
         s, -s, 0.0,     0.0, 1.0, 0.0, 1.0,   0.0, 0.0, 1.0,  # Green
        0.0, s, 0.0,     0.0, 0.0, 1.0, 1.0,   0.0, 0.0, 1.0   # Blue
    ], dtype=np.float32)
    
    return Model(vertex_data, num_points=3)

def create_cube_model(size: float = 1.0, color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)) -> Model:
    """Create a simple cube model for testing."""
    # Create a colored cube
    s = size / 2
    vertex_data = []
    
    # Define the 8 vertices of the cube
    vertices = [
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
    ]
    
    # Define the 6 face normals
    normals = [
        [0, 0, -1], [0, 0, 1], [0, -1, 0],
        [0, 1, 0], [-1, 0, 0], [1, 0, 0]
    ]
    
    # Define faces with different colors
    face_colors = [
        [1.0, 0.0, 0.0, 1.0],  # Red - back
        [0.0, 1.0, 0.0, 1.0],  # Green - front
        [0.0, 0.0, 1.0, 1.0],  # Blue - bottom
        [1.0, 1.0, 0.0, 1.0],  # Yellow - top
        [0.0, 1.0, 1.0, 1.0],  # Cyan - left
        [1.0, 0.0, 1.0, 1.0]   # Magenta - right
    ]
    
    # Define the faces using indices
    faces = [
        [0, 1, 2, 3],  # back
        [4, 7, 6, 5],  # front
        [0, 4, 5, 1],  # bottom
        [3, 2, 6, 7],  # top
        [0, 3, 7, 4],  # left
        [1, 5, 6, 2]   # right
    ]
    
    # Create vertex data for each face
    for face_idx, face in enumerate(faces):
        normal = normals[face_idx]
        face_color = face_colors[face_idx]
        
        # Create two triangles per face
        tri1 = [face[0], face[1], face[2]]
        tri2 = [face[0], face[2], face[3]]
        
        for tri in [tri1, tri2]:
            for vertex_idx in tri:
                # position
                vertex_data.extend(vertices[vertex_idx])
                # color
                vertex_data.extend(face_color)
                # normal
                vertex_data.extend(normal)
    
    
    return Model(np.array(vertex_data, dtype=np.float32), num_points=36)

def create_colored_test_cube(size: float = 2.0) -> Model:
    """Create a test cube with distinct bright colors for each face."""
    s = size / 2
    vertex_data = []
    
    # Define cube vertices
    vertices = [
        [s, -s, -s], [-s, -s, -s], [-s, s, -s], [s, s, -s],  # 0-3 back face
        [s, -s, s], [-s, -s, s], [-s, s, s], [s, s, s]       # 4-7 front face
    ]
    
    # Define face normals
    normals = [
        [0, 0, -1],  # back - red
        [0, 0, 1],   # front - green
        [0, -1, 0],  # bottom - blue
        [0, 1, 0],   # top - yellow
        [-1, 0, 0],  # left - magenta
        [1, 0, 0]    # right - cyan
    ]
    
    alpha = 0.3
    # Bright colors for each face
    colors = [
        [1.0, 0.0, 0.0, alpha],  # red - back
        [0.0, 1.0, 0.0, alpha],  # green - front
        [0.0, 0.0, 1.0, alpha],  # blue - bottom
        [1.0, 1.0, 0.0, alpha],  # yellow - top
        [1.0, 0.0, 1.0, alpha],  # magenta - left
        [0.0, 1.0, 1.0, alpha]   # cyan - right
    ]
    
    # Face definitions (vertices comprising each face)
    faces = [
        [0, 1, 2, 3],  # back
        [4, 7, 6, 5],  # front
        [0, 4, 5, 1],  # bottom
        [3, 2, 6, 7],  # top
        [0, 3, 7, 4],  # left
        [1, 5, 6, 2]   # right
    ]
    
    # Create vertex data for each face
    for face_idx, face in enumerate(faces):
        normal = normals[face_idx]
        color = colors[face_idx]
        
        # Create two triangles per face
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
    
    return Model(np.array(vertex_data, dtype=np.float32), num_points=36)