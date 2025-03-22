import pytest
import numpy as np
from pythonopenscad.m3dapi import triangulate_3d_face

def test_simple_square():
    """Test triangulation of a simple square face in XY plane."""
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=np.float32)
    
    face = [0, 1, 2, 3]
    
    triangles = triangulate_3d_face(verts, face)
    assert len(triangles) == 2  # Should produce 2 triangles
    assert all(0 <= idx < len(verts) for tri in triangles for idx in tri)
    
    # Verify that all original edges are preserved
    edges = {(face[i], face[(i+1)%4]) for i in range(4)}
    triangulation_edges = set()
    for tri in triangles:
        triangulation_edges.add((tri[0], tri[1]))
        triangulation_edges.add((tri[1], tri[2]))
        triangulation_edges.add((tri[2], tri[0]))
        triangulation_edges.add((tri[1], tri[0]))
        triangulation_edges.add((tri[2], tri[1]))
        triangulation_edges.add((tri[0], tri[2]))
    
    # Check that each original edge appears in the triangulation
    for e1, e2 in edges:
        assert ((e1, e2) in triangulation_edges or 
                (e2, e1) in triangulation_edges), f"Edge {(e1, e2)} not found in triangulation"

def test_already_triangle():
    """Test handling of an already triangular face."""
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)
    
    face = [0, 1, 2]
    
    triangles = triangulate_3d_face(verts, face)
    assert triangles == [face]  # Should return the same face unchanged

def test_non_planar_face():
    """Test triangulation of a non-planar face."""
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 1],  # Note the z=1
        [0, 1, 0]
    ], dtype=np.float32)
    
    face = [0, 1, 2, 3]
    
    triangles = triangulate_3d_face(verts, face)
    assert len(triangles) == 2  # Should still produce 2 triangles
    assert all(0 <= idx < len(verts) for tri in triangles for idx in tri)

def test_pentagon():
    """Test triangulation of a pentagon (5-sided polygon)."""
    verts = np.array([
        [0, 0, 0],
        [2, 0, 0],
        [3, 2, 0],
        [1.5, 3, 0],
        [0, 2, 0]
    ], dtype=np.float32)
    
    face = [0, 1, 2, 3, 4]
    
    triangles = triangulate_3d_face(verts, face)
    assert len(triangles) == 3  # A pentagon should be triangulated into 3 triangles
    assert all(0 <= idx < len(verts) for tri in triangles for idx in tri)
    
    # Verify that all original edges are preserved
    edges = {(face[i], face[(i+1)%5]) for i in range(5)}
    triangulation_edges = set()
    for tri in triangles:
        triangulation_edges.add((tri[0], tri[1]))
        triangulation_edges.add((tri[1], tri[2]))
        triangulation_edges.add((tri[2], tri[0]))
        triangulation_edges.add((tri[1], tri[0]))
        triangulation_edges.add((tri[2], tri[1]))
        triangulation_edges.add((tri[0], tri[2]))
    
    # Check that each original edge appears in the triangulation
    for e1, e2 in edges:
        assert ((e1, e2) in triangulation_edges or 
                (e2, e1) in triangulation_edges), f"Edge {(e1, e2)} not found in triangulation"

def test_vertical_face():
    """Test triangulation of a vertical face (perpendicular to XY plane)."""
    verts = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 1]
    ], dtype=np.float32)
    
    face = [0, 1, 2, 3]
    
    triangles = triangulate_3d_face(verts, face)
    assert len(triangles) == 2  # Should produce 2 triangles
    assert all(0 <= idx < len(verts) for tri in triangles for idx in tri)

def test_concave_face():
    """Test triangulation of a concave polygon (U-shape)."""
    verts = np.array([
        [0, 0, 0.5],    # 0
        [3, 0, -0.5],   # 1
        [3, 1, 0],      # 2
        [2, 1, 0.2],    # 3
        [2, 2, 0.3],    # 4
        [1, 2, 0.1],    # 5
        [1, 1, 0.2],    # 6
        [0, 1, 0]       # 7
    ], dtype=np.float32)
    
    face = [0, 1, 2, 3, 4, 5, 6, 7]
    
    triangles = triangulate_3d_face(verts, face)
    assert len(triangles) == 6  # A U-shape should be triangulated into 6 triangles
    assert all(0 <= idx < len(verts) for tri in triangles for idx in tri)

def test_slanted_face():
    """Test triangulation of a face not aligned with any primary plane."""
    verts = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [1, 1, 2],
        [0, 1, 1]
    ], dtype=np.float32)
    
    face = [0, 1, 2, 3]
    
    triangles = triangulate_3d_face(verts, face)
    assert len(triangles) == 2  # Should produce 2 triangles
    assert all(0 <= idx < len(verts) for tri in triangles for idx in tri)

def test_area_preservation():
    """Test that the triangulation preserves the area of the original polygon."""
    # Create a simple square of known area
    verts = np.array([
        [0, 0, 0],
        [2, 0, 0],
        [2, 2, 0],
        [0, 2, 0]
    ], dtype=np.float32)
    
    face = [0, 1, 2, 3]
    original_area = 4.0  # 2x2 square
    
    triangles = triangulate_3d_face(verts, face)
    
    # Calculate total area of triangles
    total_area = 0
    for tri in triangles:
        v1 = verts[tri[1]] - verts[tri[0]]
        v2 = verts[tri[2]] - verts[tri[0]]
        # Area = magnitude of cross product / 2
        area = np.linalg.norm(np.cross(v1, v2)) / 2
        total_area += area
    
    np.testing.assert_almost_equal(total_area, original_area, decimal=5)

def test_winding_order():
    """Test that the triangulation preserves the winding order of vertices."""
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=np.float32)
    
    face = [0, 1, 2, 3]  # CCW order
    triangles = triangulate_3d_face(verts, face)
    
    # Check each triangle maintains CCW order
    for tri in triangles:
        v1 = verts[tri[1]] - verts[tri[0]]
        v2 = verts[tri[2]] - verts[tri[0]]
        normal = np.cross(v1, v2)
        # For CCW order in XY plane, normal should point in +Z direction
        assert normal[2] > 0

if __name__ == '__main__':
    pytest.main([__file__]) 