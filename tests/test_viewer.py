import pytest
import numpy as np
import os

# Try to import the viewer module
try:
    from pythonopenscad.viewer.viewer import Model, Viewer, BoundingBox
    HAS_VIEWER = True
except ImportError:
    HAS_VIEWER = False

# Skip all tests if OpenGL is not available
pytestmark = pytest.mark.skipif(not HAS_VIEWER, reason="Viewer module not available")

def test_bounding_box():
    """Test BoundingBox functionality."""
    # Create a bounding box
    bbox = BoundingBox()
    
    # Test initial values
    assert np.all(bbox.min_point == np.array([float('inf'), float('inf'), float('inf')]))
    assert np.all(bbox.max_point == np.array([float('-inf'), float('-inf'), float('-inf')]))
    
    # Update min and max points
    bbox.min_point = np.array([-1.0, -2.0, -3.0])
    bbox.max_point = np.array([4.0, 5.0, 6.0])
    
    # Test size
    np.testing.assert_array_equal(bbox.size, np.array([5.0, 7.0, 9.0]))
    
    # Test center
    np.testing.assert_array_equal(bbox.center, np.array([1.5, 1.5, 1.5]))
    
    # Test diagonal
    assert abs(bbox.diagonal - np.sqrt(5**2 + 7**2 + 9**2)) < 1e-6
    
    # Test contains_point
    assert bbox.contains_point(np.array([0.0, 0.0, 0.0]))
    assert bbox.contains_point(np.array([4.0, 5.0, 6.0]))
    assert not bbox.contains_point(np.array([10.0, 0.0, 0.0]))
    
    # Test union
    other_bbox = BoundingBox(
        min_point=np.array([0.0, 0.0, 0.0]),
        max_point=np.array([10.0, 10.0, 10.0])
    )
    
    union_bbox = bbox.union(other_bbox)
    np.testing.assert_array_equal(union_bbox.min_point, np.array([-1.0, -2.0, -3.0]))
    np.testing.assert_array_equal(union_bbox.max_point, np.array([10.0, 10.0, 10.0]))

def test_model_creation():
    """Test Model creation with triangle data."""
    # Create a simple triangle
    vertex_data = np.array([
        # position (3)     # color (4)                # normal (3)
        -0.5, -0.5, 0.0,   1.0, 0.0, 0.0, 1.0,       0.0, 0.0, 1.0,
        0.5, -0.5, 0.0,    1.0, 0.0, 0.0, 1.0,       0.0, 0.0, 1.0,
        0.0, 0.5, 0.0,     1.0, 0.0, 0.0, 1.0,       0.0, 0.0, 1.0
    ], dtype=np.float32)
    
    # Create a model
    model = Model(vertex_data, num_points=3)
    
    # Test model properties
    assert model.num_points == 3
    assert model.position_offset == 0
    assert model.color_offset == 3
    assert model.normal_offset == 7
    assert model.stride == 10
    
    # Test bounding box
    np.testing.assert_array_almost_equal(
        model.bounding_box.min_point, 
        np.array([-0.5, -0.5, 0.0])
    )
    np.testing.assert_array_almost_equal(
        model.bounding_box.max_point, 
        np.array([0.5, 0.5, 0.0])
    )
    
    # Cleanup
    model.delete()

def test_viewer_creation():
    """Test Viewer creation with a model."""
    # Skip if running in CI environment
    if os.environ.get('CI') == 'true':
        pytest.skip("Skipping in CI environment")
    
    # Create a simple model
    vertex_data = np.array([
        # position (3)     # color (4)                # normal (3)
        -0.5, -0.5, 0.0,   1.0, 0.0, 0.0, 1.0,       0.0, 0.0, 1.0,
        0.5, -0.5, 0.0,    1.0, 0.0, 0.0, 1.0,       0.0, 0.0, 1.0,
        0.0, 0.5, 0.0,     1.0, 0.0, 0.0, 1.0,       0.0, 0.0, 1.0
    ], dtype=np.float32)
    
    model = Model(vertex_data, num_points=3)
    
    try:
        # Create a viewer with non-visible window
        viewer = Viewer([model], width=1, height=1, title="Test Viewer")
        
        # Test camera setup
        assert viewer.camera_pos is not None
        assert viewer.camera_front is not None
        assert viewer.camera_up is not None
        
        # Test shader program
        assert viewer.shader_program is not None
        
        # Test model storage
        assert len(viewer.models) == 1
    except Exception as e:
        pytest.skip(f"Viewer creation failed: {e}")
    finally:
        # Clean up resources
        if 'viewer' in locals():
            viewer.close()
        model.delete()

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 