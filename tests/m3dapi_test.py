import pytest
import numpy as np
import manifold3d as m3d
from pythonopenscad.m3dapi import (
    M3dRenderer,
    RenderContext,
    RenderContextManifold,
    triangulate_3d_face,
    _make_array,
    manifold_to_stl,
    Mode
)


@pytest.fixture
def m3d_api():
    return M3dRenderer()


def assert_rendercontext(rc: RenderContext):
    assert isinstance(rc, RenderContext), "not a RenderContext"
    assert all(isinstance(m, m3d.Manifold) for m in rc.solid_objs), (
        "solid_manifold not all Manifold"
    )
    assert all(isinstance(m, m3d.Manifold) for m in rc.shell_objs), (
        "shell_manifold not all Manifold"
    )


def test_cube():
    api = M3dRenderer()
    result = api._cube((1.0, 2.0, 3.0))
    assert_rendercontext(result)

    solid_manifold: m3d.Manifold = result.get_solid_manifold()
    assert isinstance(solid_manifold, m3d.Manifold)
    

def test_write_stl():
    api = M3dRenderer()
    result = api._cube((1.0, 2.0, 3.0))

    solid_manifold: m3d.Manifold = result.get_solid_manifold()
    assert isinstance(solid_manifold, m3d.Manifold)

    # Use BytesIO instead of a file
    import io
    stl_file = io.BytesIO()
    manifold_to_stl(solid_manifold, "test_cube.stl", file_obj=stl_file)
    
    # Reset buffer position to start for reading
    stl_file.seek(0)
    
    import stl

    mesh = stl.mesh.Mesh.from_file("test_cube.stl", fh=stl_file)
    assert isinstance(mesh, stl.mesh.Mesh)
    assert mesh.v0.shape == (12, 3)

def test_sphere():
    api = M3dRenderer()
    result = api._sphere(radius=1.0, fn=32)
    assert_rendercontext(result)


def test_cylinder():
    api = M3dRenderer()
    # Test regular cylinder
    result = api._cylinder(h=2.0, r_base=1.0)
    assert_rendercontext(result)

    # Test cone
    result = api._cylinder(h=2.0, r_base=1.0, r_top=0.5)
    assert_rendercontext(result)

    # Test centered cylinder
    result = api._cylinder(h=2.0, r_base=1.0, center=True)
    assert_rendercontext(result)


def test_polyhedron():
    api = M3dRenderer()
    # Create a cylinder.
    count: int = 64
    radius: float = 10
    height: float = 15
    points2d: np.ndarray = (
        np.array([
            (
                np.cos(i * 2 * np.pi / count),
                np.sin(i * 2 * np.pi / count),
            )
            for i in range(count)
        ])
        * radius
    )

    upper_points3d: np.ndarray = np.hstack((points2d, np.tile((height,), (count, 1))))
    lower_points3d: np.ndarray = np.hstack((points2d, np.tile((0.0,), (count, 1))))

    points: np.ndarray = np.vstack((lower_points3d, upper_points3d))

    faces: list[tuple[int, int, int, int]] = []
    
    faces.append(list(range(count - 1, -1, -1)))
    faces.append(list(range(count, 2 * count)))
    
    for i in range(count):
        j = (i + 1) % count
        faces.append((i, j, j + count, i + count))


    angle = 0 * np.pi / 4
    # Rotate the points3d by around the Y-axis to make the tesselator do something.
    cosr = np.cos(angle)
    sinr = np.sin(angle)
    # rotation_matrix = np.array([
    #     [cosr, 0, sinr],
    #     [0, 1, 0],
    #     [-sinr, 0, cosr],
    # ])
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, cosr, -sinr],
        [0, sinr, cosr],
    ])
    points = points @ rotation_matrix

    result = api._polyhedron(points, faces)
    assert isinstance(result, RenderContext)

    manifold = result.get_solid_manifold()
    #manifold_to_stl(manifold, "test_polyhedron.stl", mode=Mode.ASCII)
    m_mesh = manifold.to_mesh()
    tri_verts = m_mesh.tri_verts
    points = m_mesh.vert_properties[:, :3]
    
    assert len(tri_verts) == count * 4 - 4
    assert points.shape == (256, 3) # Normals added doubles points.


def test_triangulate_3d_face():
    # Test with a simple square face
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)

    face = [0, 1, 2, 3]

    triangles = triangulate_3d_face(verts, [face])
    assert len(triangles) == 2  # Should produce 2 triangles (6 indices)

    # Verify that all indices are within bounds
    assert all(0 <= idx < len(verts) for tri in triangles for idx in tri)


def test_triangulate_3d_face_already_triangle():
    # Test with an already triangular face
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

    face = [0, 1, 2]

    triangles = triangulate_3d_face(verts, [face])
    assert triangles == [face]  # Should return the same face unchanged


def test_triangulate_3d_face_non_planar():
    # Test with a non-planar face
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 1],  # Note the z=1
            [0, 1, 0],
        ],
        dtype=np.float32,
    )

    face = [0, 1, 2, 3]

    triangles = triangulate_3d_face(verts, [face])
    assert len(triangles) == 2  # Should still produce 2 triangles
    assert all(0 <= idx < len(verts) for tri in triangles for idx in tri)


def test_make_array():
    api = M3dRenderer()

    # Test with list
    input_list = [1, 2, 3]
    result = _make_array(np.array(input_list), np.uint32)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint32
    assert result.flags.c_contiguous
    assert result.flags.writeable

    # Test with None
    assert _make_array(None, np.float32) is None

    # Test with non-contiguous array
    non_contiguous = np.array([[1, 2], [3, 4]])[:, :1]
    result = _make_array(non_contiguous, np.float32)
    assert result.flags.c_contiguous
    assert result.flags.writeable


def test_transform():
    api = M3dRenderer()
    cube = api._cube((1.0, 2.0, 3.0))

    # Test translation
    transform = np.eye(4)
    transform[0:3, 3] = [1.0, 2.0, 3.0]  # Translation vector
    result = cube.transform(transform)
    assert isinstance(result, RenderContextManifold)

    # Test rotation
    angle = np.pi / 2  # 45 degrees
    transform = np.eye(4)
    transform[0:3, 0:3] = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    result = cube.transform(transform)
    assert isinstance(result, RenderContextManifold)


def test_get_resize_scale():
    api = M3dRenderer()
    cube = api._cube((2.0, 3.0, 4.0))  # Create a cube with known dimensions

    # Test exact resize
    newsize = np.array([4.0, 6.0, 8.0])  # Double all dimensions
    scale = cube.getResizeScale(newsize, auto=False)
    np.testing.assert_array_almost_equal(scale, np.array([2.0, 2.0, 2.0]))

    # Test auto resize with one dimension specified
    newsize = np.array([10.0, 0.0, 0.0])
    auto = [False, True, True]
    scale = cube.getResizeScale(newsize, auto)
    np.testing.assert_array_almost_equal(scale, np.array([5.0, 5.0, 5.0]))

    # Test mixed resize (some auto, some exact)
    newsize = np.array([10.0, 3.0, 0.0])
    auto = np.array([False, False, True])
    scale = cube.getResizeScale(newsize, auto)
    np.testing.assert_array_almost_equal(scale, np.array([5.0, 1.0, 5.0]))


if __name__ == "__main__":
    # test_polyhedron()
    pytest.main([__file__])
