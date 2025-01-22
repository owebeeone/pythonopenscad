from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar
import manifold3d as m3d
import numpy as np
import mapbox_earcut


T = TypeVar('T')

class NpArray(np.ndarray, Generic[T]):
    pass

class Vector3(np.ndarray, Generic[T]):
    pass

def is_iterable(v):
    """Return True if v is an iterable."""
    return isinstance(v, Iterable) and not isinstance(v, (str, bytes))

def _make_array(v: NpArray[T] | None, t: type[T]) -> NpArray[T] | None:
    """Condition array to be C-style contiguous and writeable.
    """
    if v is None:
        return None
    if not isinstance(v, np.ndarray) or not (v.flags.c_contiguous and v.flags.writeable and v.dtype == t):
        v = np.array(v, dtype=t, order="C")
    return v

def _rotVSinCos(v: np.ndarray, 
               sinr: float, 
               cosr: float) -> np.ndarray:
    '''Returns a matrix that causes a rotation about an axis vector v by the 
    given sin and cos of the rotation angle.'''
    u = v / np.linalg.norm(v)
    ux = u[0]
    uy = u[1]
    uz = u[2]
    u2 = u * u
    ux2 = u2[0]
    uy2 = u2[1]
    uz2 = u2[2]
    uxz = ux * uz
    uxy = ux * uy
    uyz = uy * uz
    lcosr = 1 - cosr
    return np.array(
        [[cosr + ux2 * lcosr, uxy * lcosr - uz * sinr, uxz * lcosr + uy * sinr, 0],
         [uxy * lcosr + uz * sinr, cosr + uy2 * lcosr, uyz * lcosr - ux * sinr, 0],
         [uxz * lcosr - uy * sinr, uyz * lcosr + ux * sinr, cosr + uz2 * lcosr, 0],
         [0.0, 0, 0, 1]])

@dataclass
class RenderContext:
    """Each level of the rendering phase produces a RenderContext.
    While also wrapping manifold3d.Manifold, it also handles the OpenSCAD
    operations and flags.
    """
    api: 'M3dRenderer'
    manifold: m3d.Manifold
    
    def transform(self, transform: np.ndarray) -> 'RenderContext':
        if transform.shape == (4, 4):
            # assert the bottom row is [0, 0, 0, 1]
            assert np.allclose(transform[-1], [0, 0, 0, 1])
            transform = transform[:3, :]
        else:
            assert transform.shape == (3, 4)
        return RenderContext(self.api, self.manifold.transform(transform))
    
    def translate(self, v: np.ndarray) -> 'RenderContext':
        return RenderContext(self.api, self.manifold.translate(v))
    
    def rotate(self, a: float | np.ndarray, v: np.ndarray | None = None) -> 'RenderContext':
        if is_iterable(a):
            if v is not None:
                raise ValueError("Cannot specify both a vector and v for rotation")
            a = _make_array(a, np.double)
            return self.manifold.rotate(a)
        else:
            if v is None:
                return self.manifold.rotate(_make_array([a, 0, 0], np.double))
            # Give exact values for the most common angles.
            if a == 0:
                sinr, cosr = np.double(0), np.double(1)
            elif a == 90:
                sinr, cosr = np.double(1), np.double(0)
            elif a == 180:
                sinr, cosr = np.double(0), np.double(-1)
            elif a == 270:
                sinr, cosr = np.double(-1), np.double(0)
            else:
                r = np.double(a) * np.pi / 180
                sinr, cosr = np.sin(r), np.cos(r)
            transform = _rotVSinCos(v, sinr, cosr)
            return self.transform(transform)
    
    def mirror(self, normal: np.ndarray) -> 'RenderContext':
        return RenderContext(self.api, self.manifold.mirror(_make_array(normal, np.float64)))
    
    def property(self, property_values: NpArray[np.float32], idx: int = 3) -> 'RenderContext':
        property_values = _make_array(property_values, np.float32)
        mesh = self.manifold.to_mesh()
        properties = mesh.vert_properties
        # Calculate required width for the properties array
        required_width = idx + len(property_values)
        current_width = properties.shape[1] if properties.size > 0 else 0
        
        # Create new array with required size
        new_properties = np.zeros((properties.shape[0], required_width))
        # Copy existing data
        new_properties[:, :current_width] = properties
        # Assign new values
        new_properties[:, idx:idx + len(property_values)] = property_values
        
        return self.api.mesh(
            vert_properties=new_properties,
            tri_verts=mesh.tri_verts
        )
        
    def union(self, other: 'RenderContext') -> 'RenderContext':
        return RenderContext(self.api, self.manifold + other.manifold)
    
    def intersect(self, other: 'RenderContext') -> 'RenderContext':
        return RenderContext(self.api, self.manifold % other.manifold)
    
    def difference(self, other: 'RenderContext') -> 'RenderContext':
        return RenderContext(self.api, self.manifold - other.manifold)
    
    def getResizeScale(self, newsize: np.ndarray, auto: np.ndarray | bool) -> np.ndarray:
        # This is the code from OpenSCAD's getResizeTransform() function in:
        # https://github.com/openscad/openscad/blob/master/src/geometry/GeometryUtils.h
        #
        #   // Find largest dimension
        #   int maxdim = 0;
        #   for (int i = 1; i < 3; ++i) if (newsize[i] > newsize[maxdim]) maxdim = i;

        #   // Default scale (scale with 1 if the new size is 0)
        #   Vector3d scale(1, 1, 1);
        #   for (int i = 0; i < 3; ++i) if (newsize[i] > 0) scale[i] = newsize[i] / bbox.sizes()[i];

        #   // Autoscale where applicable
        #   double autoscale = scale[maxdim];
        #   Vector3d newscale;
        #   for (int i = 0; i < 3; ++i) newscale[i] = !autosize[i] || (newsize[i] > 0) ? scale[i] : autoscale;

        #   Transform3d t;
        #   t.matrix() <<
        #     newscale[0], 0, 0, 0,
        #     0, newscale[1], 0, 0,
        #     0, 0, newscale[2], 0,
        #     0, 0, 0, 1;
        #   return t;
        
        # Convert auto to a numpy array if it's not already.
        if not isinstance(auto, np.ndarray):
            if is_iterable(auto):
                auto = np.array(auto)
            else:
                auto = np.array([bool(auto), bool(auto), bool(auto)])
        newsize = _make_array(newsize, np.float32)
        
        # Get current bounding box sizes
        bbox: tuple[float, float, float, float, float, float] = self.manifold.bounding_box()
        sizes = np.array(bbox[3:]) - np.array(bbox[:3])  # high - low gives dimensions
        
        # Find largest dimension
        maxdim = np.argmax(newsize)
        
        # Default scale (scale with 1 if the new size is 0)
        scale = np.ones(3)
        scale[newsize > 0] = newsize[newsize > 0] / sizes[newsize > 0]
        
        # Autoscale where applicable
        autoscale = scale[maxdim]
        newscale = np.where(~auto | (newsize > 0), scale, autoscale)
        
        return newscale
        
    def resize(self, newsize: np.ndarray, auto: np.ndarray | bool) -> 'RenderContext':
        
        newscale: np.ndarray = self.getResizeScale(newsize, auto)
        # Create a scaling transformation matrix.
        transform = np.array([
            [newscale[0], 0, 0, 0],
            [0, newscale[1], 0, 0],
            [0, 0, newscale[2], 0],
            [0, 0, 0, 1]
        ])
        
        return self.transform(transform)
    
    def scale(self, v: np.ndarray | float) -> 'RenderContext':
        if is_iterable(v):
            xform = np.array([
                [v[0], 0, 0, 0],
                [0, v[1], 0, 0],
                [0, 0, v[2], 0],
                [0, 0, 0, 1]
                ])
        else:
            xform = np.array([
                [v, 0, 0, 0],
                [0, v, 0, 0],
                [0, 0, v, 0],
                [0, 0, 0, 1]
                ])
        return self.transform(xform)
    
    

class M3dRenderer:
    def cube(self, size: tuple[float, float, float] | float, center: bool = False) -> RenderContext:
        if is_iterable(size):
            size = np.array(size)
        else:
            size = np.array([size, size, size]) 
        return RenderContext(self, m3d.Manifold.cube(size, center))

    def sphere(self, radius: float, fn: int = 16) -> RenderContext:
        return RenderContext(self, m3d.Manifold.sphere(radius=radius, circular_segments=fn))

    def cylinder(
        self, h: float, r_base: float, r_top: float = -1.0, fn: int = 0, center: bool = False
    ) -> RenderContext:
        return RenderContext(
            self,
            m3d.Manifold.cylinder(
                height=h, radius_low=r_base, radius_high=r_top, circular_segments=fn, center=center
            )
        )

    def mesh(
        self,
        vert_properties: NpArray[np.float32],
        tri_verts: NpArray[np.uint32],
        merge_from_vert: NpArray[np.uint32] | None = None,
        merge_to_vert: NpArray[np.uint32] | None = None,
        run_index: NpArray[np.uint32] | None = None,
        run_original_id: NpArray[np.uint32] | None = None,
        run_transform: NpArray[np.float32] | None = None,
        face_id: NpArray[np.uint32] | None = None,
        halfedge_tangent: NpArray[np.float32] | None = None,
        tolerance: float = 0,
    ) -> RenderContext:
        """
        /// Number of property vertices
        I NumVert() const { return vertProperties.size() / numProp; };
        /// Number of triangles
        I NumTri() const { return triVerts.size() / 3; };
        /// Number of properties per vertex, always >= 3.
        I numProp = 3;
        /// Flat, GL-style interleaved list of all vertex properties: propVal =
        /// vertProperties[vert * numProp + propIdx]. The first three properties are
        /// always the position x, y, z.
        std::vector<Precision> vertProperties;
        /// The vertex indices of the three triangle corners in CCW (from the outside)
        /// order, for each triangle.
        std::vector<I> triVerts;
        /// Optional: A list of only the vertex indicies that need to be merged to
        /// reconstruct the manifold.
        std::vector<I> mergeFromVert;
        /// Optional: The same length as mergeFromVert, and the corresponding value
        /// contains the vertex to merge with. It will have an identical position, but
        /// the other properties may differ.
        std::vector<I> mergeToVert;
        /// Optional: Indicates runs of triangles that correspond to a particular
        /// input mesh instance. The runs encompass all of triVerts and are sorted
        /// by runOriginalID. Run i begins at triVerts[runIndex[i]] and ends at
        /// triVerts[runIndex[i+1]]. All runIndex values are divisible by 3. Returned
        /// runIndex will always be 1 longer than runOriginalID, but same length is
        /// also allowed as input: triVerts.size() will be automatically appended in
        /// this case.
        std::vector<I> runIndex;
        /// Optional: The OriginalID of the mesh this triangle run came from. This ID
        /// is ideal for reapplying materials to the output mesh. Multiple runs may
        /// have the same ID, e.g. representing different copies of the same input
        /// mesh. If you create an input MeshGL that you want to be able to reference
        /// as one or more originals, be sure to set unique values from ReserveIDs().
        std::vector<uint32_t> runOriginalID;
        /// Optional: For each run, a 3x4 transform is stored representing how the
        /// corresponding original mesh was transformed to create this triangle run.
        /// This matrix is stored in column-major order and the length of the overall
        /// vector is 12 * runOriginalID.size().
        std::vector<Precision> runTransform;
        /// Optional: Length NumTri, contains the source face ID this
        /// triangle comes from. When auto-generated, this ID will be a triangle index
        /// into the original mesh. This index/ID is purely for external use (e.g.
        /// recreating polygonal faces) and will not affect Manifold's algorithms.
        std::vector<I> faceID;
        /// Optional: The X-Y-Z-W weighted tangent vectors for smooth Refine(). If
        /// non-empty, must be exactly four times as long as Mesh.triVerts. Indexed
        /// as 4 * (3 * tri + i) + j, i < 3, j < 4, representing the tangent value
        /// Mesh.triVerts[tri][i] along the CCW edge. If empty, mesh is faceted.
        std::vector<Precision> halfedgeTangent;
        /// Tolerance for mesh simplification. When creating a Manifold, the tolerance
        /// used will be the maximum of this and a baseline tolerance from the size of
        /// the bounding box. Any edge shorter than tolerance may be collapsed.
        /// Tolerance may be enlarged when floating point error accumulates.
        Precision tolerance = 0;
        """

        mesh = m3d.Mesh(
            vert_properties=_make_array(vert_properties, np.float32),
            tri_verts=_make_array(tri_verts, np.uint32),
            merge_from_vert=_make_array(merge_from_vert, np.uint32),
            merge_to_vert=_make_array(merge_to_vert, np.uint32),
            run_index=_make_array(run_index, np.uint32),
            run_original_id=_make_array(run_original_id, np.uint32),
            run_transform=_make_array(run_transform, np.float32),
            face_id=_make_array(face_id, np.uint32),
            halfedge_tangent=_make_array(halfedge_tangent, np.float32),
            tolerance=tolerance
        )
        return RenderContext(self, m3d.Manifold(mesh))



    def polyhedron(
        self,
        verts: NpArray[np.float32] | list[list[float]],
        faces: NpArray[np.uint32] | list[list[int]] | None = None,
        triangles: NpArray[np.uint32] | list[list[int]] | None = None
    ) -> RenderContext:
        if triangles is not None and faces is not None:
            raise ValueError("Cannot specify both faces and triangles")
        
        if triangles is None:
            # Convert inputs to numpy arrays if they aren't already
            verts_array = np.array(verts, dtype=np.float32)
            
            # Triangulate each face and collect the indices
            tri_verts = []
            for face in faces:
                tri_verts.extend(triangulate_3d_face(verts_array, face))
        
            return self.mesh(
                vert_properties=verts_array,
                tri_verts=np.array(tri_verts, dtype=np.uint32)
            )
        elif faces is not None:
            return self.mesh(
                vert_properties=verts_array,
                tri_verts=triangles
            )
        else:
            raise ValueError("Must specify either faces or triangles but not both.")
        
    def intersection(self, manifolds: list[RenderContext]) -> RenderContext:
        result = manifolds[0]
        for manifold in manifolds[1:]:
            result = result.intersect(manifold)
        return result
    
    def union(self, manifolds: list[RenderContext]) -> RenderContext:
        result = manifolds[0]
        for manifold in manifolds[1:]:
            result = result.union(manifold)
        return result
    
    def difference(self, manifolds: list[RenderContext]) -> RenderContext:
        result = manifolds[0]
        for manifold in manifolds[1:]:
            result = result.difference(manifold)
        return result

def triangulate_3d_face(verts_array: np.ndarray, face: list[int]) -> list[list[int]]:
    """Triangulate a 3D face using earcut after projecting to 2D.
    
    Args:
        verts_array: Array of vertex coordinates
        face: List of vertex indices defining the face
        
    Returns:
        List of lists of vertex indices defining triangles
    """
    # Skip triangulation for triangles
    if len(face) == 3:
        return [face]
        
    # Get the vertices for this face, taking only x,y,z coordinates
    face_verts = np.array([verts_array[idx][:3] for idx in face])
    
    # Compute face normal using Newell's method with vectorized operations
    v1 = face_verts
    v2 = np.roll(face_verts, -1, axis=0)  # Shifted vertices for pairs
    normal = np.sum([
        (v1[:, 1] - v2[:, 1]) * (v1[:, 2] + v2[:, 2]),  # x component
        (v1[:, 2] - v2[:, 2]) * (v1[:, 0] + v2[:, 0]),  # y component
        (v1[:, 0] - v2[:, 0]) * (v1[:, 1] + v2[:, 1])   # z component
    ], axis=1)
    normal = normal / np.linalg.norm(normal)
    
    # Create rotation matrix to align normal with Z axis
    if not np.allclose(normal, [0, 0, 1]):
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(normal, z_axis)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        cos_theta = np.dot(normal, z_axis)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        # Rodriguez rotation formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        R = np.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)
        
        # Apply rotation to vertices
        face_verts = face_verts @ R.T
    
    # Project to 2D by dropping Z coordinate
    verts_2d = face_verts[:, :2]
    
    # Create ring array (number of vertices in each ring/polygon)
    rings = [len(face)]
    
    # Triangulate
    triangles = mapbox_earcut.triangulate_float64(verts_2d, rings)
    
    # Convert triangle indices back to original vertex indices and group into triplets
    return [[face[triangles[i]], face[triangles[i+1]], face[triangles[i+2]]] 
            for i in range(0, len(triangles), 3)]
