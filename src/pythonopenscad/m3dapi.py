from dataclasses import dataclass, field, replace
from itertools import chain
from typing import Any, Callable, Generic, Iterable, Self, TypeVar
import manifold3d as m3d
import numpy as np
import mapbox_earcut
import stl
from stl import mesh, Mode


TM3d = TypeVar("T")

# Colour map derived from:
# https://github.com/openscad/openscad/blob/master/src/core/ColorNode.cc#L51
COLOUR_MAP = {
    "aliceblue": (240/255, 248/255, 255/255),
    'antiquewhite': (250/255, 235/255, 215/255),
    'aqua': (0/255, 255/255, 255/255),
    'aquamarine': (127/255, 255/255, 212/255),
    'azure': (240/255, 255/255, 255/255),
    'beige': (245/255, 245/255, 220/255),
    'bisque': (255/255, 228/255, 196/255),
    'black': (0/255, 0/255, 0/255),
    'blanchedalmond': (255/255, 235/255, 205/255),
    'blue': (0/255, 0/255, 255/255),
    'blueviolet': (138/255, 43/255, 226/255),
    'brown': (165/255, 42/255, 42/255),
    'burlywood': (222/255, 184/255, 135/255),
    'cadetblue': (95/255, 158/255, 160/255),
    'chartreuse': (127/255, 255/255, 0/255),
    'chocolate': (210/255, 105/255, 30/255),
    'coral': (255/255, 127/255, 80/255),
    'cornflowerblue': (100/255, 149/255, 237/255),
    'cornsilk': (255/255, 248/255, 220/255),
    'crimson': (220/255, 20/255, 60/255),
    'cyan': (0/255, 255/255, 255/255),
    'darkblue': (0/255, 0/255, 139/255),
    'darkcyan': (0/255, 139/255, 139/255),
    'darkgoldenrod': (184/255, 134/255, 11/255),
    'darkgray': (169/255, 169/255, 169/255),
    'darkgreen': (0/255, 100/255, 0/255),
    'darkgrey': (169/255, 169/255, 169/255),
    'darkkhaki': (189/255, 183/255, 107/255),
    'darkmagenta': (139/255, 0/255, 139/255),
    'darkolivegreen': (85/255, 107/255, 47/255),
    'darkorange': (255/255, 140/255, 0/255),
    'darkorchid': (153/255, 50/255, 204/255),
    'darkred': (139/255, 0/255, 0/255),
    'darksalmon': (233/255, 150/255, 122/255),
    'darkseagreen': (143/255, 188/255, 143/255),
    'darkslateblue': (72/255, 61/255, 139/255),
    'darkslategray': (47/255, 79/255, 79/255),
    'darkslategrey': (47/255, 79/255, 79/255),
    'darkturquoise': (0/255, 206/255, 209/255),
    'darkviolet': (148/255, 0/255, 211/255),
    'deeppink': (255/255, 20/255, 147/255),
    'deepskyblue': (0/255, 191/255, 255/255),
    'dimgray': (105/255, 105/255, 105/255),
    'dimgrey': (105/255, 105/255, 105/255),
    'dodgerblue': (30/255, 144/255, 255/255),
    'firebrick': (178/255, 34/255, 34/255),
    'floralwhite': (255/255, 250/255, 240/255),
    'forestgreen': (34/255, 139/255, 34/255),
    'fuchsia': (255/255, 0/255, 255/255),
    'gainsboro': (220/255, 220/255, 220/255),
    'ghostwhite': (248/255, 248/255, 255/255),
    'gold': (255/255, 215/255, 0/255),
    'goldenrod': (218/255, 165/255, 32/255),
    'gray': (128/255, 128/255, 128/255),
    'green': (0/255, 128/255, 0/255),
    'greenyellow': (173/255, 255/255, 47/255),
    'grey': (128/255, 128/255, 128/255),
    'honeydew': (240/255, 255/255, 240/255),
    'hotpink': (255/255, 105/255, 180/255),
    'indianred': (205/255, 92/255, 92/255),
    'indigo': (75/255, 0/255, 130/255),
    'ivory': (255/255, 255/255, 240/255),
    'khaki': (240/255, 230/255, 140/255),
    'lavender': (230/255, 230/255, 250/255),
    'lavenderblush': (255/255, 240/255, 245/255),
    'lawngreen': (124/255, 252/255, 0/255),
    'lemonchiffon': (255/255, 250/255, 205/255),
    'lightblue': (173/255, 216/255, 230/255),
    'lightcoral': (240/255, 128/255, 128/255),
    'lightcyan': (224/255, 255/255, 255/255),
    'lightgoldenrodyellow': (250/255, 250/255, 210/255),
    'lightgray': (211/255, 211/255, 211/255),
    'lightgreen': (144/255, 238/255, 144/255),
    'lightgrey': (211/255, 211/255, 211/255),
    'lightpink': (255/255, 182/255, 193/255),
    'lightsalmon': (255/255, 160/255, 122/255),
    'lightseagreen': (32/255, 178/255, 170/255),
    'lightskyblue': (135/255, 206/255, 250/255),
    'lightslategray': (119/255, 136/255, 153/255),
    'lightslategrey': (119/255, 136/255, 153/255),
    'lightsteelblue': (176/255, 196/255, 222/255),
    'lightyellow': (255/255, 255/255, 224/255),
    'lime': (0/255, 255/255, 0/255),
    'limegreen': (50/255, 205/255, 50/255),
    'linen': (250/255, 240/255, 230/255),
    'magenta': (255/255, 0/255, 255/255),
    'maroon': (128/255, 0/255, 0/255),
    'mediumaquamarine': (102/255, 205/255, 170/255),
    'mediumblue': (0/255, 0/255, 205/255),
    'mediumorchid': (186/255, 85/255, 211/255),
    'mediumpurple': (147/255, 112/255, 219/255),
    'mediumseagreen': (60/255, 179/255, 113/255),
    'mediumslateblue': (123/255, 104/255, 238/255),
    'mediumspringgreen': (0/255, 250/255, 154/255),
    'mediumturquoise': (72/255, 209/255, 204/255),
    'mediumvioletred': (199/255, 21/255, 133/255),
    'midnightblue': (25/255, 25/255, 112/255),
    'mintcream': (245/255, 255/255, 250/255),
    'mistyrose': (255/255, 228/255, 225/255),
    'moccasin': (255/255, 228/255, 181/255),
    'navajowhite': (255/255, 222/255, 173/255),
    'navy': (0/255, 0/255, 128/255),
    'oldlace': (253/255, 245/255, 230/255),
    'olive': (128/255, 128/255, 0/255),
    'olivedrab': (107/255, 142/255, 35/255),
    'orange': (255/255, 165/255, 0/255),
    'orangered': (255/255, 69/255, 0/255),
    'orchid': (218/255, 112/255, 214/255),
    'palegoldenrod': (238/255, 232/255, 170/255),
    'palegreen': (152/255, 251/255, 152/255),
    'paleturquoise': (175/255, 238/255, 238/255),
    'palevioletred': (219/255, 112/255, 147/255),
    'papayawhip': (255/255, 239/255, 213/255),
    'peachpuff': (255/255, 218/255, 185/255),
    'peru': (205/255, 133/255, 63/255),
    'pink': (255/255, 192/255, 203/255),
    'plum': (221/255, 160/255, 221/255),
    'powderblue': (176/255, 224/255, 230/255),
    'purple': (128/255, 0/255, 128/255),
    'rebeccapurple': (102/255, 51/255, 153/255),
    'red': (255/255, 0/255, 0/255),
    'rosybrown': (188/255, 143/255, 143/255),
    'royalblue': (65/255, 105/255, 225/255),
    'saddlebrown': (139/255, 69/255, 19/255),
    'salmon': (250/255, 128/255, 114/255),
    'sandybrown': (244/255, 164/255, 96/255),
    'seagreen': (46/255, 139/255, 87/255),
    'seashell': (255/255, 245/255, 238/255),
    'sienna': (160/255, 82/255, 45/255),
    'silver': (192/255, 192/255, 192/255),
    'skyblue': (135/255, 206/255, 235/255),
    'slateblue': (106/255, 90/255, 205/255),
    'slategray': (112/255, 128/255, 144/255),
    'slategrey': (112/255, 128/255, 144/255),
    'snow': (255/255, 250/255, 250/255),
    'springgreen': (0/255, 255/255, 127/255),
    'steelblue': (70/255, 130/255, 180/255),
    'tan': (210/255, 180/255, 140/255),
    'teal': (0/255, 128/255, 128/255),
    'thistle': (216/255, 191/255, 216/255),
    'tomato': (255/255, 99/255, 71/255),
    'turquoise': (64/255, 224/255, 208/255),
    'violet': (238/255, 130/255, 238/255),
    'wheat': (245/255, 222/255, 179/255),
    'white': (255/255, 255/255, 255/255),
    'whitesmoke': (245/255, 245/255, 245/255),
    'yellow': (255/255, 255/255, 0/255),
    'yellowgreen': (154/255, 205/255, 50/255),
    'transparent': (0/255, 0/255, 0/255, 0/255),
}


def manifold_to_stl(
    manifold: m3d.Manifold, filename: str, file_obj=None, mode=Mode.AUTOMATIC, update_normals=True
):
    """Convert a manifold to STL format and either save to a file or write to a file-like object.

    Args:
        manifold: The manifold to convert
        filename: Path to save the STL file (ignored if file_obj is provided)
        file_obj: Optional file-like object to write to instead of a file
        mode: Mode to use for the STL file
        update_normals: Whether to update the normals of the mesh
    """
    m_mesh = manifold.to_mesh()
    tri_verts = m_mesh.tri_verts
    points = m_mesh.vert_properties[:, :3]

    # Count number of triangles
    num_triangles = len(tri_verts)

    # Create an empty data array for the mesh
    data = np.zeros(num_triangles, dtype=mesh.Mesh.dtype)

    # Efficiently assign triangle vertices
    data["vectors"] = points[tri_verts]

    # Create the mesh
    stl_mesh = mesh.Mesh(data)

    # Save the mesh to file or write to file object
    stl_mesh.save(filename, fh=file_obj, mode=mode, update_normals=update_normals)


class NpArray(np.ndarray, Generic[TM3d]):
    pass


class Vector3(np.ndarray, Generic[TM3d]):
    pass


def is_iterable(v):
    """Return True if v is an iterable."""
    return isinstance(v, Iterable) and not isinstance(v, (str, bytes))


def _make_array(v: NpArray[np.float32 | np.float64] | None, t: type[np.float32 | np.float64]) -> NpArray[np.float32 | np.float64] | None:
    """Condition array to be C-style contiguous and writeable."""
    if v is None:
        return None
    if not isinstance(v, np.ndarray) or not (
        v.flags.c_contiguous and v.flags.writeable and v.dtype == t
    ):
        v = np.array(v, dtype=t, order="C")
    return v


def _rotVSinCos(v: np.ndarray, sinr: float, cosr: float) -> np.ndarray:
    """Returns a matrix that causes a rotation about an axis vector v by the
    given sin and cos of the rotation angle."""
    u = v / np.linalg.norm(v[:3])
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
    return np.array([
        [cosr + ux2 * lcosr, uxy * lcosr - uz * sinr, uxz * lcosr + uy * sinr, 0],
        [uxy * lcosr + uz * sinr, cosr + uy2 * lcosr, uyz * lcosr - ux * sinr, 0],
        [uxz * lcosr - uy * sinr, uyz * lcosr + ux * sinr, cosr + uz2 * lcosr, 0],
        [0.0, 0, 0, 1],
    ])


def _rotXSinCos(sinr, cosr) -> np.ndarray:
    """Returns a Gmatrix for a rotation about the X axis given a sin/cos pair."""
    return np.array([[1.0, 0, 0, 0], [0, cosr, -sinr, 0], [0, sinr, cosr, 0], [0, 0, 0, 1]])


def _rotYSinCos(sinr, cosr) -> np.ndarray:
    return np.array([[cosr, 0.0, sinr, 0], [0, 1, 0, 0], [-sinr, 0, cosr, 0], [0, 0, 0, 1]])


def _rotZSinCos(sinr, cosr) -> np.ndarray:
    return np.array([[cosr, -sinr, 0, 0], [sinr, cosr, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _to_radians(degs: float) -> float:
    """Convert degrees to radians."""
    return degs * np.pi / 180


def _exactSinCos(degs: float) -> tuple[float, float]:
    """Returns the sin and cos of an angle in degrees, with exact values for the

    most common angles."""
    if degs == 0:
        return 0, 1
    elif degs == 90:
        return 1, 0
    elif degs == 180:
        return 0, -1
    elif degs == 270:
        return -1, 0
    else:
        return np.sin(_to_radians(degs)), np.cos(_to_radians(degs))


def _rotPitchRollYaw(pitch: float, roll: float, yaw: float) -> np.ndarray:
    """Returns a matrix that causes a rotation about the X, Y, and Z axes by the
    given pitch, roll, and yaw angles."""
    return (
        _rotZSinCos(*_exactSinCos(yaw))
        @ _rotYSinCos(*_exactSinCos(pitch))
        @ _rotXSinCos(*_exactSinCos(roll))
    )


# Define constant axes and mirror matrices
X_AXIS = np.array([1, 0, 0])
Y_AXIS = np.array([0, 1, 0])

MIRROR_X = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

MIRROR_Y = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

IDENTITY_TRANSFORM = np.eye(4)
IDENTITY_TRANSFORM.flags.writeable = False


def _mirror(axis: np.ndarray) -> np.ndarray:
    """Mirror at the origin about any plane. The axis provided is the normal to the mirror plane.

    Uses the Householder reflection matrix formula: I - 2 * (n ⊗ n) where n is the normalized normal vector
    and ⊗ is the outer product.
    """
    # Normalize the axis vector
    n = axis / np.linalg.norm(axis)

    # Calculate Householder reflection matrix for the 3x3 part
    # H = I - 2 * (n ⊗ n)
    outer_product = np.outer(n, n)
    reflect_3x3 = np.eye(3) - 2.0 * outer_product

    # Create full 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = reflect_3x3

    return transform


TM3d = TypeVar("TM3d", bound=m3d.Manifold | m3d.CrossSection)

@dataclass(init=False)
class RenderContext(Generic[TM3d]):
    """Each level of the rendering phase produces a RenderContext.
    While also wrapping manifold3d.Manifold, it also handles the OpenSCAD
    operations and flags.
    """

    api: "M3dRenderer"
    transform_mat: np.ndarray = field(default_factory=lambda: IDENTITY_TRANSFORM)
    solid_objs: tuple[TM3d, ...] = ()
    shell_objs: tuple[TM3d, ...] = ()

    def as_transparent(self) -> Self:
        """Applies the OpenSCAD % modifier."""
        return RenderContext(
            api=self.api,
            transform_mat=self.transform_mat,
            solid_objs=(),
            shell_objs=self.shell_objs + self.solid_objs,
        )

    def transform(self, transform: np.ndarray) -> Self:
        if transform.shape == (4, 4):
            # assert the bottom row is [0, 0, 0, 1]
            assert np.allclose(transform[-1], [0, 0, 0, 1])
        else:
            assert transform.shape == (3, 4)
            # Add a row of [0, 0, 0, 1] to the bottom
            transform = np.concatenate([transform, [[0, 0, 0, 1]]])

        new_transform = transform @ self.transform_mat
        cls = type(self)
        return cls(self.api, new_transform, self.solid_objs, self.shell_objs)

    def translate(self, v: np.ndarray) -> Self:
        return self.transform(
            np.array([[1, 0, 0, v[0]], [0, 1, 0, v[1]], [0, 0, 1, v[2]], [0, 0, 0, 1]])
        )

    def rotate(self, a: float | np.ndarray, v: np.ndarray | None = None) -> Self:
        if is_iterable(a):
            if v is not None:
                raise ValueError("Cannot specify both a vector and v for rotation")
            return self.transform(_rotPitchRollYaw(*a))
        else:
            if v is None:
                return self.transform(_rotXSinCos(*_exactSinCos(a)))

            transform = _rotVSinCos(v, *_exactSinCos(a))
            return self.transform(transform)

    def mirror(self, normal: np.ndarray) -> Self:
        return RenderContext(self.api, self.manifold.mirror(_make_array(normal, np.float64)))

    def property(self, property_values: NpArray[np.float32], idx: int = 3) -> Self:
        # TODO: Implement this
        pass

    def get_solids(self) -> tuple[TM3d, ...]:
        return self.solid_objs

    def get_shells(self) -> tuple[TM3d, ...]:
        return self.shell_objs
    
    def _to_object_transform(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this")

    def _apply_transforms(
        self, get_type_func: Callable[[], tuple[TM3d, ...]]
    ) -> tuple[TM3d, ...]:
        if self.transform_mat is IDENTITY_TRANSFORM:
            return get_type_func()
        
        obj_transform = self._to_object_transform()
        manifs: tuple[TM3d, ...] = tuple(m.transform(obj_transform) for m in get_type_func())
        return manifs

    def _apply_and_merge(
        self, get_type_func: Callable[[], tuple[TM3d, ...]]
    ) -> tuple[TM3d, ...]:
        manifs = self._apply_transforms(get_type_func)
        result_manifs = (
            (sum(manifs[1:], start=manifs[0]),)
            if len(manifs) > 1
            else manifs
        )
        return result_manifs

    def _apply_and_merge_helper(
        self, other: Self
    ) -> tuple[tuple[TM3d], tuple[TM3d], tuple[TM3d], tuple[TM3d]]:
        solids = self._apply_and_merge(self.get_solids)
        shells = self._apply_and_merge(self.get_shells)
        other_solids = other._apply_and_merge(other.get_solids)
        other_shells = other._apply_and_merge(other.get_shells)

        return solids, shells, other_solids, other_shells

    def union(self, other: Self) -> Self:
        solids, shells, other_solids, other_shells = self._apply_and_merge_helper(other)

        cls = type(self)
        return cls(
            self.api, IDENTITY_TRANSFORM, solids + other_solids, shells + other_shells
        )

    def intersect(self, other: Self) -> Self:
        solids, shells, other_solids, other_shells = self._apply_and_merge_helper(other)

        if solids and other_solids:
            result_solids = (solids[0] ^ other_solids[0],)
        elif solids:
            result_solids = (solids[0],)
        elif other_solids:
            result_solids = (other_solids[0],)
        else:
            result_solids = ()

        # We union the shells since the significance of the shell is to show it's
        # encolsing volume for development purposes.
        result_shells = shells + other_shells

        cls = type(self)
        return cls(self.api, IDENTITY_TRANSFORM, result_solids, result_shells)

    def difference(self, other: Self) -> Self:
        solids, shells, other_solids, other_shells = self._apply_and_merge_helper(other)

        if solids and other_solids:
            result_solids = (solids[0] - other_solids[0],)
        elif solids:
            result_solids = (solids[0],)
        elif other_solids:
            result_solids = ()  # When we remove a volume from nothing we get nothing.
        else:
            result_solids = ()

        # We union the shells since the significance of the shell is to show it's
        # encolsing volume for development purposes.
        result_shells = shells + other_shells
        
        cls = type(self)
        return cls(self.api, IDENTITY_TRANSFORM, result_solids, result_shells)

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

        manifold = self.get_solid_manifold()
        # Get current bounding box sizes
        bbox: tuple[float, float, float, float, float, float] = manifold.bounding_box()
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

    def resize(self, newsize: np.ndarray, auto: np.ndarray | bool) -> Self:
        newscale: np.ndarray = self.getResizeScale(newsize, auto)
        # Create a scaling transformation matrix.
        transform = np.array([
            [newscale[0], 0, 0, 0],
            [0, newscale[1], 0, 0],
            [0, 0, newscale[2], 0],
            [0, 0, 0, 1],
        ])

        return self.transform(transform)

    def scale(self, v: np.ndarray | float) -> Self:
        if is_iterable(v):
            xform = np.array([[v[0], 0, 0, 0], [0, v[1], 0, 0], [0, 0, v[2], 0], [0, 0, 0, 1]])
        else:
            xform = np.array([[v, 0, 0, 0], [0, v, 0, 0], [0, 0, v, 0], [0, 0, 0, 1]])
        return self.transform(xform)

    def to_single_solid(self) -> Self:
        solids = self._apply_and_merge(self.get_solids)
        shells = self._apply_and_merge(self.get_shells)
        cls = type(self)
        return cls(self.api, self.transform_mat, solids, shells)


@dataclass
class RenderContextManifold(RenderContext[m3d.Manifold]):
    
    def __post_init__(self):
        for solid in self.solid_objs:
            if not isinstance(solid, m3d.Manifold):
                raise ValueError("All solid objects must be manifolds")
        for shell in self.shell_objs:
            if not isinstance(shell, m3d.Manifold):
                raise ValueError("All shell objects must be manifolds")
            
    def _to_object_transform(self) -> np.ndarray:
        transform_43 = self.transform_mat[:3, :]
        return transform_43
    
    @staticmethod
    def with_manifold(api: "M3dRenderer", manifold: m3d.Manifold) -> "RenderContextManifold":
        return RenderContextManifold(api=api, transform_mat=IDENTITY_TRANSFORM, solid_objs=(manifold,))

    def with_solid(self, manifold: m3d.Manifold) -> Self:
        solids, shells = self._apply_transforms()
        return RenderContextManifold(self.api, self.transform_mat, solids + (manifold,), shells)

    def with_shell(self, manifold: m3d.Manifold) -> Self:
        solids, shells = self._apply_transforms()
        return RenderContextManifold(self.api, self.transform_mat, solids, shells + (manifold,))
    
    def get_solid_manifold(self) -> m3d.Manifold:
        solids = self._apply_and_merge(self.get_solids)
        return solids[0] if solids else m3d.Manifold()

    def get_shell_manifolds(self) -> tuple[m3d.Manifold, ...]:
        shells = self._apply_and_merge(self.get_shells)
        return shells[0] if shells else m3d.Manifold()

    def write_solid_stl(
        self, filename: str, mode: Mode = Mode.AUTOMATIC, update_normals: bool = True
    ):
        """Write the solid manifold to an STL file."""
        manifold = self.get_solid_manifold()
        manifold_to_stl(manifold, filename, mode=mode, update_normals=update_normals)

    def write_shell_stl(
        self, filename: str, mode: Mode = Mode.AUTOMATIC, update_normals: bool = True
    ):
        """Write the shell manifolds to an STL file."""
        manifold = self.get_shell_manifolds()
        manifold_to_stl(manifold, filename, mode=mode, update_normals=update_normals)

@dataclass
class RenderContextCrossSection(RenderContext[m3d.CrossSection]):
    
    def __post_init__(self):
        for solid in self.solid_objs:
            if not isinstance(solid, m3d.CrossSection):
                raise ValueError("All solid objects must be manifolds")
        for shell in self.shell_objs:
            if not isinstance(shell, m3d.CrossSection):
                raise ValueError("All shell objects must be manifolds")

    def _to_object_transform(self) -> np.ndarray:
        transform_23 = self.transform_mat[:2, [0,1,3]]
        return transform_23
            
    def get_bbox(self) -> tuple[float, float, float, float]:
        solids = self.get_solids()
        if len(solids) != 1:
            solids = self._apply_and_merge(self.get_solids)
        return solids[0].bounds()
            

def set_property(manifold: m3d.Manifold, prop: np.ndarray, prop_index: int) -> m3d.Manifold:
    """
    Set a constant property on a manifold on all vertices.
    """
    
    min_num_props = prop_index + len(prop)
    curr_num_props = manifold.num_prop()
    if curr_num_props < min_num_props:
        num_props = min_num_props
    else:
        num_props = curr_num_props

    prop_len = len(prop)
    
    apply_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    # Select the appropriate apply function based on the property index and current 
    # number of properties.
    if prop_index == 0:
        if curr_num_props > len(prop):
            apply_func = lambda pos, curr_props: np.concatenate((prop, curr_props[prop_len:]))
        else:
            apply_func = lambda pos, curr_props: prop
    elif num_props > curr_num_props:
        apply_func = lambda pos, curr_props: np.concatenate((curr_props[:prop_index], prop))
    elif curr_num_props < prop_index:
        padding = np.zeros(prop_index - curr_num_props)
        apply_func = lambda pos, curr_props: np.concatenate((curr_props[:prop_index], padding, prop))
    else:
        apply_func = lambda pos, curr_props: np.concatenate(
                (curr_props[:prop_index], 
                 prop, 
                 curr_props[prop_index + prop_len:]))

    manifold = manifold.set_properties(num_props, apply_func)
    
    return manifold


@dataclass(frozen=True)
class M3dRenderer:
    
    NORMALS_PROP_INDEX = 4
    COLOR_PROP_INDEX = 0

    color_prop: np.ndarray = field(default_factory=lambda: np.array([0.976, 0.843, 0.173, 1.0]))
    
    def with_color(self, color: np.ndarray | list[float]) -> Self:
        """Returns a new renderer with the specified color and alpha.
        """
        if not isinstance(color, np.ndarray):
            color = np.array(color)
        if color.shape != (4,):
            raise ValueError("Color must be a 4-element array")
        return replace(self, color_prop=color)
    
    def _apply_properties(self, manifold: m3d.Manifold) -> m3d.Manifold:
        manifold = manifold.calculate_normals(self.NORMALS_PROP_INDEX)
        manifold = set_property(manifold, self.color_prop, self.COLOR_PROP_INDEX)
        return manifold
    
    def cube(self, size: tuple[float, float, float] | float, center: bool = False) -> RenderContextManifold:
        if is_iterable(size):
            size = np.array(size)
        else:
            size = np.array([size, size, size])
        return RenderContextManifold.with_manifold(self, self._apply_properties(m3d.Manifold.cube(size, center)))

    def sphere(self, radius: float, fn: int = 16) -> RenderContext:
        return RenderContextManifold.with_manifold(
            self, self._apply_properties(m3d.Manifold.sphere(radius=radius, circular_segments=fn)))

    def cylinder(
        self, h: float, r_base: float, r_top: float = -1.0, fn: int = 0, center: bool = False
    ) -> RenderContext:
        return RenderContextManifold.with_manifold(
            self,
            self._apply_properties(m3d.Manifold.cylinder(
                height=h,
                radius_low=r_base,
                radius_high=r_top,
                circular_segments=fn,
                center=center,
            ))
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
    ) -> RenderContextManifold:
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
        // corresponding original mesh was transformed to create this triangle run.
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
        
        vert_properties = _make_array(vert_properties, np.float32)
        
        if vert_properties is None:
            raise ValueError("vert_properties must be a numpy array")
        
        if vert_properties.shape[1] == 3:
            # Create array with right shape and fill with color values
            color_array = np.tile(self.color_prop, (vert_properties.shape[0], 1))
            vert_properties = np.column_stack((vert_properties, color_array))

        mesh = m3d.Mesh(
            vert_properties=vert_properties,
            tri_verts=tri_verts,
            merge_from_vert=merge_from_vert,
            merge_to_vert=merge_to_vert,
            run_index=run_index,
            run_original_id=_make_array(run_original_id, np.uint32),
            run_transform=_make_array(run_transform, np.float32),
            face_id=_make_array(face_id, np.uint32),
            halfedge_tangent=_make_array(halfedge_tangent, np.float32),
            tolerance=tolerance,
        )
        
        manifold = m3d.Manifold(mesh)
        manifold = manifold.calculate_normals(self.NORMALS_PROP_INDEX)

        return RenderContextManifold.with_manifold(self, manifold)

    def polyhedron(
        self,
        verts: NpArray[np.float32] | list[list[float]],
        faces: NpArray[np.uint32] | list[list[int]] | None = None,
        triangles: NpArray[np.uint32] | list[list[int]] | None = None,
    ) -> RenderContext:
        if triangles is not None and faces is not None:
            raise ValueError("Cannot specify both faces and triangles")

        if triangles is None:
            # Convert inputs to numpy arrays if they aren't already
            verts_array = np.array(verts, dtype=np.float32)

            # Triangulate each face and collect the indices
            tri_verts = []
            for face in faces:
                tri_verts.extend(triangulate_3d_face(verts_array, [face]))

            return self.mesh(
                vert_properties=verts_array, tri_verts=np.array(tri_verts, dtype=np.uint32)
            )
        elif faces is not None:
            return self.mesh(vert_properties=verts_array, tri_verts=triangles)
        else:
            raise ValueError("Must specify either faces or triangles but not both.")

    def difference(self, ops: list[RenderContextManifold | RenderContextCrossSection]) \
        -> RenderContextManifold | RenderContextCrossSection:
        
        if len(ops) == 0:
            raise ValueError("Must specify at least one other render context")
        
        if len(ops) == 1:
            return ops[0]
        
        cls = type(ops[0])
        
        cls = type(ops[0])
        solids = tuple(chain(*(op._apply_and_merge(op.get_solids) for op in ops[1:])))
        shells = tuple(chain(*(op._apply_and_merge(op.get_shells) for op in ops[1:])))
        
        rhs = cls(self, transform_mat=IDENTITY_TRANSFORM, solid_objs=solids, shell_objs=shells)
        
        return ops[0].difference(rhs)
    
    def union(self, ops: list[RenderContextManifold | RenderContextCrossSection]) \
        -> RenderContextManifold | RenderContextCrossSection:
        if len(ops) == 0:
            raise ValueError("Must specify at least one other render context")
        
        if len(ops) == 1:
            return ops[0]   
        
        cls = type(ops[0])  
        
        solids = tuple(chain(*(op._apply_and_merge(op.get_solids) for op in ops)))
        shells = tuple(chain(*(op._apply_and_merge(op.get_shells) for op in ops)))
        
        return cls(self, transform_mat=IDENTITY_TRANSFORM, solid_objs=solids, shell_objs=shells)
    
    def intersection(self, ops: list[RenderContextManifold | RenderContextCrossSection]) \
        -> RenderContextManifold | RenderContextCrossSection:
        if len(ops) == 0:
            raise ValueError("Must specify at least one other render context")
        
        if len(ops) == 1:
            return ops[0]
        
        cls = type(ops[0])
        solids = tuple(chain(*(op._apply_transforms(op.get_solids) for op in ops)))
        shells = tuple(chain(*(op._apply_and_merge(op.get_shells) for op in ops)))

        intersected = solids[0]
        for solid in solids[1:]:
            intersected = intersected ^ solid
        
        rhs = cls(self, transform_mat=IDENTITY_TRANSFORM, solid_objs=(intersected,), shell_objs=shells)
        
        return ops[0].intersect(rhs)

    
    def import_file(self, file: str, layer: str, convexity: int) \
        -> RenderContextManifold | RenderContextCrossSection:
        raise NotImplementedError("import_file is not implemented")
    
    def surface(self, file: str, center: bool, invert: bool, convexity: int) \
        -> RenderContextManifold:
        raise NotImplementedError("surface is not implemented")
    
    def fill(self, ops: "list[RenderContext]") -> "RenderContext":
        raise NotImplementedError("fill is not implemented")
    
    def text(self, 
             text: str, 
             size: float, 
             font: str, 
             halign: str, 
             valign: str, 
             spacing: float, 
             direction: str, 
             language: str, 
             script: str, 
             fa: float, 
             fs: float, 
             fn: float) -> RenderContextCrossSection:
        raise NotImplementedError("text is not implemented")
    
    def polygon(self, 
                points: list[list[float]], 
                paths: list[list[int]], 
                convexity: int) -> RenderContextCrossSection:
        
        if not paths:
            paths = [list(range(len(points)))]
        
        points = _make_array(points, np.float32)
        paths = [_make_array(path, np.uint32) for path in paths]
        
        contours = [points[path] for path in paths]
        cross_section = m3d.CrossSection(contours, m3d.FillRule.Positive)
        
        return RenderContextCrossSection(self, solid_objs=(cross_section,))
    
    def square(self, size: float | tuple[float, float], center: bool = False) -> RenderContextCrossSection:
        return RenderContextCrossSection(self,
            solid_objs=(m3d.CrossSection.square(size, True if center else False),))
    
    def circle(self, radius: float, fn: int) -> RenderContextCrossSection:
        return RenderContextCrossSection(self,
            solid_objs=(m3d.CrossSection.circle(radius, fn),))
    
    def rotate_extrude(self, 
                       context: RenderContextCrossSection, 
                       angle: float, 
                       convexity: int, 
                       fn: int) -> RenderContextManifold:
        assert isinstance(context, RenderContextCrossSection)
        solids = context.get_solids()
        if len(solids) != 1:
            solids = context._apply_and_merge(context.get_solids)
        solid = solids[0]
        manifold = solid.revolve(fn, angle)
        
        rctxt =  RenderContextManifold(self,
            solid_objs=(self._apply_properties(manifold),))
        return rctxt
    
    
    
    def linear_extrude(self, 
                       contexts: list[RenderContextCrossSection], 
                       height: float, 
                       center: bool = False, 
                       convexity: int | None = None, 
                       twist: float | None = None, 
                       slices: int | None = None, 
                       scale: tuple[float, float] | None = None,
                       fn: float | None = None) -> RenderContextManifold:
        assert all(isinstance(c, RenderContextCrossSection) for c in contexts)
        union_solids = self.union(contexts)
        solids = union_solids._apply_and_merge(union_solids.get_solids)
        if not scale:
            scale = (1.0, 1.0)
        if twist is None:
            twist = 0.0
        if slices is None:
            slices = 16 if twist else 1
        rctxt =  RenderContextManifold(self,
            solid_objs=(self._apply_properties(solids[0].extrude(
                height, 
                slices, 
                twist, 
                scale)),))
        
        if center:
            return rctxt.translate([0, 0, -height / 2])
        return rctxt
        
    def hull(self, ops: list[RenderContextManifold | RenderContextCrossSection]) \
    -> RenderContextManifold | RenderContextCrossSection:
        raise NotImplementedError("hull is not implemented")
    
    def minkowski(self, ops: list[RenderContextManifold]) -> RenderContextManifold:
        raise NotImplementedError("minkowski is not implemented")
    
    def render(self, ops: list[RenderContextManifold], convexity: int) -> RenderContextManifold:
        return self.union(ops)
    
    def projection(self, ops: list[RenderContextManifold], cut: bool) -> RenderContextCrossSection:
        raise NotImplementedError("projection is not implemented")
    
    def offset(self, 
               ops: list[RenderContextCrossSection], 
               r: float, 
               delta: float, 
               chamfer: bool, 
               fa: float, 
               fs: float, 
               fn: float) -> RenderContextCrossSection:
        raise NotImplementedError("offset is not implemented")
    
    def color(self, c: str | np.ndarray | None = None, alpha: float | None = None) -> "M3dRenderer":
        """Returns a new renderer with the specified color and alpha.
        """
        if c is None:
            if alpha is None:
                raise ValueError("Both color and alpha cannot be None")
            if alpha > 1.0 or alpha < 0.0:
                raise ValueError("Alpha must be between 0.0 and 1.0")
            return self.with_color(np.concatenate((self.color_prop[:3], [alpha])))
        elif isinstance(c, str):
            color = COLOUR_MAP.get(c.lower())
            if color is None:
                raise ValueError(f"Invalid color: {c}")
            if alpha is None:
                alpha = 1.0
            if alpha > 1.0 or alpha < 0.0:
                raise ValueError("Alpha must be between 0.0 and 1.0")
            return self.with_color(np.concatenate((color, [alpha])))
        elif isinstance(c, np.ndarray):
            if c.shape == (3,):
                if alpha is None:
                    alpha = 1.0
                if alpha > 1.0 or alpha < 0.0:
                    raise ValueError("Alpha must be between 0.0 and 1.0")   
                c = np.concatenate((c, [alpha]))
            elif c.shape != (4,):
                raise ValueError("Color must be a 3 or 4 element numpy array")
        else:
            raise ValueError("Color must be a string or a numpy array")

        return self.with_color(c)
    

def _triangulate(verts_array: np.ndarray, rings: list[int]) -> list[list[int]]:
    """Calls mapbox_earcut.triangulate_float32 or float64 depending on the given dtype."""
    if verts_array.dtype == np.float32:
        return mapbox_earcut.triangulate_float32(verts_array, rings)
    elif verts_array.dtype == np.float64:
        return mapbox_earcut.triangulate_float64(verts_array, rings)
    else:
        raise ValueError("verts_array must be a numpy array of float32 or float64")

def triangulate_3d_face(verts_array: np.ndarray, face: list[list[int]]) -> list[list[int]]:
    """Triangulate a 3D face using earcut. The face is assumed to be close to
    planar and the surface is rotated to ensure the normal is pointing in the
    +Z direction.
    
    face is a list of lists. This may consist of multiple polygons where the winding order
    defines holes.

    Args:
        verts_array: Array of vertex coordinates
        face: List of vertex indices defining the face

    Returns:
        List of lists of vertex indices defining triangles
    """
    # Skip triangulation for triangles
    if len(face) == 1 and len(face[0]) == 3:
        return [face[0]]

    # Get the vertices for this face, taking only x,y,z coordinates
    face_verts = np.array([verts_array[idx][:3] for idx in np.concatenate(face)])

    # Compute face normal using Newell's method with vectorized operations
    v1 = face_verts
    v2 = np.roll(face_verts, -1, axis=0)  # Shifted vertices for pairs
    normal = np.sum(
        [
            (v1[:, 1] - v2[:, 1]) * (v1[:, 2] + v2[:, 2]),  # x component
            (v1[:, 2] - v2[:, 2]) * (v1[:, 0] + v2[:, 0]),  # y component
            (v1[:, 0] - v2[:, 0]) * (v1[:, 1] + v2[:, 1]),  # z component
        ],
        axis=1,
    )
    normal = normal / np.linalg.norm(normal)

    # Create rotation matrix to align normal with Z axis
    # We always want the normal to point approximately in +Z direction to ensure
    # consistent triangulation
    cos_theta = np.dot(normal, [0, 0, 1])
    flip_order = False
    # Always rotate to align with +Z if the normal is not already close to +Z.
    if abs(cos_theta) < 0.93:
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(normal, z_axis)
        norm_rotation_axis = np.linalg.norm(rotation_axis)

        if norm_rotation_axis < 1e-10:  # If rotation axis is too small, use X axis
            rotation_axis = np.array([1, 0, 0])
        else:
            rotation_axis = rotation_axis / norm_rotation_axis

        # Calculate sin_theta from cos_theta, always producing a positive sine value
        # Note: This is a simplification that works for our purposes because:
        # 1. When cos_theta > 0 (normal pointing toward +Z), positive sin_theta gives correct rotation
        # 2. When cos_theta < 0 (normal pointing toward -Z), positive sin_theta creates a rotation that
        #    effectively flips the face's winding order, which is what we want for projection to +Z
        # This "incorrect" sine calculation actually works in our favor by automatically handling
        # the winding order change needed when rotating from -Z to +Z direction.
        sin_theta = np.sqrt(1 - cos_theta**2)

        # Rodriguez rotation formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0],
        ])
        R = np.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)

        # Apply rotation to vertices
        face_verts = face_verts @ R.T
    else:
        if cos_theta < 0:
            flip_order = True

    # Project to 2D by dropping Z coordinate (now facing +/-Z)
    verts_2d = face_verts[:, :2]

    # Create ring array (cumulative number of vertices in each ring/polygon)
    rings = np.cumsum([len(poly) for poly in face])
    
    # Triangulate
    triangles = _triangulate(verts_2d, rings)

    # Convert triangle indices back to original vertex indices and group into triplets
    # First create a mapping from flattened index to (polygon, vertex) indices
    offsets = np.concatenate(([0], rings[:-1]))  # Starting index of each polygon
    polygon_indices = np.searchsorted(rings, np.array(triangles), side='right')
    vertex_indices = np.array(triangles) - offsets[polygon_indices]
    
    tris = [
        [face[p][v] for p, v in zip(polygon_indices[i:i+3], vertex_indices[i:i+3])]
        for i in range(0, len(triangles), 3)
    ]

    if flip_order and len(tris) > 0:
        tris = [[tri[0], tri[2], tri[1]] for tri in tris]

    return tris
