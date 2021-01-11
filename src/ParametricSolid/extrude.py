'''
Created on 7 Jan 2021

@author: gianni
'''

from dataclasses import dataclass

from frozendict import frozendict
from scipy.constants.constants import degree

import ParametricSolid.core as core
import ParametricSolid.linear as l
import numpy as np


class DuplicateNameException(Exception):
    '''The name requested is already used.'''

class MoveNotAllowedException(Exception):
    '''Attempt to insert a move in a closed path.'''
    
class InvalidSplineParametersException(Exception):
    '''PathBuiler.spine requires 3 control points or 2 points and a length.'''
    
class IncorrectAnchorArgsException(Exception):
    '''Unable to interpret args..'''
    
class UnknownOperationException(Exception):
    '''Requested anchor is not found...'''

def strict_t_or_none(v, t):
    if v is None or v == 'None':
        return None
    
    if isinstance(v, str):
        raise TypeError(
            'Was provided a string value but expecting a numeric value or None.')
    return t(v)


def strict_int_or_none(v):
    return strict_t_or_none(v, int)

def strict_float_or_none(v):
    return strict_t_or_none(v, float)

LIST_2_FLOAT_OR_NONE = l.list_of(strict_float_or_none, len_min_max=(2, 2), fill_to_min='None')
LIST_2_INT_OR_NONE = l.list_of(strict_float_or_none, len_min_max=(2, 2), fill_to_min='None')
LIST_2_FLOAT = l.list_of(l.strict_float, len_min_max=(2, 3), fill_to_min=0.0)
LIST_3_FLOAT = l.list_of(l.strict_float, len_min_max=(3, 3), fill_to_min=0.0)
LIST_3X2_FLOAT = l.list_of(LIST_2_FLOAT, len_min_max=(3, 3), fill_to_min=None)
LIST_23X2_FLOAT = l.list_of(LIST_2_FLOAT, len_min_max=(2, 3), fill_to_min=None)

def _normalize(v):
    l = np.sqrt(np.sum(v**2))
    return v / l

@dataclass(frozen=True)
class CubicSpline():
    '''Cubic spline evaluator, extents and inflection point finder.'''
    p: object
    dimensions: int=2
    
    COEFFICIENTS=np.array([
        [-1.,  3, -3,  1 ],
        [  3, -6,  3,  0 ],
        [ -3,  3,  0,  0 ],
        [  1,  0,  0,  0 ]])
    
    def _dcoeffs_builder(dims):
        zero_order_derivative_coeffs=np.array([[1.] * dims, [1] * dims, [1] * dims, [1] * dims])
        derivative_coeffs=np.array([[3.] * dims, [2] * dims, [1] * dims, [0] * dims])
        second_derivative=np.array([[6] * dims, [2] * dims, [0] * dims, [0] * dims])
        return (zero_order_derivative_coeffs, derivative_coeffs, second_derivative)
    
    DERIVATIVE_COEFFS = tuple((
        _dcoeffs_builder(1), 
        _dcoeffs_builder(2), 
        _dcoeffs_builder(3), ))
    
    def _dcoeffs(self, deivative_order):
        return self.DERIVATIVE_COEFFS[self.dimensions - 1][deivative_order]
    
    class InvalidTypeForP(Exception):
        '''The parameter p must be a numpy.ndarray.'''
        
    def __post_init__(self):
        object.__setattr__(self, 'coefs', np.matmul(self.COEFFICIENTS, self.p))
    
    def _make_ta3(self, t):
        t2 = t * t
        t3 = t2 * t
        ta = [c * self.dimensions for c in [[t3], [t2], [t], [1]]]
        return ta
        
    def _make_ta2(self, t):
        t2 = t * t
        ta = [c * self.dimensions for c in [[t2], [t], [1], [0]]]
        return ta
    
    def evaluate(self, t):
        return np.sum(np.multiply(self.coefs, self._make_ta3(t)), axis=0)
    
  
    @classmethod
    def find_roots(cls, a, b, c, *, t_range=(0.0, 1.0)):
        '''Find roots of quaratic polynomial that are between t_range.'''
        # a, b, c are quadratic coefficients i.e. at^2 + bt + c
        if a == 0:
            # Degenerate curve is a linear. Only one possible root.
            if b == 0:
                # Degenerate curve is constant so there is no 0 gradient.
                return ()
            t = -c / b
            
            return (t,) if  t >= t_range[0] and t <= t_range[1] else ()
    
        b2_4ac = b * b - 4 * a * c;
        if b2_4ac < 0:
            # Complex roots - no answer.
            return ()
    
        sqrt_b2_4ac = np.sqrt(b2_4ac)
        two_a = 2 * a
    
        values = ((-b + sqrt_b2_4ac) / two_a, (-b - sqrt_b2_4ac) / two_a)
        return tuple(t for t in values if t >= t_range[0] and t <= t_range[1])
    
    # Solve for minima and maxima over t. There are two possible locations 
    # for each axis. The results for t outside of the bounds 0-1 are ignored
    # since the cubic spline is only interpolated in those bounds.
    def cuve_maxima_minima_t(self, t_range=(0.0, 1.0)):
        '''Returns a dict with an entry for each dimension containing a list of
        t for each minima or maxima found.'''
        # Splines are defined only for t in the range [0..1] however the curve may
        # go beyond those points. Each axis has a potential of two roots.
        d_coefs = self.coefs * self._dcoeffs(1)
        return dict((i, self.find_roots(*(d_coefs[0:3, i]), t_range=t_range)) 
                    for i in range(self.dimensions))
    
    
    def cuve_inflexion_t(self, t_range=(0.0, 1.0)):
        '''Returns a dict with an entry for each dimension containing a list of
        t for each inflection point found.'''
        # Splines are defined only for t in the range [0..1] however the curve may
        # go beyond those points. Each axis has a potential of two roots.
        d_coefs = self.coefs * self._dcoeffs(2)
        return dict((i, self.find_roots(0., *(d_coefs[0:2, i]), t_range=t_range))
                    for i in range(self.dimensions))
    
    def derivative(self, t):
        return -np.sum(
            np.multiply(
                np.multiply(self.coefs, self._dcoeffs(1)), self._make_ta2(t)), axis=0)
    
    def normal2d(self, t, dims=[0, 1]):
        '''Returns the normal to the curve at t for the 2 given dimensions.'''
        d = self.derivative(t)
        vr = np.array([d[dims[0]], -d[dims[1]]])
        l = np.sqrt(np.sum(vr**2))
        return vr / l
    
    def extents(self):
        roots = self.cuve_maxima_minima_t()
        
        minima_maxima = []

        start = self.p[0]
        end = self.p[3]
        for i in range(self.dimensions):
            v = [float(start[i]), float(end[i])]
            v.extend(tuple((self.evaluate(t)[i] for t in roots[i] if t >= 0 and t <= 1),))
            minima_maxima.append([np.min(v), np.max(v)])
    
        return np.transpose(minima_maxima)


def _normal_of_2d(v1, v2, dims=[0, 1]):
    vr = np.array(v1)
    vr[dims[0]] = v1[dims[1]] - v2[dims[1]]
    vr[dims[1]] = v2[dims[0]] - v1[dims[0]]
    l = np.sqrt(np.sum(vr * vr))
    return vr / l

    
@dataclass(frozen=True)
class Path():
    ops: tuple
    name_map: frozendict

    def get_node(self, name):
        return self.name_map.get(name, None)
    
    def extents(self):
        extents = None
        for op in self.ops:
            cur = op.extents()
            if cur is None:
                continue
            cur = np.transpose(op.extents())
            if extents is None:
                extents = cur
            else:
                extents = [
                    [min(v1[0], v2[0]), max(v1[1], v2[1])] 
                    for v1, v2 in zip(extents, cur)]
                
        return np.transpose(extents)
    
    def build(self, meta_data):
        path_builder = []
        start_indexes = []
        map_builder = []
        for op in self.ops:
            op.populate(path_builder, start_indexes, map_builder, meta_data)
        return (np.array(path_builder), start_indexes, map_builder)
    
    def points(self, meta_data):
        points, _, _ = self.build(meta_data)
        return points

    def polygons(self, meta_data):
        points, start_indexes, map_ops = self.build(meta_data)
        if len(start_indexes) == 1:
            return (points,)
        
        indexes = start_indexes + [len(points),]
        return (points, 
                (tuple(tuple(range(indexes[i], indexes[i+1])) for i in range(len(start_indexes)))))

@dataclass()
class PathBuilder():
    ops: list
    name_map: dict
    multi: bool=False
    
    @dataclass(frozen=True)
    class _LineTo:
        '''Line segment from current position.'''
        point: np.array
        prev_op: object
        name: str=None
            
        def lastPosition(self):
            return self.point
        
        def populate(self, path_builder, start_indexes, map_builder, meta_data):
            path_builder.append(self.point)
            map_builder.append((self,))
            
        def direction(self, t):
            return self.point - self.prev_op.lastPosition()
        
        def direction_normalized(self, t):
            return _normalize(self.direction(t))
        
        def normal2d(self, dims=[0, 1]):
            return _normal_of_2d(self.prev_op.lastPosition(), self.point, dims)
        
        def extents(self):
            p0 = self.prev_op.lastPosition()
            p1 = self.point
            return np.transpose(
                list(([p0[k], p1[k]] if p0[k] < p1[k] else [p1[k], p0[k]]) for k in range(len(p0))))
            
        def position(self, t):
            return self.point + (t - 1) * self.direction(0)
            
    
    @dataclass(frozen=True)
    class _MoveTo:
        '''Move to position.'''
        point: np.array
        prev_op: object
        name: str=None
            
        def lastPosition(self):
            return self.point
        
        def populate(self, path_builder, start_indexes, map_builder, meta_data):
            path_builder.append(self.point)
            start_indexes.append(len(path_builder))
            map_builder.append((self,))
            
        def direction(self, t):
            return None
            
        def direction_normalized(self, t):
            return None
        
        def normal2d(self, t, dims=[0, 1]):
            return None
        
        def extents(self):
            return None
        
        def position(self, t):
            return self.point  # Move is associated only with the move point. 


    @dataclass(frozen=True)
    class _SplineTo:
        '''Cubic Bezier Spline to.'''
        points: np.array
        prev_op: object
        name: str=None
        meta_data: object=None
        
        def __post_init__(self):
            to_cat = [[self.prev_op.lastPosition()],  self.points]
            spline_points = np.concatenate(to_cat)
            object.__setattr__(self, 'spline', CubicSpline(spline_points))
            
        def lastPosition(self):
            return self.points[2]
            
        def populate(self, path_builder, start_indexes, map_builder, meta_data):
            if (self.meta_data):
                meta_data = self.meta_data
    
            count = meta_data.fn
            if not count:
                count = 10
    
            for i in range(1, count + 1):
                t = float(i) / float(count)
                point = self.spline.evaluate(t)
                path_builder.append(point)
                map_builder.append((self, t, count))
    
        def direction(self, t):
            return -self.spline.derivative(t)
        
        def direction_normalized(self, t):
            return _normalize(self.direction(t))
        
        def normal2d(self, t, dims=[0, 1]):
            return self.spline.normal2d(t, dims)
        
        def extents(self):
            return self.spline.extents()
        
        def position(self, t):
            if t < 0:
                return self.direction(0) * t + self.prev_op.lastPosition()
            elif t > 1:
                return self.direction(1) * t + self.points[2]
            return self.spline.evaluate(t)
    
    def __init__(self, multi=False):
        self.ops = []
        self.name_map = {}
        self.multi = multi
        
    def add_op(self, op):
        if op.name:
            if op.name in self.name_map:
                raise DuplicateNameException(f'Duplicate name ({op.name!r}) is already used.')
            self.name_map[op.name] = op
        self.ops.append(op)
        return self
        
    def last_op(self):
        return self.ops[-1] if self.ops else None
        
    def move(self, point, name=None):
        if not self.multi and self.ops:
            raise MoveNotAllowedException(f'Move is not allowed in non multi-path builder.')
        return self.add_op(self._MoveTo(np.array(LIST_2_FLOAT(point)), self.last_op(), name))
                        
    def line(self, point, name=None):
        assert len(self.ops) > 0, "Cannot line to without starting point"
        return self.add_op(self._LineTo(np.array(LIST_2_FLOAT(point)), self.ops[-1], name))
             
    def spline(self, points, name=None, metadata=None, 
               cv_len=(None, None), degrees=(0, 0), radians=(0, 0), rel_len=None):
        '''Adds a spline node to the path.
        Args:
            points: Either 3 point list (first control point is the last point) or a 
                    2 point list and cv_len with the first element set to the distance 
                    the control point follows along the previous operations last direction.
            cv_len: If provided will force the length of the control point (1 an 2)
                    to be the given length.
            name: The name of this node. Naming a node will make it an anchor.
            metadata: Provides parameters for rendering that override the renderer metadata.
            degrees: A 2 tuple that contains a rotation angle for control points 1 and 2
                    respectively.
            radians: line degrees but in radians. If radians are provided they override any
                    degrees values provided.
            rel_len: Forces control points to have relatively the same length as the
                    distance from the end points. If cv_len is set it is used as a multiplier.
        '''
        assert len(self.ops) > 0, "Cannot line to without starting point"
        degrees = LIST_2_INT_OR_NONE(degrees) if degrees else (None, None)
        radians = LIST_2_INT_OR_NONE(radians) if radians else (None, None)
        cv_len = LIST_2_FLOAT_OR_NONE(cv_len) if cv_len else (None, None)
        points = np.array(LIST_23X2_FLOAT(points))
        if len(points) == 2:
            if cv_len[0] is None:
                raise InvalidSplineParametersException(
                    'Only 2 control points provided so the direction of the previous operation'
                    ' will be used but a size (in cv_len. This needs a control vector size.')
            if self.ops[-1].direction_normalized(1.0) is None:
                raise InvalidSplineParametersException(
                    'Only 2 control points provided so the direction of the previous operation'
                    ' will be used but the previous operation (move) does not provide direction.')
            cv0 = self.ops[-1].lastPosition()
            cv1 = self.ops[-1].direction_normalized(1.0) * cv_len[0] + cv0
            cv2 = points[0]
            cv3 = points[1]
        else:
            cv0 = self.ops[-1].lastPosition()
            cv1 = points[0]
            cv2 = points[1]
            cv3 = points[2]
        if not rel_len is None:
            l = np.sqrt(np.sum((cv0 - cv3)**2))
            cv_len = tuple(rel_len * l if v is None else v * l * rel_len for v in cv_len)
        cv1 = self.squeeze_and_rot(cv0, cv1, cv_len[0], degrees[0], radians[0])
        cv2 = self.squeeze_and_rot(cv3, cv2, cv_len[1], degrees[1], radians[1])
        
        points = np.array(LIST_3X2_FLOAT([cv1, cv2, cv3]))
        return self.add_op(self._SplineTo(points, self.ops[-1], name, metadata))
    
    def squeeze_and_rot(self, point, control, cv_len, degrees, radians):
        if cv_len is None and not degrees and not radians:
            return control
        gpoint = l.GVector(LIST_3_FLOAT(point))
        gcontrol = l.GVector(LIST_3_FLOAT(control))
        g_rel = (gcontrol - gpoint)
        if not cv_len is None:
            g_rel = g_rel.N * cv_len

        if radians:
            g_rel = l.rotZ(radians=radians) * g_rel
        elif degrees:
            g_rel = l.rotZ(degrees=degrees) * g_rel
            
        return (gpoint + g_rel).A[0:len(point)]
        

    def get_node(self, name):
        return self.name_map.get(name, None)
    
    def build(self):
        return Path(tuple(self.ops), frozendict(self.name_map))
    

class ExtrudedShape(core.Shape):
        
    def has_anchor(self, name):
        return name in self.anchorscad.anchors or name in self.path.name_map
    
    def anchor_names(self):
        return tuple(set(self.anchorscad.anchors.keys()) + tuple(self.path.name_map.keys()))
    
    def at(self, anchor_name, *args, **kwds):
        spec = self.anchorscad.get(anchor_name)
        if spec:
            func = spec[0]
            try:
                return func(self, *args, **kwds)
            except TypeError as ex:
                raise IncorrectAnchorArgsException(
                    f'{ex}\nAttempted to call {anchor_name} on {self.__class__.__name__}'
                    f' with args={args!r} kwds={kwds!r}') from ex
        else:
            return self.node(anchor_name, *args, forward=False, **kwds)
            
    def to_3d_from_2d(self, vec_2d, h=0):
        return l.IDENTITY * l.GVector([vec_2d[0], vec_2d[1], h])
    
    @core.anchor('Anchor to the path for a given operation.')
    def node(self, path_node_name, *args, op='edge', forward=True, **kwds):
        if op == 'edge':
            return self.edge(path_node_name, *args, **kwds)
        
        op = self.path.name_map.get(path_node_name)
        if core.Shape.has_anchor(self, op) and forward:
            return self.at(op, path_node_name, *args, forward=False, **kwds)
        raise UnknownOperationException(
            f'Undefined anchor operation {op!r} for node {path_node_name!r}.')
        
    def eval_z_vector(self, h):
        return l.GVector([0, 0, h])

@core.shape('linear_extrude')
@dataclass
class LinearExtrude(ExtrudedShape):
    '''Generates an extrusion of a given Path.'''
    path: Path
    h: float=100
    twist: float=0.0
    slices: int=None
    scale: float=(1.0, 1.0)  # (x, y)
    fn: int=None
    
    SCALE=0.8
    
    EXAMPLE_SHAPE_ARGS=core.args(
        PathBuilder()
            .move([0, 0])
            .line([100 * SCALE, 0], 'linear')
            .spline([[150 * SCALE, 100 * SCALE], [20 * SCALE, 100 * SCALE]],
                     name='curve', cv_len=(0.5,0.4), degrees=(90,), rel_len=0.8)
            .line([0, 100 * SCALE], 'linear2')
            .line([0, 0], 'linear3')
            .build(),
        h=40,
        fn=30,
        twist=0,
        scale=(1, 0.3)
        )

    EXAMPLE_ANCHORS=(
                core.surface_args('edge', 'linear', 0.5),
                core.surface_args('linear2', 0.5, 10),
                core.surface_args('linear3', 0.5, 20),
                core.surface_args('curve', 0, 40),
                core.surface_args('curve', 0.1, rh=0.9),
                core.surface_args('curve', 0.2, 40),
                core.surface_args('curve', 0.3, 40),
                core.surface_args('curve', 0.4, 40),
                core.surface_args('curve', 0.5, 40),
                core.surface_args('curve', 0.6, 40),
                core.surface_args('curve', 0.7, 40),
                core.surface_args('curve', 0.8, 40),
                core.surface_args('curve', 0.9, 40),
                core.surface_args('curve', 1.0, 40),
                )

    def render(self, renderer):
        polygon = renderer.model.Polygon(*self.path.polygons(renderer.get_current_attributes()))
        params = core.fill_params(
            self, renderer, ('fn',), exclude=('path',), xlation_table={'h': 'height'})
        return renderer.add(renderer.model.linear_extrude(**params)(polygon))
    
    
    @core.anchor('Anchor to the path edge and surface.')
    def edge(self, path_node_name, t=0, h=0, rh=None, align_twist=False):
        '''Anchors to the edge and surface of the linear extrusion.
        Args:
            path_node_name: The path node name to attach to.
            t: 0 to 1 being the beginning and end of the segment. Numbers out of 0-1
               range will depart the path linearly.
               
        '''
        if not rh is None:
            h = rh * self.h
        op = self.path.name_map.get(path_node_name)
        pos = self.to_3d_from_2d(op.position(t), h)
        normal_t = 0 if t < 0 else 1 if t > 1 else t 
        twist_vector = self.to_3d_from_2d(op.position(normal_t), 0)
        twist_radius = twist_vector.length()
        plane_dir = op.direction_normalized(normal_t)
        x_direction = self.to_3d_from_2d([plane_dir[0], -plane_dir[1]])
        z_direction = self.eval_z_vector(1)
        y_direction = z_direction.cross3D(x_direction)
        orientation = l.GMatrix.from_zyx_axis(x_direction, y_direction, z_direction) * l.rotX(90)
        
        # The twist andle is simply a rotation about Z depending on height.
        rel_h = h / self.h
        twist_angle = self.twist * rel_h
        twist_rot = l.rotZ(-twist_angle)
        
        twist_align = l.IDENTITY
        z_to_centre = l.IDENTITY
        if align_twist:
            # Aligning to the twist requires rotation about a axis perpendicular to the
            # axis of the twist (which is at (0, 0, h).
            z_to_centre = l.rot_to_V(twist_vector, [0, 0, 1])
            twist_align = l.rotZ(
                radians=np.arctan2(self.twist * np.pi / 180 * twist_radius , self.h))

        twisted = twist_rot * l.translate(pos) * z_to_centre.I * twist_align * z_to_centre * orientation 
        
        # The scale factors are for the x and y axii.
        scale = l.scale(
            tuple(self.scale[i] * rel_h + (1 - rel_h) for i in range(2)) + (1,))
        
        result = scale * twisted 

        # Descaling the matrix so the co-ordinates don't skew.
        result = result.descale()
        return result
    

if __name__ == "__main__":
    core.anchorscad_main(False)
    
