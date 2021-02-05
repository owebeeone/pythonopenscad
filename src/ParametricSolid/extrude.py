'''
Created on 7 Jan 2021

@author: gianni
'''

from dataclasses import dataclass

from frozendict import frozendict

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
    '''Unable to interpret args.'''
    
class UnknownOperationException(Exception):
    '''Requested anchor is not found.'''
    
class UnableToFitCircleWithGivenParameters(Exception):
    '''There was no solution to the requested arc. Try a spline.'''


EPSILON=1e-12

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

def _vlen(v):
    return np.sqrt(np.sum(v**2))

def _normalize(v):
    return v / _vlen(v)


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
    def curve_maxima_minima_t(self, t_range=(0.0, 1.0)):
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
        roots = self.curve_maxima_minima_t()
        
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

    def transform_to_builder(self, m):
        '''Returns a PathBuilder with the new transformed path.'''
        builder = PathBuilder()
        
        for op in self.ops:
            builder.add_op_with_params(op.transform(m), op.name)
        return builder
            
    def transform(self, m):
        return self.transform_to_builder(m).build()
    
def to_gvector(np_array):
    if len(np_array) == 2:
        return l.GVector([np_array[0], np_array[1], 0, 1])
    else:
        return l.GVector(np_array)
    
    
# Solution derived from https://planetcalc.com/8116/
def solve_circle_3_points(p1, p2, p3): 
    '''Returns the centre and radius of a circle that passes the 3 given points or and empty
    tuple if the points are colinear.'''
    
    p = np.array([p1, p2, p3])
    
    m = np.array([
        np.concatenate((p[0] * 2, [1])),
        np.concatenate((p[1] * 2, [1])),
        np.concatenate((p[2] * 2, [1]))
        ])

    try:
        mi = np.linalg.inv(m)
    except np.linalg.LinAlgError:
        return (None, None)
    
    v1 = -np.sum(p **2, axis=1)
    
    abc = np.matmul(mi, v1)
    centre = -abc[0:2]
    radius = np.sqrt(np.sum(centre **2) - abc[2])
    
    return (centre, radius)

def find_a_b_c_from_point_tangent(p, t):
    '''Given a point and direction of a line (t) compute the parameter (a, b, c) for the line:
    described by ax + by = c. Returns [a, b, c], p (as an numpy array) and t but also  normalized 
    (in both length and direction).
    '''
    p = np.array(p)
    t = np.array(t)
    tn = t / _vlen(t)
     
    a = tn[1]
    b = -tn[0]
    c = np.linalg.det([p, tn])
    
    l = np.array([a, b, c])
    if a < 0:
        l = -l
    elif a == 0 and b < 0:
        l = -l
        
    return l, p, t

def find_2d_line_intersection(l1, l2):
    '''Finds the point of intersection of l1 and l2. l1 and l2 are 3x1 quantities
    defined by [a, b, c] where ax + by = c defines the line.
    [a, b] should be normalized (vector length = 1).
    Returns the point of intersection, 0 if the lines are parallel or 1 if the lines are
    identical.
    Derived from the use of Cramer's rule.:
    https://math.libretexts.org/Bookshelves/Precalculus/Book%3A_Precalculus_(OpenStax)/\
    09%3A_Systems_of_Equations_and_Inequalities/9.08%3A_Solving_Systems_with_Cramer's_Rule
    '''
    m = np.array([l1[0:2], l2[0:2]])
    d = np.linalg.det(m)
    mT = np.transpose(m)
    
    if np.abs(d) < EPSILON:
        # if the c values are the same then the normals are the same line meaning that
        # the lines are colinear. If the c values are the same, then the lines are identical.
        if np.abs(l1[2] - l2[2]) < EPSILON:
            # lines are identical.
            return ('identical')
        else:
            return ('parallel')
    else:
        cn = np.array([l1[2], l2[2]])
        return np.array([np.linalg.det([cn, mT[1]]) / d, np.linalg.det([mT[0], cn]) / d])

def solve_circle_tangent_point(p1, t1, p2):
    '''Returns the (centre, radius) tuple of the circle whose tangent is p1, t1 second
    point is p2.'''
    # The centre must lie in the perpendicular to the tangent.
    l1, p1, tn1 = find_a_b_c_from_point_tangent(p1, [-t1[1], t1[0]])
    
    # The second line is defined by keeping the centre equidistant from p1 and p2
    # i.e.
    # len(p1-C) == len(p2-c)
    a = 2 * (p2[0] - p1[0])
    b = 2 * (p2[1] - p1[1])
    c = p2[0]**2 - p1[0]**2 + p2[1]**2 - p1[1]**2
    
    l2 = np.array([a, b, c])
    l2 = l2 / (np.sign(a if a != 0 else b) * _vlen(l2[0:2]))
    
    centre = find_2d_line_intersection(l1, l2)
    if isinstance(centre[0], str):
        return (None, None)
    
    radius = np.sqrt(np.sum((centre - p1) ** 2))
    return (centre, radius)

def solve_circle_tangent_radius(p, t, r):
    '''Finds the centre of the circle described by a tangent and a radius.
    Returns (centre, radius)'''
    p = np.array(p)
    t = np.array(t)
    tn = _normalize(t)
    centre = p + r * np.array(-tn[1, t[0]])
    return (centre, r)


def solve_circle_2_point_radius(p1, p2, r, left=True):
    '''Finds the centre of the circle described by a tangent and a circle and a radius.
    Returns (centre, radius)'''
    p1 = np.array(p1)
    p2 = np.array(p2)
    pd = p2 - p1
    leng = np.sqrt(np.sum(pd **2)) / 2
    pdn = pd / (2 * leng)
    if len * r:
        return (None, None)
    if np.abs(len - r) < EPSILON:
        centre = (p1 + p2) / 2
        return (centre, r)
    opp_side = np.sqrt(r**2 - len **2)
    if left:
        dir = np.array([-pdn[1], pdn[0]]) #+90 degrees
    else:
        dir = np.array([pdn[1], -pdn[0]]) #-90 degrees
    centre = (p1 + p2) / 2 + dir * opp_side
    return (centre, r)

@dataclass()
class CircularArc:
    start_angle: float  # Angles in radians
    span_angle: float   # Angles in radians
    radius: float
    centre: np.array
    
    def derivative(self, t):
        '''Returns the derivative (direction of the curve at t).'''
        angle = t * self.span_angle + self.start_angle
        # Derivative direction depends on sense of angle.
        d = 1 if self.span_angle < 0 else -1
        return np.array([np.sin(angle), -np.cos(angle)]) * d
    
    def normal2d(self, t):
        '''Returns the normal to the curve at t.'''
        ddt = self.derivative(t)
        return np.array([ddt[1], -ddt[0]])
    
    def extents(self):
        sa = self.start_angle
        ea = sa + self.span_ang
        r = self.radius
        result = [
            np.array([r * np.sin(sa), r * np.cos(sa) ]) + self.centre,
            np.array([r * np.sin(ea), r * np.cos(ea) ]) + self.centre]
        
        sai = sa * 2 / np.pi
        eai = sa * 2 / np.pi
        angle_dir = 1. if self.span_angle >= 0 else -1.
        count = 1
        while count < (eai - sai) * angle_dir and count < 4:
            _, ai = np.modf(sai + count * angle_dir)
            angle = ai * (np.pi / 2)
            result.append(
                np.array([r * np.sin(angle), r * np.cos(angle) ]) + self.centre)
            ++count

        return np.transpose(result)

    def evaluate(self, t):
        angle = t * self.span_angle + self.start_angle
        return np.array([np.cos(angle), np.sin(angle)]) * self.radius + self.centre
  

@dataclass()
class PathBuilder():
    ops: list
    name_map: dict
    multi: bool=False
    
    class OpBase:
        def _as_non_defaults_dict(self):
            return dict((k, getattr(self, k)) 
                        for k in self.__annotations__.keys() 
                            if not getattr(self, k) is None and k != 'prev_op')
    
    @dataclass(frozen=True)
    class _LineTo(OpBase):
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
        
        def normal2d(self, t, dims=[0, 1]):
            return _normal_of_2d(self.prev_op.lastPosition(), self.point, dims)
        
        def extents(self):
            p0 = self.prev_op.lastPosition()
            p1 = self.point
            return np.transpose(
                list(([p0[k], p1[k]] if p0[k] < p1[k] else [p1[k], p0[k]]) for k in range(len(p0))))
            
        def position(self, t):
            return self.point + (t - 1) * self.direction(0)
        
        def transform(self, m):
            params = self._as_non_defaults_dict()
            params['point'] = (m * to_gvector(self.point)).A[0:len(self.point)]
            return (self.__class__, params)
            
    
    @dataclass(frozen=True)
    class _MoveTo(OpBase):
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

        def transform(self, m):
            params = self._as_non_defaults_dict()
            params['point'] = (m * to_gvector(self.point)).A[0:len(self.point)]
            return (self.__class__, params)
            

    @dataclass(frozen=True)
    class _SplineTo(OpBase):
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
        
        def transform(self, m):
            points = list((m * to_gvector(p)).A[0:len(p)] for p in self.points)
            points = np.array(LIST_23X2_FLOAT(points))
            params = self._as_non_defaults_dict()
            params['points'] = points
            return (self.__class__, params)
    
        
    @dataclass(frozen=True)
    class _ArcTo(OpBase):
        '''Draw a circular arc.'''
        end_point: np.array
        centre: np.array
        path_direction: bool
        prev_op: object
        name: str=None
        meta_data: object=None
        
        def __post_init__(self):
            
            start_point = self.prev_op.lastPosition()
            r_start = start_point - self.centre
            radius_start = _vlen(r_start)
            r_end = self.end_point - self.centre
            radius_end = _vlen(r_end)
            assert np.abs(radius_start - radius_end) < EPSILON, (
                'start and end point radius should be the same')
            start_angle = np.arctan2(r_start[1] / radius_start, r_start[0] / radius_start)
            end_angle = np.arctan2(r_end[1] / radius_start, r_end[0] / radius_start)
            span_angle = end_angle - start_angle
            if self.path_direction:
                span_angle = -span_angle
                    
            object.__setattr__(self, 'arcto', CircularArc(
                start_angle, span_angle, radius_start, self.centre))
            
        def lastPosition(self):
            return self.end_point
            
        def populate(self, path_builder, start_indexes, map_builder, meta_data):
            if (self.meta_data):
                meta_data = self.meta_data
    
            count = meta_data.fn
            if not count:
                count = 10
                
            for i in range(1, count + 1):
                t = float(i) / float(count)
                point = self.arcto.evaluate(t)
                path_builder.append(point)
                map_builder.append((self, t, count))
    
        def direction(self, t):
            return self.arcto.derivative(t)
        
        def direction_normalized(self, t):
            return _normalize(self.direction(t))
        
        def normal2d(self, t, dims=[0, 1]):
            return self.arcto.normal2d(t)
        
        def extents(self):
            return self.arcto.extents()
        
        def position(self, t):
            if t < 0:
                return self.direction(0) * t + self.prev_op.lastPosition()
            elif t > 1:
                return self.direction(1) * t + self.end_point
            return self.arcto.evaluate(t)
        
        def transform(self, m):
            end_point = (m * to_gvector(self.end_point)).A[0:len(self.end_point)]
            centre = (m * to_gvector(self.centre)).A[0:len(self.centre)]
            params = {
                'end_point': end_point,
                'centre': centre,
                'path_direction': self.path_direction}
            return (self.__class__, params)
    
    
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
    
    def add_op_with_params(self, op_parts, op_name=None):
        params_dict = op_parts[1]
        params_dict['prev_op'] = self.last_op()
        if op_name:
            params_dict['name'] = op_name
        return self.add_op((op_parts[0])(**params_dict))

    def last_op(self):
        return self.ops[-1] if self.ops else None
        
    def move(self, point, name=None):
        if not self.multi and self.ops:
            raise MoveNotAllowedException(f'Move is not allowed in non multi-path builder.')
        return self.add_op(self._MoveTo(np.array(LIST_2_FLOAT(point)),
                                        prev_op=self.last_op(), name=name))
                        
    def line(self, point, name=None):
        assert len(self.ops) > 0, "Cannot line to without starting point"
        return self.add_op(self._LineTo(np.array(LIST_2_FLOAT(point)), 
                                        prev_op=self.last_op(), name=name))
             
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
            if self.last_op().direction_normalized(1.0) is None:
                raise InvalidSplineParametersException(
                    'Only 2 control points provided so the direction of the previous operation'
                    ' will be used but the previous operation (move) does not provide direction.')
            cv0 = self.last_op().lastPosition()
            cv1 = self.last_op().direction_normalized(1.0) * cv_len[0] + cv0
            cv2 = points[0]
            cv3 = points[1]
        else:
            cv0 = self.last_op().lastPosition()
            cv1 = points[0]
            cv2 = points[1]
            cv3 = points[2]
        if not rel_len is None:
            l = np.sqrt(np.sum((cv0 - cv3)**2))
            cv_len = tuple(rel_len * l if v is None else v * l * rel_len for v in cv_len)
        cv1 = self.squeeze_and_rot(cv0, cv1, cv_len[0], degrees[0], radians[0])
        cv2 = self.squeeze_and_rot(cv3, cv2, cv_len[1], degrees[1], radians[1])
        
        points = np.array(LIST_3X2_FLOAT([cv1, cv2, cv3]))
        return self.add_op(
            self._SplineTo(points, prev_op=self.last_op(), name=name, meta_data=metadata))
    
    def arc_points(self, middle, last, name=None, metadata=None):
        '''Defines a circular arc starting at the previous operator's end point
        and passing through middle and ending at last.'''
        start = self.last_op().lastPosition()
        centre, radius = solve_circle_3_points(start, middle, last)
        n_points = np.array([start - centre, middle - centre, last - centre]) / radius
        start_angle = np.arctan2(n_points[0][0], n_points[0][1])
        middle_delta = np.arctan2(n_points[0][0], n_points[0][1]) - start_angle
        end_delta = np.arctan2(n_points[0][0], n_points[0][1]) - start_angle

        angle_dir = -1 if end_delta > 0 else  1
        
        path_direction = angle_dir > 0
        
        return self.add_op(self._ArcTo(last, centre, path_direction, name, metadata))
    
    def arc_tangent_point(self, last, degrees=0, radians=None, direction=None, 
                          name=None, metadata=None):
        '''Defines a circular arc starting at the previous operator's end point
        and ending at last. The tangent  .'''
        start = self.last_op().lastPosition()
        if direction is None:
            direction = self.last_op().direction_normalized(1.0)
        else:
            direction = _normalize(direction)
        
        t_dir = (
            l.rotZ(degrees=degrees, radians=radians) * to_gvector(direction))
        direction = t_dir.A[0:len(direction)]
        centre, radius = solve_circle_tangent_point(start, direction, last)
        if centre is None:
            # This degenerates to a line.
            return self.line(last, name=name)
        n_points = np.array([start - centre, last - centre]) / radius
        start_angle = np.arctan2(n_points[0][1], n_points[0][0])
        end_angle = np.arctan2(n_points[1][1], n_points[1][0])
        end_delta = end_angle - start_angle
        c_dir = l.GVector([-np.sin(start_angle), np.cos(start_angle), 0])
        
        path_direction = (t_dir.dot3D(c_dir) < 0) == (end_delta > 0)
        
        return self.add_op(self._ArcTo(
            last, centre, path_direction, 
            prev_op=self.last_op(), name=name, meta_data=metadata))
    
    def squeeze_and_rot(self, point, control, cv_len, degrees, radians):
        if cv_len is None and not degrees and not radians:
            return control
        gpoint = l.GVector(LIST_3_FLOAT(point))
        gcontrol = l.GVector(LIST_3_FLOAT(control))
        g_rel = (gcontrol - gpoint)
        if not cv_len is None and g_rel.length() > EPSILON:
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
        return tuple(self.anchorscad.anchors.keys()) + tuple(self.path.name_map.keys())
    
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
    '''Generates a linear extrusion of a given Path.'''
    path: Path
    h: float=100
    twist: float=0.0
    slices: int=None
    scale: float=(1.0, 1.0)  # (x, y)
    fn: int=None
    
    SCALE=2
    
    EXAMPLE_SHAPE_ARGS=core.args(
        PathBuilder()
            .move([0, 0])
            .line([100 * SCALE, 0], 'linear')
            .spline([[150 * SCALE, 100 * SCALE], [20 * SCALE, 100 * SCALE]],
                     name='curve', cv_len=(0.5,0.4), degrees=(90,), rel_len=0.8)
            .line([0, 100 * SCALE], 'linear2')
            .line([0, 0], 'linear3')
            .build(),
        h=80,
        fn=30,
        twist=45,
        slices=40,
        scale=(1, 0.3)
        )

    EXAMPLE_ANCHORS=(
                core.surface_args('edge', 'linear', 0.5),
                core.surface_args('linear2', 0.5, 10),
                core.surface_args('linear2', 0, 40),
                core.surface_args('linear2', 1, 40),
                core.surface_args('linear3', 0.5, 20, None, True, True),
                core.surface_args('curve', 0, 40),
                core.surface_args('curve', 0.1, rh=0.9),
                core.surface_args('curve', 0.2, 40),
                core.surface_args('curve', 0.3, 40),
                core.surface_args('curve', 0.4, 40),
                core.surface_args('curve', 0.5, 40, None, True, True),
                core.surface_args('curve', 0.6, 40, None, True, True),
                core.surface_args('curve', 0.7, 40, None, True, True),
                core.surface_args('curve', 0.8, 40, None, True, True),
                core.surface_args('curve', 0.9, 40, None, True, True),
                core.surface_args('curve', 1, 40, None, True, True),
                core.surface_args('linear2', 0.1, rh=0.9),
                core.surface_args('linear2', 0.5, 0.9, True, True),
                core.surface_args('linear2', 1.0, rh=0.9),
                )

    def render(self, renderer):
        polygon = renderer.model.Polygon(*self.path.polygons(renderer.get_current_attributes()))
        params = core.fill_params(
            self, renderer, ('fn',), exclude=('path',), xlation_table={'h': 'height'})
        return renderer.add(renderer.model.linear_extrude(**params)(polygon))
    
    def _z_radians_scale_align(self, rel_h, twist_vector):
        xelipse_max = self.scale[0] * rel_h + (1 - rel_h)
        yelipse_max = self.scale[1] * rel_h + (1 - rel_h)
        eliplse_angle = np.arctan2(xelipse_max * twist_vector.y, yelipse_max * twist_vector.x)
        circle_angle = np.arctan2(twist_vector.y, twist_vector.x)
        return eliplse_angle - circle_angle
        
    
    @core.anchor('Anchor to the path edge and surface.')
    def edge(self, path_node_name, t=0, h=0, rh=None, align_twist=False, align_scale=False):
        '''Anchors to the edge and surface of the linear extrusion.
        Args:
            path_node_name: The path node name to attach to.
            t: 0 to 1 being the beginning and end of the segment. Numbers out of 0-1
               range will depart the path linearly.
            h: The absolute height of the anchor location.
            rh: The relative height (0-1).
            align_twist: Align the anchor for the twist factor.
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

        # The scale factors are for the x and y axii.
        scale = l.scale(
            tuple(self.scale[i] * rel_h + (1 - rel_h) for i in range(2)) + (1,))
        
        scale_zalign = l.IDENTITY
        scale_xalign = l.IDENTITY
        if align_scale:
            # Scaling adjustment along the Z plane is equivalent to a z rotation 
            # of the difference of the angle of a circle and the scaleg cirle.
            scale_zalign = l.rotY(radians=self._z_radians_scale_align(rel_h, twist_vector)) 
            
            scaled_vector = scale * twist_vector
            scale_xalign = l.rotZ(radians=-np.arctan2(
                twist_vector.length() - scaled_vector.length(), rel_h * self.h)) 
                      

        twisted = (twist_rot * l.translate(pos) * z_to_centre.I 
                   * twist_align * z_to_centre * orientation * scale_zalign * scale_xalign)
        
        
        result = scale * twisted 

        # Descaling the matrix so the co-ordinates don't skew.
        result = result.descale()
        return result


@core.shape('arc_extrude')
@dataclass
class RotateExtrude(ExtrudedShape):
    '''Generates a circular/arc extrusion of a given Path.'''
    path: Path
    degrees: float=360
    radians: float=None
    convexity: int=10
    fn: int=None
    fa: float=None
    fs: float=None

    
    SCALE=1.0
    
    EXAMPLE_SHAPE_ARGS=core.args(
        PathBuilder()
            .move([0, 0])
            .line([110 * SCALE, 0], 'linear')
            .arc_tangent_point([10 * SCALE, 100 * SCALE], name='curve', degrees=120)
            .line([0, 100 * SCALE], 'linear2')
            .line([0, 0], 'linear3')
            .build(),
        degrees=120,
        fn=80,
        )

    EXAMPLE_ANCHORS=(
                core.surface_args('edge', 'linear', 0.5),
                core.surface_args('linear2', 0.5, 10),
                core.surface_args('linear2', 0, 40),
                core.surface_args('linear2', 1, 40),
                core.surface_args('linear3', 0.5, 20),
                core.surface_args('curve', 0, 45),
                core.surface_args('curve', 0.1, 40),
                core.surface_args('curve', 0.2, 40),
                core.surface_args('curve', 0.3, 40),
                core.surface_args('curve', 0.4, 40),
                core.surface_args('curve', 0.5, 40),
                core.surface_args('curve', 0.6, 40),
                core.surface_args('curve', 0.7, 40),
                core.surface_args('curve', 0.8, 40),
                core.surface_args('curve', 0.9, 40),
                core.surface_args('curve', 1, 70),
                core.surface_args('linear2', 0.1, 0.9),
                core.surface_args('linear2', 0.5, 0.9),
                core.surface_args('linear2', 1.0, 0.9),
                )
    
    EXAMPLES_EXTENDED={
        'example2': core.ExampleParams(
            shape_args=core.args(
                PathBuilder()
                    .move([0, 0])
                    .line([110 * SCALE, 0], 'linear')
                    .arc_tangent_point([10 * SCALE, 100 * SCALE], name='curve', degrees=150)
                    .line([0, 100 * SCALE], 'linear2')
                    .line([0, 0], 'linear3')
                    .build(),
                degrees=120,
                fn=80,
                ),
            anchors=(core.surface_args('linear', 0.5),))
        }

    def render(self, renderer):
        polygon = renderer.model.Polygon(*self.path.polygons(renderer.get_current_attributes()))
        params = core.fill_params(
            self, renderer, tuple(core.ARGS_XLATION_TABLE.keys()), exclude=('path', 'degrees', 'radians'))
        angle = self.degrees
        if self.radians:
            angle = self.radians * 180 / np.pi
        params['angle'] = angle
        
        return renderer.add(renderer.model.rotate_extrude(**params)(polygon))

    def to_3d_from_2d(self, vec_2d, angle=0., degrees=0, radians=None):
        return l.rotZ(
            degrees=degrees, radians=radians) * l.rotX(90) * l.GVector([vec_2d[0], vec_2d[1], 0])
    
    def _z_radians_scale_align(self, rel_h, twist_vector):
        xelipse_max = self.scale[0] * rel_h + (1 - rel_h)
        yelipse_max = self.scale[1] * rel_h + (1 - rel_h)
        eliplse_angle = np.arctan2(xelipse_max * twist_vector.y, yelipse_max * twist_vector.x)
        circle_angle = np.arctan2(twist_vector.y, twist_vector.x)
        return eliplse_angle - circle_angle

    @core.anchor('Anchor to the path edge projected to surface.')
    def edge(self, path_node_name, t=0, degrees=0, radians=None):
        '''Anchors to the edge projected to the surface of the rotated extrusion.
        Args:
            path_node_name: The path node name to attach to.
            t: 0 to 1 being the beginning and end of the segment. Numbers out of 0-1
               range will depart the path linearly.
            degrees or radians: The angle along the rotated extrusion.
        '''
        op = self.path.name_map.get(path_node_name)
        normal = op.normal2d(t)
        pos = op.position(t)

        return (l.rotZ(degrees=degrees, radians=radians)
                     * l.ROTX_90  # Projection from 2D Path to 3D space
                     * l.translate([pos[0], pos[1], 0])
                     * l.ROTY_90  
                     * l.rotXSinCos(normal[1], -normal[0]))
    
    @core.anchor('Centre of the extrusion arc.')
    def centre(self):
        return l.IDENTITY

if __name__ == "__main__":
    core.anchorscad_main(False)
    
