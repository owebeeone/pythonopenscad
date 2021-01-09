'''
Created on 7 Jan 2021

@author: gianni
'''

from dataclasses import dataclass

import bezier
from frozendict import frozendict

import ParametricSolid.linear as l
import numpy as np


class DuplicateNameException(Exception):
    '''The name requested is already used.'''

LIST_2_FLOAT = l.list_of(l.strict_float, len_min_max=(2, 2), fill_to_min=0.0)
LIST_3X2_FLOAT = l.list_of(LIST_2_FLOAT, len_min_max=(2, 2), fill_to_min=None)

def _normal_of_2d(v1, v2):
    vr = np.array([v1[1] - v2[1], v2[0] - v1[0]])
    l = np.sqrt(np.sum(vr * vr))
    return vr / l

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
        return np.sum(
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
            print(f'{v!r}')
            minima_maxima.append([np.min(v), np.max(v)])
    
        return np.transpose(minima_maxima)
        
    

@dataclass(frozen=True)
class LineNormal():
    op: object
    prev_op: object
    
    def has_normal(self):
        return True
    
    def normal(self, t):
        pass
    
@dataclass(frozen=True)
class SplineNormal():
    op: object
    prev_op: object
    
    def has_normal(self):
        return True
    
    def normal(self, t):
        pass

@dataclass(frozen=True)
class NoNormal():
    op: object
    prev_op: object
    
    def has_normal(self):
        return False
    
    def normal(self, t):
        pre_point = self.prev_op.lastPosition()
        next_point = self.op.point
        
    
@dataclass(frozen=True)
class Path():
    ops: tuple
    points: np.array
    ops_map: tuple
    name_map: frozendict

    def get_node(self, name):
        return self.name_map.get(name, None)

@dataclass()
class PathBuilder():
    ops: list
    name_map: dict
    
    @dataclass(frozen=True)
    class _LineTo:
        '''Line segment from current position.'''
        point: np.array
        prev_op: object
        name: str=None
            
        def lastPosition(self):
            return self.point
        
        def populate(self, path_builder, map_builder, meta_data):
            path_builder.append(self.point)
            map_builder.append((self,))
            
        def normal(self):
            return LineNormal(self.prev_op, self)
    
    
    @dataclass(frozen=True)
    class _MoveTo:
        '''Move to position.'''
        point: np.array
        prev_op: object
        name: str=None
            
        def lastPosition(self):
            return self.point
        
        def populate(self, path_builder, map_builder, meta_data):
            path_builder.append(self.point)
            map_builder.append((self,))
            
        def normal(self, t, dims=[0, 1]):
            return NoNormal(self.prev_op, self)
        
        def extents(self):
            pass

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
            
        def populate(self, path_builder, map_builder, meta_data):
            if (self.meta_data):
                meta_data = self.meta_data
    
            count = meta_data.fn
    
            for i in range(1, count + 1):
                t = float(i) / float(count)
                point = self.spline.evaluate(t)
                path_builder.append(point)
                map_builder.extend((self, t, count))
    
        def normal(self, t, dims=[0, 1]):
            return self.spline.extents(t, dims)
        
        def extents(self):
            return self.spline.extents()
    
    def __init__(self):
        self.ops = []
        self.name_map = {}
        
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
        return self.add_op(self._MoveTo(np.array(LIST_2_FLOAT(point)), self.last_op(), name))
                        
    def line(self, point, name=None):
        assert len(self.ops) > 0, "Cannot line to without starting point"
        return self.add_op(self._LineTo(np.array(LIST_2_FLOAT(point)), self.ops[-1], name))
                        
    def spline(self, point, name=None, metadata=None):
        assert len(self.ops) > 0, "Cannot line to without starting point"
        return self.add_op(self._SplineTo(
            np.array(LIST_3X2_FLOAT(point)), self.ops[-1], name, metadata))

    def build(self, meta_data):
        path_builder = []
        map_builder = []
        for op in self.ops:
            op.populate(path_builder, map_builder, meta_data)
        return Path(self.ops, np.array(path_builder), map_builder, frozendict(self.name_map))

