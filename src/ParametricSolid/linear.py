'''
Linear algebra tools for 3D transformations.

This is not a generic 3D matrix library but is design to work with the limitations of
OpenScad's multmatrix transform function. It will perform all linear non skewing 3D
linear transformations.

Examples:

from linear import *

>>> rotX(90) * GVector([1, 2, 3])
GVector([1.0, -3.0, 2.0, 1.0])

# Because we're dealing with floating point numbers, there is an is_approx_equal function
# to handle the resulting small inexactness.
# Rotation by 90, 3 times is the same as rotating back 90.
>>> (rotX(90) * rotX(90) * rotX(90)).is_approx_equal(rotX(-90))
True

# Rotations are cumulative.
>>> (rotX(90) * rotX(90) * rotX(90) * rotX(90)).is_approx_equal(IDENTITY)
True

# Rotate about an arbitrary axis.
>>> rotV([1, 1, 0], 45) * GVector([0, 0, 1])
GVector([0.4999999999999999, -0.4999999999999999, 0.7071067811865476, 1.0])

'''

from builtins import isinstance

import numpy as np


# Exceptions for dealing with argument checking.
class BaseException(Exception):
    '''Base exception functionality'''
    def __init__(self, message):
        self.message = message


class ConversionException(BaseException):
    '''Exception for conversion errors.'''


class MatrixShapeError(BaseException):
    '''Thrown when attempting to use a matrix of the wrong dimensions.'''


class MatrixInvalidError(BaseException):
    '''Failed consistency check of GMatrix.'''


class VectorInvalidError(BaseException):
    '''Failed consistency check for GVector.'''


def list_of(typ, len_min_max=(3, 3), fill_to_min=None):
    '''Defines a converter for an iterable to a list of elements of a given type.
    Args:
        typ: The type of list elements.
        len_min_max: A tuple of the (min,max) length, (0, 0) indicates no limits.
        fill_to_min: If the provided list is too short then use this value.
    Returns:
        A function that performs the conversion.
    '''
    description = 'list_of(%s, len_min_max=%r, fill_to_min=%r)' % (
        typ.__name__, len_min_max, fill_to_min)

    def list_converter(value):
        '''Converts provided value as a list of the given type.
        value: The value to be converted
        '''
        converted_value = []
        for v in value:
            if len_min_max[1] and len(converted_value) > len_min_max[1]:
                raise ConversionException(
                    'provided length too large, max is %d' % len_min_max[1])
            converted_value.append(typ(v))
        if len_min_max[0] and len(value) < len_min_max[0]:
            if fill_to_min is None:
                raise ConversionException(
                    'provided length (%d) too small and fill_to_min is None, min is %d'
                    % (len(converted_value), len_min_max[0]))
            fill_converted = typ(fill_to_min)
            for _ in range(len_min_max[0] - len(converted_value)):
                converted_value.append(fill_converted)
        return converted_value

    list_converter.__name__ = description
    return list_converter


def strict_float(v):
    '''Converter for a floating point value. Specifically does not allow str.
    Returns a numpy.float64 value.
    '''
    if isinstance(v, str):
        raise TypeError(
            'Was provided a string value but expecting a numeric value')
    return np.float64(v)

LIST_3_FLOAT = list_of(strict_float, len_min_max=(3, 3), fill_to_min=1.0)
LIST_4_FLOAT = list_of(strict_float, len_min_max=(4, 4), fill_to_min=0.0)
LIST_3_4_FLOAT = list_of(strict_float, len_min_max=(3, 4), fill_to_min=None)
LIST_3_4X4_FLOAT = list_of(LIST_4_FLOAT, len_min_max=(3, 4))


class GVector(object):
    '''A 3D (4x) vector.
    
    GVectors are not general 4 length vectors. The last/4th element is always 1.
    
    '''
    def __init__(self, v):
        '''
        Args:
            v: a length 3 or 4 list, iterable, numpy.matrix or numpy.ndarray. The
            If not provided, the last value will be defaulted to 1. If provided it 
            must be 1.
        '''
        self.v = self._validate(v)
        if np.abs(self.v[0, 3] - 1.0) > np.float64(1e-14):
            raise VectorInvalidError(
                'Last value must be 1 (or approx 1) was %f' % self.v[0, 3])

    @classmethod
    def _validate(cls, v):
        if isinstance(v, GVector):
            return v.v.copy()
        if isinstance(v, np.matrix):
            if np.shape(v) == (1, 4):
                return v
            elif np.shape(v) == (4, 1):
                return v.T
            elif np.shape(v) == (1, 3):
                return np.matrix(v.tolist()[0] + [1.])
            elif np.shape(v) == (3, 1):
                return np.matrix(v.T.tolist()[0] + [1.])
            else:
                raise MatrixShapeError(
                    'Matrix supplied is not a 4x1 or 1x4, Shape is %s' %
                    'x'.join(str(n) for n in np.shape(v)))
        elif isinstance(v, np.ndarray):
            if np.shape(v) == (4, ):
                return np.matrix(v)
            elif np.shape(v) == (3, ):
                return np.matrix(v.tolist() + [1.])
            elif np.shape(v) == (1, 4):
                return np.matrix(v)
            elif np.shape(v) == (4, 1):
                return np.matrix(v).T
            elif np.shape(v) == (1, 3):
                return np.matrix(v.tolist()[0] + [1.])
            elif np.shape(v) == (3, 1):
                return np.matrix(v.T.tolist()[0] + [1.])
            else:
                raise MatrixShapeError(
                    'Array supplied is not a 1x4 or 4x4, Shape is %s' %
                    'x'.join(str(n) for n in np.shape(v)))
        else:
            # Converts a len 3 iterable into a len 4 iterable (last elenent defaults to 1) or
            # a len 4 iterable.
            l = LIST_3_4_FLOAT(v)
            if len(l) == 3:
                l += [np.float64(1.0)]
            if len(l) == 4:
                return np.matrix(l)
            raise MatrixShapeError(
                'Array supplied is not a 3 or 4 length value, Length is %d' %
                len(v))
            
    def __str__(self):
        return str(self.L)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.L) + ')'

    def __add__(self, other):
        if not isinstance(other, GVector):
            other = GVector(other)
        return GVector(self.A3 + other.A3)

    def __radd__(self, other):
        if not isinstance(other, GVector):
            other = GVector(other)
        return GVector(other.A3 + self.A3)

    def __sub__(self, other):
        if not isinstance(other, GVector):
            other = GVector(other)
        return GVector(self.A3 - other.A3)

    def __rsub__(self, other):
        if not isinstance(other, GVector):
            other = GVector(other)
        return GVector(other.A3 - self.A3)
    
    def __mul__(self, scalar):
        v = strict_float(scalar)
        return GVector(self.A3 * v)

    def __rmul__(self, scalar):
        v = strict_float(scalar)
        return GVector(self.A3 * v)
    
    def __truediv__(self, scalar):
        v = strict_float(scalar)
        return GVector(self.A3 / v)

    def __rtruediv__(self, scalar):
        v = strict_float(scalar)
        return GVector(v / self.A3)
        
    def __neg__(self):
        return GVector(-(self.A3))

    def __pos__(self):
        return GVector(self.v.copy())

    def __getitem__(self, index):
        if isinstance(index, tuple):
            if len(index) == 1:
                return self.v.A1[index]
            return self.v[index]
        return self.v[0, index]

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            if len(index) == 1:
                self.v.A1[index] = value
            else:
                self.v[index] = value
        else:
            self.v[0, index] = value

    def __eq__(self, other):
        return isinstance(other, GVector) and np.array_equal(self.v, other.v)

    def __ne__(self, other):
        return not self == other
    
    def __len__(self):
        return np.shape(self.v)[1]

    def dot3D(self, other):
        '''
        Returns the dot product being the product of the length of the self and other
        vectors and cos(andgle between the vectors self and other).
        '''
        return np.sum(self.v.A1[0:3] * other.v.A1[0:3])

    def length(self):
        '''Returns the length of this 3D vector.'''
        return np.sqrt(self.dot3D(self))

    def is_approx_equal(self, other, error=1.e-12):
        '''Returns true of the other vector is approximately equal to this.
        Args:
            other: The value to compare this to.
            error: The allowable size of error to compare equal.
        '''
        return (self - other).length() < error

    @property
    def N(self):
        '''
        Returns a new normalized (length 1 same direction) vector.
        '''
        return GVector(self.v.A1[0:3] / self.length())

    def cross3D(self, other):
        '''
        Returns a cross product of this vector and other.
        The resulting vector is perpendicular to both self and other 
        and it's length is the product of the lengths of self and other
        and sin(the angle between self and other).
        '''
        if not isinstance(other, GVector):
            other = GVector(other)
        a = self.v.A1
        b = other.v.A1
        return GVector([
                    a[1]*b[2] - a[2]*b[1],
                    a[2]*b[0] - a[0]*b[2],
                    a[0]*b[1] - a[1]*b[0],
                    1.0])

    @property
    def L(self):
        '''Returns the Python list equivalent of this vector.'''
        return self.v.tolist()[0]

    @property
    def A(self):
        '''Returns the numpy.array equivalent of this vector.'''
        return self.v.A1

    @property
    def A3(self):
        '''Returns the numpy.array equivalent of this vector's first 3 elements.'''
        return self.v.A1[0:3]
    
    @property
    def x(self):
        '''Returns the x component of this GVector.'''
        return self.v[0, 0]
    
    @property
    def y(self):
        '''Returns the y component of this GVector.'''
        return self.v[0, 1]
    
    @property
    def z(self):
        '''Returns the z component of this GVector.'''
        return self.v[0, 2]


# GVector for the X axis.
X_AXIS = GVector([1, 0, 0])

# GVector for the y axis.
Y_AXIS = GVector([0, 1, 0])

# GVector for the z axis.
Z_AXIS = GVector([0, 0, 1])


class GMatrix(object):
    '''A 4x4 matrix for 3D geometric transformations.
    
    This does not perform generic matrix operations. Subtract and add do not 
    follow generic matrix rules. The last row of the matrix is maintained as 
    [0, 0, 0,1].
    '''

    LAST_ROW = np.array([[0., 0., 0., 1.]])

    def __init__(self, v):
        '''
        Args:
          v: A 4x4 or 3x4 numpy.matrix, numpy.ndarray or a list of lists containing 
          or something that can be converted to a 4x4 of floating point numbers.
        '''
        self.m = self._validate(v)
        if self.m.A[3].tolist() != [0., 0., 0., 1.]:
            raise MatrixInvalidError(
                'Last row of GMatrix must be [0, 0, 0, 1] but found %r.' %
                self.m.A[3].tolist())

    @classmethod
    def _validate(cls, v):
        if isinstance(v, GMatrix):
            return v.m.copy()
        
        # Sometimes isinstance breaks.
#         if v.__class__.__name__ == GMatrix.__name__:
#             if isinstance(v.m, np.matrix):
#                 return v.m.copy()

        if isinstance(v, np.matrix):
            shape = np.shape(v)
            if shape == (4, 4):
                return v
            elif shape == (3, 4):
                return cls._add_last_row(v)
            else:
                raise MatrixShapeError(
                    'Matrix supplied is not a 4x4 or 3x4, Shape is %s' %
                    'x'.join(str(n) for n in np.shape(v)))
        elif isinstance(v, np.ndarray):
            shape = np.shape(v)
            if shape == (4, 4):
                return np.matrix(v)
            elif shape == (3, 4):
                return np.matrix(cls._add_last_row(v))
            else:
                raise MatrixShapeError(
                    'Array supplied is not a 4x4 or 3x4, Shape is %s' %
                    'x'.join(str(n) for n in np.shape(v)))
        else:
            vm = np.matrix(LIST_3_4X4_FLOAT(v))
            if np.shape(vm) == (4, 4):
                return vm
            elif shape == (3, 4):
                return cls._add_last_row(v)
            else:
                raise MatrixShapeError(
                    'Matrix supplied is not a 4x4 or 3x4, Shape is %s' %
                    'x'.join(str(n) for n in np.shape(vm)))

    @classmethod
    def _add_last_row(cls, m):
        return np.append(m, cls.LAST_ROW, axis=0)

    def __str__(self):
        return '[\n    ' + ',\n    '.join([str(i) for i in self.L]) + ']'
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self) + ')'

    def __mul__(self, other):
        if isinstance(other, GMatrix):
            return GMatrix(self.m * other.m)
        if isinstance(other, GVector):
            return GVector(self.m * other.v.T)
        return GMatrix(self.m[0:3] * other)

    def __rmul__(self, other):
        if isinstance(other, GMatrix):
            return GMatrix(other.m * self.m)
        if isinstance(other, GVector):
            return GVector(other.v * self.m)
        return GMatrix(other * self.m)

    def __add__(self, other):
        if isinstance(other, GMatrix):
            return GMatrix(self.m[0:3] + other.m[0:3])
        return GMatrix(self.m[0:3] + GMatrix(other).m[0:3])

    def __radd__(self, other):
        if isinstance(other, GMatrix):
            return GMatrix(self.m[0:3] + other.m[0:3])
        return GMatrix(self.m[0:3] + GMatrix(other).m[0:3])

    def __sub__(self, other):
        if isinstance(other, GMatrix):
            return GMatrix(self.m[0:3] - other.m[0:3])
        return GMatrix(self.m[0:3] - GMatrix(other).m[0:3])

    def __rsub__(self, other):
        if isinstance(other, GMatrix):
            return GMatrix(other.m[0:3] - self.m[0:3])
        return GMatrix(GMatrix(other).m[0:3] - self.m[0:3])

    def __neg__(self):
        return GMatrix(-self.m[0:3])

    def __pos__(self):
        return GMatrix(self.m.copy())

    def __invert__(self):
        return GMatrix(self.m.I)

    def __getitem__(self, index):
        return self.m[index]

    def __setitem__(self, index, value):
        self.m[index] = value

    def __eq__(self, other):
        return isinstance(other, GMatrix) and np.array_equal(self.m, other.m)

    def __ne__(self, other):
        return not self == other

    def length(self):
        '''Returns the Euclidian length of all components in the matrix.'''
        a1 = self.m[0:3].A1
        return np.sqrt((a1 * a1).sum())

    def is_approx_equal(self, other, error=1.e-12):
        return (self - other).length() < error

    def copy(self):
        return GMatrix(self)

    @property
    def I(self):
        '''Returns the inverted matrix.
        i.e.
           M.I * M == IDENTITY
        '''
        return GMatrix(self.m.I)

    @property
    def L(self):
        '''Returns the Python list equivalent of this matrix.'''
        return self.m.tolist()

    @property
    def A(self):
        '''Returns the numpy.array equivalent of this matrix.'''
        return self.m.A


# The identity matrix.
IDENTITY = GMatrix([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]])

# Mirrors the X axis.
MIRROR_X = GMatrix([
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]])

# Mirrors the Y axis.
MIRROR_Y = GMatrix([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]])

# Mirrors the Z axis.
MIRROR_Z = GMatrix([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]])

def clean(v, epsilon=1.e-13):
    if np.abs(v) < epsilon:
        return 0
    return v

def rotZ(degrees=90, radians=None):
    '''Returns a GMatrix that causes a rotation about Z a given number of degrees.'''
    if radians is None:
        radians = np.pi * (degrees / 180.0)
    cosr = clean(np.cos(radians))
    sinr = clean(np.sin(radians))
    return GMatrix(np.matrix([[cosr, -sinr, 0.0, 0], 
                              [sinr, cosr, 0, 0], 
                              [0, 0, 1, 0], 
                              [0, 0, 0, 1]]))

def rotX(degrees=90, radians=None):
    '''Returns a GMatrix that causes a rotation about X a given number of degrees.'''
    if radians is None:
        radians = np.pi * (degrees / 180.0)
    cosr = clean(np.cos(radians))
    sinr = clean(np.sin(radians))
    return GMatrix(np.matrix([[1.0, 0, 0, 0], 
                              [0, cosr, -sinr, 0], 
                              [0, sinr, cosr, 0], 
                              [0, 0, 0, 1]]))
    
def rotY(degrees=90, radians=None):
    '''Returns a GMatrix that causes a rotation about Y a given number of degrees.'''
    if radians is None:
        radians = np.pi * (degrees / 180.0)
    cosr = clean(np.cos(radians))
    sinr = clean(np.sin(radians))
    return GMatrix(np.matrix([[cosr, 0.0, sinr, 0], 
                              [0, 1, 0, 0],
                              [-sinr, 0, cosr, 0], 
                              [0, 0, 0, 1]]))
    
def rotV(v, degrees=90, radians=None):
    '''Returns a GMatrix that causes a rotation about an axis vector V a given number of degrees.'''
    if not isinstance(v, GVector):
        v = GVector(v)
    if radians is None:
        radians = np.pi * (degrees / 180.0)
    cosr = clean(np.cos(radians))
    sinr = clean(np.sin(radians))
    u = v.N
    ux = u.x
    uy = u.y
    uz = u.z
    u2 = u.A * u.A
    ux2 = u2[0]
    uy2 = u2[1]
    uz2 = u2[2]
    uxz = ux * uz
    uxy = ux * uy
    uyz = uy * uz
    lcosr = 1 - cosr
    return GMatrix(np.matrix(
        [[cosr + ux2 * lcosr, uxy * lcosr - uz * sinr, uxz * lcosr + uy * sinr, 0],
         [uxy * lcosr + uz * sinr, cosr + uy2 * lcosr, uyz * lcosr - ux * sinr, 0],
         [uxz * lcosr - uy * sinr, uyz * lcosr + ux * sinr, cosr + uz2 * lcosr, 0],
         [0.0, 0, 0, 1]]))

ROTV111_240=GMatrix([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]])

ROTV111_120=GMatrix([
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]])

def scale(s):
    '''Returns a GMatrix that scales by a vector [x,y,z] scalars or [s,s,s].'''
    try:
        v = LIST_3_FLOAT(s)
    except:
        v = LIST_3_FLOAT([s, s, s])
    
    return GMatrix(
        np.matrix([[v[0], 0.0, 0, 0], [0, v[1], 0, 0], [0, 0, v[2], 0], [0, 0, 0, 1]]))

def translate(v):
    '''Returns GMatrix that scales by the given vector.'''
    if not isinstance(v, GVector):
        v = GVector(v)
    return GMatrix(np.matrix(
        [[1., 0, 0, v.x], [0, 1, 0, v.y], [0, 0, 1, v.z], [0, 0, 0, 1]]))

def rot_to_V(from_v, to_v):
    '''Computes the rotation so that transformation from from_v becomes parallel to to_v'''
    if not isinstance(from_v, GVector):
        from_v = GVector(from_v)
    if not isinstance(to_v, GVector):
        to_v = GVector(to_v)

    from_vn = from_v.N
    to_vn = to_v.N

    cross = from_vn.cross3D(to_vn)
    sinr = cross.length()
    # If the rotation is very small just return the identity
    if abs(sinr) < 1e-12:
        return IDENTITY.copy()
    cosr = from_vn.dot3D(to_vn)
    angle = np.arctan2(sinr, cosr)

    return rotV(cross, radians=angle)

def mirror(axis):
    '''Mirror at the origin about any plane. The axis provided is the normal to the mirror plane.
    '''
    if not isinstance(axis, GVector):
        axis = GVector(axis)
    axis = axis.N
    dotx = X_AXIS.dot3D(axis)
    # Use one of the predefined mirror matricies, We first have to choose one that is
    # not colinear so we don't break rotV with a zero length cross vector.
    if abs(dotx) > 0.5:
        # closer to X axis than Y axis so pick Y axis mirror.
        ref_axis = Y_AXIS
        mm = MIRROR_Y
    else:
        # closer to Y axis so pick X axis.
        ref_axis = X_AXIS
        mm = MIRROR_X
    # Finds a rotation matrix that will rotate from the given axis to the reference axis.
    m = rot_to_V(axis, ref_axis)
    # Chain the matrix multiplications so we first rotate to the plane then mirror and
    # then rotate back to the original frame of reference.
    return m.I * mm * m

