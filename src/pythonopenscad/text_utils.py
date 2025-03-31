from datatrees.datatrees import datatree, dtfield
import anchorscad_lib.linear as l
import numpy as np
import logging

log = logging.getLogger(__name__)


def extentsof(p: np.ndarray) -> np.ndarray:
    return np.array((p.min(axis=0), p.max(axis=0)))

def to_gvector(np_array):
    if len(np_array) == 2:
        return l.GVector([np_array[0], np_array[1], 0, 1])
    else:
        return l.GVector(np_array)
    

EPSILON = 1e-6

@datatree(frozen=True)
class CubicSpline:
    """Cubic spline evaluator, extents and inflection point finder."""

    p: object = dtfield(doc="The control points for the spline, shape (4, N).")
    dimensions: int = dtfield(
        self_default=lambda s: np.asarray(s.p).shape[1],  # Get dim from shape
        init=True,
        doc="The number of dimensions in the spline.",
    )
    coefs: np.ndarray=dtfield(init=False)

    COEFFICIENTS = np.array([
        [-1.0, 3, -3, 1],
        [3, -6, 3, 0],
        [-3, 3, 0, 0],
        [1, 0, 0, 0],
    ])  # Shape (4, 4)

    # @staticmethod # For some reason this breaks on Raspberry Pi OS.
    def _dcoeffs_builder(dims):
        # ... (keep as before) ...
        zero_order_derivative_coeffs = np.array([[1.0] * dims, [1] * dims, [1] * dims, [1] * dims])
        derivative_coeffs = np.array([[3.0] * dims, [2] * dims, [1] * dims, [0] * dims])
        second_derivative = np.array([[6] * dims, [2] * dims, [0] * dims, [0] * dims])
        return (zero_order_derivative_coeffs, derivative_coeffs, second_derivative)

    DERIVATIVE_COEFFS = tuple((
        _dcoeffs_builder(1),
        _dcoeffs_builder(2),
        _dcoeffs_builder(3),
    ))

    def _dcoeffs(self, deivative_order):
        # ... (keep as before) ...
        if 1 <= self.dimensions <= len(self.DERIVATIVE_COEFFS):
            return self.DERIVATIVE_COEFFS[self.dimensions - 1][deivative_order]
        else:
            log.warning(
                f"Unsupported dimension {self.dimensions} for derivative coeffs, using dim 2"
            )
            return self.DERIVATIVE_COEFFS[1][deivative_order]  # Default to 2D

    def __post_init__(self):
        # Ensure p is a numpy array (should be (4, dims))
        p_arr = np.asarray(self.p, dtype=float)
        if p_arr.shape[0] != 4 or p_arr.ndim != 2:
            raise ValueError(
                f"CubicSpline control points 'p' must have shape (4, dims), got {p_arr.shape}"
            )
        object.__setattr__(self, "p", p_arr)
        # Calculate coefficients: (4, 4) @ (4, dims) -> (4, dims)
        object.__setattr__(self, "coefs", np.matmul(self.COEFFICIENTS, self.p))

    def _make_ta3(self, t):
        """
        Create properly shaped array of t powers for vectorized evaluation.
        Creates appropriate arrays for matrix multiplication.
        """
        t_arr = np.asarray(t)
        
        if t_arr.ndim == 0:  # Single t value
            # Create powers [t³, t², t, 1] - shape (4,)
            t2 = t_arr * t_arr
            t3 = t2 * t_arr
            return np.array([t3, t2, t_arr, 1.0])
        else:  # Array of t values
            # Create powers with shape (4, len(t))
            t2 = t_arr * t_arr
            t3 = t2 * t_arr
            t_powers = np.vstack([t3, t2, t_arr, np.ones_like(t_arr)])
            return t_powers  # Shape (4, N)

    def _make_ta2(self, t):
        # t2 = t * t
        # # Correct usage: Create column vector and tile horizontally
        # t_powers = np.array([[t2], [t], [1], [0]])  # Shape (4, 1)
        # ta = np.tile(t_powers, (1, self.dimensions))  # Shape (4, dims)
        # return ta
    
        """
        Create properly shaped array of t powers for vectorized evaluation.
        Creates appropriate arrays for matrix multiplication.
        """
        t_arr = np.asarray(t)
        
        if t_arr.ndim == 0:  # Single t value
            # Create powers [t², t, 1, 0] - shape (4,)
            t2 = t * t
            return np.array([t2, t_arr, 1.0, 0])
        else:  # Array of t values
            # Create powers with shape (4, len(t))
            t2 = t_arr * t_arr
            t_powers = np.vstack([t2, t_arr, np.ones_like(t_arr), np.zeros_like(t_arr)])
            return t_powers  # Shape (4, N)

    # --- evaluate (Iterative version as requested by user) ---
    def evaluate(self, t):
        """
        Evaluates the spline at one or more t values.
        
        Args:
            t: Scalar or array of t values where to evaluate the spline
            
        Returns:
            For scalar t: array of shape (dimensions,) with the point coordinates
            For array t: array of shape (len(t), dimensions) with point coordinates
        """
        t_arr = np.asarray(t)

        # Get powers with shape (4, N)
        powers = self._make_ta3(t_arr)
        
        # Matrix multiply coefficients (4, dims).T with powers (4, N)
        # Result shape: (dims, N)
        result = np.matmul(self.coefs.T, powers)
        
        # Transpose to get shape (N, dims) as expected
        return result.T

    # --- Keep find_roots, curve_maxima_minima_t, curve_inflexion_t ---
    @classmethod
    def find_roots(cls, a, b, c, *, t_range: tuple[float, float] = (0.0, 1.0)):
        # ... (keep as before, using np.isclose maybe) ...
        if np.isclose(a, 0):
            if np.isclose(b, 0):
                return ()
            t = -c / b
            return (t,) if t_range[0] - EPSILON <= t <= t_range[1] + EPSILON else ()
        b2_4ac = b * b - 4 * a * c
        if b2_4ac < 0 and not np.isclose(b2_4ac, 0):
            return ()
        elif b2_4ac < 0:
            b2_4ac = 0
        sqrt_b2_4ac = np.sqrt(b2_4ac)
        two_a = 2 * a
        if np.isclose(two_a, 0):  # Avoid division by zero if a is extremely small
            return ()
        values = ((-b + sqrt_b2_4ac) / two_a, (-b - sqrt_b2_4ac) / two_a)
        return tuple(t for t in values if t_range[0] - EPSILON <= t <= t_range[1] + EPSILON)

    def curve_maxima_minima_t(self, t_range: tuple[float, float] = (0.0, 1.0)):
        d_coefs_scaled = self.coefs * self._dcoeffs(1)  # Shape (4, dims)
        # Derivative coeffs are 3A, 2B, C (rows 0, 1, 2)
        return dict(
            (i, self.find_roots(*(d_coefs_scaled[0:3, i]), t_range=t_range))
            for i in range(self.dimensions)
        )

    def curve_inflexion_t(self, t_range: tuple[float, float] = (0.0, 1.0)):
        d2_coefs_scaled = self.coefs * self._dcoeffs(2)  # Shape (4, dims)
        # Second derivative coeffs are 6A, 2B (rows 0, 1)
        # Solve 6At + 2B = 0 -> find_roots(6A, 2B)
        return dict(
            (
                i,
                QuadraticSpline.find_roots(*(d2_coefs_scaled[0:2, i]), t_range=t_range),
            )  # Use linear root finder
            for i in range(self.dimensions)
        )

    def derivative(self, t):
        t_arr = np.asarray(t)

        # Get powers with shape (4, N)
        powers = self._make_ta2(t_arr)
        
        # Matrix multiply coefficients (4, dims).T with powers (4, N)
        # Result shape: (dims, N)
        coefs = np.multiply(self.coefs, self._dcoeffs(1))
        result = np.matmul(coefs.T, powers)
        
        # Transpose to get shape (N, dims) as expected
        return -result.T

    def normal2d(self, t, dims=[0, 1]):
        t_arr = np.asarray(t)
        if t_arr.ndim == 0:
            d = self.derivative(t_arr)
            if d.shape[0] < 2:
                return np.array([0.0, 0.0])
            vr = np.array([d[dims[1]], -d[dims[0]]])
            mag = np.linalg.norm(vr)
            return vr / mag if mag > EPSILON else np.array([0.0, 0.0])
        else:
            # Vectorized implementation
            d = self.derivative(t_arr)  # shape (N, dims)
            
            # Check if we have enough dimensions
            if d.shape[1] < 2:
                return np.zeros((len(t_arr), 2))
                
            # Create normals array: swap and negate to get perpendicular vector
            vr = np.column_stack([d[:, dims[1]], -d[:, dims[0]]])  # shape (N, 2)
            
            # Calculate magnitudes
            mag = np.linalg.norm(vr, axis=1)  # shape (N,)
            
            # Create result array
            result = np.zeros_like(vr)  # shape (N, 2)
            
            # Only normalize where magnitude is significant
            mask = mag > EPSILON
            if np.any(mask):
                # Properly reshape mag for broadcasting
                result[mask] = vr[mask] / mag[mask, np.newaxis]
            
            return result

    def extremes(self):
        roots = self.curve_maxima_minima_t()
        t_values = {0.0, 1.0}
        for v in roots.values():
            t_values.update(v)
        valid_t_values = sorted([t for t in t_values if 0.0 - EPSILON <= t <= 1.0 + EPSILON])
        clamped_t_values = np.clip(valid_t_values, 0.0, 1.0)
        if not clamped_t_values.size:
            return np.empty((0, self.dimensions))
        # Use iterative evaluate
        return np.array([self.evaluate(t) for t in clamped_t_values])

    def extents(self):
        extr = self.extremes()
        return extentsof(extr)

    def transform(self, m: l.GMatrix) -> 'CubicSpline':
        '''Returns a new spline transformed by the matrix m.'''
        new_p = list((m * to_gvector(p)).A[0:self.dimensions] for p in self.p)
        return CubicSpline(np.array(new_p), self.dimensions)

    def azimuth_t(self, angle: float | l.Angle=0, t_end: bool=False, 
                t_range: tuple[float, float]=(0.0, 1.0)) -> tuple[float, ...]:
        '''Returns the list of t where the tangent is at the given angle from the beginning of the
        given t_range. The angle is in degrees or Angle.'''
        
        angle = l.angle(angle)
        
        start_slope = self.normal2d(t_range[1 if t_end else 0])
        start_rot: l.GMatrix = l.rotZ(sinr_cosr=(start_slope[1], -start_slope[0]))
        
        qs: CubicSpline = self.transform(l.rotZ(angle.inv()) * start_rot)
        
        roots = qs.curve_maxima_minima_t(t_range)

        return sorted(roots[0])


@datatree(frozen=True)
class QuadraticSpline:
    """Quadratic spline evaluator, extents and inflection point finder."""

    p: object = dtfield(doc="The control points for the spline, shape (3, N).")
    dimensions: int = dtfield(
        self_default=lambda s: np.asarray(s.p).shape[1],  # Get dim from shape
        init=True,
        doc="The number of dimensions in the spline.",
    )
    coefs: np.ndarray=dtfield(init=False)

    COEFFICIENTS = np.array([[1.0, -2, 1], [-2.0, 2, 0], [1.0, 0, 0]])  # Shape (3, 3)

    # @staticmethod # For some reason this breaks on Raspberry Pi OS.
    def _dcoeffs_builder(dims):
        # ... (keep as before) ...
        zero_order_derivative_coeffs = np.array([[1.0] * dims, [1] * dims, [1] * dims])
        derivative_coeffs = np.array([[2] * dims, [1] * dims, [0] * dims])
        second_derivative = np.array([[2] * dims, [0] * dims, [0] * dims])
        return (zero_order_derivative_coeffs, derivative_coeffs, second_derivative)

    DERIVATIVE_COEFFS = tuple((
        _dcoeffs_builder(1),
        _dcoeffs_builder(2),
        _dcoeffs_builder(3),
    ))

    def _dcoeffs(self, deivative_order):
        # ... (keep as before) ...
        if 1 <= self.dimensions <= len(self.DERIVATIVE_COEFFS):
            return self.DERIVATIVE_COEFFS[self.dimensions - 1][deivative_order]
        else:
            log.warning(
                f"Unsupported dimension {self.dimensions} for derivative coeffs, using dim 2"
            )
            return self.DERIVATIVE_COEFFS[1][deivative_order]  # Default to 2D

    def __post_init__(self):
        # Ensure p is a numpy array (should be (3, dims))
        p_arr = np.asarray(self.p, dtype=float)
        if p_arr.shape[0] != 3 or p_arr.ndim != 2:
            raise ValueError(
                f"QuadraticSpline control points 'p' must have shape (3, dims), got {p_arr.shape}"
            )
        object.__setattr__(self, "p", p_arr)
        # Calculate coefficients: (3, 3) @ (3, dims) -> (3, dims)
        object.__setattr__(self, "coefs", np.matmul(self.COEFFICIENTS, self.p))

    def _qmake_ta2(self, t):
        """
        Create properly shaped array of t powers for vectorized evaluation.
        Creates appropriate arrays for matrix multiplication.
        """
        t_arr = np.asarray(t)
        
        if t_arr.ndim == 0:  # Single t value
            # Create powers [t², t, 1] - shape (3,)
            return np.array([t_arr**2, t_arr, 1.0])
        else:  # Array of t values
            # Create powers with shape (3, len(t))
            t_powers = np.vstack([t_arr**2, t_arr, np.ones_like(t_arr)])
            return t_powers  # Shape (3, N)

    def _qmake_ta1(self, t):
        # Correct usage: Create column vector and tile horizontally
        t_powers = np.array([[t], [1], [0]])  # Shape (3, 1)
        ta = np.tile(t_powers, (1, self.dimensions))  # Shape (3, dims)
        return ta

    def evaluate(self, t):
        """
        Evaluates the spline at one or more t values.
        
        Args:
            t: Scalar or array of t values where to evaluate the spline
            
        Returns:
            For scalar t: array of shape (dimensions,) with the point coordinates
            For array t: array of shape (len(t), dimensions) with point coordinates
        """
        t_arr = np.asarray(t)
        
        if t_arr.ndim == 0:  # Single scalar t
            # Get powers [t², t, 1] - shape (3,)
            powers = self._qmake_ta2(t_arr)
            
            # Matrix multiply coefficients (3, dims) with powers (3,)
            # Result shape: (dims,)
            return np.dot(self.coefs.T, powers)
        else:  # Multiple t values
            # Get powers [t²_1...t²_n, t_1...t_n, 1...1] - shape (3, N)
            powers = self._qmake_ta2(t_arr)
            
            # Matrix multiply coefficients (3, dims) with powers (3, N)
            # Result shape: (dims, N)
            result = np.matmul(self.coefs.T, powers)
            
            # Transpose to get shape (N, dims) as expected
            return result.T

    @classmethod
    def find_roots(cls, a, b, *, t_range: tuple[float, float] = (0.0, 1.0)):
        if np.isclose(a, 0):
            return ()
        t = -b / a
        return (t,) if t_range[0] - EPSILON <= t <= t_range[1] + EPSILON else ()

    def curve_maxima_minima_t(self, t_range: tuple[float, float] = (0.0, 1.0)):
        d_coefs_scaled = self.coefs * self._dcoeffs(1)  # Shape (3, dims)
        # Derivative coeffs are 2A, B (rows 0, 1)
        return dict(
            (i, self.find_roots(*(d_coefs_scaled[0:2, i]), t_range=t_range))
            for i in range(self.dimensions)
        )

    def curve_inflexion_t(self, t_range: tuple[float, float] = (0.0, 1.0)):
        return dict((i, ()) for i in range(self.dimensions))  # No inflection points

    def derivative(self, t):
        """
        Calculates the derivative of the spline at one or more t values.
        
        Args:
            t: Scalar or array of t values
            
        Returns:
            For scalar t: array of shape (dimensions,) with the derivatives
            For array t: array of shape (len(t), dimensions) with derivatives
        """
        t_arr = np.asarray(t)
        
        # Coefs are [A, B, C] for each dimension (shape (3, dims))
        A = self.coefs[0]  # Shape (dims,)
        B = self.coefs[1]  # Shape (dims,)
        
        # Derivative coefficients: [2A, B, 0]
        deriv_poly_coefs = np.vstack([2 * A, B, np.zeros_like(A)]).T  # Shape (dims, 3)

        if t_arr.ndim == 0:  # Scalar t
            # For a single t, create powers [t, 1, 0] - shape (3,)
            t_powers = np.array([t_arr, 1.0, 0.0])
            
            # Matrix multiply: (dims, 3) @ (3,) -> (dims,)
            return np.dot(deriv_poly_coefs, t_powers)
        else:  # Array of t values
            # For multiple t values, create powers shape (3, N)
            t_powers = np.vstack([t_arr, np.ones_like(t_arr), np.zeros_like(t_arr)])
            
            # Matrix multiply: (dims, 3) @ (3, N) -> (dims, N)
            result = np.matmul(deriv_poly_coefs, t_powers)
            
            # Transpose to shape (N, dims)
            return result.T

    def normal2d(self, t, dims=[0, 1]):
        t_arr = np.asarray(t)
        if t_arr.ndim == 0:
            d = self.derivative(t_arr)
            if d.shape[0] < 2:
                return np.array([0.0, 0.0])
            vr = np.array([d[dims[1]], -d[dims[0]]])
            mag = np.linalg.norm(vr)
            return vr / mag if mag > EPSILON else np.array([0.0, 0.0])
        else:
            # Vectorized implementation
            d = self.derivative(t_arr)  # shape (N, dims)
            
            # Check if we have enough dimensions
            if d.shape[1] < 2:
                return np.zeros((len(t_arr), 2))
                
            # Create normals array: swap and negate to get perpendicular vector
            vr = np.column_stack([d[:, dims[1]], -d[:, dims[0]]])  # shape (N, 2)
            
            # Calculate magnitudes
            mag = np.linalg.norm(vr, axis=1)  # shape (N,)
            
            # Create result array
            result = np.zeros_like(vr)  # shape (N, 2)
            
            # Only normalize where magnitude is significant
            mask = mag > EPSILON
            if np.any(mask):
                # Properly reshape mag for broadcasting
                result[mask] = vr[mask] / mag[mask, np.newaxis]
            
            return result

    def extremes(self):
        roots = self.curve_maxima_minima_t()
        t_values = {0.0, 1.0}
        for v in roots.values():
            t_values.update(v)
        valid_t_values = sorted([t for t in t_values if 0.0 - EPSILON <= t <= 1.0 + EPSILON])
        clamped_t_values = np.clip(valid_t_values, 0.0, 1.0)
        if not clamped_t_values.size:
            return np.empty((0, self.dimensions))
        return np.array([self.evaluate(t) for t in clamped_t_values])

    def extents(self):
        extr = self.extremes()
        return extentsof(extr)

    
    def transform(self, m: l.GMatrix) -> 'QuadraticSpline':
        '''Returns a new spline transformed by the matrix m.'''
        new_p = list((m * to_gvector(p)).A[0:self.dimensions] for p in self.p)
        return QuadraticSpline(np.array(new_p), self.dimensions)

    def azimuth_t(self, angle: float | l.Angle=0, t_end: bool=False, 
                t_range: tuple[float, float]=(0.0, 1.0)) -> tuple[float, ...]:
        '''Returns the list of t where the tangent is at the given angle from the beginning of the
        given t_range. The angle is in degrees or Angle.'''
        
        angle = l.angle(angle)
        
        start_slope = self.normal2d(t_range[1 if t_end else 0])
        start_rot: l.GMatrix = l.rotZ(sinr_cosr=(-start_slope[1], start_slope[0]))
        
        qs: QuadraticSpline = self.transform(angle.inv().rotZ * start_rot)
        
        roots = qs.curve_maxima_minima_t(t_range)

        return sorted(roots[0])

