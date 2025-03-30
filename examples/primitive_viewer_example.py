r"""
Doc for OpenSCAD text module.
... (docstring remains the same) ...
"""

import numpy as np
from datatrees import datatree, dtfield
import anchorscad_lib.linear as l
import logging
import sys
import os
import copy # For deepcopy
import io   # <-- Import io for BytesIO

# --- Add HarfBuzz Import ---
try:
    import uharfbuzz as hb
except ImportError:
    print("ERROR: uharfbuzz library not found. Text shaping will be unavailable.")
    print("Please install it: pip install uharfbuzz fonttools[ufo]")
    hb = None # Set to None if import fails

# --- Library Imports for Font Handling (moved inside __post_init__) ---
# import fontTools.ttLib
# from fontTools.pens.recordingPen import RecordingPen
# from PIL import ImageFont, ImageDraw, Image

logging.basicConfig(level=logging.INFO) # Or logging.WARNING
log = logging.getLogger(__name__)

EPSILON = 1e-6

# --- Keep Vector/Math Helpers ---
def to_gvector(np_array):
    """Converts a 2D or 3D numpy array to a GVector."""
    arr = np.asarray(np_array)
    if arr.shape == (2,):
        return l.GVector([arr[0], arr[1], 0, 1])
    elif arr.shape == (3,):
        return l.GVector([arr[0], arr[1], arr[2], 1])
    elif arr.shape == (4,):
         return l.GVector(arr)
    else:
         raise ValueError(f"Cannot convert array of shape {arr.shape} to GVector")

def extentsof(p: np.ndarray) -> np.ndarray:
    """Calculates the min and max coordinates (bounding box) of points."""
    if p.shape[0] == 0: # Handle empty array
        # Return a zero-size box at origin
        return np.array([[0., 0.], [0., 0.]])
    return np.array((p.min(axis=0), p.max(axis=0)))

# --- Spline Classes (with corrected np.tile) ---
@datatree(frozen=True)
class CubicSpline():
    '''Cubic spline evaluator, extents and inflection point finder.'''
    p: object=dtfield(doc='The control points for the spline, shape (4, N).')
    dimensions: int=dtfield(
        self_default=lambda s: np.asarray(s.p).shape[1], # Get dim from shape
        init=True,
        doc='The number of dimensions in the spline.')

    COEFFICIENTS=np.array([
        [-1.,  3, -3,  1 ],
        [  3, -6,  3,  0 ],
        [ -3,  3,  0,  0 ],
        [  1,  0,  0,  0 ]]) # Shape (4, 4)

    #@staticmethod # For some reason this breaks on Raspberry Pi OS.
    def _dcoeffs_builder(dims):
        # ... (keep as before) ...
        zero_order_derivative_coeffs=np.array([[1.] * dims, [1] * dims, [1] * dims, [1] * dims])
        derivative_coeffs=np.array([[3.] * dims, [2] * dims, [1] * dims, [0] * dims])
        second_derivative=np.array([[6] * dims, [2] * dims, [0] * dims, [0] * dims])
        return (zero_order_derivative_coeffs, derivative_coeffs, second_derivative)

    DERIVATIVE_COEFFS = tuple((
        _dcoeffs_builder(1),
        _dcoeffs_builder(2),
        _dcoeffs_builder(3), ))

    def _dcoeffs(self, deivative_order):
        # ... (keep as before) ...
        if 1 <= self.dimensions <= len(self.DERIVATIVE_COEFFS):
            return self.DERIVATIVE_COEFFS[self.dimensions - 1][deivative_order]
        else:
            log.warning(f"Unsupported dimension {self.dimensions} for derivative coeffs, using dim 2")
            return self.DERIVATIVE_COEFFS[1][deivative_order] # Default to 2D


    def __post_init__(self):
        # Ensure p is a numpy array (should be (4, dims))
        p_arr = np.asarray(self.p, dtype=float)
        if p_arr.shape[0] != 4 or p_arr.ndim != 2:
             raise ValueError(f"CubicSpline control points 'p' must have shape (4, dims), got {p_arr.shape}")
        object.__setattr__(self, 'p', p_arr)
        # Calculate coefficients: (4, 4) @ (4, dims) -> (4, dims)
        object.__setattr__(self, 'coefs', np.matmul(self.COEFFICIENTS, self.p))

    def _make_ta3(self, t):
        t2 = t * t
        t3 = t2 * t
        # Correct usage: Create column vector and tile horizontally
        t_powers = np.array([[t3], [t2], [t], [1]]) # Shape (4, 1)
        ta = np.tile(t_powers, (1, self.dimensions)) # Shape (4, dims)
        return ta

    def _make_ta2(self, t):
        t2 = t * t
        # Correct usage: Create column vector and tile horizontally
        t_powers = np.array([[t2], [t], [1], [0]]) # Shape (4, 1)
        ta = np.tile(t_powers, (1, self.dimensions)) # Shape (4, dims)
        return ta

    # --- evaluate (Iterative version as requested by user) ---
    def evaluate(self, t):
        """Evaluates the spline at one or more t values."""
        t_arr = np.asarray(t)
        if t_arr.ndim == 0: # Scalar input
             ta = self._make_ta3(t_arr.item()) # Pass scalar t
             # coefs=(4,dims), ta=(4,dims) -> multiply=(4,dims) -> sum(axis=0)=(dims,)
             return np.sum(np.multiply(self.coefs, ta), axis=0)
        else: # Array input
            results = []
            for t_val in t_arr:
                results.append(self.evaluate(t_val)) # Recursive call for scalar
            return np.array(results) # Shape (N, dims)

    # --- Keep find_roots, curve_maxima_minima_t, curve_inflexion_t ---
    @classmethod
    def find_roots(cls, a, b, c, *, t_range: tuple[float, float]=(0.0, 1.0)):
         # ... (keep as before, using np.isclose maybe) ...
         if np.isclose(a, 0):
            if np.isclose(b, 0): return ()
            t = -c / b
            return (t,) if t_range[0] - EPSILON <= t <= t_range[1] + EPSILON else ()
         b2_4ac = b * b - 4 * a * c
         if b2_4ac < 0 and not np.isclose(b2_4ac, 0): return ()
         elif b2_4ac < 0: b2_4ac = 0
         sqrt_b2_4ac = np.sqrt(b2_4ac)
         two_a = 2 * a
         if np.isclose(two_a, 0): # Avoid division by zero if a is extremely small
              return ()
         values = ((-b + sqrt_b2_4ac) / two_a, (-b - sqrt_b2_4ac) / two_a)
         return tuple(t for t in values if t_range[0] - EPSILON <= t <= t_range[1] + EPSILON)

    def curve_maxima_minima_t(self, t_range: tuple[float, float]=(0.0, 1.0)):
         d_coefs_scaled = self.coefs * self._dcoeffs(1) # Shape (4, dims)
         # Derivative coeffs are 3A, 2B, C (rows 0, 1, 2)
         return dict((i, self.find_roots(*(d_coefs_scaled[0:3, i]), t_range=t_range))
                     for i in range(self.dimensions))

    def curve_inflexion_t(self, t_range: tuple[float, float]=(0.0, 1.0)):
         d2_coefs_scaled = self.coefs * self._dcoeffs(2) # Shape (4, dims)
         # Second derivative coeffs are 6A, 2B (rows 0, 1)
         # Solve 6At + 2B = 0 -> find_roots(6A, 2B)
         return dict((i, QuadraticSpline.find_roots(*(d2_coefs_scaled[0:2, i]), t_range=t_range)) # Use linear root finder
                     for i in range(self.dimensions))

    # --- derivative (using iterative evaluate) ---
    def derivative(self, t):
        """Calculates the derivative of the spline."""
        # B'(t) = 3(1-t)^2(P1-P0) + 6(1-t)t(P2-P1) + 3t^2(P3-P2)
        # Using polynomial form: d/dt (At^3 + Bt^2 + Ct + D) = 3At^2 + 2Bt + C
        t_arr = np.asarray(t)
        # Coefs are [A, B, C, D] for each dimension (shape (4, dims))
        A = self.coefs[0] # Shape (dims,)
        B = self.coefs[1] # Shape (dims,)
        C = self.coefs[2] # Shape (dims,)
        # Derivative coefficients: [3A, 2B, C]
        deriv_poly_coefs = np.vstack([3*A, 2*B, C]) # Shape (3, dims)

        if t_arr.ndim == 0: # Scalar t
            t_val = t_arr.item()
            t2 = t_val * t_val
            # Create t-power array matching deriv_poly_coefs shape
            ta = np.tile(np.array([[t2], [t_val], [1]]), (1, self.dimensions)) # Shape (3, dims)
            return np.sum(np.multiply(deriv_poly_coefs, ta), axis=0) # Shape (dims,)
        else: # Array t
            results = []
            for t_val in t_arr:
                results.append(self.derivative(t_val)) # Recursive call
            return np.array(results) # Shape (N, dims)

    # --- Keep normal2d, extremes, extents, transform, azimuth_t ---
    # (Make sure they use the updated derivative/evaluate if needed)
    def normal2d(self, t, dims=[0, 1]):
        d = self.derivative(t)
        if d.shape[0] < 2: return np.array([0., 0.])
        vr = np.array([d[dims[1]], -d[dims[0]]])
        mag = np.linalg.norm(vr)
        return vr / mag if mag > EPSILON else np.array([0., 0.])

    def extremes(self):
        roots = self.curve_maxima_minima_t()
        t_values = {0.0, 1.0}
        for v in roots.values(): t_values.update(v)
        valid_t_values = sorted([t for t in t_values if 0.0 - EPSILON <= t <= 1.0 + EPSILON])
        clamped_t_values = np.clip(valid_t_values, 0.0, 1.0)
        if not clamped_t_values.size: return np.empty((0, self.dimensions))
        # Use iterative evaluate
        return np.array([self.evaluate(t) for t in clamped_t_values])

    def extents(self):
        extr = self.extremes(); return extentsof(extr)

    def transform(self, m: l.GMatrix) -> 'CubicSpline':
         # This transform assumes self.p is array (4, dims)
         try:
             new_p_list = []
             for i in range(self.p.shape[0]): # Iterate through rows (points)
                  pt = self.p[i, :] # Get the i-th point (shape (dims,))
                  gv = to_gvector(pt) # Convert to GVector (adds 0 for z if needed)
                  transformed_gv = m * gv
                  new_p_list.append(transformed_gv.A[0, :self.dimensions]) # Extract relevant dims
             # Create new spline with list of points, let __post_init__ handle array conversion
             return CubicSpline(p=new_p_list)
         except Exception as e:
              log.error(f"CubicSpline transform failed: {e}. Input p shape: {self.p.shape}")
              return self # Return self on error? Or raise?

    def azimuth_t(self, angle: float | l.Angle=0, t_end: bool=False,
                t_range: tuple[float, float]=(0.0, 1.0)) -> tuple[float, ...]:
         # ... (keep as before, relies on derivative) ...
         if self.dimensions < 2: return ()
         angle = l.angle(angle)
         ref_t = t_range[1] if t_end else t_range[0]
         d_ref = self.derivative(ref_t)
         if np.linalg.norm(d_ref) < EPSILON: return ()
         ref_angle_rad = np.arctan2(d_ref[1], d_ref[0])
         target_angle_rad = ref_angle_rad + angle.radians
         cos_target, sin_target = np.cos(target_angle_rad), np.sin(target_angle_rad)
         # D(t) = 3At^2 + 2Bt + C
         A, B, C = self.coefs[0], self.coefs[1], self.coefs[2]
         # Dy(t)*cos - Dx(t)*sin = 0
         # (3Ay*t^2+2By*t+Cy)*cos - (3Ax*t^2+2Bx*t+Cx)*sin = 0
         # t^2*(3Ay*cos-3Ax*sin) + t*(2By*cos-2Bx*sin) + (Cy*cos-Cx*sin) = 0
         a = 3*A[1]*cos_target - 3*A[0]*sin_target
         b = 2*B[1]*cos_target - 2*B[0]*sin_target
         c =   C[1]*cos_target -   C[0]*sin_target
         return self.find_roots(a, b, c, t_range=t_range)


@datatree(frozen=True)
class QuadraticSpline():
    '''Quadratic spline evaluator, extents and inflection point finder.'''
    p: object=dtfield(doc='The control points for the spline, shape (3, N).')
    dimensions: int=dtfield(
        self_default=lambda s: np.asarray(s.p).shape[1], # Get dim from shape
        init=True,
        doc='The number of dimensions in the spline.')

    COEFFICIENTS=np.array([
        [  1., -2,  1 ],
        [ -2.,  2,  0 ],
        [  1.,  0,  0 ]]) # Shape (3, 3)

    #@staticmethod # For some reason this breaks on Raspberry Pi OS.
    def _dcoeffs_builder(dims):
        # ... (keep as before) ...
        zero_order_derivative_coeffs=np.array([[1.] * dims, [1] * dims, [1] * dims])
        derivative_coeffs=np.array([[2] * dims, [1] * dims, [0] * dims])
        second_derivative=np.array([[2] * dims, [0] * dims, [0] * dims])
        return (zero_order_derivative_coeffs, derivative_coeffs, second_derivative)

    DERIVATIVE_COEFFS = tuple((
        _dcoeffs_builder(1),
        _dcoeffs_builder(2),
        _dcoeffs_builder(3), ))

    def _dcoeffs(self, deivative_order):
        # ... (keep as before) ...
        if 1 <= self.dimensions <= len(self.DERIVATIVE_COEFFS):
            return self.DERIVATIVE_COEFFS[self.dimensions - 1][deivative_order]
        else:
            log.warning(f"Unsupported dimension {self.dimensions} for derivative coeffs, using dim 2")
            return self.DERIVATIVE_COEFFS[1][deivative_order] # Default to 2D

    def __post_init__(self):
        # Ensure p is a numpy array (should be (3, dims))
        p_arr = np.asarray(self.p, dtype=float)
        if p_arr.shape[0] != 3 or p_arr.ndim != 2:
             raise ValueError(f"QuadraticSpline control points 'p' must have shape (3, dims), got {p_arr.shape}")
        object.__setattr__(self, 'p', p_arr)
        # Calculate coefficients: (3, 3) @ (3, dims) -> (3, dims)
        object.__setattr__(self, 'coefs', np.matmul(self.COEFFICIENTS, self.p))

    def _qmake_ta2(self, t):
        # Correct usage: Create column vector and tile horizontally
        t_powers = np.array([[t**2], [t], [1]]) # Shape (3, 1)
        ta = np.tile(t_powers, (1, self.dimensions)) # Shape (3, dims)
        return ta

    def _qmake_ta1(self, t):
        # Correct usage: Create column vector and tile horizontally
        t_powers = np.array([[t], [1], [0]]) # Shape (3, 1)
        ta = np.tile(t_powers, (1, self.dimensions)) # Shape (3, dims)
        return ta

    # --- evaluate (Iterative version as requested by user) ---
    def evaluate(self, t):
        """Evaluates the spline at one or more t values."""
        t_arr = np.asarray(t)
        if t_arr.ndim == 0: # Scalar input
             ta = self._qmake_ta2(t_arr.item()) # Pass scalar t
             # coefs=(3,dims), ta=(3,dims) -> multiply=(3,dims) -> sum(axis=0)=(dims,)
             return np.sum(np.multiply(self.coefs, ta), axis=0)
        else: # Array input
            results = []
            for t_val in t_arr:
                results.append(self.evaluate(t_val)) # Recursive call for scalar
            return np.array(results) # Shape (N, dims)

    # --- Keep find_roots (linear), curve_maxima_minima_t, curve_inflexion_t ---
    @classmethod
    def find_roots(cls, a, b, *, t_range: tuple[float, float]=(0.0, 1.0)):
         if np.isclose(a, 0): return ()
         t = -b / a
         return (t,) if t_range[0] - EPSILON <= t <= t_range[1] + EPSILON else ()

    def curve_maxima_minima_t(self, t_range: tuple[float, float]=(0.0, 1.0)):
         d_coefs_scaled = self.coefs * self._dcoeffs(1) # Shape (3, dims)
         # Derivative coeffs are 2A, B (rows 0, 1)
         return dict((i, self.find_roots(*(d_coefs_scaled[0:2, i]), t_range=t_range))
                     for i in range(self.dimensions))

    def curve_inflexion_t(self, t_range: tuple[float, float]=(0.0, 1.0)):
         return dict((i, ()) for i in range(self.dimensions)) # No inflection points

    # --- derivative (using iterative evaluate) ---
    def derivative(self, t):
        """Calculates the derivative of the spline."""
        # B'(t) = 2(1-t)(P1-P0) + 2t(P2-P1)
        # Using polynomial form: d/dt (At^2 + Bt + C) = 2At + B
        t_arr = np.asarray(t)
        # Coefs are [A, B, C] for each dimension (shape (3, dims))
        A = self.coefs[0] # Shape (dims,)
        B = self.coefs[1] # Shape (dims,)
        # Derivative coefficients: [2A, B]
        deriv_poly_coefs = np.vstack([2*A, B]) # Shape (2, dims)

        if t_arr.ndim == 0: # Scalar t
            t_val = t_arr.item()
            # Create t-power array matching deriv_poly_coefs shape
            ta = np.tile(np.array([[t_val], [1]]), (1, self.dimensions)) # Shape (2, dims)
            return np.sum(np.multiply(deriv_poly_coefs, ta), axis=0) # Shape (dims,)
        else: # Array t
            results = []
            for t_val in t_arr:
                results.append(self.derivative(t_val)) # Recursive call
            return np.array(results) # Shape (N, dims)

    # --- Keep normal2d, extremes, extents, transform, azimuth_t ---
    # (Make sure they use the updated derivative/evaluate if needed)
    def normal2d(self, t, dims=[0, 1]):
        d = self.derivative(t)
        if d.shape[0] < 2: return np.array([0., 0.])
        vr = np.array([d[dims[1]], -d[dims[0]]])
        mag = np.linalg.norm(vr)
        return vr / mag if mag > EPSILON else np.array([0., 0.])

    def extremes(self):
        roots = self.curve_maxima_minima_t()
        t_values = {0.0, 1.0}
        for v in roots.values(): t_values.update(v)
        valid_t_values = sorted([t for t in t_values if 0.0 - EPSILON <= t <= 1.0 + EPSILON])
        clamped_t_values = np.clip(valid_t_values, 0.0, 1.0)
        if not clamped_t_values.size: return np.empty((0, self.dimensions))
        # Use iterative evaluate
        return np.array([self.evaluate(t) for t in clamped_t_values])

    def extents(self):
        extr = self.extremes(); return extentsof(extr)

    def transform(self, m: l.GMatrix) -> 'QuadraticSpline':
         # This transform assumes self.p is array (3, dims)
         try:
             new_p_list = []
             for i in range(self.p.shape[0]): # Iterate through rows (points)
                  pt = self.p[i, :] # Get the i-th point (shape (dims,))
                  gv = to_gvector(pt)
                  transformed_gv = m * gv
                  new_p_list.append(transformed_gv.A[0, :self.dimensions])
             # Create new spline with list of points, let __post_init__ handle array conversion
             return QuadraticSpline(p=new_p_list)
         except Exception as e:
             log.error(f"QuadraticSpline transform failed: {e}. Input p shape: {self.p.shape}")
             return self

    def azimuth_t(self, angle: float | l.Angle=0, t_end: bool=False,
                t_range: tuple[float, float]=(0.0, 1.0)) -> tuple[float, ...]:
         # ... (keep as before, relies on derivative) ...
         if self.dimensions < 2: return ()
         angle = l.angle(angle)
         ref_t = t_range[1] if t_end else t_range[0]
         d_ref = self.derivative(ref_t)
         if np.linalg.norm(d_ref) < EPSILON: return ()
         ref_angle_rad = np.arctan2(d_ref[1], d_ref[0])
         target_angle_rad = ref_angle_rad + angle.radians
         cos_target, sin_target = np.cos(target_angle_rad), np.sin(target_angle_rad)
         # D(t) = 2At + B
         A, B = self.coefs[0], self.coefs[1]
         # Dy(t)*cos - Dx(t)*sin = 0
         # (2Ay*t+By)*cos - (2Ax*t+Bx)*sin = 0
         # t*(2Ay*cos-2Ax*sin) + (By*cos-Bx*sin) = 0
         a = 2*A[1]*cos_target - 2*A[0]*sin_target
         b =   B[1]*cos_target -   B[0]*sin_target
         return self.find_roots(a, b, t_range=t_range)


# --- TextContext Class (with HarfBuzz) ---
@datatree
class TextContext:
    text: str=dtfield(default="")
    size: float=dtfield(default=10.0)
    font: str=dtfield(default="Liberation Sans")
    halign: str=dtfield(default="left")
    valign: str=dtfield(default="baseline")
    spacing: float=dtfield(default=1.0)
    direction: str=dtfield(default="ltr") # Note: Used for final transform, hb uses base_direction
    language: str=dtfield(default="en")
    script: str=dtfield(default="latin") # Used as hint for HarfBuzz
    fa: float=dtfield(default=12.0)
    fs: float=dtfield(default=2.0)
    fn: int=dtfield(default=0)
    base_direction: str=dtfield(default="ltr") # Used for HarfBuzz direction

    # --- Internal fields ---
    _font: object = dtfield(init=False, repr=False, default=None) # fontTools TTFont object
    _font_path: str = dtfield(init=False, repr=False, default=None) # Store path if available
    _hb_font: object = dtfield(init=False, repr=False, default=None) # uharfbuzz Font object
    _glyph_set: object = dtfield(init=False, repr=False, default=None) # fontTools glyph set
    _scale_factor: float = dtfield(init=False, repr=False, default=1.0)
    _pil_font: object = dtfield(init=False, repr=False, default=None) # Keep PIL for fallback/metrics if needed
    _y_axis_inverted: bool = dtfield(init=False, repr=False, default=False)
    _fallback_fonts: dict = dtfield(init=False, repr=False, default_factory=dict)
    # Store imported modules to avoid repeated imports in methods
    _font_tools: object = dtfield(init=False, repr=False, default=None)
    _recording_pen_cls: type = dtfield(init=False, repr=False, default=None)
    _image_font_cls: type = dtfield(init=False, repr=False, default=None)
    _image_draw_cls: type = dtfield(init=False, repr=False, default=None)
    _image_cls: type = dtfield(init=False, repr=False, default=None)


    def __post_init__(self):
        # --- Import heavy libraries once ---
        try:
            import fontTools.ttLib
            from fontTools.pens.recordingPen import RecordingPen
            from PIL import ImageFont, ImageDraw, Image
            self._font_tools = fontTools
            self._recording_pen_cls = RecordingPen
            self._image_font_cls = ImageFont
            self._image_draw_cls = ImageDraw
            self._image_cls = Image
        except ImportError as e:
             log.error(f"Error importing PIL or fontTools: {e}. Text rendering will likely fail.")
             self._font_tools = None; self._recording_pen_cls = None; self._image_font_cls = None
             self._image_draw_cls = None; self._image_cls = None
             raise ImportError(f"Missing required libraries (Pillow or fontTools): {e}") from e

        if hb is None:
            raise ImportError("uharfbuzz library is required for text shaping but could not be imported.")

        # --- Keep validations ---
        if not self.font: self.font = "Liberation Sans"
        if self.halign not in ("left", "center", "right"): raise ValueError(f"Invalid halign: {self.halign}")
        if self.valign not in ("top", "center", "baseline", "bottom"): raise ValueError(f"Invalid valign: {self.valign}")
        if self.direction not in ("ltr", "rtl", "ttb", "btt"): raise ValueError(f"Invalid direction: {self.direction}")
        if self.base_direction not in ("ltr", "rtl"): raise ValueError(f"Invalid base_direction: {self.base_direction}")
        # --- End validations ---

        # --- Load Font (sets self._font, self._pil_font, self._font_path) ---
        self._load_font()

        if self._font is None:
            # If fontTools failed even with fallbacks, we cannot proceed with shaping
            raise ValueError(f"Cannot perform text shaping: fontTools failed to load '{self.font}' or any fallback.")

        # --- Setup HarfBuzz Font, Scale Factor, GlyphSet ---
        face_data = None
        # Try getting data from stored path first
        if self._font_path and os.path.exists(self._font_path):
            try:
                with open(self._font_path, 'rb') as f:
                    face_data = f.read()
                log.info(f"Read font data for HarfBuzz from path: {self._font_path}")
            except Exception as e:
                log.warning(f"Failed to read font data from path {self._font_path}: {e}")
                face_data = None

        # If path failed or wasn't available, try saving font to buffer
        if face_data is None and hasattr(self._font, 'save'):
            log.info("Font path not available or failed, attempting to save font data to buffer for HarfBuzz.")
            try:
                 buffer = io.BytesIO()
                 self._font.save(buffer)
                 face_data = buffer.getvalue()
                 log.info("Successfully saved font data to buffer for HarfBuzz.")
            except Exception as e:
                log.warning(f"Failed to save font data to buffer: {e}")
                face_data = None

        # If still no data, raise error
        if face_data is None:
             raise ValueError("Could not obtain font byte data for HarfBuzz Face creation.")

        # Create HarfBuzz Face and Font
        try:
            hb_face = hb.Face(face_data)
            self._hb_font = hb.Font(hb_face)
            units_per_em = hb_face.upem
            if units_per_em <= 0: units_per_em = 1000 # Fallback
            # Set font pt size? HarfBuzz uses font units by default. Scaling happens later.
            # self._hb_font.ptem = self.size # Experiment if needed, but scaling later is typical
        except Exception as e:
            log.error(f"Failed to create HarfBuzz font object from data: {e}", exc_info=True)
            self._hb_font = None
            raise ValueError(f"Failed to create HarfBuzz font object from data: {e}") from e

        # Calculate scale factor
        self._scale_factor = self.size / units_per_em

        # Get glyph set for drawing outlines
        try:
             self._glyph_set = self._font.getGlyphSet()
        except Exception as e:
             log.error(f"Failed to get glyphSet from font: {e}", exc_info=True)
             self._glyph_set = None
             raise ValueError(f"Failed to get glyphSet from font: {e}") from e

        # --- Keep orientation detection and fallback font loading ---
        self._detect_font_orientation()
        self._load_fallback_fonts()

    # --- _load_font (ensure it sets self._font, self._pil_font, self._font_path) ---
    def _load_font(self):
        """Load the font specified in self.font using PIL and fontTools"""
        # Reset first
        self._pil_font = None
        self._font = None
        self._font_path = None # Reset path
        font_path_found = None
        font_name_used_for_pil = None

        if not self._image_font_cls or not self._font_tools:
             raise RuntimeError("Font loading dependencies (PIL/fontTools) not available.")

        try:
            font_name, font_style = self.font, None
            if ":" in self.font:
                parts = self.font.split(":", 1)
                font_name = parts[0]
                if len(parts) > 1 and parts[1].startswith("style="):
                    font_style = parts[1][6:]

            # Try loading with PIL (using path or name)
            try:
                # Attempt direct path/name load with PIL
                log.debug(f"Attempting direct PIL load: {self.font}")
                self._pil_font = self._image_font_cls.truetype(self.font, size=int(self.size * 3.937))
                font_name_used_for_pil = self.font # Store the name/path that worked
                # If successful, try to get path for fontTools
                if hasattr(self._pil_font, 'path'):
                     font_path_found = self._pil_font.path
                     log.debug(f"PIL loaded, path found: {font_path_found}")
                else:
                     # Try finding path based on name PIL resolved to
                     pil_resolved_name = self._pil_font.getname()[0]
                     log.debug(f"PIL loaded, no path attribute. Resolved name: {pil_resolved_name}. Searching for path...")
                     font_path_found = self._find_font_file(pil_resolved_name, font_style) # Search using resolved name

            except OSError:
                # If direct load fails, try finding font file path
                log.info(f"Direct PIL load failed for '{self.font}', searching system...")
                font_path_found = self._find_font_file(font_name, font_style)
                if font_path_found:
                    try:
                        log.debug(f"Attempting PIL load with found path: {font_path_found}")
                        self._pil_font = self._image_font_cls.truetype(font_path_found, size=int(self.size * 3.937))
                        font_name_used_for_pil = font_path_found # Store path that worked
                    except OSError as e:
                        log.warning(f"PIL failed to load found path '{font_path_found}': {e}")
                        font_path_found = None # Reset if PIL fails on found path
                else:
                     log.warning(f"Could not find font file for '{font_name}' with style '{font_style}'.")

            # If PIL loading failed entirely, try fallbacks
            if self._pil_font is None:
                log.warning(f"Could not load primary font '{self.font}' with PIL. Trying fallbacks...")
                fallback_fonts = ["Arial", "Times New Roman", "Verdana", "DejaVu Sans", "Liberation Sans"]
                for fallback in fallback_fonts:
                    try:
                        log.debug(f"Attempting PIL fallback: {fallback}")
                        self._pil_font = self._image_font_cls.truetype(fallback, size=int(self.size * 3.937))
                        log.info(f"Using fallback PIL font '{fallback}'.")
                        font_name_used_for_pil = fallback # Store fallback name
                        # Try to get path for fontTools fallback
                        if hasattr(self._pil_font, 'path'): font_path_found = self._pil_font.path
                        else: font_path_found = self._find_font_file(fallback, None)
                        break
                    except OSError: continue
                if self._pil_font is None:
                    raise ValueError(f"Could not load font '{self.font}' or any fallback fonts with PIL.")

            # --- Now load with fontTools ---
            log.info(f"Attempting fontTools load. Path found: {font_path_found}")
            if font_path_found and os.path.exists(font_path_found):
                 try:
                      self._font = self._font_tools.ttLib.TTFont(font_path_found)
                      self._font_path = font_path_found # Store the confirmed path
                      log.info(f"fontTools loaded successfully from path: {font_path_found}")
                 except self._font_tools.ttLib.TTLibError as e:
                      log.error(f"fontTools failed to load path '{font_path_found}': {e}")
                      self._font = None
                      self._font_path = None # Clear path if fontTools failed with it
            else:
                 # If no path, try loading by name (using the name PIL resolved to or fallback name)
                 effective_font_name = font_name_used_for_pil or font_name # Use name that worked for PIL
                 log.info(f"No valid path found, attempting fontTools load by name: {effective_font_name}")
                 try:
                      self._font = self._font_tools.ttLib.TTFont(effective_font_name)
                      log.info(f"fontTools loaded successfully by name: {effective_font_name}")
                      # Try to get path from fontTools object if possible
                      if hasattr(self._font, 'reader') and hasattr(self._font.reader, 'file') and hasattr(self._font.reader.file, 'name'):
                           self._font_path = self._font.reader.file.name
                           log.info(f"Retrieved font path from fontTools object: {self._font_path}")
                      else:
                           log.warning("Could not retrieve font path from fontTools object after loading by name.")
                           self._font_path = None # Ensure path is None if not retrieved

                 except self._font_tools.ttLib.TTLibError as e:
                      log.error(f"fontTools failed to load font by name '{effective_font_name}': {e}")
                      self._font = None
                      self._font_path = None

            if self._font is None:
                 log.error(f"fontTools could not load font '{self.font}'. Outlines will be unavailable/rectangular.")
                 # Keep self._pil_font for basic metrics, but shaping won't work

        except Exception as e:
            log.error(f"Unexpected error loading font {self.font}: {e}", exc_info=True)
            raise ValueError(f"Error loading font {self.font}: {str(e)}") from e

    # --- Keep _find_font_file ---
    def _find_font_file(self, family, style):
         """Tries to find a font file path matching family and style."""
         # This is platform dependent and complex. Using a basic search.
         # Consider using matplotlib.font_manager for a more robust search.
         import platform
         font_dirs = []
         system = platform.system()
         if system == "Windows":
             windir = os.environ.get("WINDIR", "C:\\Windows")
             font_dirs.append(os.path.join(windir, "Fonts"))
         elif system == "Darwin": # macOS
             font_dirs.extend(["/Library/Fonts", "/System/Library/Fonts", os.path.expanduser("~/Library/Fonts")])
         else: # Linux/Unix
             font_dirs.extend(["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"), os.path.expanduser("~/.local/share/fonts")])

         font_dirs = [d for d in font_dirs if d and os.path.isdir(d)] # Filter valid dirs
         if not font_dirs:
             log.warning("No valid system font directories found.")
             return None

         family_lower = family.lower()
         style_lower = style.lower() if style else "regular"

         found_path = None
         best_match_score = -1

         log.debug(f"Searching for font: Family='{family}', Style='{style}' in dirs: {font_dirs}")

         for d in font_dirs:
             try:
                 for root, _, files in os.walk(d, followlinks=True): # followlinks might be needed
                     for fname in files:
                         if fname.lower().endswith((".ttf", ".otf")): # Only check ttf/otf for fontTools
                             fpath = os.path.join(root, fname)
                             try:
                                 # Use fontTools to check name/style if possible
                                 # Use lazy=True for faster check, only load 'name' table if needed
                                 temp_font = self._font_tools.ttLib.TTFont(fpath, lazy=True)
                                 if 'name' not in temp_font:
                                     temp_font.close()
                                     continue # Skip if no name table

                                 # Check Name table for Family and Style
                                 f_family, f_style = None, None
                                 # Prioritize platform-specific names? For now, check common IDs
                                 for rec in temp_font['name'].names:
                                     # Name ID 1: Font Family name
                                     if rec.nameID == 1: f_family = rec.toUnicode()
                                     # Name ID 2: Font Subfamily name (Style)
                                     if rec.nameID == 2: f_style = rec.toUnicode()
                                     # Name ID 16: Typographic Family name (Preferred Family)
                                     if rec.nameID == 16: f_family = rec.toUnicode()
                                     # Name ID 17: Typographic Subfamily name (Preferred Style)
                                     if rec.nameID == 17: f_style = rec.toUnicode()
                                     # Break early if both found?
                                     if f_family and f_style: break

                                 temp_font.close() # Close lazy loaded font

                                 if f_family and f_family.lower() == family_lower:
                                      current_style = (f_style or "Regular").lower()
                                      # Simple scoring for style match
                                      score = 0
                                      if current_style == style_lower: score = 10 # Exact match is best
                                      # Check for substring matches (e.g., 'Bold Italic' vs 'Bold')
                                      elif style_lower in current_style: score += 3
                                      elif current_style in style_lower: score += 1 # Less specific match
                                      # Check common style keywords
                                      if "bold" in style_lower and "bold" in current_style: score += 2
                                      if "italic" in style_lower and "italic" in current_style: score += 2
                                      if "oblique" in style_lower and "oblique" in current_style: score += 2
                                      if "regular" in style_lower and "regular" in current_style: score += 2
                                      # Prefer 'Regular' if no style was specified
                                      if style is None and current_style == "regular": score = 8

                                      if score > best_match_score:
                                           log.debug(f"  Found potential match: {fpath} (Family: {f_family}, Style: {f_style or 'Regular'}, Score: {score})")
                                           best_match_score = score
                                           found_path = fpath
                                           # Don't return early unless score is perfect, keep searching for better match
                                           if score == 10: break # Found exact style match

                             except Exception as e_inner:
                                 # log.debug(f"Could not read font metadata for {fpath}: {e_inner}")
                                 continue # Ignore fonts that fail to load/parse
                     if best_match_score == 10: break # Stop searching directory if exact match found
             except OSError: continue # Ignore dirs we can't access
             if best_match_score == 10: break # Stop searching all dirs if exact match found

         if found_path:
             log.info(f"Best match font file: '{found_path}' for family '{family}' style '{style}' (Score: {best_match_score})")
         else:
             log.warning(f"Could not find suitable font file for family '{family}' style '{style}'.")
         return found_path

    # --- Keep _detect_font_orientation ---
    def _detect_font_orientation(self):
         # ... (keep as before) ...
         self._y_axis_inverted = False
         known_inverted_fonts = ["New Gulim", "Gulim", "GulimChe", "Dotum", "DotumChe", "MS Mincho", "MS Gothic", "MS PMincho", "MS PGothic", "SimSun", "NSimSun", "SimHei", "FangSong", "KaiTi"]
         # Check using self.font string first
         font_name_check = self.font.split(":")[0] # Get base name
         if any(inverted_font.lower() in font_name_check.lower() for inverted_font in known_inverted_fonts):
             self._y_axis_inverted = True
             log.info(f"Detected inverted Y axis for font '{self.font}' based on known list.")
             return

         # Then try detection using glyph metrics if fontTools loaded
         if self._font and hasattr(self._font, 'getBestCmap') and self._recording_pen_cls:
             try:
                 cmap = self._font.getBestCmap(); glyph_set = self._font.getGlyphSet()
                 if ord('A') in cmap:
                     pen = self._recording_pen_cls()
                     glyph_name = cmap.get(ord('A'))
                     if glyph_name and glyph_name in glyph_set:
                         glyph_set[glyph_name].draw(pen)
                         y_values = [args[0][1] for cmd, args in pen.value if cmd in ('moveTo', 'lineTo') and args]
                         if y_values:
                             if sum(1 for y in y_values if y < 0) / len(y_values) > 0.7:
                                 self._y_axis_inverted = True
                                 log.info(f"Detected inverted Y axis for font '{self.font}' based on metrics.")
             except Exception as e: log.warning(f"Warning: Error detecting font orientation via metrics: {e}")


    # --- Keep _load_fallback_fonts ---
    def _load_fallback_fonts(self):
         # ... (keep as before, uses self._image_font_cls) ...
         self._fallback_fonts = {}
         if not self._image_font_cls: return # Skip if PIL failed
         fallback_font_names = ["Arial", "Times New Roman", "DejaVu Sans", "Liberation Sans"]
         try:
             for font_name in fallback_font_names:
                 # Avoid reloading the primary font as fallback
                 primary_family = self.font.split(":")[0]
                 if font_name != primary_family:
                     try:
                         fallback_font = self._image_font_cls.truetype(font_name, size=10)
                         self._fallback_fonts[font_name] = fallback_font
                     except OSError: pass # Ignore if fallback not found
         except Exception as e: log.warning(f"Warning: Error loading fallback fonts: {e}")


    # --- Keep _get_glyph_outlines_by_name (uses iterative evaluate) ---
    def _get_glyph_outlines_by_name(self, glyph_name):
        """Gets raw outlines in font units for a specific glyph name."""
        if not self._glyph_set or glyph_name not in self._glyph_set:
            log.warning(f"Glyph name '{glyph_name}' not found in glyph set.")
            units_per_em = 1000
            try: units_per_em = self._font['head'].unitsPerEm if self._font else 1000
            except: pass
            fallback_size = units_per_em * 0.6
            return [np.array([[0,0], [fallback_size,0], [fallback_size, fallback_size], [0, fallback_size], [0,0]], dtype=float)]

        pen = self._recording_pen_cls()
        try:
            glyph = self._glyph_set[glyph_name]
            glyph.draw(pen)
        except Exception as e:
            log.error(f"Error drawing glyph '{glyph_name}': {e}", exc_info=True)
            return []

        raw_polygons = []
        current_contour = [] # Store tuples
        for command, args in pen.value:
            try:
                if command == 'moveTo':
                    if current_contour: raw_polygons.append(np.array(current_contour, dtype=float))
                    if args and len(args[0]) == 2: current_contour = [tuple(args[0])]
                    else: current_contour = []
                elif command == 'lineTo':
                     if current_contour and args and len(args[0]) == 2: current_contour.append(tuple(args[0]))
                elif command == 'qCurveTo':
                    if not args or not current_contour: continue
                    start_point_tuple = current_contour[-1]
                    args_tuples = [tuple(p) for p in args if isinstance(p, (list, tuple)) and len(p) == 2]
                    if len(args_tuples) != len(args): continue
                    full_pts_tuples = [start_point_tuple] + args_tuples
                    if len(full_pts_tuples) > 1 and np.allclose(full_pts_tuples[0], full_pts_tuples[1]): full_pts_tuples = [full_pts_tuples[0]] + full_pts_tuples[2:]
                    if len(full_pts_tuples) < 3:
                         if not np.allclose(current_contour[-1], full_pts_tuples[-1]): current_contour.append(full_pts_tuples[-1])
                         continue
                    num_bezier_segments = len(full_pts_tuples) - 2
                    for i in range(num_bezier_segments):
                        p1_tuple = full_pts_tuples[i+1]
                        if i == 0: p0_tuple = full_pts_tuples[0]
                        else: p0_tuple = tuple((np.asarray(full_pts_tuples[i]) + np.asarray(p1_tuple)) / 2.0)
                        if i == num_bezier_segments - 1: p2_tuple = full_pts_tuples[i+2]
                        else: p2_tuple = tuple((np.asarray(p1_tuple) + np.asarray(full_pts_tuples[i+2])) / 2.0)
                        p0_arr, p1_arr, p2_arr = (np.asarray(p0_tuple), np.asarray(p1_tuple), np.asarray(p2_tuple))
                        if p0_arr.shape != (2,) or p1_arr.shape != (2,) or p2_arr.shape != (2,): continue
                        bezier_points = np.array([p0_arr, p1_arr, p2_arr])
                        if np.allclose(p0_arr, p1_arr) and np.allclose(p1_arr, p2_arr):
                             if not np.allclose(current_contour[-1], p2_tuple): current_contour.append(p2_tuple)
                             continue
                        radius = max(np.linalg.norm(p1_arr-p0_arr), np.linalg.norm(p2_arr-p1_arr), EPSILON)
                        num_steps = max(int(get_fragments_from_fn_fa_fs(radius, self.fn, self.fa, self.fs) * 1.5), 8)
                        spline = QuadraticSpline(bezier_points)
                        t_values = np.linspace(0, 1, num_steps + 1)[1:]
                        if len(t_values) > 0:
                             new_points = np.array([spline.evaluate(t) for t in t_values])
                             if new_points.ndim == 2 and new_points.shape[1] == 2: current_contour.extend([tuple(pt) for pt in new_points])
                             elif new_points.shape == (2,): current_contour.append(tuple(new_points))
                        final_pt_tuple = tuple(p2_arr)
                        if not current_contour or not np.allclose(current_contour[-1], final_pt_tuple): current_contour.append(final_pt_tuple)
                elif command == 'curveTo':
                    if not args or len(args) != 3 or not current_contour: continue
                    p0_tuple = current_contour[-1]
                    p1_tuple, p2_tuple, p3_tuple = tuple(args[0]), tuple(args[1]), tuple(args[2])
                    p0_arr, p1_arr, p2_arr, p3_arr = (np.asarray(p0_tuple), np.asarray(p1_tuple), np.asarray(p2_tuple), np.asarray(p3_tuple))
                    if p0_arr.shape != (2,) or p1_arr.shape != (2,) or p2_arr.shape != (2,) or p3_arr.shape != (2,): continue
                    points = np.array([p0_arr, p1_arr, p2_arr, p3_arr])
                    if np.allclose(p0_arr,p1_arr) and np.allclose(p1_arr,p2_arr) and np.allclose(p2_arr,p3_arr):
                         if not np.allclose(current_contour[-1], p3_tuple): current_contour.append(p3_tuple)
                         continue
                    radius = max(np.linalg.norm(p1_arr-p0_arr), np.linalg.norm(p2_arr-p1_arr), np.linalg.norm(p3_arr-p2_arr), EPSILON)
                    num_steps = max(int(get_fragments_from_fn_fa_fs(radius, self.fn, self.fa, self.fs) * 1.5), 12)
                    spline = CubicSpline(points)
                    t_values = np.linspace(0, 1, num_steps + 1)[1:]
                    if len(t_values) > 0:
                         new_points = np.array([spline.evaluate(t) for t in t_values])
                         if new_points.ndim == 2 and new_points.shape[1] == 2: current_contour.extend([tuple(pt) for pt in new_points])
                         elif new_points.shape == (2,): current_contour.append(tuple(new_points))
                    final_pt_tuple = tuple(p3_arr)
                    if not current_contour or not np.allclose(current_contour[-1], final_pt_tuple): current_contour.append(final_pt_tuple)
                elif command == 'closePath':
                    if current_contour:
                        if not np.allclose(current_contour[0], current_contour[-1]):
                            current_contour.append(current_contour[0])
                        raw_polygons.append(np.array(current_contour, dtype=float))
                        current_contour = []
            except Exception as e:
                log.error(f"Error processing pen command {command} {args}: {e}", exc_info=True)
                current_contour = [] # Reset contour on error within loop

        if current_contour: raw_polygons.append(np.array(current_contour, dtype=float))
        if self._y_axis_inverted:
            for i in range(len(raw_polygons)):
                 if isinstance(raw_polygons[i], np.ndarray): raw_polygons[i][:, 1] *= -1
        return raw_polygons

    # --- Keep _apply_alignment ---
    def _apply_alignment(self, polygons, min_coord, max_coord):
        """Apply alignment transformations based on calculated bounding box."""
        # ... (Implementation remains the same as previous version) ...
        if not polygons: return polygons
        min_coord = np.asarray(min_coord); max_coord = np.asarray(max_coord)
        bbox_width = max_coord[0] - min_coord[0]; bbox_height = max_coord[1] - min_coord[1]
        dx, dy = 0.0, 0.0
        if self.halign == "center": dx = -(min_coord[0] + bbox_width / 2.0)
        elif self.halign == "right": dx = -max_coord[0]
        else: dx = -min_coord[0]
        if self.valign == "top": dy = -max_coord[1]
        elif self.valign == "center": dy = -(min_coord[1] + bbox_height / 2.0)
        elif self.valign == "bottom": dy = -min_coord[1]
        if abs(dx) > EPSILON or abs(dy) > EPSILON:
            translation = np.array([dx, dy])
            return [poly + translation for poly in polygons]
        else:
            return polygons

    # --- Keep _apply_text_direction ---
    def _apply_text_direction(self, polygons, min_coord, max_coord):
        """Apply text direction transformations based on actual bounds."""
        # ... (Implementation remains the same as previous version) ...
        if not polygons or self.direction == "ltr": return polygons
        min_coord = np.asarray(min_coord); max_coord = np.asarray(max_coord)
        width = max_coord[0] - min_coord[0]; height = max_coord[1] - min_coord[1]
        directed_polygons = [poly.copy() for poly in polygons]
        if self.direction == "rtl":
            center_x = min_coord[0] + width / 2.0
            for poly in directed_polygons: poly[:, 0] = 2 * center_x - poly[:, 0]; poly[:] = poly[::-1]
        elif self.direction == "ttb":
            center_x = min_coord[0] + width / 2.0; center_y = min_coord[1] + height / 2.0
            for poly in directed_polygons:
                 x_new = center_x + (poly[:, 1] - center_y); y_new = center_y - (poly[:, 0] - center_x)
                 poly[:, 0], poly[:, 1] = x_new, y_new; poly[:] = poly[::-1]
        elif self.direction == "btt":
            center_x = min_coord[0] + width / 2.0; center_y = min_coord[1] + height / 2.0
            for poly in directed_polygons:
                 x_new = center_x - (poly[:, 1] - center_y); y_new = center_y + (poly[:, 0] - center_x)
                 poly[:, 0], poly[:, 1] = x_new, y_new
        return directed_polygons

    # --- Rewrite get_polygons ---
    def get_polygons(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """Generates shaped text polygons using HarfBuzz."""
        if not self.text or self._hb_font is None or self._glyph_set is None:
            log.warning("Text is empty or HarfBuzz/GlyphSet not initialized.")
            return np.empty((0, 2)), []

        # 1. Setup HarfBuzz Buffer
        buf = hb.Buffer()
        buf.add_str(self.text)
        script_map = {"latin": "Latn", "arabic": "Arab", "hebrew": "Hebr", "cyrillic": "Cyrl", "greek": "Grek"}
        hb_script = script_map.get(self.script.lower(), self.script.upper()[:4].ljust(4))
        buf.direction = "rtl" if self.base_direction == "rtl" else "ltr"
        buf.script = hb_script
        buf.language = self.language
        buf.cluster_level = hb.BufferClusterLevel.MONOTONE_CHARACTERS
        log.info(f"Shaping text: '{self.text}' | Direction: {buf.direction} | Script: {buf.script} | Lang: {buf.language}")

        # 2. Shape the text
        try:
            features = {"kern": True, "liga": True}
            hb.shape(self._hb_font, buf, features)
        except Exception as e:
            log.error(f"HarfBuzz shaping failed: {e}", exc_info=True)
            return np.empty((0, 2)), []

        glyph_infos = buf.glyph_infos
        glyph_positions = buf.glyph_positions
        if not glyph_infos:
             log.warning("HarfBuzz returned no glyphs.")
             return np.empty((0, 2)), []

        # 3. Process Shaped Glyphs
        all_polygons_scaled_positioned = []
        pen_pos = np.array([0.0, 0.0])
        log.info(f"Processing {len(glyph_infos)} shaped glyphs...")
        for i, (info, pos) in enumerate(zip(glyph_infos, glyph_positions)):
            gid = info.codepoint; cluster = info.cluster
            dx, dy = pos.x_advance, pos.y_advance; xoff, yoff = pos.x_offset, pos.y_offset
            try: glyph_name = self._font.getGlyphName(gid)
            except Exception: glyph_name = ".notdef"
            log.debug(f"Glyph {i}: GID={gid}, Name='{glyph_name}', Cluster={cluster}, Adv=({dx},{dy}), Off=({xoff},{yoff})")
            raw_contours = self._get_glyph_outlines_by_name(glyph_name)
            if raw_contours:
                draw_pos = pen_pos + np.array([xoff * self._scale_factor, yoff * self._scale_factor])
                for raw_contour in raw_contours:
                     if raw_contour.ndim == 2 and raw_contour.shape[1] == 2:
                          final_contour = (raw_contour * self._scale_factor) + draw_pos
                          all_polygons_scaled_positioned.append(final_contour)
                     else: log.warning(f"Skipping invalid raw contour shape {raw_contour.shape} for glyph '{glyph_name}'")
            advance_scale = self.spacing if self.direction in ("ltr", "rtl") else 1.0
            pen_pos += np.array([dx * self._scale_factor * advance_scale, dy * self._scale_factor * advance_scale])

        # --- Post-Processing ---
        if not all_polygons_scaled_positioned:
             log.warning("No polygons generated after processing glyphs.")
             return np.empty((0, 2)), []
        # Skip orientation check for now, assuming HB + _get_glyph_outlines handles it
        oriented_polygons = all_polygons_scaled_positioned
        try:
             if not oriented_polygons: raise ValueError("No polygons")
             combined_points = np.vstack(oriented_polygons)
             min_coord, max_coord = extentsof(combined_points)
        except ValueError: min_coord, max_coord = np.array([0.,0.]), np.array([0.,0.])
        aligned_polygons = self._apply_alignment(oriented_polygons, min_coord, max_coord)
        directed_polygons = self._apply_text_direction(aligned_polygons, min_coord, max_coord)

        # --- Final Output Format Conversion ---
        final_all_points = []; final_contours = []; point_offset = 0
        for poly in directed_polygons:
             if poly.ndim == 2 and poly.shape[1] == 2 and len(poly) > 0:
                 num_points = len(poly)
                 final_all_points.append(poly)
                 final_contours.append(np.arange(point_offset, point_offset + num_points))
                 point_offset += num_points
             else: log.warning(f"Skipping polygon with invalid shape {getattr(poly, 'shape', 'N/A')} during final conversion.")
        if not final_all_points: return np.empty((0, 2)), []
        log.info(f"Successfully generated {len(final_contours)} contours.")
        return np.vstack(final_all_points), final_contours

    # --- get_polygons_at (Simplified version using filtering) ---
    def get_polygons_at(self, pos: int) -> tuple[np.ndarray, list[np.ndarray]]:
         """
         Returns polygons for the character at logical index `pos`.
         NOTE: Less efficient. Relies on full shaping first. May not work perfectly for complex ligatures spanning multiple clusters.
         """
         if pos < 0 or pos >= len(self.text):
             raise ValueError(f"Position {pos} out of range for text '{self.text}'")
         if self._hb_font is None: return np.empty((0, 2)), []
         buf = hb.Buffer(); buf.add_str(self.text)
         script_map = {"latin": "Latn", "arabic": "Arab", "hebrew": "Hebr"}
         hb_script = script_map.get(self.script.lower(), self.script.upper()[:4].ljust(4))
         buf.direction = "rtl" if self.base_direction == "rtl" else "ltr"
         buf.script = hb_script; buf.language = self.language
         buf.cluster_level = hb.BufferClusterLevel.MONOTONE_CHARACTERS
         try: hb.shape(self._hb_font, buf, {"kern": True, "liga": True})
         except Exception as e: return np.empty((0, 2)), []
         glyph_infos = buf.glyph_infos; glyph_positions = buf.glyph_positions
         if not glyph_infos: return np.empty((0, 2)), []
         target_glyph_indices = [i for i, info in enumerate(glyph_infos) if info.cluster == pos]
         if not target_glyph_indices: return np.empty((0, 2)), []

         # Generate ALL polygons first
         all_polygons_scaled_positioned = []; pen_pos = np.array([0.0, 0.0])
         glyph_to_polygons_map = {}
         for i, (info, glyph_pos) in enumerate(zip(glyph_infos, glyph_positions)):
             gid = info.codepoint; dx, dy = glyph_pos.x_advance, glyph_pos.y_advance; xoff, yoff = glyph_pos.x_offset, glyph_pos.y_offset
             try: glyph_name = self._font.getGlyphName(gid)
             except: glyph_name = ".notdef"
             raw_contours = self._get_glyph_outlines_by_name(glyph_name)
             current_glyph_poly_indices = []
             if raw_contours:
                 draw_pos = pen_pos + np.array([xoff * self._scale_factor, yoff * self._scale_factor])
                 for raw_contour in raw_contours:
                      if raw_contour.ndim == 2 and raw_contour.shape[1] == 2:
                           final_contour = (raw_contour * self._scale_factor) + draw_pos
                           current_glyph_poly_indices.append(len(all_polygons_scaled_positioned))
                           all_polygons_scaled_positioned.append(final_contour)
             glyph_to_polygons_map[i] = current_glyph_poly_indices
             advance_scale = self.spacing if self.direction in ("ltr", "rtl") else 1.0
             pen_pos += np.array([dx * self._scale_factor * advance_scale, dy * self._scale_factor * advance_scale])

         # Filter polygons
         target_polygons = []
         for glyph_idx in target_glyph_indices:
              poly_indices = glyph_to_polygons_map.get(glyph_idx, [])
              for poly_idx in poly_indices:
                   if 0 <= poly_idx < len(all_polygons_scaled_positioned): target_polygons.append(all_polygons_scaled_positioned[poly_idx])
         if not target_polygons: return np.empty((0, 2)), []

         # Post-process target polygons
         try: combined_points = np.vstack(target_polygons); min_coord, max_coord = extentsof(combined_points)
         except ValueError: return np.empty((0, 2)), []
         aligned_polygons = self._apply_alignment(target_polygons, min_coord, max_coord)
         directed_polygons = aligned_polygons # Skip direction transform for single char

         # Final Output Format
         final_all_points = []; final_contours = []; point_offset = 0
         for poly in directed_polygons:
             if poly.ndim == 2 and poly.shape[1] == 2 and len(poly) > 0:
                 num_points = len(poly); final_all_points.append(poly)
                 final_contours.append(np.arange(point_offset, point_offset + num_points)); point_offset += num_points
         if not final_all_points: return np.empty((0, 2)), []
         return np.vstack(final_all_points), final_contours


# --- Keep text() function wrapper ---
def text(text: str, size: float=10, font: str="Liberation Sans", halign: str="left",
         valign: str="baseline", spacing: float=1.0, direction: str="ltr",
         language: str="en", script: str="latin", fa: float=12.0, fs: float=2.0,
         fn: int=0, base_direction: str="ltr") -> tuple[np.ndarray, list[np.ndarray]]:
    try:
        context = TextContext(text=text, size=size, font=font, halign=halign, valign=valign,
                              spacing=spacing, direction=direction, language=language,
                              script=script, fa=fa, fs=fs, fn=fn, base_direction=base_direction)
        return context.get_polygons()
    except (ImportError, ValueError, RuntimeError, TypeError) as e: # Catch potential init errors
        log.error(f"Error creating TextContext or getting polygons: {e}", exc_info=True)
        return np.empty((0, 2)), [] # Return empty on error
    except Exception as e: # Catch unexpected errors
        log.error(f"Unexpected error in text(): {e}", exc_info=True)
        return np.empty((0, 2)), []


# --- Keep font utility functions (get_available_fonts, get_fonts_list) ---
# ... (Assume they are present and correct) ...
def get_available_fonts():
     # ... (implementation from previous code) ...
     from PIL import ImageFont
     import platform
     available_fonts = {}
     font_dirs = []
     system = platform.system()
     # ... (logic to find font_dirs based on OS) ...
     # Windows
     if system == "Windows":
         windir = os.environ.get("WINDIR", "C:\\Windows")
         font_dirs.append(os.path.join(windir, "Fonts"))
     # macOS
     elif system == "Darwin":
         font_dirs.extend(["/Library/Fonts", "/System/Library/Fonts", os.path.expanduser("~/Library/Fonts")])
     # Linux/Unix
     else:
         font_dirs.extend(["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"), os.path.expanduser("~/.local/share/fonts")])

     font_dirs = [d for d in font_dirs if d and os.path.isdir(d)] # Filter valid dirs

     for d in font_dirs:
         try:
             for root, _, files in os.walk(d, followlinks=True): # followlinks might be needed
                 for file in files:
                     if file.lower().endswith(('.ttf', '.otf', '.ttc')): # Include TTC
                         try:
                             font_path = os.path.join(root, file)
                             # Use size=10 as default for probing
                             font = ImageFont.truetype(font_path, size=10)
                             family, style = font.getname()
                             if family not in available_fonts: available_fonts[family] = []
                             if style not in available_fonts[family]: available_fonts[family].append(style)
                         except Exception: continue # Ignore fonts PIL can't read
         except OSError: continue # Ignore dirs we can't access

     # Add known PIL builtins if they weren't found via path
     try:
         default_fonts = ["arial.ttf", "times.ttf", "cour.ttf"] # Basic ones
         for font_name in default_fonts:
             try:
                 font = ImageFont.truetype(font_name, size=10)
                 family, style = font.getname()
                 if family not in available_fonts: available_fonts[family] = []
                 if style not in available_fonts[family]: available_fonts[family].append(style)
             except OSError: pass
     except Exception: pass
     return available_fonts

def get_fonts_list():
     fonts_dict = get_available_fonts()
     fonts_list = []
     for family, styles in fonts_dict.items():
         for style in styles:
             # Prefer 'Regular' as default style name if applicable
             style_name = "Regular" if "regular" in style.lower() and "bold" not in style.lower() and "italic" not in style.lower() else style
             if style_name == "Regular":
                 fonts_list.append(family)
             else:
                 # Attempt to format style nicely
                 style_formatted = style.replace("Italic", " Italic").replace("Bold", " Bold").strip()
                 fonts_list.append(f"{family}:style={style_formatted}")
     # Remove duplicates that might arise from style name normalization
     return sorted(list(set(fonts_list)))


# --- Keep Example Usage (__main__) ---
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # ... (Keep --list-fonts logic) ...
    if len(sys.argv) > 1 and sys.argv[1] == '--list-fonts':
         print("Available Fonts:")
         fonts = get_fonts_list()
         for font in fonts: print(f"  {font}")
         sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == '--list-fonts-by-family':
         print("Available Font Families:")
         fonts = get_available_fonts()
         for family, styles in sorted(fonts.items()):
             print(f"{family}: {', '.join(sorted(styles))}")
         sys.exit(0)

    font_to_use = "Arial" # Default if no arg
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        font_to_use = sys.argv[1]
    log.info(f"Using font: '{font_to_use}'")

    print("Generating text...")
    mixed_text = "Hello! مرحباً שלום" # Mix English, Arabic, Hebrew
    print(f"Testing text: {mixed_text}")

    try:
        # Regular text first
        log.info("Generating LTR example...")
        points1, contours1 = text(
            "Test!", size=25, font=font_to_use, halign="center", valign="center", fn=12)

        # Bidirectional text with LTR base direction
        log.info("Generating Bidi LTR example...")
        points2, contours2 = text(
            mixed_text, size=25, font=font_to_use, halign="center", valign="center",
            fn=24, base_direction="ltr", script="latin") # Script hint might need adjustment

        # Bidirectional text with RTL base direction
        log.info("Generating Bidi RTL example...")
        points3, contours3 = text(
            mixed_text, size=25, font=font_to_use, halign="center", valign="center",
            fn=24, base_direction="rtl", script="arabic") # Script hint might need adjustment

    except Exception as e:
        log.error(f"Error during text generation: {e}", exc_info=True)
        print("\nTry running with --list-fonts to see available fonts.")
        sys.exit(1)

    print(f"Generated {len(contours1)} polygons for 'Test!'")
    print(f"Generated {len(contours2)} polygons for bidi text (LTR base)")
    print(f"Generated {len(contours3)} polygons for bidi text (RTL base)")

    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=False) # Don't share X

    def plot_polygons(ax, title, points, contours, color):
         if points is None or points.shape[0] == 0 or not contours:
             ax.text(0.5, 0.5, "No polygons generated", ha='center', va='center')
             ax.set_title(f"{title} (No polygons)")
             return
         for contour_indices in contours:
             # Ensure indices are valid
             valid_indices = contour_indices[contour_indices < points.shape[0]]
             if len(valid_indices) > 1:
                  contour_points = points[valid_indices]
                  patch = plt.Polygon(contour_points, closed=True, facecolor=color, edgecolor='black', alpha=0.6)
                  ax.add_patch(patch)
             else:
                  log.warning(f"Skipping contour with invalid indices: {contour_indices}")

         ax.set_aspect('equal', adjustable='box')
         min_coord, max_coord = extentsof(points)
         ax.set_xlim(min_coord[0] - 5, max_coord[0] + 5)
         ax.set_ylim(min_coord[1] - 5, max_coord[1] + 5)
         ax.set_title(title)
         ax.grid(True)

    plot_polygons(axes[0], f"Regular Text: 'Test!'", points1, contours1, 'blue')
    plot_polygons(axes[1], f"Bidirectional Text (LTR base): '{mixed_text}'", points2, contours2, 'green')
    plot_polygons(axes[2], f"Bidirectional Text (RTL base): '{mixed_text}'", points3, contours3, 'red')

    plt.tight_layout()
    plt.savefig("text_render_bidi_test_hb.png", dpi=150)
    print("Figure saved to 'text_render_bidi_test_hb.png'")
    try: plt.show()
    except Exception: print("Could not display plot - but image was saved to file")

