from datatrees import datatree, dtfield, Node
from typing import ClassVar, List, Tuple
import numpy as np

import OpenGL.GL as gl
import OpenGL.GLU as glu

from HersheyFonts import HersheyFonts

from pythonopenscad.viewer.viewer_base import ViewerBase
from pythonopenscad.viewer.glctxt import PYOPENGL_VERBOSE

import anchorscad_lib.linear as l


@datatree
class ScreenContext:
    """Context for the screen."""
    
    ws_width_per_px: float
    scene_diagonal_px: float
    
    def __post_init__(self):
        self.ws_width_per_px = float(self.ws_width_per_px)
        self.scene_diagonal_px = float(self.scene_diagonal_px)
    
    def get_scene_diagonal_ws(self):
        return self.scene_diagonal_px * self.ws_width_per_px

@datatree
class AxesRenderer:
    """Renders X, Y, Z axes using immediate mode."""

    show_graduation_ticks: bool = True
    show_graduation_values: bool = True
    factor: float = 1  # Length of axes relative to scene diagonal
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Color for the axes lines
    line_width_px: float = 1.5  # Line width for the axes
    grad_line_width_px: float = 1.3  # Line width for the graduations
    stipple_factor: int = 4  # Scaling factor for the dash pattern negtive of axes.
    negative_stipple_pattern: int = 0xAAAA  # For negative side of axes.
    
    grad_tick_size_px: list[float] = (10, 20, 25) # Size of the graduations in pixels
    grad_size_px_min: float = 9 # Min space between graduations
    grad_colors: tuple[tuple[float, float, float], 
                       tuple[float, float, float], 
                       tuple[float, float, float]] = ((1, 0, 0), (0, 0.5, 0), (0, 0, 1))
    
    font_name: str = None # None means pick the default font.
    font_size_px: float = 30
    text_margin_px: float = 10
    
    _font: HersheyFonts = dtfield(default_factory=HersheyFonts)
    
    def __post_init__(self):
        if self.show_graduation_values:
            self._font.load_default_font(self.font_name)

    AXES: ClassVar[List[np.ndarray]] = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    
    GRAD_DIR: ClassVar[List[np.ndarray]] = [
        np.array([0.0, 1.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
    ]
    
    def get_ws_width_per_px(self, viewer: ViewerBase):
        # Get how wide one pixel is in world space.

        if hasattr(viewer, 'custom_glu_project'): # Indicates PoscGLWidget or similar
            modelview_mat = viewer.get_view_mat()
            projection_mat = viewer.get_projection_mat()
            viewport = viewer.get_viewport()

            try:
                origin_sx, origin_sy, origin_sz = viewer.custom_glu_project(0.0, 0.0, 0.0, modelview_mat, projection_mat, viewport)
                world_at_origin_screen = np.array(viewer.custom_glu_unproject(origin_sx, origin_sy, origin_sz, modelview_mat, projection_mat, viewport))
                world_at_origin_plus_1px_x_screen = np.array(viewer.custom_glu_unproject(origin_sx + 1.0, origin_sy, origin_sz, modelview_mat, projection_mat, viewport))
                
                delta_world = world_at_origin_plus_1px_x_screen - world_at_origin_screen
                model_1px_len = np.sqrt(np.dot(delta_world, delta_world))

                if model_1px_len == 0.0: return 0.001
                return model_1px_len
            except ValueError as ve:
                if PYOPENGL_VERBOSE: print(f"AxesRenderer.get_ws_width_per_px (custom path): ValueError: {ve}")
                return 0.01
            except Exception as e:
                if PYOPENGL_VERBOSE: print(f"AxesRenderer.get_ws_width_per_px (custom path): Exception: {e}")
                return 0.01
        else: # Fallback for original Viewer (GLUT-based)
            try:
                origin_screen_coords = np.array(glu.gluProject(0.0, 0.0, 0.0))
                world_at_origin_screen = np.array(glu.gluUnProject(origin_screen_coords[0], origin_screen_coords[1], origin_screen_coords[2]))
                world_at_origin_plus_1px_x_screen = np.array(glu.gluUnProject(origin_screen_coords[0] + 1.0, origin_screen_coords[1], origin_screen_coords[2]))
                
                delta_world = world_at_origin_plus_1px_x_screen - world_at_origin_screen
                model_1px_len = np.sqrt(np.dot(delta_world, delta_world))

                if model_1px_len == 0.0: return 0.001
                return model_1px_len
            except ValueError as ve:
                if PYOPENGL_VERBOSE: print(f"AxesRenderer.get_ws_width_per_px (GLU path): ValueError: {ve}")
                return 0.01
            except Exception as e:
                if PYOPENGL_VERBOSE: print(f"AxesRenderer.get_ws_width_per_px (GLU path): Exception: {e}")
                return 0.01

    
    def get_scene_diagonal_px(self, viewer: ViewerBase):
        win_xy = np.asarray(viewer.get_current_window_dims())
        return np.sqrt(np.dot(win_xy, win_xy))
    
    def compute_screen_context(self, viewer: ViewerBase):
        """Compute the screen context."""
        return ScreenContext(
            ws_width_per_px=self.get_ws_width_per_px(viewer),
            scene_diagonal_px=self.get_scene_diagonal_px(viewer),
        )

    def draw(self, viewer: ViewerBase):
        """Draw the axes lines."""
        screen_context = self.compute_screen_context(viewer)
        self.draw_axes(viewer, screen_context)
        if self.show_graduation_ticks: 
            self.draw_graduations(viewer, screen_context)
    
    def draw_text(self, text: str, max_allowed_width: float, transform: np.ndarray):
        try:
            if not text or not self.show_graduation_values:
                return
            
            lines_list = np.array(list(self._font.lines_for_text(text)))
            if len(lines_list) == 0:
                return
                        # This should be an nx2x2 array.
            # Get the min max of all the points.
            min_point = np.min(lines_list, axis=(0, 1))
            max_point = np.max(lines_list, axis=(0, 1))
            
            width_with_minus = (min_point[0] + max_point[0])
            width = width_with_minus
            if width > max_allowed_width:
                return
            offset = 0
            
            # Negative numbers should be centrerd on the digits.
            if text[0] == '-':
                min_point_m = np.min(lines_list[1:], axis=(0, 1))
                max_point_m = np.max(lines_list[1:], axis=(0, 1))
                width = (min_point_m[0] + max_point_m[0])
                offset = width_with_minus - width
            
            gl.glPushMatrix()
            gl.glMultMatrixf(transform)
            gl.glTranslate(-width_with_minus / 2 + offset / 2, -max_point[1], 0)
            
            gl.glBegin(gl.GL_LINES)
            
            for a, b in lines_list:
                gl.glVertex2f(*a)
                gl.glVertex2f(*b)
            gl.glEnd()
            
            gl.glPopMatrix()
        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Failed to render text {e}")
        
    def format_for_scale(self, pos: float) -> str:
        """Format a floating point number to display on the axis.

        Rules:
            * If the number approx a whole number it should be printed without
                the decimal point or trailing zeros.
            * We don't render really small numbers (abs value < 0.01), except for 0 itself.
            * We limit to 3 significant digits.

        """
        if np.abs(pos) < 0.01:
            return ""

        # Rule 1: Check if the number is approximately a whole number.
        # Use np.isclose for robust floating-point comparison.
        # Using default tolerances of np.isclose should be fine here.
        rounded_pos = round(pos)
        if np.isclose(pos, rounded_pos, atol=0.01):
            # Format as an integer string
            return str(int(rounded_pos))

        # Rule 3: Limit to 3 significant digits for other numbers.
        # Use the 'g' format specifier for significant digits.
        return "{:.3g}".format(pos)
        
    def max_allowed_text_width_in_grad_multiples(self, ngrads_from_origin) -> float:
        if ngrads_from_origin % 100 == 0:
            return 80
        if ngrads_from_origin % 10 == 0:
            return 9
        if ngrads_from_origin % 5 == 0:
            return 5
        return 0.9
        
    def draw_graduations(self, viewer: ViewerBase, screen_context: ScreenContext):
        # Calculate minimum world-space distance required between ticks
        # to maintain grad_size_px_min pixel separation on screen.
        min_size_ws = screen_context.ws_width_per_px * self.grad_size_px_min
        
        text_margin_ws = self.text_margin_px * screen_context.ws_width_per_px
        
        # Set the font size to the appropriate width.
        if self.show_graduation_values:
            self._font.normalize_rendering(
                self.font_size_px * screen_context.ws_width_per_px)
        
        # Calculate corresponding world-space tick heights for constant pixel size.
        tick_heights_ws = [
            screen_context.ws_width_per_px * grad_size for grad_size in self.grad_tick_size_px]

        # Find the exponents needed for base-10 and base-5 spacing candidates.
        min_ws_exp_base10 = float(np.ceil(np.log10(min_size_ws)))
        min_ws_exp_base5 = float(np.ceil(np.log10(2 * min_size_ws))) # log10(2*ws) relates to 5*10^M

        # Calculate the smallest power-of-10 spacing >= min_size_ws.
        candidate_spacing_10 = 10.0 ** min_ws_exp_base10
        # Calculate the smallest 5*(power-of-10) spacing >= min_size_ws.
        candidate_spacing_5 = 10.0 ** min_ws_exp_base5 / 2

        # *** Select the actual world-space spacing between graduations ***
        if candidate_spacing_10 <= candidate_spacing_5:
            grad_spacing_ws = candidate_spacing_10
            steps_per_major_grad = 10 # Use 10 steps per major interval (e.g., 0 to 10)
            mid_step_index = 5        # Intermediate tick at step 5
        else:
            grad_spacing_ws = candidate_spacing_5
            steps_per_major_grad = 2 # Use 2 steps per major interval (e.g., 0 to 10, spaced by 5)
            mid_step_index = None    # No intermediate tick needed

        # Determine the dynamic world-space extent based on screen diagonal and factor.
        dynamic_axis_extent_ws = screen_context.get_scene_diagonal_ws() * self.factor / 2

        # Calculate how many intervals fit in half the extent.
        num_intervals_half_axis = (int(dynamic_axis_extent_ws // grad_spacing_ws) // 2)
        curr_pos = 0 #-(num_intervals_half_axis * grad_spacing_ws)

        gl.glLineWidth(self.grad_line_width_px)
        # Loop through calculated number of intervals for both negative and positive sides.
        
        for axis_idx in range(3):
            try:
                gl.glPushMatrix()
                if axis_idx == 1:
                    gl.glRotate(90, 0, 0, 1)
                if axis_idx == 2:
                    gl.glRotate(-90, 0, 1, 0)
                    gl.glRotate(90, 1, 0, 0)
                    
                gl.glColor3f(*self.grad_colors[axis_idx])
                for interval_index in range(-num_intervals_half_axis, num_intervals_half_axis + 1): # Iterate interval count
                    curr_pos = grad_spacing_ws * interval_index
                    max_size_n = self.max_allowed_text_width_in_grad_multiples(abs(interval_index))
                    
                    try:
                        gl.glPushMatrix()
                        gl.glTranslatef(curr_pos, 0, 0)
                        text = self.format_for_scale(curr_pos)
                        # Don't render if the string is wider than max_size_n grads. 
                        self.draw_text(
                            text, 
                            max_size_n * grad_spacing_ws,
                            l.translate((0, -text_margin_ws, 0)).A.T)
                        
                        # Select tick height based on position relative to major/mid steps
                        if interval_index % steps_per_major_grad == 0: # Major tick
                            current_tick_height_ws = tick_heights_ws[2]
                        elif mid_step_index is not None and interval_index % mid_step_index == 0: # Mid tick
                            current_tick_height_ws = tick_heights_ws[1]
                        else: # Minor tick
                            current_tick_height_ws = tick_heights_ws[0]

                        gl.glBegin(gl.GL_LINES)
                        gl.glVertex3f(0, 0, 0)
                        gl.glVertex3f(0, current_tick_height_ws, 0)
                        gl.glEnd()
                    finally:
                        gl.glPopMatrix()
            finally:
                gl.glPopMatrix()

    def draw_axes(self, viewer: ViewerBase, screen_context: ScreenContext):
        try:
            length = screen_context.get_scene_diagonal_ws() * self.factor / 2
            
            if length <= 0:
                length = 1.0

            lighting_was_enabled = gl.glIsEnabled(gl.GL_LIGHTING)
            if lighting_was_enabled:
                gl.glDisable(gl.GL_LIGHTING)

            gl.glLineWidth(self.line_width_px)
            gl.glColor3f(*self.color)

            self.draw_half_axes(length)
            try:
                gl.glEnable(gl.GL_LINE_STIPPLE)
                gl.glLineStipple(self.stipple_factor, self.negative_stipple_pattern)
                self.draw_half_axes(-length)
            finally:
                gl.glDisable(gl.GL_LINE_STIPPLE)

            if lighting_was_enabled:
                gl.glEnable(gl.GL_LIGHTING)

        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"AxesRenderer: Error drawing axes: {e}")

    def draw_half_axes(self, length: float):
        gl.glBegin(gl.GL_LINES)
        try:
            for i, axis_vec in enumerate(self.AXES):
                p0 = (0.0, 0.0, 0.0)
                p1 = tuple(length * axis_vec)
                gl.glVertex3f(*p0)
                gl.glVertex3f(*p1)
        finally:
            gl.glEnd()
