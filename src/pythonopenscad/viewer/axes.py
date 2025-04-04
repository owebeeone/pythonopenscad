import sys

import glm
from datatrees import datatree, dtfield, Node
from typing import ClassVar, List, Tuple
import numpy as np

import OpenGL.GL as gl
import OpenGL.GLU as glu

from pythonopenscad.viewer.viewer_base import ViewerBase
from pythonopenscad.viewer.glctxt import PYOPENGL_VERBOSE


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

    factor: float = 3  # Length of axes relative to scene diagonal
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Color for the axes lines
    line_width_px: float = 1.5  # Line width for the axes
    grad_line_width_px: float = 1.3  # Line width for the graduations
    dash_ratio: float = 0.6  # Ratio of dash length to total dash+space (for negative axes)
    stipple_factor: int = 4  # Scaling factor for the dash pattern
    negative_stipple_pattern: int = 0xAAAA
    
    grad_tick_size_px: list[float] = (10, 20, 40) # Size of the graduations in pixels
    grad_size_px_min: float = 5 # Min space between graduations
    grad_colors: tuple[tuple[float, float, float], 
                       tuple[float, float, float], 
                       tuple[float, float, float]] = ((1, 0, 0), (0, 0.5, 0), (0, 0, 1))

    AXES: ClassVar[List[np.ndarray]] = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    
    GRAD_DIR: ClassVar[List[np.ndarray]] = [
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]

    
    def get_ws_width_per_px(self, viewer: ViewerBase):
        # Get how wide one pixel is in world space.
        origin = np.array(glu.gluProject(0.0, 0.0, 0.0))
        x_first_grad = origin + self.AXES[0]
        
        model_origin_plus_1x = np.array(glu.gluUnProject(*x_first_grad))
        # Mapping back origin and x_first_grad to model space because
        # there might be some floating point precision issues.
        model_origin = np.array(glu.gluUnProject(*origin))
        
        model_1x = model_origin_plus_1x - model_origin
        model_1x_len = np.sqrt(np.dot(model_1x, model_1x))
        
        return model_1x_len 

    
    def get_scene_diagonal_px(self, viewer: ViewerBase):
        # Get the diagonal of the scene in pixels.
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
        
        self.draw_graduations(viewer, screen_context)
        
        
    def draw_graduations(self, viewer: ViewerBase, screen_context: ScreenContext):
        global PYOPENGL_VERBOSE
        PYOPENGL_VERBOSE = True
        min_size_ws = screen_context.ws_width_per_px * self.grad_size_px_min
        grad_heights = [screen_context.ws_width_per_px * grad_size for grad_size in self.grad_tick_size_px]
        
        min_size_ws_tens = float(np.ceil(np.log10(min_size_ws)))
        min_size_ws_fives = float(np.ceil(np.log10(2 * min_size_ws)))
        

        actual_min_size_tens_ws = 10.0 ** min_size_ws_tens
        actual_min_size_fives_ws = 10.0 ** min_size_ws_fives / 2
        print(f"{actual_min_size_tens_ws = }")
        print(f"{actual_min_size_fives_ws = }")
        
        if actual_min_size_tens_ws >= actual_min_size_fives_ws:
            per_grad = actual_min_size_tens_ws
            num_grads = 10
            mid_grad = 5
            print("Picked tens")
        else:
            per_grad = actual_min_size_fives_ws
            num_grads = 2
            mid_grad = None
            print("Picked fives")
        length = screen_context.get_scene_diagonal_ws() * self.factor / 2
        
        total_grads = (int(length // per_grad) // 10) * 10
        curr_pos = -(total_grads * per_grad)
        gl.glLineWidth(self.grad_line_width_px)
        for j in range(total_grads * 2):
            gl.glBegin(gl.GL_LINES)
            for i, axis, grad_dir in zip((0, 1, 2), self.AXES, self.GRAD_DIR):
                start_pos = curr_pos * axis
                if j % num_grads == 0:
                    tick_size = grad_heights[2]
                elif mid_grad is not None and j % mid_grad == 0:
                    tick_size = grad_heights[1]
                else:
                    tick_size = grad_heights[0]
                end_pos = start_pos + grad_dir * tick_size
                
                gl.glColor3f(*self.grad_colors[i])
                gl.glVertex3f(*start_pos)
                gl.glVertex3f(*end_pos)
            gl.glEnd()
            curr_pos += per_grad
        
        

    def draw_axes(self, viewer: ViewerBase, screen_context: ScreenContext):
        try:
            # Calculate axis length
            length = screen_context.get_scene_diagonal_ws() * self.factor / 2
            
            if length <= 0:
                length = 1.0  # Default length if diagonal is zero or negative
                print("Warning: Scene diagonal is zero or negative, using default length of 1.0")
            else:
                print(f"{length = }")

            # Store current OpenGL state
            lighting_enabled = gl.glIsEnabled(gl.GL_LIGHTING)
            current_line_width = gl.glGetFloatv(gl.GL_LINE_WIDTH)

            # Setup state for axes drawing
            if lighting_enabled:
                gl.glDisable(gl.GL_LIGHTING)
            gl.glLineWidth(self.line_width_px)

            # Set the color for all axes
            gl.glColor3f(*self.color)

            # --- Draw Positive Axes (Solid) ---
            self.draw_half_axes(length)
            # --- Draw Negative Axes (Dashed) ---
            try:
                gl.glEnable(gl.GL_LINE_STIPPLE)
                gl.glLineStipple(self.stipple_factor, self.negative_stipple_pattern)
                self.draw_half_axes(-length)
            finally:
                # Ensure stipple is disabled even if errors occur
                gl.glDisable(gl.GL_LINE_STIPPLE)

            # Restore previous OpenGL state
            gl.glLineWidth(current_line_width)
            # Restore depth test if it was disabled (it wasn't)
            if lighting_enabled:
                gl.glEnable(gl.GL_LIGHTING)
        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"AxesRenderer: Error drawing axes: {e}")

    def draw_half_axes(self, length: float):
        gl.glBegin(gl.GL_LINES)  # Start new block for dashed lines
        try:
            for axis in self.AXES:
                gl.glVertex3f(0.0, 0.0, 0.0)
                gl.glVertex3f(*(length * axis))
        finally:
            gl.glEnd()  # End dashed lines block
