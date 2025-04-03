import sys
from datatrees import datatree, dtfield, Node
from typing import ClassVar, List, Tuple
import numpy as np

import OpenGL.GL as gl


@datatree
class AxesRenderer:
    """Renders X, Y, Z axes using immediate mode."""

    factor: float = 5  # Length of axes relative to scene diagonal
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Color for the axes lines
    line_width_px: float = 1.0  # Line width for the axes
    dash_ratio: float = 0.6  # Ratio of dash length to total dash+space (for negative axes)
    stipple_factor: int = 4  # Scaling factor for the dash pattern
    negative_stipple_pattern: int = 0xAAAA

    AXES: ClassVar[List[np.ndarray]] = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    def draw(self, viewer: "Viewer"):
        """Draw the axes lines."""

        self.draw_axes(viewer)

    def draw_axes(self, viewer: "Viewer"):
        try:
            win_xy = np.asarray(viewer.get_current_window_dims())

            scene_diagonal = np.sqrt(np.dot(win_xy, win_xy))

            # Calculate axis length
            length = scene_diagonal * self.factor
            if length <= 0:
                length = 1.0  # Default length if diagonal is zero or negative

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
