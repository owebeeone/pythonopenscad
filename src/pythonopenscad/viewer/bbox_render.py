import sys
from typing import Any

from pythonopenscad.viewer.glctxt import PYOPENGL_VERBOSE

import OpenGL.GL as gl

class BBoxRender:
    @classmethod
    def render(cls, viewer: Any):
        """Draw the scene bounding box in the current mode (off/wireframe/solid)."""
        if viewer.bounding_box_mode == 0:
            return

        # Store current backface culling state and disable it for the bounding box
        was_culling_enabled = gl.glIsEnabled(gl.GL_CULL_FACE)
        if was_culling_enabled:
            gl.glDisable(gl.GL_CULL_FACE)

        # Get bounding box coordinates
        min_x, min_y, min_z = viewer.bounding_box.min_point
        max_x, max_y, max_z = viewer.bounding_box.max_point

        # Set up transparency for solid mode
        was_blend_enabled = False
        if viewer.bounding_box_mode == 2:
            try:
                # Save current blend state
                was_blend_enabled = gl.glIsEnabled(gl.GL_BLEND)

                # Enable blending
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

                # Semi-transparent green
                gl.glColor4f(0.0, 1.0, 0.0, 0.2)
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Blending setup failed: {e}")
                # Fall back to wireframe
                viewer.bounding_box_mode = 1
                if PYOPENGL_VERBOSE:
                    print("Viewer: Blending not supported, falling back to wireframe mode")

        # Draw the bounding box
        if viewer.bounding_box_mode == 1:  # Wireframe mode
            gl.glColor3f(0.0, 1.0, 0.0)  # Green

            # Front face
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex3f(min_x, min_y, max_z)
            gl.glVertex3f(max_x, min_y, max_z)
            gl.glVertex3f(max_x, max_y, max_z)
            gl.glVertex3f(min_x, max_y, max_z)
            gl.glEnd()

            # Back face
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex3f(min_x, min_y, min_z)
            gl.glVertex3f(max_x, min_y, min_z)
            gl.glVertex3f(max_x, max_y, min_z)
            gl.glVertex3f(min_x, max_y, min_z)
            gl.glEnd()

            # Connecting edges
            gl.glBegin(gl.GL_LINES)
            gl.glVertex3f(min_x, min_y, min_z)
            gl.glVertex3f(min_x, min_y, max_z)

            gl.glVertex3f(max_x, min_y, min_z)
            gl.glVertex3f(max_x, min_y, max_z)

            gl.glVertex3f(max_x, max_y, min_z)
            gl.glVertex3f(max_x, max_y, max_z)

            gl.glVertex3f(min_x, max_y, min_z)
            gl.glVertex3f(min_x, max_y, max_z)
            gl.glEnd()

        else:  # Solid mode
            # Front face
            gl.glBegin(gl.GL_QUADS)
            gl.glVertex3f(min_x, min_y, max_z)
            gl.glVertex3f(max_x, min_y, max_z)
            gl.glVertex3f(max_x, max_y, max_z)
            gl.glVertex3f(min_x, max_y, max_z)
            gl.glEnd()

            # Back face
            gl.glBegin(gl.GL_QUADS)
            gl.glVertex3f(min_x, min_y, min_z)
            gl.glVertex3f(max_x, min_y, min_z)
            gl.glVertex3f(max_x, max_y, min_z)
            gl.glVertex3f(min_x, max_y, min_z)
            gl.glEnd()

            # Top face
            gl.glBegin(gl.GL_QUADS)
            gl.glVertex3f(min_x, max_y, min_z)
            gl.glVertex3f(max_x, max_y, min_z)
            gl.glVertex3f(max_x, max_y, max_z)
            gl.glVertex3f(min_x, max_y, max_z)
            gl.glEnd()

            # Bottom face
            gl.glBegin(gl.GL_QUADS)
            gl.glVertex3f(min_x, min_y, min_z)
            gl.glVertex3f(max_x, min_y, min_z)
            gl.glVertex3f(max_x, min_y, max_z)
            gl.glVertex3f(min_x, min_y, max_z)
            gl.glEnd()

            # Right face
            gl.glBegin(gl.GL_QUADS)
            gl.glVertex3f(max_x, min_y, min_z)
            gl.glVertex3f(max_x, max_y, min_z)
            gl.glVertex3f(max_x, max_y, max_z)
            gl.glVertex3f(max_x, min_y, max_z)
            gl.glEnd()

            # Left face
            gl.glBegin(gl.GL_QUADS)
            gl.glVertex3f(min_x, min_y, min_z)
            gl.glVertex3f(min_x, max_y, min_z)
            gl.glVertex3f(min_x, max_y, max_z)
            gl.glVertex3f(min_x, min_y, max_z)
            gl.glEnd()

        # Clean up blending state
        if viewer.bounding_box_mode == 2 and not was_blend_enabled:
            try:
                gl.glDisable(gl.GL_BLEND)
            except Exception:
                pass

        # Restore backface culling state
        if was_culling_enabled:
            gl.glEnable(gl.GL_CULL_FACE)
