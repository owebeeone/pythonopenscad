import sys
import numpy as np
import manifold3d as m3d
from datatrees import datatree, dtfield, Node
from typing import Any, Iterable, List, Optional, Tuple, Union, Dict, Callable, ClassVar
import warnings

from pythonopenscad.viewer.bbox import BoundingBox
from pythonopenscad.viewer.glctxt import GLContext, PYOPENGL_VERBOSE

import ctypes

import OpenGL.GL as gl
import OpenGL.GLUT as glut


@datatree
class Model:
    """3D model with vertex data including positions, colors, and normals."""

    data: np.ndarray
    has_alpha_lt1: bool | None = None
    num_points: int | None = None
    position_offset: int = 0
    color_offset: int = 3
    normal_offset: int = 7
    stride: int = 10

    def __post_init__(self):
        """
        Initialize a 3D model from vertex data.

        Args:
            data: Numpy array with interleaved vertex data
            num_points: Number of vertices (computed from data if not provided)
            position_offset: Offset of position data in the vertex structure
            color_offset: Offset of color data in the vertex structure
            normal_offset: Offset of normal data in the vertex structure
            stride: Total stride of the vertex structure
        """
        self.data = self.data.astype(np.float32)
        if self.num_points is None:
            self.num_points = len(self.data) // self.stride

        self.gl_ctx = GLContext.get_instance()

        if self.has_alpha_lt1 is None:
            # Scan the data for alpha values less than 1, the sata array is single dimensional
            # containing all the vertex data.
            self.has_alpha_lt1 = np.any(self.data[self.color_offset + 3 :: self.stride] < 1.0)

        # OpenGL objects - Initialized later in initialize_gl_resources
        self.vao = None
        self.vbo = None

        # Compute bounding box
        self._compute_bounding_box()

    @staticmethod
    def from_manifold(manifold: m3d.Manifold, has_alpha_lt1: bool | None = None) -> "Model":
        """Convert a manifold3d Manifold to a viewer Model."""

        # Get the mesh from the manifold
        mesh = manifold.to_mesh()

        # Extract vertex positions and triangle indices
        positions = mesh.vert_properties
        triangles = mesh.tri_verts

        if len(triangles) > 0 and manifold.num_prop() != 7:
            raise ValueError(
                "Manifold must have exactly 7 values in its property array: "
                f"{manifold.num_prop()} values found"
            )

        tri_indices = triangles.reshape(-1)

        # Flatten triangles and use to index positions
        vertex_data = positions[tri_indices]

        # Flatten the vertex data to 1D
        flattened_vertex_data = vertex_data.reshape(-1)

        # Create a model from the vertex data
        return Model(flattened_vertex_data, has_alpha_lt1=has_alpha_lt1)

    def num_triangles(self) -> int:
        return len(self.data) // (3 * self.stride)

    def initialize_gl_resources(self):
        """Initialize OpenGL resources (VBO and VAO) for this model."""
        # Early return if resources already exist
        if self.vbo is not None or self.vao is not None:
            return

        gl_ctx = GLContext.get_instance()
        
        # Check if VBO support is available
        if not hasattr(gl_ctx, 'has_vbo') or not gl_ctx.has_vbo:
            return

        try:
            # Create VBO
            self.vbo = gl.glGenBuffers(1)
            if isinstance(self.vbo, np.ndarray):
                self.vbo = int(self.vbo[0])

            # Check for errors after VBO creation
            error = gl.glGetError()
            if error != gl.GL_NO_ERROR or not self.vbo or self.vbo <= 0:
                self.vbo = None
                return

            # Bind VBO and upload data
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
            error = gl.glGetError()
            if error != gl.GL_NO_ERROR:
                gl.glDeleteBuffers(1, [self.vbo])
                self.vbo = None
                return

            gl.glBufferData(gl.GL_ARRAY_BUFFER, self.data.nbytes, self.data, gl.GL_STATIC_DRAW)
            error = gl.glGetError()
            if error != gl.GL_NO_ERROR:
                gl.glDeleteBuffers(1, [self.vbo])
                self.vbo = None
                return

            # Try to create VAO if supported
            if gl_ctx.has_3_3:
                try:
                    self.vao = gl.glGenVertexArrays(1)
                    if isinstance(self.vao, np.ndarray):
                        self.vao = int(self.vao[0])

                    error = gl.glGetError()
                    if error != gl.GL_NO_ERROR or not self.vao or self.vao <= 0:
                        self.vao = None
                    else:
                        # Set up VAO
                        gl.glBindVertexArray(self.vao)
                        error = gl.glGetError()
                        if error != gl.GL_NO_ERROR:
                            gl.glDeleteVertexArrays(1, [self.vao])
                            self.vao = None
                        else:
                            # Bind VBO to VAO and set up attributes
                            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
                            error = gl.glGetError()
                            if error != gl.GL_NO_ERROR:
                                gl.glDeleteVertexArrays(1, [self.vao])
                                self.vao = None
                            else:
                                # Set up vertex attributes
                                self._setup_vao_attributes()
                                
                                # Unbind VAO
                                gl.glBindVertexArray(0)
                                error = gl.glGetError()
                                if error != gl.GL_NO_ERROR and PYOPENGL_VERBOSE:
                                    print(f"Model: Warning - Error unbinding VAO: {error}")

                except Exception as e:
                    if PYOPENGL_VERBOSE:
                        print(f"Model: VAO creation failed: {e}")
                    # Clean up VAO if it was partially created
                    if self.vao:
                        try:
                            gl.glDeleteVertexArrays(1, [self.vao])
                        except Exception:
                            pass
                        self.vao = None

            # Unbind VBO
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Model: GL resource initialization failed: {e}")
            # Clean up on error
            if self.vao:
                try:
                    gl.glDeleteVertexArrays(1, [self.vao])
                except Exception:
                    pass
                self.vao = None
            if self.vbo:
                try:
                    gl.glDeleteBuffers(1, [self.vbo])
                except Exception:
                    pass
                self.vbo = None

    def _setup_vao_attributes(self):
        """Set up vertex attribute pointers for the VAO."""
        # Position attribute - always in location 0
        gl.glVertexAttribPointer(
            0,
            3,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            self.stride * 4,
            ctypes.c_void_p(self.position_offset * 4),
        )
        gl.glEnableVertexAttribArray(0)

        # Color attribute - always in location 1
        gl.glVertexAttribPointer(
            1,
            4,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            self.stride * 4,
            ctypes.c_void_p(self.color_offset * 4),
        )
        gl.glEnableVertexAttribArray(1)

        # Normal attribute - always in location 2
        gl.glVertexAttribPointer(
            2,
            3,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            self.stride * 4,
            ctypes.c_void_p(self.normal_offset * 4),
        )
        gl.glEnableVertexAttribArray(2)

    def column_data_generator(self, column_index_start: int, column_index_end: int):
        """Generator that yields slices of the data array without copying."""
        for i in range(0, len(self.data), self.stride):
            yield self.data[i + column_index_start : i + column_index_end]

    @staticmethod
    def min_max(arr: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Compute min and max values in a single pass through the data.

        Args:
            arr: Iterable of numpy arrays to compute min/max for

        Returns:
            Tuple of (min_values, max_values) as numpy arrays
        """
        # Get first value without copying
        first = next(arr)
        min_val = first.copy()
        max_val = first.copy()

        # Update min/max in a single pass
        for v in arr:
            np.minimum(min_val, v, out=min_val)
            np.maximum(max_val, v, out=max_val)

        return min_val, max_val

    def _compute_bounding_box(self):
        """Compute the bounding box of the model."""
        self.bounding_box = BoundingBox()

        if self.data is None or len(self.data) == 0 or self.num_points <= 0:
            # Empty model, leave the bounding box with its default values
            return

        # Extract positions from the interleaved data
        try:
            # Get position data using generator to avoid copying
            positions = self.column_data_generator(self.position_offset, self.position_offset + 3)
            self.bounding_box.min_point, self.bounding_box.max_point = self.min_max(positions)

        except Exception as e:
            warnings.warn(f"Failed to compute bounding box: {e}")
            # Use default bounding box (centered at origin with unit size)
            self.bounding_box.min_point = np.array([-0.5, -0.5, -0.5])
            self.bounding_box.max_point = np.array([0.5, 0.5, 0.5])

    def draw(self, use_shaders: bool = True):
        """Draw the model using the best available rendering method."""
        if self.num_points == 0:
            return True

        if use_shaders:
            # Try three rendering tiers in order of preference
            # TIER 1: Modern VAO + Shaders (highest preference)
            if self.gl_ctx.has_shader:
                current_program = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
                if current_program and current_program > 0:
                    if self._draw_with_shader(current_program):
                        return True

            # TIER 2: Vertex Arrays + VBOs (middle fallback)
            if self.gl_ctx.has_legacy_vertex_arrays and self.vbo:
                if self._draw_with_vertex_arrays():
                    return True

        # TIER 3: Immediate Mode (final fallback)
        if self._draw_immediate_mode():
            return True
        else:
            # Last resort: wireframe fallback
            return self._draw_fallback_wireframe()

    def _draw_with_shader(self, active_program: int) -> bool:
        """Draw the model using the provided shader program.

        Returns:
            bool: True if drawing succeeded, False otherwise.
        """
        # Check if we have either VAO or VBO (for non-VAO fallback)
        if (not self.vao and not self.vbo) or self.num_points == 0:
            return False

        try:
            is_program = gl.glIsProgram(active_program)
        except Exception:
            return False

        if not is_program:
            return False

        # Setup blending for transparent models
        blend_was_enabled_globally = gl.glIsEnabled(gl.GL_BLEND)
        blend_enabled_before_draw = blend_was_enabled_globally
        if self.has_alpha_lt1 and not blend_was_enabled_globally:
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            blend_enabled_before_draw = True

        try:
            if self.vao:
                # Use VAO (modern path)
                gl.glBindVertexArray(self.vao)
            else:
                # Manual vertex attribute setup (OpenGL 2.1 fallback)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

                # Position attribute (location 0)
                position_loc = gl.glGetAttribLocation(active_program, "position")
                if position_loc == -1:
                    position_loc = 0
                gl.glVertexAttribPointer(
                    position_loc,
                    3,
                    gl.GL_FLOAT,
                    gl.GL_FALSE,
                    self.stride * 4,
                    ctypes.c_void_p(self.position_offset * 4),
                )
                gl.glEnableVertexAttribArray(position_loc)

                # Color attribute (location 1)
                color_loc = gl.glGetAttribLocation(active_program, "color")
                if color_loc == -1:
                    color_loc = 1
                gl.glVertexAttribPointer(
                    color_loc,
                    4,
                    gl.GL_FLOAT,
                    gl.GL_FALSE,
                    self.stride * 4,
                    ctypes.c_void_p(self.color_offset * 4),
                )
                gl.glEnableVertexAttribArray(color_loc)

                # Normal attribute (location 2)
                normal_loc = gl.glGetAttribLocation(active_program, "normal")
                if normal_loc == -1:
                    normal_loc = 2
                gl.glVertexAttribPointer(
                    normal_loc,
                    3,
                    gl.GL_FLOAT,
                    gl.GL_FALSE,
                    self.stride * 4,
                    ctypes.c_void_p(self.normal_offset * 4),
                )
                gl.glEnableVertexAttribArray(normal_loc)

            # Draw the triangles
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.num_points)
            draw_error = gl.glGetError()
            if draw_error != gl.GL_NO_ERROR and PYOPENGL_VERBOSE:
                print(f"Model: Warning - Error during glDrawArrays: {draw_error}")

            # Clean up
            if self.vao:
                gl.glBindVertexArray(0)
            else:
                # Disable vertex attributes and unbind VBO
                position_loc = gl.glGetAttribLocation(active_program, "position")
                if position_loc == -1:
                    position_loc = 0
                color_loc = gl.glGetAttribLocation(active_program, "color")
                if color_loc == -1:
                    color_loc = 1
                normal_loc = gl.glGetAttribLocation(active_program, "normal")
                if normal_loc == -1:
                    normal_loc = 2

                gl.glDisableVertexAttribArray(position_loc)
                gl.glDisableVertexAttribArray(color_loc)
                gl.glDisableVertexAttribArray(normal_loc)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

            # Restore blending state
            if self.has_alpha_lt1 and not blend_was_enabled_globally:
                gl.glDisable(gl.GL_BLEND)

            return True

        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Model: Shader rendering failed: {e}")
            # Restore blending state on error
            if self.has_alpha_lt1 and not blend_was_enabled_globally:
                try:
                    gl.glDisable(gl.GL_BLEND)
                except Exception:
                    pass
            return False

    def _draw_bounding_box(self):
        """Draw a simple bounding box for the model as a last resort."""
        try:
            # Make sure we're showing colors
            gl.glColor3f(1.0, 1.0, 0.0)  # Yellow

            # Get bounding box
            min_x, min_y, min_z = self.bounding_box.min_point
            max_x, max_y, max_z = self.bounding_box.max_point

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
        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Model._draw_bounding_box: Failed: {str(e)}")

    def _draw_immediate_mode(self):
        """Draw using legacy immediate mode (OpenGL 1.x) as a fallback.

        Returns:
            bool: True if drawing succeeded, False otherwise.
        """
        try:
            # Clear any errors before starting
            gl.glGetError()

            # Force colors to be visible
            try:
                gl.glEnable(gl.GL_COLOR_MATERIAL)
                gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
            except Exception:
                pass

            # Begin drawing triangles
            gl.glBegin(gl.GL_TRIANGLES)

            # Process only valid vertices
            max_valid = min(len(self.data), self.num_points * self.stride)
            for i in range(0, max_valid, self.stride):
                # Skip if we don't have enough data for a full vertex
                if i + max(
                    self.position_offset + 3, self.color_offset + 4, self.normal_offset + 3
                ) > len(self.data):
                    continue

                # Get position (3 floats)
                px = self.data[i + self.position_offset]
                py = self.data[i + self.position_offset + 1]
                pz = self.data[i + self.position_offset + 2]

                # Get color (4 floats)
                r = self.data[i + self.color_offset]
                g = self.data[i + self.color_offset + 1]
                b = self.data[i + self.color_offset + 2]
                a = self.data[i + self.color_offset + 3]

                # Get normal (3 floats)
                nx = self.data[i + self.normal_offset]
                ny = self.data[i + self.normal_offset + 1]
                nz = self.data[i + self.normal_offset + 2]

                # Set each component individually for maximum compatibility
                try:
                    gl.glNormal3f(nx, ny, nz)
                except Exception:
                    pass

                # Set color - CRUCIAL for visibility
                # Use alpha component for transparent rendering
                gl.glColor4f(r, g, b, a)

                # Set vertex
                gl.glVertex3f(px, py, pz)

            gl.glEnd()
            return True
        except Exception as e:
            # Immediate mode failed
            if PYOPENGL_VERBOSE:
                print(f"Model._draw_immediate_mode: Rendering failed: {str(e)}")
            return False

    def _draw_fallback_wireframe(self):
        """Draw a simple wireframe representation when all other rendering methods fail.

        This is the simplest possible fallback that should work on virtually all OpenGL versions.
        """
        try:
            # Use a basic line drawing approach - this avoids most compatibility issues
            gl.glDisable(gl.GL_LIGHTING)

            # Make sure we use vertex colors if possible
            try:
                gl.glEnable(gl.GL_COLOR_MATERIAL)
                gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
            except Exception:
                pass

            # Process triangles as wireframes by drawing each edge
            max_valid = min(len(self.data), self.num_points * self.stride)

            for i in range(0, max_valid, self.stride * 3):  # Process one triangle at a time
                # Skip if we don't have a full triangle
                if i + (self.stride * 2) + self.position_offset + 3 > len(self.data):
                    continue

                # Get the three vertices of the triangle
                v1 = self.data[i + self.position_offset : i + self.position_offset + 3]
                v2 = self.data[
                    i + self.stride + self.position_offset : i
                    + self.stride
                    + self.position_offset
                    + 3
                ]
                v3 = self.data[
                    i + (self.stride * 2) + self.position_offset : i
                    + (self.stride * 2)
                    + self.position_offset
                    + 3
                ]

                # Get the colors of the three vertices
                c1 = self.data[i + self.color_offset : i + self.color_offset + 4]
                c2 = self.data[
                    i + self.stride + self.color_offset : i + self.stride + self.color_offset + 4
                ]
                c3 = self.data[
                    i + (self.stride * 2) + self.color_offset : i
                    + (self.stride * 2)
                    + self.color_offset
                    + 4
                ]

                # Draw the three edges of the triangle with their correct colors
                gl.glBegin(gl.GL_LINES)
                # Edge 1: v1-v2
                gl.glColor4fv(c1)
                gl.glVertex3fv(v1)
                gl.glColor4fv(c2)
                gl.glVertex3fv(v2)
                gl.glEnd()

                gl.glBegin(gl.GL_LINES)
                # Edge 2: v2-v3
                gl.glColor4fv(c2)
                gl.glVertex3fv(v2)
                gl.glColor4fv(c3)
                gl.glVertex3fv(v3)
                gl.glEnd()

                gl.glBegin(gl.GL_LINES)
                # Edge 3: v3-v1
                gl.glColor4fv(c3)
                gl.glVertex3fv(v3)
                gl.glColor4fv(c1)
                gl.glVertex3fv(v1)
                gl.glEnd()

        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Model._draw_fallback_wireframe: Even wireframe rendering failed: {str(e)}")

            # Last resort - try drawing a white wireframe
            try:
                gl.glColor3f(1.0, 1.0, 1.0)  # White color

                # Draw a simple box representing the model's bounding box
                min_x, min_y, min_z = self.bounding_box.min_point
                max_x, max_y, max_z = self.bounding_box.max_point

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

                # Connect front to back
                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(min_x, min_y, min_z)
                gl.glVertex3f(min_x, min_y, max_z)
                gl.glEnd()

                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(max_x, min_y, min_z)
                gl.glVertex3f(max_x, min_y, max_z)
                gl.glEnd()

                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(max_x, max_y, min_z)
                gl.glVertex3f(max_x, max_y, max_z)
                gl.glEnd()

                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(min_x, max_y, min_z)
                gl.glVertex3f(min_x, max_y, max_z)
                gl.glEnd()
            except Exception:
                # If even this fails, give up
                pass

        finally:
            # Try to restore state
            try:
                if self.gl_ctx.has_legacy_lighting:
                    gl.glEnable(gl.GL_LIGHTING)
            except Exception:
                pass

    def delete(self):
        """Delete OpenGL resources."""
        gl_ctx: GLContext = self.gl_ctx
        if gl_ctx.has_3_3 and self.vao:
            try:
                # Convert VAO to int if it's a numpy array
                vao_id = self.vao
                if isinstance(vao_id, np.ndarray):
                    vao_id = int(vao_id[0])

                # Delete the VAO
                gl.glDeleteVertexArrays(1, [vao_id])
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Model.delete: Error deleting VAO: {e}")
            self.vao = None

        if gl_ctx.has_vbo and self.vbo:
            try:
                # Convert VBO to int if it's a numpy array
                vbo_id = self.vbo
                if isinstance(vbo_id, np.ndarray):
                    vbo_id = int(vbo_id[0])

                # Delete the VBO
                gl.glDeleteBuffers(1, [vbo_id])
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Model.delete: Error deleting VBO: {e}")
            self.vbo = None

    def __del__(self):
        """Destructor to clean up OpenGL resources."""
        self.delete()

    def _draw_lines_core_profile(self):
        """Draw a wireframe model using only modern core profile features.

        This is a minimal drawing method that should work in core profiles without
        requiring VAOs, VBOs, or fixed-function pipeline features.
        """
        try:
            # Clear any existing errors
            gl.glGetError()

            # Make sure no program is active - we're drawing without shaders
            gl.glUseProgram(0)

            # Create a simple line-based representation of the model
            # Process triangles as wireframes
            max_valid = min(len(self.data), self.num_points * self.stride)

            # Set white color using generic vertex attribute if we have shaders
            if self.gl_ctx.has_shader:
                try:
                    # Set color using generic vertex attribute 0
                    # This assumes a basic vertex shader that uses this attribute
                    gl.glVertexAttrib4f(0, 1.0, 1.0, 1.0, 1.0)
                except Exception:
                    pass

            # Draw each triangle as 3 separate lines
            for i in range(0, max_valid, self.stride * 3):
                # Skip if we don't have a full triangle
                if i + (self.stride * 2) + self.position_offset + 3 > len(self.data):
                    continue

                # Get the three vertices of the triangle
                v1 = self.data[i + self.position_offset : i + self.position_offset + 3]
                v2 = self.data[
                    i + self.stride + self.position_offset : i
                    + self.stride
                    + self.position_offset
                    + 3
                ]
                v3 = self.data[
                    i + (self.stride * 2) + self.position_offset : i
                    + (self.stride * 2)
                    + self.position_offset
                    + 3
                ]

                try:
                    # Set up a line strip
                    gl.glBegin(gl.GL_LINE_LOOP)
                    gl.glVertex3fv(v1)
                    gl.glVertex3fv(v2)
                    gl.glVertex3fv(v3)
                    gl.glEnd()
                except Exception as e:
                    if PYOPENGL_VERBOSE:
                        print(f"Model._draw_lines_core_profile: glBegin/End failed: {e}")
                    return False

            return True

        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Model._draw_lines_core_profile: Failed to draw: {e}")
            return False

    def get_triangles(self) -> np.ndarray:
        """Extract individual triangles from the model data.

        Returns:
            numpy.ndarray: Array of triangles, each containing 3 vertices with full attributes
                           Shape: (num_triangles, 3, stride)
        """
        triangle_count = self.num_points // 3

        # Reshape the data to extract triangles
        # Each triangle has 3 vertices, each vertex has self.stride attributes
        triangles = np.zeros((triangle_count, 3, self.stride), dtype=np.float32)

        for i in range(triangle_count):
            for j in range(3):  # Three vertices per triangle
                vertex_idx = i * 3 + j
                start_idx = vertex_idx * self.stride
                end_idx = start_idx + self.stride

                # Extract full vertex data for this triangle vertex
                if start_idx < len(self.data) and end_idx <= len(self.data):
                    triangles[i, j] = self.data[start_idx:end_idx]

        return triangles

    def is_triangle_transparent(self, triangle: np.ndarray) -> bool:
        """Check if a triangle has any transparent vertices.

        Args:
            triangle: Triangle data with shape (3, stride)

        Returns:
            bool: True if any vertex in the triangle has alpha < 1.0
        """
        # Check alpha value (4th component of color) for each vertex
        for vertex in triangle:
            alpha = vertex[self.color_offset + 3]
            if alpha < 1.0:
                return True
        return False

    def get_triangle_z_position(self, triangle: np.ndarray) -> float:
        """Calculate average Z position of a triangle for depth sorting.

        Args:
            triangle: Triangle data with shape (3, stride)

        Returns:
            float: Average Z coordinate of the triangle's vertices
        """
        # Extract Z coordinate (3rd component of position) for each vertex
        z_coords = [vertex[self.position_offset + 2] for vertex in triangle]
        # Return average Z position
        return sum(z_coords) / 3.0

    @staticmethod
    def create_coalesced_models(models: List["Model"]) -> Tuple["Model", "Model"]:
        """Create two consolidated models from a list of models - one opaque, one transparent.

        This method scans through all triangles in all models and separates them based on
        transparency (any vertex with alpha < 1.0). For the transparent model, triangles
        are sorted by Z position for proper back-to-front rendering.

        Args:
            models: List of Model objects to coalesce

        Returns:
            tuple: (opaque_model, transparent_model)
        """
        if not models:
            # Return empty models if no input
            empty_data = np.array([], dtype=np.float32)
            return Model(empty_data), Model(empty_data, has_alpha_lt1=True)

        # Use first model's structure as reference
        reference_model = models[0]
        stride = reference_model.stride
        position_offset = reference_model.position_offset
        color_offset = reference_model.color_offset
        normal_offset = reference_model.normal_offset

        # Collect triangles from all models
        opaque_triangles = []
        transparent_triangles = []

        for model in models:
            # Extract triangles from this model
            triangles = model.get_triangles()

            # Categorize each triangle
            for triangle in triangles:
                if model.is_triangle_transparent(triangle):
                    # Add triangle with its Z position for later sorting
                    z_pos = model.get_triangle_z_position(triangle)
                    transparent_triangles.append((z_pos, triangle))
                else:
                    opaque_triangles.append(triangle)

        # Sort transparent triangles by Z position (back to front)
        transparent_triangles.sort(key=lambda item: item[0], reverse=True)

        # Extract just the triangle data after sorting
        transparent_triangles = [t[1] for t in transparent_triangles]

        # Create the data arrays for the new models
        opaque_data = np.zeros(len(opaque_triangles) * 3 * stride, dtype=np.float32)
        transparent_data = np.zeros(len(transparent_triangles) * 3 * stride, dtype=np.float32)

        # Fill the opaque data array
        idx = 0
        for triangle in opaque_triangles:
            for vertex in triangle:
                opaque_data[idx : idx + stride] = vertex
                idx += stride

        # Fill the transparent data array
        idx = 0
        for triangle in transparent_triangles:
            for vertex in triangle:
                transparent_data[idx : idx + stride] = vertex
                idx += stride

        # Create and return the new models
        opaque_model = Model(
            opaque_data,
            has_alpha_lt1=False,
            num_points=len(opaque_triangles) * 3,
            position_offset=position_offset,
            color_offset=color_offset,
            normal_offset=normal_offset,
            stride=stride,
        )

        transparent_model = Model(
            transparent_data,
            has_alpha_lt1=True,
            num_points=len(transparent_triangles) * 3,
            position_offset=position_offset,
            color_offset=color_offset,
            normal_offset=normal_offset,
            stride=stride,
        )

        return opaque_model, transparent_model

    def _draw_with_vertex_arrays(self) -> bool:
        """Draw the model using legacy vertex arrays with VBO (reload arrays each time).

        Returns:
            bool: True if drawing succeeded, False otherwise.
        """
        if not self.vbo or self.num_points == 0:
            return False

        try:
            # Ensure VBO is bound for drawing
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
            
            # Set up vertex arrays for THIS model (don't rely on global state from initialization)
            # Enable and set up vertex array
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3,                                   # size (3 components for vertex)
                            gl.GL_FLOAT,                         # type
                            self.stride * 4,                     # stride (in bytes)
                            ctypes.c_void_p(self.position_offset * 4)) # pointer (offset into VBO)

            # Enable and set up color array
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glColorPointer(4,                                  # size (4 components for color RGBA)
                            gl.GL_FLOAT,                        # type
                            self.stride * 4,                    # stride (in bytes)
                            ctypes.c_void_p(self.color_offset * 4)) # pointer (offset into VBO)

            # Enable and set up normal array
            gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
            gl.glNormalPointer(gl.GL_FLOAT,                       # type (no size parameter for normals)
                            self.stride * 4,                   # stride (in bytes)
                            ctypes.c_void_p(self.normal_offset * 4)) # pointer (offset into VBO)
            
            # Setup blending if needed
            blend_enabled_before_draw = False
            if self.has_alpha_lt1:
                blend_was_enabled_globally = gl.glIsEnabled(gl.GL_BLEND)
                if not blend_was_enabled_globally:
                    gl.glEnable(gl.GL_BLEND)
                    blend_enabled_before_draw = True
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

            # Draw using vertex arrays
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.num_points)
            
            draw_error = gl.glGetError()
            if draw_error != gl.GL_NO_ERROR:
                return False

            # Restore blend state if we changed it
            if self.has_alpha_lt1 and blend_enabled_before_draw:
                gl.glDisable(gl.GL_BLEND)

            # Clean up vertex array state
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
            
            # Unbind VBO
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            
            return True

        except Exception as e:
            # Cleanup on error
            try:
                gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
                gl.glDisableClientState(gl.GL_COLOR_ARRAY)
                gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
                if self.has_alpha_lt1 and blend_enabled_before_draw:
                    gl.glDisable(gl.GL_BLEND)
            except:
                pass
            
            return False
