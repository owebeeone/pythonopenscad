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
        """Initialize OpenGL vertex buffer and array objects.

        This must be called when the correct OpenGL context is active.
        """
        if self.vbo is not None or self.vao is not None:
            if PYOPENGL_VERBOSE: print(f"MODEL_DEBUG: initialize_gl_resources returning early: VBO ({self.vbo}) or VAO ({self.vao}) already exists.")
            return

        gl_ctx: GLContext = self.gl_ctx
        try:
            # This print is unconditional to check has_vbo
            if PYOPENGL_VERBOSE: print(f"MODEL_DEBUG: gl_ctx.has_vbo = {gl_ctx.has_vbo}")
        except AttributeError:
            if PYOPENGL_VERBOSE: print(f"MODEL_DEBUG: gl_ctx has no attribute 'has_vbo' or gl_ctx is None.")
        # --- End Unconditional Debug Prints ---
        
        if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Starting. Model has_alpha_lt1={self.has_alpha_lt1}")

        # Skip VBO/VAO initialization if not supported
        if not gl_ctx.has_vbo:
            if PYOPENGL_VERBOSE: print("Model.initialize_gl_resources: GLContext reports no VBO support. Aborting.")
            return

        # Store current error state to check if operations succeed
        # prev_error = gl.glGetError() # Clear errors at the start
        while gl.glGetError() != gl.GL_NO_ERROR: pass # Clear any pre-existing errors

        try:
            if PYOPENGL_VERBOSE: print("Model.initialize_gl_resources: Attempting VBO creation.")
            # --- Create VBO ---
            self.vbo = gl.glGenBuffers(1)
            if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: glGenBuffers(1) returned VBO ID: {self.vbo}")
            # Handle numpy array return if necessary (older PyOpenGL versions?)
            if isinstance(self.vbo, np.ndarray):
                self.vbo = int(self.vbo[0])
                if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Converted VBO ID to int: {self.vbo}")

            # Check for immediate error after glGenBuffers
            error = gl.glGetError()
            if error != gl.GL_NO_ERROR:
                if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Error after glGenBuffers: {error}. Cleaning up VBO.")
                self.vbo = None
                return

            if self.vbo is None or (isinstance(self.vbo, int) and self.vbo <= 0):
                if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: VBO ID is invalid ({self.vbo}). Aborting.")
                self.vbo = None
                return

            if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Binding VBO {self.vbo}.")
            # Bind and Buffer Data
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
            error = gl.glGetError()
            if error != gl.GL_NO_ERROR:
                if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Error after glBindBuffer (VBO {self.vbo}): {error}. Cleaning up VBO.")
                gl.glDeleteBuffers(1, [self.vbo])  # Attempt cleanup
                self.vbo = None
                return

            if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Buffering data to VBO {self.vbo} ({self.data.nbytes} bytes).")
            gl.glBufferData(gl.GL_ARRAY_BUFFER, self.data.nbytes, self.data, gl.GL_STATIC_DRAW)
            error = gl.glGetError()
            if error != gl.GL_NO_ERROR:
                if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Error after glBufferData (VBO {self.vbo}): {error}. Cleaning up VBO.")
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)  # Unbind before deleting
                gl.glDeleteBuffers(1, [self.vbo])  # Attempt cleanup
                self.vbo = None
                return
            if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: VBO {self.vbo} created and buffered successfully.")

            # --- Use VAO if supported (OpenGL 3.3+) ---
            if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Checking for VAO support (gl_ctx.has_3_3: {gl_ctx.has_3_3}).")
            if gl_ctx.has_3_3:
                try:
                    if PYOPENGL_VERBOSE: print("Model.initialize_gl_resources: Attempting VAO creation.")
                    # Create VAO
                    self.vao = gl.glGenVertexArrays(1)
                    if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: glGenVertexArrays(1) returned VAO ID: {self.vao}")
                    if isinstance(self.vao, np.ndarray):
                        self.vao = int(self.vao[0])  # Convert from numpy array to int
                        if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Converted VAO ID to int: {self.vao}")

                    # Check for immediate error after glGenVertexArrays
                    error = gl.glGetError()
                    if error != gl.GL_NO_ERROR:
                        if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Error after glGenVertexArrays: {error}. VAO set to None.")
                        self.vao = None
                        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)  # Unbind VBO before returning
                        return

                    if self.vao is None or (isinstance(self.vao, int) and self.vao <= 0):
                        if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: VAO ID is invalid ({self.vao}). Aborting VAO setup.")
                        self.vao = None
                        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)  # Unbind VBO before returning
                        return

                    if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Binding VAO {self.vao}.")
                    # Bind the VAO and set up vertex attributes
                    gl.glBindVertexArray(self.vao)
                    error = gl.glGetError()
                    if error != gl.GL_NO_ERROR:
                        if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Error after glBindVertexArray (VAO {self.vao}): {error}. Cleaning up VAO.")
                        try:
                            gl.glDeleteVertexArrays(1, [self.vao])
                        except Exception:
                            pass
                        self.vao = None
                        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)  # Unbind VBO before returning
                        return

                    if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Binding VBO {self.vbo} to VAO {self.vao}.")
                    # Now set up vertex attributes (with VAO bound)
                    gl.glBindBuffer(
                        gl.GL_ARRAY_BUFFER, self.vbo
                    )  # Ensure VBO is bound *while* VAO is bound
                    error = gl.glGetError()
                    if error != gl.GL_NO_ERROR:
                        if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Error binding VBO {self.vbo} to VAO {self.vao}: {error}. Cleaning up VAO.")
                        gl.glBindVertexArray(0)  # Unbind VAO
                        try:
                            gl.glDeleteVertexArrays(1, [self.vao])
                        except:
                            pass
                        self.vao = None
                        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)  # Unbind VBO
                        return

                    if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Setting up VAO attributes for VAO {self.vao}.")
                    # Position attribute - always in location 0
                    gl.glVertexAttribPointer(
                        0,
                        3,
                        gl.GL_FLOAT,
                        gl.GL_FALSE,
                        self.stride * 4,
                        ctypes.c_void_p(self.position_offset * 4),
                    )
                    error = gl.glGetError() # Check error immediately after
                    if error != gl.GL_NO_ERROR and PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Error after Pos glVertexAttribPointer: {error}") # VAO cleanup handled by outer try/except
                    gl.glEnableVertexAttribArray(0)
                    error = gl.glGetError() # Check error immediately after
                    if error != gl.GL_NO_ERROR and PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Error after Pos glEnableVertexAttribArray: {error}")

                    # Color attribute - always in location 1
                    gl.glVertexAttribPointer(
                        1,
                        4,
                        gl.GL_FLOAT,
                        gl.GL_FALSE,
                        self.stride * 4,
                        ctypes.c_void_p(self.color_offset * 4),
                    )
                    error = gl.glGetError()
                    if error != gl.GL_NO_ERROR and PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Error after Color glVertexAttribPointer: {error}")
                    gl.glEnableVertexAttribArray(1)
                    error = gl.glGetError()
                    if error != gl.GL_NO_ERROR and PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Error after Color glEnableVertexAttribArray: {error}")

                    # Normal attribute - always in location 2
                    gl.glVertexAttribPointer(
                        2,
                        3,
                        gl.GL_FLOAT,
                        gl.GL_FALSE,
                        self.stride * 4,
                        ctypes.c_void_p(self.normal_offset * 4),
                    )
                    error = gl.glGetError()
                    if error != gl.GL_NO_ERROR and PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Error after Normal glVertexAttribPointer: {error}")
                    gl.glEnableVertexAttribArray(2)
                    error = gl.glGetError()
                    if error != gl.GL_NO_ERROR and PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Error after Normal glEnableVertexAttribArray: {error}")

                    if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: VAO {self.vao} attributes set. Unbinding VAO.")
                    # Unbind VAO first, then VBO to avoid state leaks
                    gl.glBindVertexArray(0)
                    error = gl.glGetError()
                    if error != gl.GL_NO_ERROR:
                        if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Error unbinding VAO {self.vao}: {error}")
                        pass  # Just note the error internally if needed
                    if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: VAO {self.vao} setup successful.")

                except Exception as e:
                    if PYOPENGL_VERBOSE:
                        print(
                            f"Model.initialize_gl_resources: VAO setup failed with exception: {e}"
                        )
                    if self.vao:
                        try:
                            # Ensure VAO isn't bound before deleting
                            if gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING) == self.vao:
                                gl.glBindVertexArray(0)
                            gl.glDeleteVertexArrays(1, [self.vao])
                        except Exception as del_e:
                            if PYOPENGL_VERBOSE:
                                print(
                                    f"Model.initialize_gl_resources: Error during VAO cleanup: {del_e}"
                                )
                    self.vao = None
            else:
                if PYOPENGL_VERBOSE: print("Model.initialize_gl_resources: GLContext reports no VAO (GL 3.3) support. VAO not created.")
                pass

            # Unbind VBO (happens whether VAO was used or not)
            if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Unbinding GL_ARRAY_BUFFER (VBO {self.vbo}).")
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            error = gl.glGetError()
            if error != gl.GL_NO_ERROR:
                if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Error unbinding VBO {self.vbo}: {error}")
                pass  # Just note the error internally if needed
            if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Resource initialization completed. VAO: {self.vao}, VBO: {self.vbo}")

        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(
                    f"Model.initialize_gl_resources: General failure initializing OpenGL resources: {e}"
                )
            # Clean up any resources that might have been partially created
            if self.vao:
                try:
                    if gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING) == self.vao:
                        gl.glBindVertexArray(0)
                    gl.glDeleteVertexArrays(1, [self.vao])
                except Exception as del_e:
                    if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Error during VAO cleanup: {del_e}")
                self.vao = None

            if self.vbo:
                try:
                    # No need to check binding for buffers like VAOs
                    gl.glDeleteBuffers(1, [self.vbo])
                except Exception as del_e:
                    if PYOPENGL_VERBOSE: print(f"Model.initialize_gl_resources: Error during VBO cleanup: {del_e}")
                self.vbo = None
        finally:
            # Final check
            final_error = gl.glGetError()
            if final_error != gl.GL_NO_ERROR and PYOPENGL_VERBOSE:
                print(f"Model.initialize_gl_resources: Exiting with uncleared GL error: {final_error}")

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

    def draw(self):
        """Draw the model using OpenGL."""
        if PYOPENGL_VERBOSE: print(f"Model.draw: Entered for model with VBO {self.vbo}, VAO {self.vao}. Shader program ID on model (if any): {getattr(self, 'shader_program_id', 'N/A')}")

        gl_ctx: GLContext = self.gl_ctx
        # current_window = glut.glutGetWindow() # REMOVED: This is GLUT-specific

        # Ensure we're drawing in a valid window context
        # if current_window == 0: # REMOVED: Context should be managed by caller (PoscGLWidget or GLUT Viewer)
        #     if PYOPENGL_VERBOSE:
        #         print("Model.draw: No valid window context")
        #     return

        # Setup blending for transparent models
        blend_enabled = False
        if self.has_alpha_lt1:
            try:
                # Enable blending for transparent models
                blend_enabled = gl.glIsEnabled(gl.GL_BLEND)
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Model.draw: Failed to enable blending: {e}")

        try:
            # First try to render using shaders if available
            if PYOPENGL_VERBOSE: print(f"Model.draw: Checking shader/VBO support. gl_ctx.has_shader={gl_ctx.has_shader}, gl_ctx.has_vbo={gl_ctx.has_vbo}")
            if gl_ctx.has_shader and gl_ctx.has_vbo:
                # If shader program is available from the viewer, use it
                current_program = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
                if PYOPENGL_VERBOSE: print(f"Model.draw: Current active shader program: {current_program}")
                if current_program and current_program != 0:
                    if PYOPENGL_VERBOSE: print(f"Model.draw: Attempting _draw_with_shader with program {current_program}.")
                    if self._draw_with_shader(current_program):
                        if PYOPENGL_VERBOSE: print("Model.draw: _draw_with_shader succeeded.")
                        # Shader-based rendering was successful
                        if self.has_alpha_lt1 and not blend_enabled:
                            try:
                                gl.glDisable(gl.GL_BLEND)
                            except Exception:
                                pass
                        return
                    elif PYOPENGL_VERBOSE:
                        print("Model.draw: _draw_with_shader returned False. Falling back.")
                elif PYOPENGL_VERBOSE:
                    print("Model.draw: No active shader program (or program is 0). Falling back.")
            elif PYOPENGL_VERBOSE:
                print("Model.draw: Shader or VBO support not available. Falling back.")

            # Make sure colors are visible
            try:
                gl.glEnable(gl.GL_COLOR_MATERIAL)
                gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
            except Exception:
                pass

            # Fallback to immediate mode if shader rendering failed or isn't available
            if PYOPENGL_VERBOSE: print("Model.draw: Attempting _draw_immediate_mode.")
            if not self._draw_immediate_mode():
                if PYOPENGL_VERBOSE: print("Model.draw: _draw_immediate_mode returned False. Falling back to wireframe.")
                # If immediate mode fails, try the wireframe fallback
                self._draw_fallback_wireframe()
            elif PYOPENGL_VERBOSE:
                print("Model.draw: _draw_immediate_mode succeeded.")

        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Model.draw: All rendering methods failed: {str(e)}")
            # Last resort - try simple bounding box
            try:
                self._draw_bounding_box()
            except Exception:
                pass

        # Restore blending state if we changed it
        if self.has_alpha_lt1 and not blend_enabled:
            try:
                gl.glDisable(gl.GL_BLEND)
            except Exception:
                pass

    def _draw_with_shader(self, active_program: int) -> bool:
        """Draw the model using the provided shader program.

        Returns:
            bool: True if drawing succeeded, False otherwise.
        """
        if PYOPENGL_VERBOSE: print(f"Model._draw_with_shader: Entered for VAO {self.vao}, program {active_program}. num_points={self.num_points}")
        if not self.vao or self.num_points == 0: # Added num_points check
            if PYOPENGL_VERBOSE: print(f"Model._draw_with_shader: Aborting - No VAO ({self.vao}) or num_points is 0 ({self.num_points}).")
            return False

        is_prog_valid = False
        try:
            is_prog_valid = gl.glIsProgram(active_program)
        except Exception as e:
            if PYOPENGL_VERBOSE: print(f"Model._draw_with_shader: glIsProgram check failed: {e}")

        if not is_prog_valid:
            if PYOPENGL_VERBOSE: print(f"Model._draw_with_shader: Program {active_program} is not valid.")
            return False

        try:
            gl.glBindVertexArray(self.vao)
            if PYOPENGL_VERBOSE: print(f"Model._draw_with_shader: VAO {self.vao} bound.")
            # gl.glEnableVertexAttribArray(0)  # Position - These are part of VAO state, should not be needed here IF VAO is correctly set up and bound
            # gl.glEnableVertexAttribArray(1)  # Color
            # gl.glEnableVertexAttribArray(2)  # Normal

            blend_enabled_before_draw = False # To track if GL_BLEND was enabled by this function specifically
            if self.has_alpha_lt1:
                # Store current blend state
                blend_was_enabled_globally = gl.glIsEnabled(gl.GL_BLEND)
                if not blend_was_enabled_globally:
                    gl.glEnable(gl.GL_BLEND)
                    blend_enabled_before_draw = True # We turned it on
                # Ensure blend func is correct (might be changed elsewhere)
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
                if PYOPENGL_VERBOSE: print(f"Model._draw_with_shader: Blend enabled (was globally {blend_was_enabled_globally}, we set to {blend_enabled_before_draw}).")
            # else:
                # blend_enabled = False  # Keep track even if we don't enable it here

            if PYOPENGL_VERBOSE: print(f"Model._draw_with_shader: Calling glDrawArrays(GL_TRIANGLES, 0, {self.num_points}).")
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.num_points)
            draw_error = gl.glGetError()
            if draw_error != gl.GL_NO_ERROR and PYOPENGL_VERBOSE:
                print(f"Model._draw_with_shader: GL error after glDrawArrays: {draw_error}")

            # Restore blend state ONLY if we changed it
            if self.has_alpha_lt1 and blend_enabled_before_draw:
                gl.glDisable(gl.GL_BLEND)
                if PYOPENGL_VERBOSE: print(f"Model._draw_with_shader: Blend disabled (was set by this function).")

            gl.glBindVertexArray(0)
            if PYOPENGL_VERBOSE: print(f"Model._draw_with_shader: VAO {self.vao} unbound.")
            # gl.glDisableVertexAttribArray(0) # Should not be needed if VAO is managed correctly
            # gl.glDisableVertexAttribArray(1)
            # gl.glDisableVertexAttribArray(2)

            return True  # Indicate success

        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(
                    f"Model._draw_with_shader: Error drawing VAO {self.vao} with program {active_program}: {e}"
                )
            # Attempt to cleanup context state on error
            try:
                gl.glBindVertexArray(0)
                # Restore blend state if it might have been left enabled
                if self.has_alpha_lt1 and not blend_enabled_before_draw:
                    try:
                        gl.glDisable(gl.GL_BLEND)
                    except:
                        pass
            except Exception:
                pass  # Avoid errors during error handling
            return False  # Indicate failure

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
