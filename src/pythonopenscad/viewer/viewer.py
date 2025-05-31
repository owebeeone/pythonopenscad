"""OpenGL 3D viewer for PyOpenSCAD models."""

import numpy as np
import ctypes
from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union, Dict, Callable, ClassVar
from datatrees import datatree, dtfield, Node
import warnings
import sys
import signal
from datetime import datetime
import manifold3d as m3d
from pythonopenscad.viewer.basic_models import (
    create_colored_test_cube,
    create_triangle_model,
    create_cube_model,
)
from pythonopenscad.viewer.bbox import BoundingBox
from pythonopenscad.viewer.bbox_render import BBoxRender
from pythonopenscad.viewer.glctxt import GLContext, PYOPENGL_VERBOSE
from pythonopenscad.viewer.model import Model
from pythonopenscad.viewer.axes import AxesRenderer

import anchorscad_lib.linear as linear
from pythonopenscad.viewer.shader import BASIC_SHADER_PROGRAM, SHADER_PROGRAM

import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
from pyglm import glm

from pythonopenscad.viewer.viewer_base import ViewerBase

#PYOPENGL_VERBOSE = True


@datatree
class Viewer(ViewerBase):
    """OpenGL viewer for 3D models."""

    models: list[Model]
    width: int = 800
    height: int = 600
    title: str = "3D Viewer"
    use_coalesced_models: bool = True

    # Rendering state
    backface_culling: bool = True
    wireframe_mode: bool = False
    bounding_box_mode: int = 0  # 0: off, 1: wireframe, 2: solid
    zbuffer_occlusion: bool = True
    antialiasing_enabled: bool = True  # Added antialiasing state
    show_axes: bool = True
    edge_rotations: bool = False

    background_color: Tuple[float, float, float, float] = (0.98, 0.98, 0.85, 1.0)

    axes_renderer_node: Node[AxesRenderer] = Node(AxesRenderer, prefix="axes_")
    axes_renderer: AxesRenderer = dtfield(self_default=lambda self: self.axes_renderer_node())
    # REMOVED ortho_projection: bool = False - replaced by projection_mode

    # Add state for projection mode and ortho scale
    projection_mode: str = dtfield(default="perspective")  # 'perspective' or 'orthographic'
    ortho_scale: float = dtfield(default=20.0)  # World-space width for ortho view

    VIEWER_HELP_TEXT = """
    Mouse Controls:
     Left button drag: Rotate camera
     Right button drag: Pan camera
     Wheel: Zoom in/out (Perspective) / Change scale (Orthographic)
    
    Keyboard Controls:
     B - Toggle backface culling
     W - Toggle wireframe mode
     Z - Toggle Z-buffer occlusion for wireframes
     C - Toggle coalesced model mode (improves transparency rendering)
     H - Toggle shader-based rendering (modern vs. legacy mode)
     O - Toggle Orthographic/Perspective projection
     D - Print diagnostic information about OpenGL capabilities
     P - Print detailed shader program diagnostics
     R - Reset view
     X - Toggle bounding box (off/wireframe/solid)
     S - Save screenshot
     + - Toggle axes visibility
     G - Toggle axes graduation ticks
     V - Toggle axes graduation values
     E - Toggle edge effect rotation interaction
     ESC - Close viewer
    """

    # Static window registry to handle GLUT callbacks
    _instances: ClassVar[Dict[int, "Viewer"]] = {}
    _initialized: ClassVar[bool] = False
    _next_id: ClassVar[int] = 0

    # OpenGL state
    window_id: int | None = dtfield(default=0, init=False)
    shader_program: Any | None = dtfield(default=None, init=False)
    # Flag to track if multisample is supported by the system
    _multisample_supported: bool = dtfield(default=False, init=False)

    def __post_init__(self):
        """
        Initialize the viewer.
        """
        # PYOPENGL_VERBOSE = True

        # Initialize projection mode and scale using defaults or passed values
        # The dtfield defaults handle this, no extra code needed here unless
        # we want to override based on other initial state.
        # For clarity, we ensure the values are set.
        # Note: datatree handles setting these from kwargs if passed in constructor
        # So, self.projection_mode and self.ortho_scale will have the passed values
        # or the dtfield defaults if not passed.

        # Example check (optional):
        # if hasattr(self, 'projection_mode') and self.projection_mode not in ['perspective', 'orthographic']:
        #    warnings.warn(f"Invalid initial projection mode '{self.projection_mode}', defaulting to perspective.")
        #    self.projection_mode = 'perspective'
        # if not hasattr(self, 'ortho_scale') or self.ortho_scale <= 0:
        #    self.ortho_scale = 20.0 # Ensure positive default if invalid

        # 1. Get GLContext instance (but don't initialize)
        self.gl_ctx = GLContext.get_instance()

        # 2. Basic GLUT Initialization (sets display mode)
        Viewer._init_glut()  # Does glutInit and glutInitDisplayMode

        # 3. Request specific context *before* creating the window
        try:
            if PYOPENGL_VERBOSE:
                print("Viewer: Requesting OpenGL 3.0 compatibility profile")
            self.gl_ctx.request_context_version(3, 0, core_profile=False)
        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Error requesting OpenGL 3.0 context version: {e}")

        # --- Model Preparation (can happen before window creation) ---
        self.original_models = self.models  # Store original models
        if self.models:
            opaque_model, transparent_model = Model.create_coalesced_models(self.models)
            self.models = []
            if opaque_model.num_points > 0:
                self.models.append(opaque_model)
            if transparent_model.num_points > 0:
                self.models.append(transparent_model)
        # -----------------------------------------------------------

        # --- Camera Setup (based on models, before window) ---
        # These are initialized here, but ortho_scale might be overridden by constructor args
        self.camera_pos = glm.vec3(10.0, -10.0, 10.0)
        self.camera_front = glm.vec3(0.0, 0.0, 1.0)
        self.camera_up = glm.vec3(0.0, 0.0, 1.0)
        self.camera_target = glm.vec3(0.0, 0.0, 0.0)  # Initialize camera target
        self.camera_speed = 0.05
        self.yaw = -90.0
        self.pitch = 0.0
        self.model_matrix = np.eye(4, dtype=np.float32)
        self.last_mouse_x = self.width // 2
        self.last_mouse_y = self.height // 2
        self.first_mouse = True
        self.left_button_pressed = False
        self.right_button_pressed = False
        self.middle_button_pressed = False
        self.mouse_start_x = 0
        self.mouse_start_y = 0
        self._compute_scene_bounds()  # Needs models
        self._setup_camera()  # Needs bounds
        # -----------------------------------------------------

        # Register this instance
        self.instance_id = Viewer._next_id
        Viewer._next_id += 1
        Viewer._instances[self.instance_id] = self

        # 4. Create the main window (makes context current)
        self._create_window()

        # 5. Initialize GLContext using the main window's context
        if self.window_id and self.window_id == glut.glutGetWindow():
            # Only initialize if not already done (e.g., by another Viewer instance)
            if not self.gl_ctx.is_initialized:
                if PYOPENGL_VERBOSE:
                    print(
                        f"Viewer: Initializing GLContext using main window ({self.window_id}) context."
                    )
                self.gl_ctx.initialize()
            elif PYOPENGL_VERBOSE:
                print("GLContext already initialized.")
        elif PYOPENGL_VERBOSE:
            # Only initialize if not already done (e.g., by another Viewer instance)
            if not self.gl_ctx.is_initialized:
                if PYOPENGL_VERBOSE:
                    print(
                        f"Viewer: Initializing GLContext using main window ({self.window_id}) context."
                    )
                self.gl_ctx.initialize()
            elif PYOPENGL_VERBOSE:
                print("GLContext already initialized.")
        elif PYOPENGL_VERBOSE:
            warnings.warn(
                "Viewer: Cannot initialize GLContext, main window creation failed or context mismatch."
            )
        # --------------------------------------------------------------------------

        # --- Setup GL state (lighting, shaders, MSAA query etc.) AFTER context is initialized ---
        self._setup_gl()
        # --------------------------------------------------------------------------

        # --- Initialize GL resources for models LAST ---
        # Now that the context and shaders are ready, initialize VBOs/VAOs for each model
        if self.window_id and self.window_id == glut.glutGetWindow() and self.gl_ctx.is_initialized:
            if PYOPENGL_VERBOSE:
                print("Viewer: Initializing GL resources for models...")
            for model in self.models:
                try:
                    model.initialize_gl_resources()
                except Exception as e:
                    if PYOPENGL_VERBOSE:
                        print(f"Viewer: Error initializing GL resources for model: {e}")
            if PYOPENGL_VERBOSE:
                print("Viewer: Finished initializing GL resources for models.")
        elif PYOPENGL_VERBOSE:
            warnings.warn(
                "Viewer: Skipping model GL resource initialization due to missing window or context."
            )

    def _compute_scene_bounds(self):
        """Compute the overall scene bounding box."""
        # Start with a default bounding box
        self.bounding_box = BoundingBox()

        if not self.models:
            # No models, use a default unit-sized bounding box centered at origin
            self.bounding_box.min_point = np.array([-0.5, -0.5, -0.5])
            self.bounding_box.max_point = np.array([0.5, 0.5, 0.5])
            return

        valid_models = []
        for model in self.models:
            # Check if the model has a valid bounding box
            if not np.all(np.isinf(model.bounding_box.min_point)) and not np.all(
                np.isinf(model.bounding_box.max_point)
            ):
                valid_models.append(model)

        if not valid_models:
            # No valid models, use a default unit-sized bounding box centered at origin
            self.bounding_box.min_point = np.array([-0.5, -0.5, -0.5])
            self.bounding_box.max_point = np.array([0.5, 0.5, 0.5])
            return

        # Start with the first valid model's bounding box
        self.bounding_box = valid_models[0].bounding_box

        # Union with all other valid models
        for model in valid_models[1:]:
            self.bounding_box = self.bounding_box.union(model.bounding_box)

    def close(self):
        """Close the viewer."""
        # Avoid calling glut functions if the window doesn't exist or GLUT isn't initialized
        if self.window_id is not None and Viewer._initialized:
            try:
                glut.glutDestroyWindow(self.window_id)
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Error destroying window {self.window_id}: {e}")
        self.window_id = None

    @classmethod
    def _init_glut(cls):
        """Initialize GLUT and set display mode. Does NOT create a context."""
        if not cls._initialized:
            if PYOPENGL_VERBOSE:
                print("Viewer: Initializing GLUT")
            # Initialize GLUT
            try:
                glut.glutInit()
            except Exception as e:
                print(f"FATAL: glutInit() failed: {e}", file=sys.stderr)
                # Potentially raise or exit here, as nothing else will work
                raise RuntimeError("Failed to initialize GLUT") from e

            # Try to set a display mode with multisampling
            display_mode = glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH
            try:
                glut.glutInitDisplayMode(display_mode | glut.GLUT_MULTISAMPLE)
                # Assume multisample is supported if this doesn't raise an error (will verify later)
                # cls._multisample_supported = True # We now verify this later using glGet
                if PYOPENGL_VERBOSE:
                    print("Viewer: Requested multisample display mode.")
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(
                        f"Viewer: Multisample display mode failed ({e}), falling back. Antialiasing might not work."
                    )
                # Fallback to non-multisample mode
                try:
                    glut.glutInitDisplayMode(display_mode)
                except Exception as e_fallback:
                    # This is very bad if it fails
                    print(
                        f"FATAL: Error setting even basic display mode: {e_fallback}",
                        file=sys.stderr,
                    )
                    raise RuntimeError("Failed to set GLUT display mode") from e_fallback
                # cls._multisample_supported = False # Verify later

            # DO NOT initialize GLContext here - it needs a window context first
            # DO NOT request context version here - needs to be done *before* glutCreateWindow

            cls._initialized = True  # Mark GLUT itself as initialized

    def _setup_camera(self):
        """Set up the camera based on the scene bounds."""
        center = self.bounding_box.center
        diagonal = self.bounding_box.diagonal

        # Safely access center components (ensuring they're not infinite)
        cx = center[0] if not np.isinf(center[0]) else 0.0
        cy = center[1] if not np.isinf(center[1]) else 0.0
        cz = center[2] if not np.isinf(center[2]) else 0.0
        center_vec = glm.vec3(cx, cy, cz)

        # Define the desired direction from the center
        direction_vec = glm.normalize(glm.vec3(1.0, -1.0, 1.0))

        # Position camera at a reasonable distance along the direction vector from the center
        # We add the direction vector scaled by distance to the center
        self.camera_pos = center_vec + direction_vec * diagonal * 1.5
        self.camera_target = center_vec  # Set target to scene center

        # Look back at the center of the scene
        # The front vector is the direction from the camera *to* the center
        # self.camera_front = glm.normalize(center_vec - self.camera_pos)
        # Or simply: self.camera_front = -direction_vec
        self.camera_front = glm.normalize(self.camera_target - self.camera_pos)

        # Set the camera up vector to align with the world Z-axis
        self.camera_up = glm.vec3(0.0, 0.0, 1.0)

        # Update the camera speed based on the scene size
        self.camera_speed = diagonal * 0.01

    def _create_window(self):
        """Create the OpenGL window."""
        # Initialize GLUT if needed
        Viewer._init_glut()

        # Create window
        glut.glutInitWindowSize(self.width, self.height)
        self.window_id = glut.glutCreateWindow(self.title.encode())
        if PYOPENGL_VERBOSE:
            print(f"Viewer: Created main window with ID: {self.window_id}")

        # Make sure we're operating on the correct window
        glut.glutSetWindow(self.window_id)

        # Register callbacks
        glut.glutDisplayFunc(self._display_callback)
        if PYOPENGL_VERBOSE:
            print(f"Viewer: Registered display callback for window ID: {self.window_id}")

        # Register other callbacks with logging
        if PYOPENGL_VERBOSE:
            print(f"Viewer: Registering reshape callback for window ID: {self.window_id}")
        glut.glutReshapeFunc(self._reshape_callback)

        if PYOPENGL_VERBOSE:
            print(f"Viewer: Registering mouse callbacks for window ID: {self.window_id}")
        glut.glutMouseFunc(self._mouse_callback)
        glut.glutMotionFunc(self._motion_callback)
        try:
            glut.glutMouseWheelFunc(self._wheel_callback)
        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Error registering mouse wheel callback: {e}")

        if PYOPENGL_VERBOSE:
            print(f"Viewer: Registering keyboard callback for window ID: {self.window_id}")
        glut.glutKeyboardFunc(self._keyboard_callback)

        # Immediate redisplay to ensure display callback is triggered
        if PYOPENGL_VERBOSE:
            print(f"Viewer: Forcing redisplay for window ID: {self.window_id}")
        glut.glutPostRedisplay()

    def _setup_gl(self):
        """Set up OpenGL state and compile shaders."""
        # Make sure we're operating on our window
        if self.window_id != glut.glutGetWindow() and self.window_id is not None:
            glut.glutSetWindow(self.window_id)

        self._query_actual_msaa_samples()

        # Enable depth testing
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Configure backface culling
        gl.glCullFace(gl.GL_BACK)
        if self.backface_culling:
            gl.glEnable(gl.GL_CULL_FACE)
        else:
            gl.glDisable(gl.GL_CULL_FACE)

        # Configure polygon mode (wireframe or fill)
        if self.wireframe_mode:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        # Configure antialiasing (multisample)
        # Check if GL_MULTISAMPLE is defined in the current OpenGL module
        if hasattr(gl, "GL_MULTISAMPLE") and Viewer._multisample_supported:
            if self.antialiasing_enabled:
                gl.glEnable(gl.GL_MULTISAMPLE)
            else:
                gl.glDisable(gl.GL_MULTISAMPLE)
        elif self.antialiasing_enabled:  # Only warn if trying to enable when not supported
            if PYOPENGL_VERBOSE:
                print(
                    "Viewer: GL_MULTISAMPLE not defined or supported by driver/GLUT. Antialiasing disabled."
                )
            self.antialiasing_enabled = False  # Force disable if not supported

        # Always enable color material mode
        try:
            gl.glEnable(gl.GL_COLOR_MATERIAL)
            gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        except Exception:
            pass

        # Set up simple lighting for better visibility
        if self.gl_ctx.has_legacy_lighting:
            try:
                # Enable lighting
                gl.glEnable(gl.GL_LIGHTING)
                gl.glEnable(gl.GL_LIGHT0)

                # Increase ambient light intensity for better visibility
                ambient_light = [0.6, 0.6, 0.6, 1.0]  # Very bright ambient
                diffuse_light = [0.8, 0.8, 0.8, 1.0]  # Strong diffuse
                specular_light = [0.5, 0.5, 0.5, 1.0]  # Moderate specular

                gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, ambient_light)
                gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, diffuse_light)
                gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, specular_light)

                # Position light above and in front of the scene
                light_position = [0.0, 5.0, 10.0, 1.0]
                gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_position)

                # Set up material properties to ensure colors are visible
                material_ambient = [0.6, 0.6, 0.6, 1.0]  # High ambient reflection
                material_diffuse = [0.8, 0.8, 0.8, 1.0]  # High diffuse reflection
                material_specular = [0.4, 0.4, 0.4, 1.0]  # Moderate specular
                material_shininess = [20.0]  # Low shininess

                gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT, material_ambient)
                gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE, material_diffuse)
                gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR, material_specular)
                gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SHININESS, material_shininess)

                # Global ambient light for better overall illumination
                global_ambient = [0.4, 0.4, 0.4, 1.0]
                gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT, global_ambient)

                # Make sure colors are used directly without recalculation
                # Crucial for ensuring vertex colors show up properly
                gl.glEnable(gl.GL_COLOR_MATERIAL)
                gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)

            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Failed to set up basic lighting: {e}")

        # Flag to control shader usage
        self.use_shaders = True

        # Try compiling shaders - prioritize this for modern rendering
        if self.gl_ctx.has_shader:
            try:
                # First try the more advanced shader
                if PYOPENGL_VERBOSE:
                    print("Viewer._setup_gl: Attempting to compile main shader...")
                self.shader_program = self._compile_shaders()
                if PYOPENGL_VERBOSE:
                    print(f"Viewer._setup_gl: _compile_shaders returned: {self.shader_program}")

                if not self.shader_program:
                    # If the main shader fails, try the basic shader
                    if PYOPENGL_VERBOSE:
                        print("Viewer._setup_gl: Main shader failed, trying basic shader...")
                    self.shader_program = self._compile_basic_shader()
                    if PYOPENGL_VERBOSE:
                        print(
                            f"Viewer._setup_gl: _compile_basic_shader returned: {self.shader_program}"
                        )

                # Add verification logging here:
                if self.shader_program:
                    # Verify the shader program is valid
                    if isinstance(self.shader_program, int) and self.shader_program > 0:
                        # Test the shader program by trying to use it
                        try:
                            gl.glUseProgram(self.shader_program)
                            # If no error, it's a valid program
                            gl.glUseProgram(0)
                            if PYOPENGL_VERBOSE:
                                print(
                                    f"Viewer: Successfully verified shader program: {self.shader_program}"
                                )
                        except Exception as e:
                            if PYOPENGL_VERBOSE:
                                print(
                                    f"Viewer: Shader program {self.shader_program} failed validation test: {e}"
                                )
                            # Delete the invalid program and set to None
                            try:
                                gl.glDeleteProgram(self.shader_program)
                            except Exception:
                                pass
                            self.shader_program = None
                    else:
                        if PYOPENGL_VERBOSE:
                            print(f"Viewer: Invalid shader program value: {self.shader_program}")
                        self.shader_program = None

                if self.shader_program:
                    if PYOPENGL_VERBOSE:
                        print(
                            f"Viewer: Successfully compiled and verified shader program: {self.shader_program}"
                        )
                else:
                    if PYOPENGL_VERBOSE:
                        print("Viewer: All shader compilation attempts failed")
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Shader compilation failed: {e}")
                self.shader_program = None

    def _check_shader_program(self, program_id=None):
        """Check the status and validity of a shader program.

        Args:
            program_id: Optional program ID to check, defaults to self.shader_program

        Returns:
            bool: True if the program is valid, False otherwise
        """
        if program_id is None:
            program_id = self.shader_program

        if not program_id or not isinstance(program_id, int) or program_id <= 0:
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Invalid shader program ID: {program_id}")
            return False

        try:
            # Clear any existing errors
            gl.glGetError()

            # Check if program exists
            is_program = gl.glIsProgram(program_id)
            if not is_program:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: ID {program_id} is not a valid shader program")
                return False

            # Get program info
            link_status = gl.glGetProgramiv(program_id, gl.GL_LINK_STATUS)
            if not link_status:
                info_log = gl.glGetProgramInfoLog(program_id)
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Shader program {program_id} is not linked: {info_log}")
                return False

            # Validate the program
            gl.glValidateProgram(program_id)
            validate_status = gl.glGetProgramiv(program_id, gl.GL_VALIDATE_STATUS)
            info_log = gl.glGetProgramInfoLog(program_id)

            if not validate_status:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Shader program {program_id} validation failed: {info_log}")
                return False

            # Test program usage
            try:
                gl.glUseProgram(program_id)
                error = gl.glGetError()
                gl.glUseProgram(0)

                if error != gl.GL_NO_ERROR:
                    if PYOPENGL_VERBOSE:
                        print(f"Viewer: Error using shader program {program_id}: {error}")
                    return False
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Exception using shader program {program_id}: {e}")
                return False

            # If we got here, the program is valid
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Shader program {program_id} is valid and usable")
                if info_log:
                    print(f"Viewer: Validation info: {info_log}")
            return True

        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Error checking shader program {program_id}: {e}")
            return False

    def _toggle_and_diagnose_shader(self):
        """Toggle the shader on/off and diagnose any issues."""
        # Toggle shader state
        self.use_shaders = not self.use_shaders

        if PYOPENGL_VERBOSE:
            if self.use_shaders:
                print("Viewer: Shader-based rendering enabled")
            else:
                print("Viewer: Shader-based rendering disabled")

        # Diagnose shader status if enabled
        if self.use_shaders and self.shader_program:
            if not self._check_shader_program(self.shader_program):
                if PYOPENGL_VERBOSE:
                    print("Viewer: Using immediate mode rendering due to shader issues")
                self.use_shaders = False

        # Request redisplay
        if self.window_id:
            glut.glutPostRedisplay()

    def _compile_shaders(self):
        """Compile and link the shader program.

        Returns:
            bool: True if shader compilation and linking was successful, False otherwise.
        """
        if not self.gl_ctx.has_shader:
            return None

        return SHADER_PROGRAM.compile()

    def _compile_basic_shader(self):
        """Compile a very minimal shader program for maximum compatibility.

        Returns:
            int: Shader program ID if successful, None otherwise.
        """
        return BASIC_SHADER_PROGRAM.compile()

    def run_without_ctrlc_handler(self):
        """Start the main rendering loop."""
        glut.glutMainLoop()

    def run(self):
        """Run the viewer with an interrupt handler."""
        stop: list = []

        def keep_alive(val):
            if stop:
                Viewer.terminate()
            else:
                glut.glutTimerFunc(200, keep_alive, 0)

        glut.glutTimerFunc(200, keep_alive, 0)

        # Define a handler for Ctrl+C to properly terminate
        def signal_handler(sig, frame):
            print("\nCtrl+C detected, safely shutting down...")
            stop.append(True)

            # Register signal handler

        signal.signal(signal.SIGINT, signal_handler)

        try:
            # Start the main loop
            print(Viewer.VIEWER_HELP_TEXT)
            self.run_without_ctrlc_handler()
        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\nSafely shutting down...")
        finally:
            # Ensure proper cleanup
            Viewer.terminate()

    @staticmethod
    def terminate():
        """Safely terminate all viewers and clean up GLUT resources.

        Call this method to properly exit the application and clean up all resources.
        """
        # Make a copy of the instances dictionary to avoid modification during iteration
        instances = list(Viewer._instances.values())

        # Close all viewer instances
        for viewer in instances:
            viewer.close()

        # Check if GLContext still has a temp window to clean up
        gl_ctx = GLContext.get_instance()
        if gl_ctx.temp_window_id is not None:
            try:
                if PYOPENGL_VERBOSE:
                    print(
                        f"Viewer: Final cleanup of GLContext temp window ID: {gl_ctx.temp_window_id}"
                    )
                glut.glutDestroyWindow(gl_ctx.temp_window_id)
                gl_ctx.temp_window_id = None
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Error in final cleanup of GLContext temp window: {e}")

        # Reset initialization flags
        Viewer._initialized = False
        gl_ctx.is_initialized = False

        # Exit GLUT if it's running
        try:
            if PYOPENGL_VERBOSE:
                print("Viewer: Exiting GLUT main loop")
            glut.glutLeaveMainLoop()
        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Error exiting GLUT main loop: {e}")
            # As a fallback, try to exit more abruptly
            try:
                glut.glutExit()
            except Exception:
                pass

    def _render_absolute_fallback(self):
        """An absolute fallback rendering mode that should work on any OpenGL version.

        This uses the simplest possible immediate mode rendering with minimal features.
        """
        try:
            # Clear all errors before starting
            while gl.glGetError() != gl.GL_NO_ERROR:
                pass  # Clear error queue

            # Attempt to set up very basic rendering state
            try:
                gl.glDisable(gl.GL_LIGHTING)
            except Exception:
                pass

            # Set up a very simple projection - even if more complex methods fail
            try:
                gl.glMatrixMode(gl.GL_PROJECTION)
                gl.glLoadIdentity()
                gl.glOrtho(-10, 10, -10, 10, -100, 100)

                gl.glMatrixMode(gl.GL_MODELVIEW)
                gl.glLoadIdentity()
            except Exception:
                # If even this fails, we're on a severely limited OpenGL
                if PYOPENGL_VERBOSE:
                    print("Viewer: Unable to set up even basic projection matrix")

            # Draw a simple wireframe cube as a last resort
            try:
                # Draw in white
                gl.glColor3f(1.0, 1.0, 1.0)

                # Front face
                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(-1.0, -1.0, 1.0)
                gl.glVertex3f(1.0, -1.0, 1.0)
                gl.glEnd()

                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(1.0, -1.0, 1.0)
                gl.glVertex3f(1.0, 1.0, 1.0)
                gl.glEnd()

                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(1.0, 1.0, 1.0)
                gl.glVertex3f(-1.0, 1.0, 1.0)
                gl.glEnd()

                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(-1.0, 1.0, 1.0)
                gl.glVertex3f(-1.0, -1.0, 1.0)
                gl.glEnd()

                # Back face
                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(-1.0, -1.0, -1.0)
                gl.glVertex3f(1.0, -1.0, -1.0)
                gl.glEnd()

                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(1.0, -1.0, -1.0)
                gl.glVertex3f(1.0, 1.0, -1.0)
                gl.glEnd()

                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(1.0, 1.0, -1.0)
                gl.glVertex3f(-1.0, 1.0, -1.0)
                gl.glEnd()

                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(-1.0, 1.0, -1.0)
                gl.glVertex3f(-1.0, -1.0, -1.0)
                gl.glEnd()

                # Connecting edges
                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(-1.0, -1.0, 1.0)
                gl.glVertex3f(-1.0, -1.0, -1.0)
                gl.glEnd()

                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(1.0, -1.0, 1.0)
                gl.glVertex3f(1.0, -1.0, -1.0)
                gl.glEnd()

                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(1.0, 1.0, 1.0)
                gl.glVertex3f(1.0, 1.0, -1.0)
                gl.glEnd()

                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(-1.0, 1.0, 1.0)
                gl.glVertex3f(-1.0, 1.0, -1.0)
                gl.glEnd()

            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Absolute fallback rendering failed: {str(e)}")

                # Display information in the console as last resort
                print("\nOpenGL rendering is not working correctly on this system.")
                print("Please check your graphics drivers and OpenGL installation.")
                print(f"OpenGL version: {self.gl_ctx.opengl_version}")
                print("Models in scene: ", len(self.models))

        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Critical rendering failure: {str(e)}")
            # Nothing more we can do here

    def _reshape_callback(self, width, height):
        """GLUT window reshape callback."""
        self.width = width
        self.height = height
        gl.glViewport(0, 0, width, height)

        # Redisplay to update the viewport
        glut.glutPostRedisplay()

    def _is_core_profile(self):
        """Check if we're running in a core profile where immediate mode is unavailable."""
        try:
            # Try a basic immediate mode operation - will fail in core profiles
            gl.glBegin(gl.GL_POINTS)
            gl.glEnd()
            # If we got here, we're in a compatibility profile
            return False
        except Exception:
            # If we got an exception, we're likely in a core profile
            return True

    def _render_core_profile_fallback(self):
        """Render a bare minimum triangle using only core profile features."""
        try:
            if not self.shader_program:
                # Try to compile a basic shader
                self.shader_program = self._compile_basic_shader()
                if not self.shader_program:
                    if PYOPENGL_VERBOSE:
                        print("Failed to create a shader program for core profile rendering")
                    return

            # Use our shader program
            gl.glUseProgram(self.shader_program)

            # Create a VAO
            vao_temp = gl.glGenVertexArrays(1)
            if isinstance(vao_temp, np.ndarray):
                vao = int(vao_temp[0])
            else:
                vao = vao_temp

            # Bind the VAO
            gl.glBindVertexArray(vao)

            # Create a VBO
            vbo_temp = gl.glGenBuffers(1)
            if isinstance(vbo_temp, np.ndarray):
                vbo = int(vbo_temp[0])
            else:
                vbo = vbo_temp

            # Bind the VBO and upload data
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)

            # Define a simple colored triangle
            vertices = np.array(
                [
                    # positions (3 floats per vertex)   # colors (4 floats per vertex)
                    0.0,
                    0.5,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,  # top - red
                    -0.5,
                    -0.5,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,  # bottom left - green
                    0.5,
                    -0.5,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,  # bottom right - blue
                ],
                dtype=np.float32,
            )

            # Upload the data
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

            # Set up vertex attributes
            # Position attribute
            position_loc = gl.glGetAttribLocation(self.shader_program, "position")
            if position_loc == -1:
                position_loc = 0  # Default to attribute 0 if not found by name

            gl.glVertexAttribPointer(position_loc, 3, gl.GL_FLOAT, gl.GL_FALSE, 7 * 4, None)
            gl.glEnableVertexAttribArray(position_loc)

            # Color attribute
            color_loc = gl.glGetAttribLocation(self.shader_program, "color")
            if color_loc == -1:
                color_loc = 1  # Default to attribute 1 if not found by name

            gl.glVertexAttribPointer(
                color_loc, 4, gl.GL_FLOAT, gl.GL_FALSE, 7 * 4, ctypes.c_void_p(3 * 4)
            )
            gl.glEnableVertexAttribArray(color_loc)

            # Set up a model-view-projection matrix for our basic shader
            mvp_loc = gl.glGetUniformLocation(self.shader_program, "modelViewProj")
            if mvp_loc != -1:
                # Create a simple model-view-projection matrix
                model = glm.mat4(1.0)  # Identity matrix
                view = glm.lookAt(
                    glm.vec3(0.0, 0.0, 3.0),  # Camera position
                    glm.vec3(0.0, 0.0, 0.0),  # Look at origin
                    glm.vec3(0.0, 1.0, 0.0),  # Up vector
                )
                projection = self.get_projection_mat()
                mvp = projection * view * model

                # Set the uniform
                gl.glUniformMatrix4fv(mvp_loc, 1, gl.GL_FALSE, glm.value_ptr(mvp))

            # Draw the triangle
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

            # Clean up
            gl.glBindVertexArray(0)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glUseProgram(0)

            # Delete the temporary objects
            gl.glDeleteVertexArrays(1, [vao])
            gl.glDeleteBuffers(1, [vbo])

        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Core profile fallback rendering failed: {str(e)}")
                # Try to clean up safely
                try:
                    gl.glBindVertexArray(0)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
                    gl.glUseProgram(0)
                except Exception:
                    pass

    def _display_callback(self):
        """GLUT display callback."""
        try:
            if PYOPENGL_VERBOSE and False:
                print(f"Viewer: Display callback executed for window ID: {self.window_id}")

            # Clear the color and depth buffers
            gl.glClearColor(
                *self.background_color
            )  # Use the background color stored in the instance
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            # Check if we're in a core profile
            is_core = self._is_core_profile()
            if is_core and PYOPENGL_VERBOSE:
                print("Detected OpenGL core profile - immediate mode not available")

            # Track whether any models rendered successfully
            rendering_success = False

            try:
                # Set up view
                self._setup_view()

                # When using coalesced models, they're already sorted with transparent last
                if not self.use_coalesced_models:
                    # Sort models - opaque first, then transparent
                    opaque_models = [model for model in self.models if not model.has_alpha_lt1]
                    transparent_models = [model for model in self.models if model.has_alpha_lt1]
                else:
                    # If we're using coalesced models, they're already properly sorted
                    # The first model is opaque, and if there's a second one, it's transparent
                    opaque_models = []
                    transparent_models = []

                    if len(self.models) > 0:
                        # First model is always opaque when coalesced
                        opaque_models = [self.models[0]]

                        # Second model (if exists) contains all transparent triangles
                        if len(self.models) > 1:
                            transparent_models = [self.models[1]]

                # Check if we can use shader-based rendering
                using_shader = False
                if self.use_shaders and self.gl_ctx.has_shader and self.shader_program:
                    # Verify shader program is valid
                    if isinstance(self.shader_program, int) and self.shader_program > 0:
                        using_shader = True
                    else:
                        if PYOPENGL_VERBOSE:
                            print(f"Viewer: Invalid shader program value: {self.shader_program}")
                        self.shader_program = None

                # Check if we should use Z-buffer occlusion
                if self.zbuffer_occlusion and self.wireframe_mode and self.backface_culling:
                    # First pass: Fill the Z-buffer but don't show geometry
                    gl.glColorMask(gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE)
                    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

                    # Use shader for first pass if available
                    if using_shader:
                        try:
                            # Clear any previous errors
                            gl.glGetError()
                            gl.glUseProgram(self.shader_program)
                            # Check for errors
                            error = gl.glGetError()
                            if error != gl.GL_NO_ERROR:
                                if PYOPENGL_VERBOSE:
                                    print(f"Viewer: Error using shader program: {error}")
                                using_shader = False
                                gl.glUseProgram(0)
                        except Exception as e:
                            if PYOPENGL_VERBOSE:
                                print(f"Viewer: Failed to use shader program: {e}")
                            using_shader = False

                    # Draw all models to fill the Z-buffer
                    for model in self.models:
                        model.draw()

                    # Second pass: Draw wireframe using the filled Z-buffer
                    gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
                    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

                    # Important: Adjust the depth test function to make wireframes visible
                    # This allows fragments with the same depth value as those already in the buffer
                    gl.glDepthFunc(gl.GL_LEQUAL)

                    # Enable polygon offset for lines
                    gl.glEnable(gl.GL_POLYGON_OFFSET_LINE)
                    gl.glPolygonOffset(
                        -1.0, -1.0
                    )  # This helps to pull the wireframe slightly forward

                    # Apply a tiny Z offset by adjusting the projection matrix
                    # This is done in projection/view space so it's camera-relative, not model-relative
                    gl.glMatrixMode(gl.GL_PROJECTION)
                    gl.glPushMatrix()

                    # Draw all models
                    for model in self.models:
                        model.draw()
                        rendering_success = True

                    # Restore the original projection matrix
                    gl.glPopMatrix()
                    gl.glMatrixMode(gl.GL_MODELVIEW)  # Restore matrix mode

                    # Restore original depth function
                    gl.glDepthFunc(gl.GL_LESS)

                    # Disable polygon offset
                    gl.glDisable(gl.GL_POLYGON_OFFSET_LINE)

                    # Clear shader program if we were using it
                    if using_shader:
                        try:
                            gl.glUseProgram(0)
                        except Exception as e:
                            if PYOPENGL_VERBOSE:
                                print(f"Viewer: Error disabling shader program: {e}")
                else:
                    # Regular rendering: Draw opaque models first
                    # Use shader if available
                    if using_shader:
                        try:
                            # Clear any previous errors
                            gl.glGetError()
                            gl.glUseProgram(self.shader_program)
                            # Check for errors
                            error = gl.glGetError()
                            if error != gl.GL_NO_ERROR:
                                if PYOPENGL_VERBOSE:
                                    print(f"Viewer: Error using shader program: {error}")
                                using_shader = False
                                gl.glUseProgram(0)
                            else:
                                # Add extra lighting info to shader
                                try:
                                    light_pos_loc = gl.glGetUniformLocation(
                                        self.shader_program, "lightPos"
                                    )
                                    if light_pos_loc != -1:
                                        # Position light relative to camera
                                        gl.glUniform3f(
                                            light_pos_loc,
                                            self.camera_pos.x + 5.0,
                                            self.camera_pos.y + 5.0,
                                            self.camera_pos.z + 10.0,
                                        )
                                except Exception as e:
                                    if PYOPENGL_VERBOSE:
                                        print(f"Viewer: Error setting shader uniforms: {e}")
                        except Exception as e:
                            if PYOPENGL_VERBOSE:
                                print(f"Viewer: Failed to use shader program: {e}")
                            using_shader = False

                    # Draw all opaque models
                    for model in opaque_models:
                        model.draw()
                        rendering_success = True

                    # For transparent models, we need proper depth testing but shouldn't update the z-buffer
                    if transparent_models:
                        try:
                            # Enable depth testing but don't write to depth buffer
                            gl.glDepthMask(gl.GL_FALSE)

                            # Draw transparent models after opaque ones
                            for model in transparent_models:
                                model.draw()
                                rendering_success = True

                            # Restore depth mask
                            gl.glDepthMask(gl.GL_TRUE)
                        except Exception as e:
                            if PYOPENGL_VERBOSE:
                                print(f"Viewer: Error during transparent rendering: {e}")
                            # Fallback - just render transparent models normally
                            for model in transparent_models:
                                model.draw()
                                rendering_success = True

                    # Turn off shader program after use
                    if using_shader:
                        try:
                            gl.glUseProgram(0)
                        except Exception as e:
                            if PYOPENGL_VERBOSE:
                                print(f"Viewer: Error disabling shader program: {e}")

                # Draw bounding box if enabled (always use immediate mode)
                self._draw_bounding_box()

            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Error during normal rendering: {str(e)}")
                rendering_success = False

            # If normal rendering failed, use appropriate fallback
            if not rendering_success:
                if PYOPENGL_VERBOSE:
                    print("Viewer: Normal rendering failed, using fallback")

                if is_core:
                    # Use special core profile fallback
                    self._render_core_profile_fallback()
                else:
                    # Use the regular fallback
                    self._render_absolute_fallback()

            # Draw axes on top before swapping
            if self.show_axes:
                self.axes_renderer.draw(self)

            # Swap buffers
            glut.glutSwapBuffers()

        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Critical error in display callback: {str(e)}")
            # Try a final fallback
            try:
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

                if self._is_core_profile():
                    self._render_core_profile_fallback()
                else:
                    self._render_absolute_fallback()

                glut.glutSwapBuffers()
            except Exception:
                # At this point there's not much more we can do
                pass

    def get_projection_mat(self) -> glm.mat4:
        """Get the current projection matrix based on the projection mode."""
        # Prevent division by zero if height is zero
        aspect_ratio = self.width / self.height if self.height > 0 else 1.0

        if self.projection_mode == "perspective":
            projection = glm.perspective(glm.radians(45.0), aspect_ratio, 0.1, 1000.0)
        else:  # orthographic
            if aspect_ratio >= 1.0:  # Wider than tall or square
                ortho_width = self.ortho_scale
                ortho_height = self.ortho_scale / aspect_ratio
            else:  # Taller than wide
                ortho_height = self.ortho_scale
                ortho_width = self.ortho_scale * aspect_ratio

            projection = glm.ortho(
                -ortho_width / 2.0,
                ortho_width / 2.0,
                -ortho_height / 2.0,
                ortho_height / 2.0,
                0.1,
                1000.0,  # Use same near/far as perspective for consistency
            )
        return projection

    def get_model_mat(self) -> glm.mat4:
        return glm.mat4(*self.model_matrix.flatten())

    def get_view_mat(self) -> glm.mat4:
        return glm.lookAt(self.camera_pos, self.camera_target, self.camera_up)

    def _setup_view(self):
        """Set up the view transformation for rendering."""
        # Set up lighting - this is crucial for seeing colors
        if self.gl_ctx.has_legacy_lighting:
            try:
                # Make sure lighting is enabled (redundant but important)
                gl.glEnable(gl.GL_LIGHTING)
                gl.glEnable(gl.GL_LIGHT0)

                # Position light relative to camera for consistent lighting
                light_position = [
                    self.camera_pos.x,
                    self.camera_pos.y + 5.0,
                    self.camera_pos.z + 10.0,
                    1.0,
                ]
                gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_position)

                # Make sure color material is enabled
                gl.glEnable(gl.GL_COLOR_MATERIAL)
                gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Error setting up lighting in view: {e}")

        # Check if we're in a core profile
        try:
            # Try a basic immediate mode operation - will fail in core profiles
            gl.glBegin(gl.GL_POINTS)
            gl.glVertex3f(0, 0, 0)
            gl.glEnd()
            is_core_profile = False
        except Exception:
            is_core_profile = True
            # if PYOPENGL_VERBOSE: # Reduced verbosity
            #     print("Viewer: Detected core profile in setup_view")

        # Get Projection Matrix
        projection = self.get_projection_mat()

        # Set up matrices for Shader pipeline
        if self.gl_ctx.has_shader and self.shader_program:
            try:
                gl.glUseProgram(self.shader_program)

                # Update view position for specular highlights
                view_pos_loc = gl.glGetUniformLocation(self.shader_program, "viewPos")
                if view_pos_loc != -1:
                    gl.glUniform3f(
                        view_pos_loc, self.camera_pos.x, self.camera_pos.y, self.camera_pos.z
                    )

                # Set up model-view-projection matrices
                model_mat = self.get_model_mat()
                view = self.get_view_mat()
                # projection is calculated above using get_projection_mat()

                # Send matrices to the shader
                model_loc = gl.glGetUniformLocation(self.shader_program, "model")
                view_loc = gl.glGetUniformLocation(self.shader_program, "view")
                proj_loc = gl.glGetUniformLocation(self.shader_program, "projection")

                if model_loc != -1:
                    gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, glm.value_ptr(model_mat))
                if view_loc != -1:
                    gl.glUniformMatrix4fv(view_loc, 1, gl.GL_FALSE, glm.value_ptr(view))
                # Use the matrix obtained from get_projection_mat()
                if proj_loc != -1 and projection is not None:
                    gl.glUniformMatrix4fv(proj_loc, 1, gl.GL_FALSE, glm.value_ptr(projection))

                # Check for combined MVP matrix (used in basic shader)
                mvp_loc = gl.glGetUniformLocation(self.shader_program, "modelViewProj")
                # Use the matrix obtained from get_projection_mat()
                if mvp_loc != -1 and projection is not None:
                    mvp = (
                        projection * view * model_mat
                    )  # projection comes from get_projection_mat()
                    gl.glUniformMatrix4fv(mvp_loc, 1, gl.GL_FALSE, glm.value_ptr(mvp))

                gl.glUseProgram(0)
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Error setting up shader uniforms: {e}")
                try:
                    gl.glUseProgram(0)
                except:
                    pass

        # Set up matrices for Fixed-Function pipeline
        if not is_core_profile:
            try:
                # Set up projection matrix based on mode (using legacy calls)
                gl.glMatrixMode(gl.GL_PROJECTION)
                gl.glLoadIdentity()
                if self.projection_mode == "perspective":
                    aspect_ratio = self.width / self.height if self.height > 0 else 1.0
                    glu.gluPerspective(45.0, aspect_ratio, 0.1, 1000.0)
                else:  # orthographic
                    aspect_ratio = self.width / self.height if self.height > 0 else 1.0
                    if aspect_ratio >= 1.0:
                        ortho_width = self.ortho_scale
                        ortho_height = self.ortho_scale / aspect_ratio
                    else:
                        ortho_height = self.ortho_scale
                        ortho_width = self.ortho_scale * aspect_ratio
                    gl.glOrtho(
                        -ortho_width / 2.0,
                        ortho_width / 2.0,
                        -ortho_height / 2.0,
                        ortho_height / 2.0,
                        0.1,
                        1000.0,
                    )

                # Set up modelview matrix
                gl.glMatrixMode(gl.GL_MODELVIEW)
                gl.glLoadIdentity()

                # Set up view with gluLookAt
                glu.gluLookAt(
                    self.camera_pos.x,
                    self.camera_pos.y,
                    self.camera_pos.z,
                    self.camera_target.x,
                    self.camera_target.y,
                    self.camera_target.z,  # Use camera_target
                    self.camera_up.x,
                    self.camera_up.y,
                    self.camera_up.z,
                )

                # Apply model matrix
                gl.glMultMatrixf(self.model_matrix.flatten())
            except Exception as e:
                # Core profile with no shaders and no fixed function pipeline
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Failed to set up legacy view matrix: {e}")

    def _mouse_callback(self, button, state, x, y):
        """GLUT mouse button callback."""
        if button == glut.GLUT_LEFT_BUTTON:
            if state == glut.GLUT_DOWN:
                self.left_button_pressed = True
                self.mouse_start_x = x
                self.mouse_start_y = y
                # Store initial point for trackball rotation
                self.trackball_start_point = self._map_to_sphere(x, y)
            elif state == glut.GLUT_UP:
                self.left_button_pressed = False
        elif button == glut.GLUT_RIGHT_BUTTON:
            if state == glut.GLUT_DOWN:
                self.right_button_pressed = True
                self.mouse_start_x = x
                self.mouse_start_y = y
            elif state == glut.GLUT_UP:
                self.right_button_pressed = False
        elif button == glut.GLUT_MIDDLE_BUTTON:
            if state == glut.GLUT_DOWN:
                self.middle_button_pressed = True
                self.mouse_start_x = x
                self.mouse_start_y = y
            elif state == glut.GLUT_UP:
                self.middle_button_pressed = False
        else:
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Unknown mouse button: {button}")

    def relative_distance_from_screen_centre(self, x, y) -> tuple[float, float, float]:
        # Get the window dimensions
        width = glut.glutGet(glut.GLUT_WINDOW_WIDTH)
        height = glut.glutGet(glut.GLUT_WINDOW_HEIGHT)
        rel_x = 2 * (x / width - 0.5)
        rel_y = 2 * (0.5 - y / height)

        rel_len = np.sqrt(rel_x * rel_x + rel_y * rel_y)

        return rel_x, rel_y, rel_len

    def _motion_callback(self, x, y):
        """GLUT mouse motion callback."""
        dx = x - self.mouse_start_x
        dy = y - self.mouse_start_y
        # We don't reset mouse_start_x/y here for trackball,
        # we always compare current x,y to the initial press point.
        # However, we will update the trackball_start_point for continuous rotation.
        
        if self.middle_button_pressed \
            or (self.right_button_pressed and self.left_button_pressed):
            # Zooming - middle button or right button and left button pressed
            pan_dx = x - self.mouse_start_x
            pan_dy = y - self.mouse_start_y
            self.mouse_start_x = x
            self.mouse_start_y = y
            direction = 1 if pan_dy > 0 else -1
            direction *= 0.5
            self._wheel_callback(0, direction, x, y)

        elif self.left_button_pressed:
            # Trackball Rotation
            p1 = self.trackball_start_point
            p2 = self._map_to_sphere(x, y)

            # Calculate rotation only if points are different enough
            if glm.length(p1 - p2) > 0.001:  # Threshold
                try:
                    # Calculate axis in view space
                    view_axis = glm.normalize(glm.cross(p1, p2))
                    angle = glm.acos(glm.clamp(glm.dot(p1, p2), -1.0, 1.0)) * 2.0  # Scale angle

                    # Prevent NaN/tiny angles
                    if np.isnan(angle) or angle < 1e-6:
                        world_rotation = glm.mat4(1.0)  # No rotation
                    else:
                        # Transform axis from view space to world space
                        inv_view_mat = glm.inverse(self.get_view_mat())
                        world_axis = glm.normalize(
                            glm.vec3(inv_view_mat * glm.vec4(view_axis, 0.0))
                        )
                        # Negate the angle to invert rotation direction
                        world_rotation = glm.rotate(glm.mat4(1.0), -angle, world_axis)

                    # Rotate camera position around target in world space
                    target_to_cam = self.camera_pos - self.camera_target
                    new_target_to_cam = glm.vec3(world_rotation * glm.vec4(target_to_cam, 0.0))
                    self.camera_pos = self.camera_target + new_target_to_cam

                    # Rotate camera up vector in world space
                    self.camera_up = glm.normalize(
                        glm.vec3(world_rotation * glm.vec4(self.camera_up, 0.0))
                    )

                    # Update camera front vector (always points from pos to target)
                    self.camera_front = glm.normalize(self.camera_target - self.camera_pos)

                    # Update the starting point for the next motion event
                    self.trackball_start_point = p2

                    glut.glutPostRedisplay()
                except (ValueError, RuntimeWarning) as e:
                    if PYOPENGL_VERBOSE:
                        print(f"Viewer: Trackball rotation error: {e}")

        elif self.right_button_pressed:
            # Panning (moves camera_pos and camera_target together)
            # Reset start coords for panning on each motion event
            pan_dx = x - self.mouse_start_x
            pan_dy = y - self.mouse_start_y
            self.mouse_start_x = x
            self.mouse_start_y = y

            pan_speed = self.camera_speed * 0.5  # Adjust sensitivity as needed

            # Calculate camera right vector (orthogonal to front and up)
            try:
                camera_right = glm.normalize(glm.cross(self.camera_front, self.camera_up))
            except ValueError:
                # If front is parallel to up, choose an arbitrary right vector
                world_x = glm.vec3(1.0, 0.0, 0.0)
                if abs(glm.dot(self.camera_front, world_x)) < 0.99:
                    camera_right = glm.normalize(glm.cross(self.camera_front, world_x))
                else:
                    world_y = glm.vec3(0.0, 1.0, 0.0)
                    camera_right = glm.normalize(glm.cross(self.camera_front, world_y))

            # Calculate the *actual* up direction for panning (cross product of right and front)
            # This ensures panning moves parallel to the view plane even after trackball rotation.
            pan_up = glm.normalize(glm.cross(camera_right, self.camera_front))

            # Note the sign adjustments for typical panning feel (dx moves right, dy moves up SCREEN)
            # Panning up/down the screen should move along the calculated pan_up vector.
            delta = (-camera_right * pan_dx * pan_speed) + (pan_up * pan_dy * pan_speed)

            self.camera_pos += delta
            self.camera_target += delta

            # Trigger redraw for continuous panning
            glut.glutPostRedisplay()

    def _wheel_callback(self, wheel, direction, x, y):
        """GLUT mouse wheel callback."""
        if self.projection_mode == "perspective":
            # Zoom in/out by changing camera position along the front vector towards/away from target
            # Calculate distance to target
            dist = glm.length(self.camera_pos - self.camera_target)
            # Calculate new distance (ensure it doesn't go negative or too small)
            new_dist = max(0.1, dist - (direction * self.camera_speed * 10.0))
            # Update position along the front vector relative to the target
            # Use -camera_front because camera_front points *towards* target
            self.camera_pos = self.camera_target - self.camera_front * new_dist

            # Alternative: Just move along front vector (simpler, might lose target focus)
            # self.camera_pos += self.camera_front * (direction * self.camera_speed * 10.0)
        else:  # orthographic
            # Zoom by changing the ortho scale (adjust sensitivity as needed)
            zoom_factor = 1.1
            if direction > 0:  # Zoom in
                self.ortho_scale /= zoom_factor
            else:  # Zoom out
                self.ortho_scale *= zoom_factor
            # Prevent scale from becoming too small or negative
            self.ortho_scale = max(0.1, self.ortho_scale)
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Ortho scale set to {self.ortho_scale}")

        glut.glutPostRedisplay()

    def _keyboard_callback(self, key, x, y):
        """GLUT keyboard callback."""
        func = KEY_BINDINGS.get(key)
        if func:
            func(self)

    def _reset_view(self):
        """Reset camera and model transformations to defaults."""
        # Reset model matrix
        self.model_matrix = np.eye(4, dtype=np.float32)

        # Recompute bounds just in case models changed (though unlikely needed here)
        # self._compute_scene_bounds()

        # Reset camera position, target, and orientation based on current bounds
        self._setup_camera()

        # Reset ortho scale to default
        self.ortho_scale = 20.0

        # Reset mouse rotation tracking (no longer used for view matrix directly)
        # self.yaw = -90.0
        # self.pitch = 0.0

        if PYOPENGL_VERBOSE:
            print("Viewer: View reset.")  # Added log

    def _print_shader_debug(self, program_id):
        """Print detailed debugging information about a shader program.

        Args:
            program_id: The shader program ID to debug
        """
        print(f"\n===== SHADER PROGRAM {program_id} DEBUG =====")

        # First check if it's a valid program object
        try:
            is_program = gl.glIsProgram(program_id)
            print(f"Is a program object: {is_program}")

            if not is_program:
                print(f"OpenGL doesn't recognize {program_id} as a valid program object")
                return

            # Get program parameters
            try:
                delete_status = gl.glGetProgramiv(program_id, gl.GL_DELETE_STATUS)
                link_status = gl.glGetProgramiv(program_id, gl.GL_LINK_STATUS)
                validate_status = gl.glGetProgramiv(program_id, gl.GL_VALIDATE_STATUS)
                info_log = gl.glGetProgramInfoLog(program_id)

                print(f"  Delete Status: {delete_status}")
                print(f"  Link Status: {link_status}")
                print(f"  Validate Status: {validate_status}")
                print(f"  Info Log: {info_log}")

                # Count and list active attributes
                num_attribs = gl.glGetProgramiv(program_id, gl.GL_ACTIVE_ATTRIBUTES)
                print(f"  Active Attributes: {num_attribs}")
                for i in range(num_attribs):
                    try:
                        attrib_info = gl.glGetActiveAttrib(program_id, i)
                        name = attrib_info[0].decode()
                        size = attrib_info[1]
                        type_enum = attrib_info[2]
                        location = gl.glGetAttribLocation(program_id, name.encode())
                        print(f"    {name}: location={location}, size={size}, type={type_enum}")
                    except Exception as e:
                        print(f"    Error getting attribute {i}: {e}")

                # Count and list active uniforms
                num_uniforms = gl.glGetProgramiv(program_id, gl.GL_ACTIVE_UNIFORMS)
                print(f"  Active Uniforms: {num_uniforms}")
                for i in range(num_uniforms):
                    try:
                        uniform_info = gl.glGetActiveUniform(program_id, i)
                        name = uniform_info[0].decode()
                        size = uniform_info[1]
                        type_enum = uniform_info[2]
                        location = gl.glGetUniformLocation(program_id, name.encode())
                        print(f"    {name}: location={location}, size={size}, type={type_enum}")
                    except Exception as e:
                        print(f"    Error getting uniform {i}: {e}")

            except Exception as e:
                print(f"  Error getting program parameters: {e}")

            # Try to test program usage
            try:
                # Save current program
                previous_program = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)

                # Try to use this program
                gl.glUseProgram(program_id)
                error = gl.glGetError()

                if error == gl.GL_NO_ERROR:
                    print("  Successfully activated program")
                else:
                    print(f"  Error activating program: {error}")

                # Restore previous program
                gl.glUseProgram(previous_program)

            except Exception as e:
                print(f"  Error testing program usage: {e}")
                try:
                    gl.glUseProgram(0)  # Reset to default program
                except Exception:
                    pass

        except Exception as e:
            print(f"Error debugging shader program: {e}")

        print("=====================================\n")

    def _print_diagnostics(self):
        """Print detailed diagnostic information about OpenGL and shader state."""
        print("\n===== OPENGL DIAGNOSTICS =====")

        # Show OpenGL version info
        print(f"OpenGL Version: {self.gl_ctx.opengl_version}")
        print(f"GLSL Version: {self.gl_ctx.glsl_version}")

        # Show OpenGL capabilities
        print("\nOpenGL Capabilities:")
        print(f"  VBO Support: {self.gl_ctx.has_vbo}")
        print(f"  Shader Support: {self.gl_ctx.has_shader}")
        print(f"  VAO Support: {self.gl_ctx.has_vao}")
        print(f"  Legacy Lighting: {self.gl_ctx.has_legacy_lighting}")
        print(f"  Legacy Vertex Arrays: {self.gl_ctx.has_legacy_vertex_arrays}")

        # Show current state
        print("\nCurrent Viewer State:")
        print(f"  Shader Program: {self.shader_program}")
        print(f"  Current Program: {gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)}")
        print(f"  Using Shaders: {self.use_shaders}")
        print(f"  Wireframe Mode: {self.wireframe_mode}")
        print(f"  Backface Culling: {self.backface_culling}")
        print(f"  Bounding Box Mode: {self.bounding_box_mode}")
        print(f"  Z-Buffer Occlusion: {self.zbuffer_occlusion}")
        print(f"  Coalesced Models: {self.use_coalesced_models}")

        # Additional details about the shader program if available
        if self.shader_program:
            print("\nShader Program Details:")
            try:
                # Test shader validation
                is_valid = self._check_shader_program(self.shader_program)
                print(f"  Shader Program Valid: {is_valid}")

                # Get active uniforms
                try:
                    num_uniforms = gl.glGetProgramiv(self.shader_program, gl.GL_ACTIVE_UNIFORMS)
                    print(f"  Active Uniforms: {num_uniforms}")

                    for i in range(num_uniforms):
                        try:
                            uniform_info = gl.glGetActiveUniform(self.shader_program, i)
                            name = uniform_info[0].decode()
                            size = uniform_info[1]
                            type_enum = uniform_info[2]
                            print(f"    {name} (size={size}, type={type_enum})")
                        except Exception as e:
                            print(f"    Error getting uniform {i}: {e}")
                except Exception as e:
                    print(f"  Error getting uniforms: {e}")

                # Get active attributes
                try:
                    num_attribs = gl.glGetProgramiv(self.shader_program, gl.GL_ACTIVE_ATTRIBUTES)
                    print(f"  Active Attributes: {num_attribs}")

                    for i in range(num_attribs):
                        try:
                            attrib_info = gl.glGetActiveAttrib(self.shader_program, i)
                            name = attrib_info[0].decode()
                            size = attrib_info[1]
                            type_enum = attrib_info[2]
                            print(f"    {name} (size={size}, type={type_enum})")
                        except Exception as e:
                            print(f"    Error getting attribute {i}: {e}")
                except Exception as e:
                    print(f"  Error getting attributes: {e}")

            except Exception as e:
                print(f"  Error getting shader details: {e}")

        # Current model information
        print("\nModel Information:")
        print(f"  Total Models: {len(self.models)}")
        for i, model in enumerate(self.models):
            print(f"  Model {i}:")
            print(f"    Points: {model.num_points}")
            print(f"    Transparent: {model.has_alpha_lt1}")
            print(f"    Has VAO: {model.vao is not None}")
            print(f"    Has VBO: {model.vbo is not None}")

        # Check current shader programs
        try:
            print("\nChecking shader programs 1-10:")
            for i in range(1, 11):
                try:
                    is_prog = gl.glIsProgram(i)
                    print(f"  Program {i}: {is_prog}")
                except Exception as e:
                    print(f"  Program {i}: Error: {e}")
        except Exception as e:
            print(f"Error checking programs: {e}")

        print("==============================\n")

    def save_screenshot(self, filename: str):
        """Save the current window contents as a PNG image.

        Args:
            filename: Path where the image should be saved
        """
        try:
            # Make sure we're operating on our window
            if self.window_id != glut.glutGetWindow() and self.window_id is not None:
                glut.glutSetWindow(self.window_id)

            # Get the window dimensions
            width = glut.glutGet(glut.GLUT_WINDOW_WIDTH)
            height = glut.glutGet(glut.GLUT_WINDOW_HEIGHT)

            # Read the pixels from the current buffer
            buffer = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

            # Convert the buffer to a numpy array and flip it vertically
            # (OpenGL reads from bottom-left, but images are typically stored top-left)
            image = np.frombuffer(buffer, dtype=np.uint8)
            image = image.reshape((height, width, 3))
            image = np.flipud(image)

            # Save the image using PIL
            from PIL import Image

            img = Image.fromarray(image)
            img.save(filename, "PNG")

            if PYOPENGL_VERBOSE:
                print(f"Viewer: Screenshot saved to {filename}")

        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Failed to save screenshot: {str(e)}")
            raise

    def _draw_bounding_box(self):
        """Draw the scene bounding box in the current mode (off/wireframe/solid)."""
        BBoxRender.render(self)

    def toggle_coalesced_mode(self):
        """Toggle between using coalesced models and original models."""
        self.use_coalesced_models = not self.use_coalesced_models

        # Update models with the new setting
        self.update_models(self.original_models)

        if PYOPENGL_VERBOSE:
            if self.use_coalesced_models:
                print("Viewer: Coalesced model mode enabled")
            else:
                print("Viewer: Using original models (coalesced mode disabled)")

        # Request redisplay
        if self.window_id:
            glut.glutPostRedisplay()

    def num_triangles(self) -> int:
        return sum((m.num_triangles() for m in self.models))

    def update_models(self, models: List[Model]):
        """Update the models displayed by the viewer and re-initialize GL resources."""
        # Clean up old model resources first
        for old_model in self.models:
            old_model.delete()

        self.original_models = models

        # Create new set of models (coalesced or original)
        if self.use_coalesced_models and models:
            opaque_model, transparent_model = Model.create_coalesced_models(models)
            self.models = []
            if opaque_model.num_points > 0:
                self.models.append(opaque_model)
            if transparent_model.num_points > 0:
                self.models.append(transparent_model)
        else:
            self.models = models

        # Recompute scene bounds and reset camera
        self._compute_scene_bounds()
        self._reset_view()

        # Initialize GL resources for the new models
        if self.window_id and self.window_id == glut.glutGetWindow():
            if PYOPENGL_VERBOSE:
                print(
                    f"Viewer: Initializing GL resources for updated models in window {self.window_id}"
                )
            for model in self.models:
                try:
                    model.initialize_gl_resources()
                except Exception as e:
                    if PYOPENGL_VERBOSE:
                        print(f"Viewer: Error initializing GL resources for an updated model: {e}")
        elif PYOPENGL_VERBOSE:
            print(
                "Viewer: Warning - Cannot initialize updated model GL resources. Window ID mismatch or invalid window."
            )

        # Request redisplay
        if self.window_id:
            glut.glutPostRedisplay()

    def get_current_window_dims(self) -> Tuple[int, int]:
        """Get the current dimensions (width, height) of the viewer window."""
        try:
            # Ensure we query the correct window if multiple exist (though unlikely with current structure)
            current_win = glut.glutGetWindow()
            if current_win != self.window_id:
                glut.glutSetWindow(self.window_id)

            width = glut.glutGet(glut.GLUT_WINDOW_WIDTH)
            height = glut.glutGet(glut.GLUT_WINDOW_HEIGHT)

            # Restore previous window context if changed
            if current_win != self.window_id and current_win != 0:
                glut.glutSetWindow(current_win)

            return (width, height)
        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Error getting window dimensions: {e}")
            return (self.width, self.height)  # Fallback to stored dimensions

    # Proposed helper function to be added inside the Viewer class

    def _map_to_sphere(self, x: int, y: int) -> glm.vec3:
        """Map window coordinates (x, y) to a point on a virtual unit sphere."""
        # Normalize x, y to range [-1, 1] with origin at center
        # Note: OpenGL window coordinates often have (0,0) at top-left,
        # while sphere mapping typically assumes bottom-left or center origin.
        # We need to adjust based on how GLUT reports coordinates.
        # Assuming GLUT's y=0 is top, we invert y.
        win_x = (1.0 * x / self.width) - 1.0
        win_y = 1.0 - (1.0 * y / self.height)  # Invert y for bottom-left origin

        # Calculate squared distance from center
        dist_sq = win_x * win_x + win_y * win_y

        if dist_sq <= 1.0:
            # Point is inside the sphere's projection
            win_z = np.sqrt(1.0 - dist_sq)
        else:
            # Point is outside; map to the edge of the sphere
            norm = np.sqrt(dist_sq)
            win_x /= norm
            win_y /= norm
            win_z = 0.0

        return glm.vec3(win_x, win_y, win_z)

    def _save_numpy_buffer_to_png(self, buffer: np.ndarray, width: int, height: int, filename: str):
        """Saves a numpy pixel buffer (RGB, uint8) to a PNG file."""
        try:
            # Image needs flipping vertically (OpenGL reads bottom-left)
            image = np.flipud(buffer.reshape((height, width, 3)))

            # Save the image using PIL
            from PIL import Image

            img = Image.fromarray(image)
            img.save(filename, "PNG")

            if PYOPENGL_VERBOSE:
                print(f"Viewer: Image saved to {filename}")

        except ImportError:
            # Added file=sys.stderr for error messages
            print(
                "Error: PIL (Pillow) library is required to save images. Please install it (`pip install Pillow`).",
                file=sys.stderr,
            )
            raise  # Re-raise to indicate failure
        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Failed to save image buffer: {str(e)}", file=sys.stderr)
            raise  # Re-raise to indicate failure

    def offscreen_render(self, filename: str):
        """Renders the model to an offscreen buffer and saves the screen as a PNG image."""
        if self.window_id is None or not Viewer._initialized:
            raise RuntimeError(
                "Viewer window is not initialized. Offscreen rendering requires an active OpenGL context."
            )

        # Ensure the viewer's context is current
        original_window = 0
        try:
            original_window = glut.glutGetWindow()
            if original_window != self.window_id:
                glut.glutSetWindow(self.window_id)
        except Exception as e:
            raise RuntimeError(f"Failed to set OpenGL context for offscreen rendering: {e}")

        fbo = None
        color_rbo = None
        depth_rbo = None
        original_viewport = None  # Initialize to avoid reference before assignment warning

        try:
            # FBO Setup
            # Check for errors before starting
            gl.glGetError()

            fbo = gl.glGenFramebuffers(1)
            if gl.glGetError() != gl.GL_NO_ERROR or not fbo:
                raise RuntimeError("Failed to generate Framebuffer Object (FBO).")
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
            if gl.glGetError() != gl.GL_NO_ERROR:
                raise RuntimeError("Failed to bind FBO.")

            # Color Renderbuffer
            color_rbo = gl.glGenRenderbuffers(1)
            if gl.glGetError() != gl.GL_NO_ERROR or not color_rbo:
                raise RuntimeError("Failed to generate color Renderbuffer Object (RBO).")
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, color_rbo)
            if gl.glGetError() != gl.GL_NO_ERROR:
                raise RuntimeError("Failed to bind color RBO.")
            # Use GL_RGB8 for color, GL_DEPTH_COMPONENT24 for depth (common choices)
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGB8, self.width, self.height)
            if gl.glGetError() != gl.GL_NO_ERROR:
                raise RuntimeError("Failed to allocate color RBO storage.")
            gl.glFramebufferRenderbuffer(
                gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, color_rbo
            )
            if gl.glGetError() != gl.GL_NO_ERROR:
                raise RuntimeError("Failed to attach color RBO to FBO.")

            # Depth Renderbuffer
            depth_rbo = gl.glGenRenderbuffers(1)
            if gl.glGetError() != gl.GL_NO_ERROR or not depth_rbo:
                raise RuntimeError("Failed to generate depth RBO.")
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_rbo)
            if gl.glGetError() != gl.GL_NO_ERROR:
                raise RuntimeError("Failed to bind depth RBO.")
            gl.glRenderbufferStorage(
                gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, self.width, self.height
            )
            if gl.glGetError() != gl.GL_NO_ERROR:
                raise RuntimeError("Failed to allocate depth RBO storage.")
            gl.glFramebufferRenderbuffer(
                gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_rbo
            )
            if gl.glGetError() != gl.GL_NO_ERROR:
                raise RuntimeError("Failed to attach depth RBO to FBO.")

            # Unbind the RBO to avoid accidental modification
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

            # Check FBO status
            status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
            if status != gl.GL_FRAMEBUFFER_COMPLETE:
                # Map common status codes to strings for better error messages
                status_map = {
                    gl.GL_FRAMEBUFFER_UNDEFINED: "GL_FRAMEBUFFER_UNDEFINED",
                    gl.GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT",
                    gl.GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT",
                    gl.GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER: "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER",
                    gl.GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER: "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER",
                    gl.GL_FRAMEBUFFER_UNSUPPORTED: "GL_FRAMEBUFFER_UNSUPPORTED",
                    gl.GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE: "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE",
                    # GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS might not be defined in older PyOpenGL
                    # 36065: "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS",
                }
                raise RuntimeError(
                    f"Framebuffer is not complete: Status {status_map.get(status, status)}"
                )

            # --- Rendering to FBO ---
            original_viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
            gl.glViewport(0, 0, self.width, self.height)
            if gl.glGetError() != gl.GL_NO_ERROR:
                print("Warning: Error setting viewport for FBO.", file=sys.stderr)

            # Clear the FBO buffers
            gl.glClearColor(*self.background_color)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            if gl.glGetError() != gl.GL_NO_ERROR:
                print("Warning: Error clearing FBO buffers.", file=sys.stderr)

            # --- Replicate core rendering logic from _display_callback ---
            # NOTE: This duplicates logic. A refactor of _display_callback is recommended.
            try:
                self._setup_view()  # Set up projection and view matrices

                # Determine models to render based on coalesce setting
                if not self.use_coalesced_models:
                    opaque_models = [
                        model for model in self.original_models if not model.has_alpha_lt1
                    ]
                    transparent_models = [
                        model for model in self.original_models if model.has_alpha_lt1
                    ]
                    # Create temporary coalesced models just for sorting if needed (less efficient)
                    if opaque_models or transparent_models:
                        temp_opaque, temp_transparent = Model.create_coalesced_models(
                            opaque_models + transparent_models
                        )
                        # This approach might be complex, sticking to the coalesced/original logic for now
                        # Reverting to simpler logic based on self.models which ARE already coalesced or not
                        if not self.use_coalesced_models:
                            opaque_models = [
                                model for model in self.models if not model.has_alpha_lt1
                            ]
                            transparent_models = [
                                model for model in self.models if model.has_alpha_lt1
                            ]
                            # Need to sort transparent models if not coalesced
                            # This requires Z positions which are not readily available without recalculation
                            # Sticking to the assumption that self.models is correct for now
                        else:
                            opaque_models = [self.models[0]] if len(self.models) > 0 else []
                            transparent_models = [self.models[1]] if len(self.models) > 1 else []

                else:  # Already using coalesced models
                    opaque_models = [self.models[0]] if len(self.models) > 0 else []
                    transparent_models = [self.models[1]] if len(self.models) > 1 else []

                # Shader usage check
                using_shader = False
                active_program = 0
                if self.use_shaders and self.gl_ctx.has_shader and self.shader_program:
                    if isinstance(self.shader_program, int) and self.shader_program > 0:
                        # Further check if the program is valid before using
                        if self._check_shader_program(self.shader_program):
                            using_shader = True
                            active_program = self.shader_program

                # Set polygon mode, culling, color material based on viewer state
                if self.wireframe_mode:
                    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                else:
                    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

                if self.backface_culling:
                    gl.glEnable(gl.GL_CULL_FACE)
                    gl.glCullFace(gl.GL_BACK)
                else:
                    gl.glDisable(gl.GL_CULL_FACE)

                try:  # Ensure color material is enabled for fixed-function or basic shaders
                    gl.glEnable(gl.GL_COLOR_MATERIAL)
                    gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
                except Exception:
                    pass  # Ignore if not available

                # Render opaque models
                if using_shader:
                    try:
                        gl.glUseProgram(active_program)
                        # Uniforms (lightPos, viewPos) should be set if using the main shader
                        if active_program == self.shader_program:  # Check if it's the main shader
                            light_pos_loc = gl.glGetUniformLocation(active_program, "lightPos")
                            view_pos_loc = gl.glGetUniformLocation(active_program, "viewPos")
                            if light_pos_loc != -1:
                                gl.glUniform3f(
                                    light_pos_loc,
                                    self.camera_pos.x + 5.0,
                                    self.camera_pos.y + 5.0,
                                    self.camera_pos.z + 10.0,
                                )
                            if view_pos_loc != -1:
                                gl.glUniform3f(
                                    view_pos_loc,
                                    self.camera_pos.x,
                                    self.camera_pos.y,
                                    self.camera_pos.z,
                                )
                    except Exception as e:
                        if PYOPENGL_VERBOSE:
                            print(
                                f"Viewer (offscreen): Error setting shader uniforms: {e}",
                                file=sys.stderr,
                            )
                        gl.glUseProgram(0)  # Fallback if uniforms fail
                        active_program = 0

                for model in opaque_models:
                    # Pass the active shader program to draw, if any
                    model.draw()  # Model.draw now handles using the current program

                # Render transparent models
                if transparent_models:
                    try:
                        gl.glDepthMask(gl.GL_FALSE)  # Don't write to depth buffer
                        # Blending is enabled within model.draw if needed
                        for model in transparent_models:
                            model.draw()  # Pass active shader
                        gl.glDepthMask(gl.GL_TRUE)  # Restore depth writes
                    except Exception as e:
                        if PYOPENGL_VERBOSE:
                            print(
                                f"Viewer (offscreen): Error during transparent rendering: {e}",
                                file=sys.stderr,
                            )
                        gl.glDepthMask(gl.GL_TRUE)  # Ensure depth mask is restored on error
                        # Fallback rendering without depth mask modification
                        for model in transparent_models:
                            model.draw()

                # Unbind shader if it was used
                if active_program != 0:
                    try:
                        gl.glUseProgram(0)
                    except Exception:
                        pass

                # Draw bounding box if enabled (uses immediate mode)
                self._draw_bounding_box()

                # Draw axes (uses immediate mode)
                if self.show_axes:
                    self.axes_renderer.draw(self)

            except Exception as render_err:
                # Handle rendering errors specifically
                print(
                    f"Viewer (offscreen): Error during scene rendering: {render_err}",
                    file=sys.stderr,
                )
                # Continue to cleanup if possible, but report the error
                # We might still have partial results in the buffer

            # --- Read Pixels ---
            # Ensure reading from the correct FBO attachment
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
            if gl.glGetError() != gl.GL_NO_ERROR:
                print("Warning: Error setting read buffer to FBO attachment.", file=sys.stderr)

            buffer = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            read_error = gl.glGetError()
            if read_error != gl.GL_NO_ERROR:
                raise RuntimeError(f"Failed to read pixels from FBO: OpenGL Error {read_error}")
            if buffer is None:
                raise RuntimeError("glReadPixels returned None.")

            pixel_data = np.frombuffer(buffer, dtype=np.uint8)
            # Check if the buffer size matches expected size
            expected_size = self.width * self.height * 3
            if pixel_data.size != expected_size:
                print(
                    f"Warning: Read pixel buffer size ({pixel_data.size}) does not match expected size ({expected_size}). Image may be incorrect.",
                    file=sys.stderr,
                )
                # Attempt to reshape anyway, might fail if size is wrong
                # pixel_data = pixel_data[:expected_size] # Truncate/Pad? Risky.

            # --- Save Image ---
            # Check size again before saving
            if pixel_data.size == expected_size:
                self._save_numpy_buffer_to_png(pixel_data, self.width, self.height, filename)
            else:
                raise RuntimeError(
                    f"Cannot save image due to incorrect pixel buffer size. Expected {expected_size}, got {pixel_data.size}."
                )

        except Exception as e:
            print(f"Error during offscreen rendering: {e}", file=sys.stderr)
            # Re-raise the exception after attempting cleanup
            raise
        finally:
            # --- Cleanup ---
            # Unbind FBO and restore default framebuffer (0)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

            # Delete FBO and renderbuffers if they were created
            if fbo is not None:
                # Check if fbo is still a valid framebuffer before deleting
                # try:
                #      if gl.glIsFramebuffer(fbo):
                #           gl.glDeleteFramebuffers(1, [fbo])
                # except Exception: pass # Ignore cleanup errors
                gl.glDeleteFramebuffers(1, [fbo])  # Simpler deletion
            if color_rbo is not None:
                # try:
                #      if gl.glIsRenderbuffer(color_rbo):
                #           gl.glDeleteRenderbuffers(1, [color_rbo])
                # except Exception: pass
                gl.glDeleteRenderbuffers(1, [color_rbo])
            if depth_rbo is not None:
                # try:
                #      if gl.glIsRenderbuffer(depth_rbo):
                #           gl.glDeleteRenderbuffers(1, [depth_rbo])
                # except Exception: pass
                gl.glDeleteRenderbuffers(1, [depth_rbo])

            # Restore viewport
            if original_viewport is not None:
                gl.glViewport(*original_viewport)

            # Restore original window context if necessary
            if original_window != 0 and original_window != self.window_id:
                try:
                    # Only set if the original window ID is still valid (might have been destroyed)
                    # This check is difficult, simply trying might be best
                    glut.glutSetWindow(original_window)
                except Exception:
                    # This is expected if the original window (e.g. temp init window) was destroyed
                    pass

    def _query_actual_msaa_samples(self):
        """Queries OpenGL for the actual number of samples and sample buffers."""
        if not self.window_id or not Viewer._initialized:
            if PYOPENGL_VERBOSE:
                print("Viewer: Cannot query MSAA samples, window/GLUT not ready.")
            return
        # Get the current window before potentially changing it
        original_window = glut.glutGetWindow()
        context_changed = False
        if self.window_id != original_window:
            # Try setting the context, but don't proceed if it fails
            try:
                glut.glutSetWindow(self.window_id)
                context_changed = True
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Error setting window context for MSAA query: {e}")
                # Try to restore original context if possible
                if original_window != 0:
                    try:
                        glut.glutSetWindow(original_window)
                    except:
                        pass
                return

        try:
            self._actual_msaa_samples = 0  # Default if query fails
            sample_buffers = 0
            samples = 0

            # Check if multisampling is reported as available by sample buffers
            if hasattr(gl, "GL_SAMPLE_BUFFERS"):
                error_before = gl.glGetError()  # Clear errors before query
                sample_buffers = gl.glGetIntegerv(gl.GL_SAMPLE_BUFFERS)
                error_after_sb = gl.glGetError()  # Check errors after query
                if PYOPENGL_VERBOSE:
                    print(
                        f"Viewer: Queried GL_SAMPLE_BUFFERS = {sample_buffers} (Error before: {error_before}, after: {error_after_sb})"
                    )

                # If sample buffers exist, query the number of samples
                if sample_buffers > 0 and hasattr(gl, "GL_SAMPLES"):
                    samples = gl.glGetIntegerv(gl.GL_SAMPLES)
                    error_after_s = gl.glGetError()  # Check errors after query
                    if PYOPENGL_VERBOSE:
                        print(
                            f"Viewer: Queried GL_SAMPLES = {samples} (Error after: {error_after_s})"
                        )
                    self._actual_msaa_samples = samples
                    # Refine the static support flag based on actual query
                    # Only truly supported if we have buffers AND samples > 1
                    Viewer._multisample_supported = samples > 1
                elif sample_buffers == 0:
                    # If no sample buffers, confirm multisample is not supported
                    Viewer._multisample_supported = False
                    if PYOPENGL_VERBOSE:
                        print(
                            "Viewer: No sample buffers available, MSAA confirmed not supported for this context."
                        )

            else:
                # If GL_SAMPLE_BUFFERS doesn't exist, MSAA is likely not supported
                Viewer._multisample_supported = False
                if PYOPENGL_VERBOSE:
                    print("Viewer: GL_SAMPLE_BUFFERS not defined, assuming MSAA not supported.")

            # Update antialiasing_enabled state based on actual support
            # If MSAA isn't supported, force antialiasing_enabled to False
            if not Viewer._multisample_supported and self.antialiasing_enabled:
                if PYOPENGL_VERBOSE:
                    print("Viewer: Disabling antialiasing as queries indicate no MSAA support.")
                self.antialiasing_enabled = False

        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Error querying MSAA samples: {e}")
            # Ensure support flag is false if query failed
            Viewer._multisample_supported = False
            self.antialiasing_enabled = False  # Disable if query failed
            self._actual_msaa_samples = 0
        finally:
            # Restore original window context if it was changed
            if context_changed and original_window != 0:
                try:
                    glut.glutSetWindow(original_window)
                except Exception as e_restore:
                    if PYOPENGL_VERBOSE:
                        print(
                            f"Viewer: Error restoring original window context after MSAA query: {e_restore}"
                        )


KEY_BINDINGS: dict[bytes, Callable[[Viewer], None]] = {}


def keybinding(key: bytes):
    """Decorator to add a key binding to the viewer."""

    def decorator(func: Callable[[Viewer], None]):
        if key in KEY_BINDINGS:
            raise ValueError(f"Key binding {key} already exists.")
        KEY_BINDINGS[key] = func
        return func

    return decorator


@keybinding(b"\x1b")
def terminate(viewer: Viewer):
    """Terminate the application."""
    if PYOPENGL_VERBOSE:
        print("Viewer: ESC key pressed, terminating application")
    viewer.terminate()


@keybinding(b"r")
def reset_view(viewer: Viewer):
    """Reset the view."""
    viewer._reset_view()
    glut.glutPostRedisplay()
    if PYOPENGL_VERBOSE:
        print("Viewer: R key pressed, resetting view")


@keybinding(b"a")
def toggle_antialiasing(viewer: Viewer):
    """Toggle antialiasing."""
    # Toggle antialiasing
    if hasattr(gl, "GL_MULTISAMPLE") and Viewer._multisample_supported:
        viewer.antialiasing_enabled = not viewer.antialiasing_enabled
        if viewer.antialiasing_enabled:
            gl.glEnable(gl.GL_MULTISAMPLE)
            if PYOPENGL_VERBOSE:
                print("Viewer: Antialiasing enabled")
        else:
            gl.glDisable(gl.GL_MULTISAMPLE)
            if PYOPENGL_VERBOSE:
                print("Viewer: Antialiasing disabled")
        glut.glutPostRedisplay()
    elif PYOPENGL_VERBOSE:
        print("Viewer: Antialiasing (GL_MULTISAMPLE) not supported on this system.")


@keybinding(b"b")
def toggle_backface_culling(viewer: Viewer):
    # Toggle backface culling
    viewer.backface_culling = not viewer.backface_culling
    if viewer.backface_culling:
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        if PYOPENGL_VERBOSE:
            print("Viewer: Backface culling enabled")
    else:
        gl.glDisable(gl.GL_CULL_FACE)
        if PYOPENGL_VERBOSE:
            print("Viewer: Backface culling disabled")
    glut.glutPostRedisplay()


@keybinding(b"w")
def toggle_wireframe_mode(viewer: Viewer):
    """Toggle wireframe mode."""
    # Toggle wireframe mode
    viewer.wireframe_mode = not viewer.wireframe_mode
    if viewer.wireframe_mode:
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        if PYOPENGL_VERBOSE:
            print("Viewer: Wireframe mode enabled")
    else:
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        if PYOPENGL_VERBOSE:
            print("Viewer: Wireframe mode disabled")
    glut.glutPostRedisplay()


@keybinding(b"z")
def toggle_zbuffer_occlusion(viewer: Viewer):
    # Toggle Z-buffer occlusion for wireframes
    viewer.zbuffer_occlusion = not viewer.zbuffer_occlusion
    if PYOPENGL_VERBOSE:
        if viewer.zbuffer_occlusion:
            print("Viewer: Z-buffer occlusion enabled for wireframes")
        else:
            print("Viewer: Z-buffer occlusion disabled for wireframes")
    glut.glutPostRedisplay()


@keybinding(b"x")
def toggle_bounding_box_mode(viewer: Viewer):
    # Toggle bounding box mode
    viewer.bounding_box_mode = (viewer.bounding_box_mode + 1) % 3
    if PYOPENGL_VERBOSE:
        print(f"Viewer: Bounding box mode set to {viewer.bounding_box_mode}")
    glut.glutPostRedisplay()


@keybinding(b"c")
def toggle_coalesced_mode(viewer: Viewer):
    # Toggle coalesced model mode
    viewer.toggle_coalesced_mode()
    # Already calls glutPostRedisplay()


@keybinding(b"h")
def toggle_shader_based_rendering(viewer: Viewer):
    # Toggle shader-based rendering
    viewer._toggle_and_diagnose_shader()
    # Already calls glutPostRedisplay()


@keybinding(b"d")
def print_diagnostics(viewer: Viewer):
    # Print diagnostic information
    viewer._print_diagnostics()
    # No need to redisplay


@keybinding(b"p")
def print_detailed_shader_debug(viewer: Viewer):
    """Print detailed shader debug information."""
    # Print detailed shader program debug information
    try:
        # Get current program
        current_program = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
        print(f"Current program: {current_program}")

        # Debug our shader program
        if viewer.shader_program:
            viewer._print_shader_debug(viewer.shader_program)

        # Debug the special program '3'
        viewer._print_shader_debug(3)

    except Exception as e:
        print(f"Error during shader debugging: {e}")
    if PYOPENGL_VERBOSE:
        print("Viewer: P key pressed, printing detailed shader debug information")


@keybinding(b"s")
def save_screenshot(viewer: Viewer):
    """Save a screenshot."""
    try:
        # Generate a default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        # Save screenshot
        viewer.save_screenshot(filename)
        if PYOPENGL_VERBOSE:
            print(f"Viewer: Screenshot saved to {filename}")
    except Exception as e:
        print(f"Error during screenshot saving: {e}")
        if PYOPENGL_VERBOSE:
            print(f"Viewer: Failed to save screenshot: {str(e)}")


@keybinding(b"?")
def print_help(viewer: Viewer):
    print(Viewer.VIEWER_HELP_TEXT)


@keybinding(b"o")
def toggle_projection(viewer: Viewer):
    """Toggle between perspective and orthographic projection."""
    if viewer.projection_mode == "perspective":
        viewer.projection_mode = "orthographic"
        if PYOPENGL_VERBOSE:
            print("Viewer: Switched to Orthographic projection")
    else:
        viewer.projection_mode = "perspective"
        if PYOPENGL_VERBOSE:
            print("Viewer: Switched to Perspective projection")
    glut.glutPostRedisplay()


@datatree
class TestKeybindings:
    secs: float = dtfield(default=2)
    viewer: Viewer = dtfield(default=None, init=False)
    _cycle: Iterator = dtfield(default=None, init=False)
    skip_keys: Tuple[bytes, ...] = (b"*", b"&", b"\x1b")

    def do_cycle(self, viewer: Viewer):
        if self._cycle is None:
            self._cycle = iter(KEY_BINDINGS.items())
        self.viewer = viewer

    def run_next(self, val):
        if self._cycle is None:
            return
        try:
            while True:
                key, func = next(self._cycle)
                if key in self.skip_keys:
                    continue
                glut.glutTimerFunc(self.getGlutTime(), self.run_next, 0)
                print(f"Running '{str(key)}'")
                func(self.viewer)
                break
        except StopIteration:
            self._cycle = None

    def getGlutTime(self) -> int:
        return int(self.secs * 1000)


def run_test_keybindings(viewer: Viewer, secs: float = 2):
    print("Running a test for all the keybindings")
    test_keybindings = TestKeybindings(secs=secs)
    test_keybindings.do_cycle(viewer)
    glut.glutTimerFunc(test_keybindings.getGlutTime(), test_keybindings.run_next, 0)


@keybinding(b"*")
def run_test_keybindings_key_fast(viewer: Viewer):
    run_test_keybindings(viewer, secs=0.1)


@keybinding(b"&")
def run_test_keybindings_key_slow(viewer: Viewer):
    run_test_keybindings(viewer, secs=2)


@keybinding(b"+")
def toggle_axes_visibility(viewer: Viewer):
    """Toggle the visibility of the main axes lines."""
    viewer.show_axes = not viewer.show_axes
    if PYOPENGL_VERBOSE:
        print(f"Viewer: Axes visibility set to {viewer.show_axes}")
    glut.glutPostRedisplay()


@keybinding(b"g")
def toggle_graduation_ticks(viewer: Viewer):
    """Toggle the visibility of the graduation ticks on the axes."""
    viewer.axes_renderer.show_graduation_ticks = not viewer.axes_renderer.show_graduation_ticks
    if PYOPENGL_VERBOSE:
        print(
            f"Viewer: Graduation ticks visibility set to {viewer.axes_renderer.show_graduation_ticks}"
        )
    glut.glutPostRedisplay()


@keybinding(b"v")
def toggle_graduation_values(viewer: Viewer):
    """Toggle the visibility of the graduation values (text) on the axes."""
    viewer.axes_renderer.show_graduation_values = not viewer.axes_renderer.show_graduation_values
    if PYOPENGL_VERBOSE:
        print(
            f"Viewer: Graduation values visibility set to {viewer.axes_renderer.show_graduation_values}"
        )
    glut.glutPostRedisplay()


@keybinding(b"e")
def toggle_edge_effect_rotations(viewer: Viewer):
    """Toggles behaviour of rotations at the edge of the window."""
    viewer.edge_rotations = not viewer.edge_rotations
    if PYOPENGL_VERBOSE:
        print(f"Viewer: Edge rotations set or {viewer.edge_rotations}")
    glut.glutPostRedisplay()


# Helper function to create a viewer with models
def create_viewer_with_models(models, width=800, height=600, title="3D Viewer") -> Viewer:
    """Create and return a viewer with the given models."""
    viewer = Viewer(models, width, height, title)
    return viewer


def get_opengl_capabilities(gl_ctx: GLContext):
    """Get a dictionary describing the available OpenGL capabilities."""

    capabilities = {
        "opengl_available": True,
        "vbo_support": gl_ctx.has_vbo,
        "shader_support": gl_ctx.has_shader,
        "vao_support": gl_ctx.has_3_3,
    }

    if gl_ctx.has_3_3:
        capabilities["message"] = "Full modern OpenGL 3.3+ support available."
    elif gl_ctx.has_shader and gl_ctx.has_vbo:
        capabilities["message"] = "OpenGL with shaders and VBOs available, but no VAO support."
    elif gl_ctx.has_vbo:
        capabilities["message"] = "OpenGL with VBO support available, but no shader support."
    else:
        capabilities["message"] = "Basic OpenGL available, using legacy immediate mode."

    return capabilities


# If this module is run directly, show a simple demo
if __name__ == "__main__":
    # Enable verbose mode
    # PYOPENGL_VERBOSE = True

    # Initialize the OpenGL context
    gl_ctx = GLContext.get_instance()
    gl_ctx.initialize()

    # Print OpenGL capabilities
    capabilities = get_opengl_capabilities(gl_ctx)
    print(f"OpenGL capabilities: {capabilities['message']}")

    # Create our test models
    triangle = create_triangle_model()
    color_cube = create_colored_test_cube(2.0)  # Use our special test cube with bright colors

    # Create a viewer with both models - using just the color cube for testing colors
    viewer: Viewer = create_viewer_with_models(
        [color_cube, triangle], title="PyOpenSCAD Viewer Color Test"
    )

    viewer.run()
