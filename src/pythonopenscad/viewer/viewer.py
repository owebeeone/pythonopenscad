"""OpenGL 3D viewer for PyOpenSCAD models."""

import numpy as np
import ctypes
from typing import Any, Iterable, List, Optional, Tuple, Union, Dict, Callable, ClassVar
from datatrees import datatree, dtfield, Node
import warnings
import sys
import signal
from datetime import datetime
import manifold3d as m3d
from pythonopenscad.viewer.bbox import BoundingBox
from pythonopenscad.viewer.glctxt import GLContext, PYOPENGL_VERBOSE
from pythonopenscad.viewer.model import Model
from pythonopenscad.viewer.axes import AxesRenderer

import anchorscad_lib.linear as linear

# Try importing OpenGL libraries, but make them optional
try:
    import OpenGL.GL as gl
    import OpenGL.GLUT as glut
    import OpenGL.GLU as glu
    # Enable PyOpenGL's error checking
    OpenGL = sys.modules['OpenGL']
    OpenGL.ERROR_CHECKING = True
    OpenGL.ERROR_LOGGING = True
    # Ensure PyOpenGL allows the deprecated APIs
    OpenGL.FORWARD_COMPATIBLE_ONLY = False
    import glm
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False


@datatree
class Viewer:
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
    antialiasing_enabled: bool = True # Added antialiasing state
    
    background_color: Tuple[float, float, float, float] = (0.98, 0.98, 0.85, 1.0)
    
    axes_renderer_node: Node[AxesRenderer] = Node(AxesRenderer, prefix="axes_")
    axes_renderer: AxesRenderer = dtfield(self_default=lambda self: self.axes_renderer_node())

    # Shader source code - using GLSL 1.20 for broader compatibility
    VERTEX_SHADER = """
    #version 120
    
    attribute vec3 aPos;
    attribute vec4 aColor;
    attribute vec3 aNormal;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    varying vec3 FragPos;
    varying vec4 VertexColor;
    varying vec3 Normal;
    
    void main() {
        FragPos = vec3(model * vec4(aPos, 1.0));
        // Simple normal transformation - avoiding inverse which fails on some drivers
        Normal = normalize(mat3(model) * aNormal);
        VertexColor = aColor;
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
    """

    FRAGMENT_SHADER = """
    #version 120
    
    varying vec3 FragPos;
    varying vec4 VertexColor;
    varying vec3 Normal;
    
    uniform vec3 lightPos;
    uniform vec3 viewPos;
    
    void main() {
        // Ambient - increase to make colors more visible
        float ambientStrength = 0.5;  // Increased from 0.3
        vec3 ambient = ambientStrength * VertexColor.rgb;
        
        // Diffuse - increase strength
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * VertexColor.rgb * 0.8;  // More diffuse influence
        
        // Specular - keep the same
        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
        
        // Result - ensure colors are visible regardless of lighting
        vec3 result = ambient + diffuse + specular;
        
        // Add a minimum brightness to ensure visibility
        result = max(result, VertexColor.rgb * 0.4);
        
        // Preserve alpha from vertex color for transparent objects
        gl_FragColor = vec4(result, VertexColor.a);
    }
    """

    # Basic fallback shader for maximum compatibility
    BASIC_VERTEX_SHADER = """
    #version 110
    
    attribute vec3 position;
    attribute vec4 color;
    
    varying vec4 fragColor;
    
    uniform mat4 modelViewProj;
    
    void main() {
        // Pass the color directly to the fragment shader
        fragColor = color;
        
        // Transform the vertex position
        gl_Position = modelViewProj * vec4(position, 1.0);
        
        // Set point size for better visibility
        gl_PointSize = 5.0;
    }
    """

    BASIC_FRAGMENT_SHADER = """
    #version 110
    
    varying vec4 fragColor;
    
    void main() {
        // Use the interpolated color from the vertex shader
        gl_FragColor = fragColor;
    }
    """
    
    VIEWER_HELP_TEXT = """
    Mouse Controls:
     Left button drag: Rotate camera
     Right button drag: Pan camera
     Wheel: Zoom in/out
    
    Keyboard Controls:
     B - Toggle backface culling
     W - Toggle wireframe mode
     Z - Toggle Z-buffer occlusion for wireframes
     C - Toggle coalesced model mode (improves transparency rendering)
     H - Toggle shader-based rendering (modern vs. legacy mode)
     D - Print diagnostic information about OpenGL capabilities
     P - Print detailed shader program diagnostics
     R - Reset view
     X - Toggle bounding box (off/wireframe/solid)
     S - Save screenshot
     ESC - Close viewer
    """

    # Static window registry to handle GLUT callbacks
    _instances: ClassVar[Dict[int, 'Viewer']] = {}
    _initialized: ClassVar[bool] = False
    _next_id: ClassVar[int] = 0
    
            # OpenGL state
    window_id: int | None = dtfield(default=0, init=False)
    shader_program: Any | None = dtfield(default=None, init=False)
    # Flag to track if multisample is supported by the system
    _multisample_supported: bool = dtfield(default=False, init=False)

    def __post_init__(self):
        """
        Initialize the viewer with a list of models.
        
        Args:
            models: List of Model objects to display
            width: Window width
            height: Window height
            title: Window title
            use_coalesced_models: Whether to coalesce models into opaque/transparent pairs for better rendering
        """
        if not HAS_OPENGL:
            raise ImportError("OpenGL libraries (PyOpenGL and PyGLM) are required for the viewer")
        
        # Get the OpenGL context and capabilities
        self.gl_ctx = GLContext.get_instance()
        
        # Store original models
        self.original_models = self.models
        
        # Create coalesced models
        if self.models:
            # Create coalesced models (one opaque, one transparent)
            opaque_model, transparent_model = Model.create_coalesced_models(self.models)
            
            # Set models for rendering
            self.models = []
            # Add opaque model if it has data
            if opaque_model.num_points > 0:
                self.models.append(opaque_model)
            # Add transparent model if it has data
            if transparent_model.num_points > 0:
                self.models.append(transparent_model)
 
        
        # Camera parameters
        self.camera_pos = glm.vec3(0.0, 0.0, 5.0)
        self.camera_front = glm.vec3(0.0, 0.0, -1.0)
        self.camera_up = glm.vec3(0.0, 1.0, 0.0)
        self.camera_speed = 0.05
        self.yaw = -90.0
        self.pitch = 0.0
        
        # Model transformation matrix (for rotation) - using NumPy instead of GLM
        self.model_matrix = np.eye(4, dtype=np.float32)
        
        # Mouse state
        self.last_mouse_x = self.width // 2
        self.last_mouse_y = self.height // 2
        self.first_mouse = True
        self.left_button_pressed = False
        self.right_button_pressed = False
        self.mouse_start_x = 0
        self.mouse_start_y = 0
        
        # Compute bounding box and set up camera
        self._compute_scene_bounds()
        self._setup_camera()

        
        # Register this instance
        self.instance_id = Viewer._next_id
        Viewer._next_id += 1
        Viewer._instances[self.instance_id] = self
        
        # Create the window and set up OpenGL
        self._create_window()
        self._setup_gl()
        
        # Initialize GL resources for each model now that the main context is active
        # Ensure the correct window context is current first
        if self.window_id and self.window_id == glut.glutGetWindow():
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Initializing GL resources for {len(self.models)} models in window {self.window_id}")
            for model in self.models:
                try:
                    model.initialize_gl_resources()
                except Exception as e:
                    if PYOPENGL_VERBOSE:
                        print(f"Viewer: Error initializing GL resources for a model: {e}")
        elif PYOPENGL_VERBOSE:
            print(f"Viewer: Warning - Cannot initialize model GL resources. Window ID mismatch or invalid window.")
            print(f"  Expected Window ID: {self.window_id}, Current Window ID: {glut.glutGetWindow()}")
    
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
            if (not np.all(np.isinf(model.bounding_box.min_point)) and 
                not np.all(np.isinf(model.bounding_box.max_point))):
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
        """Initialize GLUT if not already initialized."""
        if not cls._initialized:
            if PYOPENGL_VERBOSE:
                print("Viewer: Initializing GLUT")
            # Initialize GLUT
            glut.glutInit()

            # Try to set a display mode with multisampling
            display_mode = glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH
            try:
                glut.glutInitDisplayMode(display_mode | glut.GLUT_MULTISAMPLE)
                # Assume multisample is supported if this doesn't raise an error
                cls._multisample_supported = True
                if PYOPENGL_VERBOSE:
                    print("Viewer: Requested multisample display mode.")
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Multisample display mode failed, falling back ({e}). Antialiasing might not work.")
                # Fallback to non-multisample mode
                try:
                    glut.glutInitDisplayMode(display_mode)
                except Exception as e_fallback:
                    if PYOPENGL_VERBOSE:
                         print(f"Viewer: Error setting even basic display mode: {e_fallback}")
                cls._multisample_supported = False # Mark as not supported


            # Initialize OpenGL context and detect capabilities
            if PYOPENGL_VERBOSE:
                print("Viewer: Getting GLContext instance")
            gl_ctx = GLContext.get_instance()
            gl_ctx.initialize()
            
            # Always prefer compatibility profile to ensure immediate mode works
            try:
                if PYOPENGL_VERBOSE:
                    print("Viewer: Requesting compatibility profile to ensure immediate mode functions work")
                # Request compatibility profile for legacy OpenGL
                gl_ctx.request_context_version(2, 1, core_profile=False)
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Error requesting OpenGL context version: {e}")
                # Continue with default OpenGL version
            
            cls._initialized = True
    
    def _setup_camera(self):
        """Set up the camera based on the scene bounds."""
        center = self.bounding_box.center
        diagonal = self.bounding_box.diagonal
        
        # Safely access center components (ensuring they're not infinite)
        cx = center[0] if not np.isinf(center[0]) else 0.0
        cy = center[1] if not np.isinf(center[1]) else 0.0
        cz = center[2] if not np.isinf(center[2]) else 0.0
        
        # Position camera at a reasonable distance
        self.camera_pos = glm.vec3(
            cx,
            cy,
            cz + diagonal * 1.5
        )
        
        # Look at the center of the scene
        self.camera_front = glm.normalize(
            glm.vec3(cx, cy, cz) - self.camera_pos
        )
        
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
        glut.glutMouseWheelFunc(self._wheel_callback)
        
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
        if hasattr(gl, 'GL_MULTISAMPLE') and Viewer._multisample_supported:
            if self.antialiasing_enabled:
                gl.glEnable(gl.GL_MULTISAMPLE)
            else:
                gl.glDisable(gl.GL_MULTISAMPLE)
        elif self.antialiasing_enabled: # Only warn if trying to enable when not supported
            if PYOPENGL_VERBOSE:
                 print("Viewer: GL_MULTISAMPLE not defined or supported by driver/GLUT. Antialiasing disabled.")
            self.antialiasing_enabled = False # Force disable if not supported

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
                material_ambient = [0.6, 0.6, 0.6, 1.0]   # High ambient reflection
                material_diffuse = [0.8, 0.8, 0.8, 1.0]   # High diffuse reflection
                material_specular = [0.4, 0.4, 0.4, 1.0]  # Moderate specular
                material_shininess = [20.0]               # Low shininess
                
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
                self.shader_program = self._compile_shaders()
                
                if not self.shader_program:
                    # If the main shader fails, try the basic shader
                    if PYOPENGL_VERBOSE:
                        print("Viewer: Main shader failed, trying basic shader")
                    self.shader_program = self._compile_basic_shader()
                
                if self.shader_program:
                    # Verify the shader program is valid
                    if isinstance(self.shader_program, int) and self.shader_program > 0:
                        # Test the shader program by trying to use it
                        try:
                            gl.glUseProgram(self.shader_program)
                            # If no error, it's a valid program
                            gl.glUseProgram(0)
                            if PYOPENGL_VERBOSE:
                                print(f"Viewer: Successfully verified shader program: {self.shader_program}")
                        except Exception as e:
                            if PYOPENGL_VERBOSE:
                                print(f"Viewer: Shader program {self.shader_program} failed validation test: {e}")
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
                        print(f"Viewer: Successfully compiled and verified shader program: {self.shader_program}")
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
            
        try:
            # Clear any previous shader-related errors
            gl.glGetError()
            
            # Create vertex shader
            vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
            if vertex_shader == 0:
                if PYOPENGL_VERBOSE:
                    print("Viewer: Failed to create vertex shader object")
                return None
                
            gl.glShaderSource(vertex_shader, self.VERTEX_SHADER)
            gl.glCompileShader(vertex_shader)
            
            # Check for vertex shader compilation errors
            compile_status = gl.glGetShaderiv(vertex_shader, gl.GL_COMPILE_STATUS)
            if not compile_status:
                error = gl.glGetShaderInfoLog(vertex_shader)
                gl.glDeleteShader(vertex_shader)
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Vertex shader compilation failed: {error}")
                return None
            
            # Create fragment shader
            fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
            if fragment_shader == 0:
                gl.glDeleteShader(vertex_shader)
                if PYOPENGL_VERBOSE:
                    print("Viewer: Failed to create fragment shader object")
                return None
                
            gl.glShaderSource(fragment_shader, self.FRAGMENT_SHADER)
            gl.glCompileShader(fragment_shader)
            
            # Check for fragment shader compilation errors
            compile_status = gl.glGetShaderiv(fragment_shader, gl.GL_COMPILE_STATUS)
            if not compile_status:
                error = gl.glGetShaderInfoLog(fragment_shader)
                gl.glDeleteShader(vertex_shader)
                gl.glDeleteShader(fragment_shader)
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Fragment shader compilation failed: {error}")
                return None
            
            # Create and link shader program
            program = gl.glCreateProgram()
            if program == 0:
                gl.glDeleteShader(vertex_shader)
                gl.glDeleteShader(fragment_shader)
                if PYOPENGL_VERBOSE:
                    print("Viewer: Failed to create shader program object")
                return None
                
            gl.glAttachShader(program, vertex_shader)
            gl.glAttachShader(program, fragment_shader)
            
            # Bind attribute locations for GLSL 1.20 (before linking)
            gl.glBindAttribLocation(program, 0, "aPos")
            gl.glBindAttribLocation(program, 1, "aColor")
            gl.glBindAttribLocation(program, 2, "aNormal")
            
            gl.glLinkProgram(program)
            
            # Check for linking errors
            link_status = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
            if not link_status:
                error = gl.glGetProgramInfoLog(program)
                gl.glDeleteShader(vertex_shader)
                gl.glDeleteShader(fragment_shader)
                gl.glDeleteProgram(program)
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Shader program linking failed: {error}")
                return None
            
            # Delete shaders (they're not needed after linking)
            gl.glDeleteShader(vertex_shader)
            gl.glDeleteShader(fragment_shader)
            
            # Validate the program
            gl.glValidateProgram(program)
            validate_status = gl.glGetProgramiv(program, gl.GL_VALIDATE_STATUS)
            if not validate_status:
                error = gl.glGetProgramInfoLog(program)
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Shader program validation failed: {error}")
                gl.glDeleteProgram(program)
                return None
                
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Successfully compiled and linked shader program: {program}")
            return program
            
        except Exception as e:
            # Handle any unexpected errors
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Error during shader compilation: {str(e)}")
            # Make sure we clean up any resources
            if 'program' in locals() and program:
                try:
                    gl.glDeleteProgram(program)
                except Exception:
                    pass
            return None
    
    def _compile_basic_shader(self):
        """Compile a very minimal shader program for maximum compatibility.
        
        Returns:
            int: Shader program ID if successful, None otherwise.
        """
        if not self.gl_ctx.has_shader:
            return None
            
        try:
            # Clear any previous shader-related errors
            gl.glGetError()
            
            # Create vertex shader
            vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
            if vertex_shader == 0:
                if PYOPENGL_VERBOSE:
                    print("Viewer: Failed to create basic vertex shader object")
                return None
                
            gl.glShaderSource(vertex_shader, self.BASIC_VERTEX_SHADER)
            gl.glCompileShader(vertex_shader)
            
            # Check for vertex shader compilation errors
            compile_status = gl.glGetShaderiv(vertex_shader, gl.GL_COMPILE_STATUS)
            if not compile_status:
                error = gl.glGetShaderInfoLog(vertex_shader)
                gl.glDeleteShader(vertex_shader)
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Basic vertex shader compilation failed: {error}")
                return None
            
            # Create fragment shader
            fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
            if fragment_shader == 0:
                gl.glDeleteShader(vertex_shader)
                if PYOPENGL_VERBOSE:
                    print("Viewer: Failed to create basic fragment shader object")
                return None
                
            gl.glShaderSource(fragment_shader, self.BASIC_FRAGMENT_SHADER)
            gl.glCompileShader(fragment_shader)
            
            # Check for fragment shader compilation errors
            compile_status = gl.glGetShaderiv(fragment_shader, gl.GL_COMPILE_STATUS)
            if not compile_status:
                error = gl.glGetShaderInfoLog(fragment_shader)
                gl.glDeleteShader(vertex_shader)
                gl.glDeleteShader(fragment_shader)
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Basic fragment shader compilation failed: {error}")
                return None
            
            # Create and link shader program
            program = gl.glCreateProgram()
            if program == 0:
                gl.glDeleteShader(vertex_shader)
                gl.glDeleteShader(fragment_shader)
                if PYOPENGL_VERBOSE:
                    print("Viewer: Failed to create basic shader program object")
                return None
                
            gl.glAttachShader(program, vertex_shader)
            gl.glAttachShader(program, fragment_shader)
            
            # Bind attribute locations for position and color
            gl.glBindAttribLocation(program, 0, "position")
            gl.glBindAttribLocation(program, 1, "color")
            
            gl.glLinkProgram(program)
            
            # Check for linking errors
            link_status = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
            if not link_status:
                error = gl.glGetProgramInfoLog(program)
                gl.glDeleteShader(vertex_shader)
                gl.glDeleteShader(fragment_shader)
                gl.glDeleteProgram(program)
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Basic shader program linking failed: {error}")
                return None
            
            # Delete shaders (they're not needed after linking)
            gl.glDeleteShader(vertex_shader)
            gl.glDeleteShader(fragment_shader)
            
            # Validate the program
            gl.glValidateProgram(program)
            validate_status = gl.glGetProgramiv(program, gl.GL_VALIDATE_STATUS)
            if not validate_status:
                error = gl.glGetProgramInfoLog(program)
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Basic shader program validation failed: {error}")
                gl.glDeleteProgram(program)
                return None
            
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Successfully compiled and linked basic shader program: {program}")
            return program
            
        except Exception as e:
            # Handle any unexpected errors
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Error during basic shader compilation: {str(e)}")
            # Clean up any resources
            if 'program' in locals() and program:
                try:
                    gl.glDeleteProgram(program)
                except Exception:
                    pass
            return None
    
    def run(self):
        """Start the main rendering loop."""
        glut.glutMainLoop()
    
    @staticmethod
    def create_triangle_model(color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)) -> Model:
        """Create a simple triangle model for testing."""
        # Create a simple colored triangle with different colors for each vertex
        vertex_data = np.array([
            # position (3)     # color (4)           # normal (3)
            -1.5, -1.5, 0.0,   1.0, 0.0, 0.0, 1.0,   0.0, 0.0, 1.0,  # Red
            1.5, -1.5, 0.0,    0.0, 1.0, 0.0, 1.0,   0.0, 0.0, 1.0,  # Green
            0.0, 1.5, 0.0,     0.0, 0.0, 1.0, 1.0,   0.0, 0.0, 1.0   # Blue
        ], dtype=np.float32)
        
        return Model(vertex_data, num_points=3)
    
    @staticmethod
    def create_cube_model(size: float = 1.0, color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)) -> Model:
        """Create a simple cube model for testing."""
        # Create a colored cube
        s = size / 2
        vertex_data = []
        
        # Define the 8 vertices of the cube
        vertices = [
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
        ]
        
        # Define the 6 face normals
        normals = [
            [0, 0, -1], [0, 0, 1], [0, -1, 0],
            [0, 1, 0], [-1, 0, 0], [1, 0, 0]
        ]
        
        # Define faces with different colors
        face_colors = [
            [1.0, 0.0, 0.0, 1.0],  # Red - back
            [0.0, 1.0, 0.0, 1.0],  # Green - front
            [0.0, 0.0, 1.0, 1.0],  # Blue - bottom
            [1.0, 1.0, 0.0, 1.0],  # Yellow - top
            [0.0, 1.0, 1.0, 1.0],  # Cyan - left
            [1.0, 0.0, 1.0, 1.0]   # Magenta - right
        ]
        
        # Define the faces using indices
        faces = [
            [0, 1, 2, 3],  # back
            [4, 7, 6, 5],  # front
            [0, 4, 5, 1],  # bottom
            [3, 2, 6, 7],  # top
            [0, 3, 7, 4],  # left
            [1, 5, 6, 2]   # right
        ]
        
        # Create vertex data for each face
        for face_idx, face in enumerate(faces):
            normal = normals[face_idx]
            face_color = face_colors[face_idx]
            
            # Create two triangles per face
            tri1 = [face[0], face[1], face[2]]
            tri2 = [face[0], face[2], face[3]]
            
            for tri in [tri1, tri2]:
                for vertex_idx in tri:
                    # position
                    vertex_data.extend(vertices[vertex_idx])
                    # color
                    vertex_data.extend(face_color)
                    # normal
                    vertex_data.extend(normal)
        
        
        return Model(np.array(vertex_data, dtype=np.float32), num_points=36)

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
                    print(f"Viewer: Final cleanup of GLContext temp window ID: {gl_ctx.temp_window_id}")
                glut.glutDestroyWindow(gl_ctx.temp_window_id)
                gl_ctx.temp_window_id = None
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Error in final cleanup of GLContext temp window: {e}")
        
        # Reset initialization flags
        Viewer._initialized = False
        gl_ctx.is_initialized = False
        
        # Exit GLUT if it's running
        if HAS_OPENGL:
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
            vertices = np.array([
                # positions (3 floats per vertex)   # colors (4 floats per vertex)
                0.0, 0.5, 0.0,                     1.0, 0.0, 0.0, 1.0,  # top - red
                -0.5, -0.5, 0.0,                   0.0, 1.0, 0.0, 1.0,  # bottom left - green
                0.5, -0.5, 0.0,                    0.0, 0.0, 1.0, 1.0   # bottom right - blue
            ], dtype=np.float32)
            
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
                
            gl.glVertexAttribPointer(color_loc, 4, gl.GL_FLOAT, gl.GL_FALSE, 7 * 4, ctypes.c_void_p(3 * 4))
            gl.glEnableVertexAttribArray(color_loc)
            
            # Set up a model-view-projection matrix for our basic shader
            mvp_loc = gl.glGetUniformLocation(self.shader_program, "modelViewProj")
            if mvp_loc != -1:
                # Create a simple model-view-projection matrix
                model = glm.mat4(1.0)  # Identity matrix
                view = glm.lookAt(
                    glm.vec3(0.0, 0.0, 3.0),  # Camera position
                    glm.vec3(0.0, 0.0, 0.0),  # Look at origin
                    glm.vec3(0.0, 1.0, 0.0)   # Up vector
                )
                projection = glm.perspective(glm.radians(45.0), self.width / self.height, 0.1, 100.0)
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
            gl.glClearColor(*self.background_color) # Use the background color stored in the instance
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
                    gl.glPolygonOffset(-1.0, -1.0)  # This helps to pull the wireframe slightly forward
                    
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
                                    light_pos_loc = gl.glGetUniformLocation(self.shader_program, "lightPos")
                                    if light_pos_loc != -1:
                                        # Position light relative to camera
                                        gl.glUniform3f(
                                            light_pos_loc, 
                                            self.camera_pos.x + 5.0, 
                                            self.camera_pos.y + 5.0, 
                                            self.camera_pos.z + 10.0
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
                    1.0
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
            if PYOPENGL_VERBOSE:
                print("Viewer: Detected core profile in setup_view")
        
        # Set up matrices
        if self.gl_ctx.has_shader and self.shader_program:
            # Use the shader program for modern pipeline
            gl.glUseProgram(self.shader_program)
            
            try:
                # Update view position for specular highlights
                view_pos_loc = gl.glGetUniformLocation(self.shader_program, "viewPos")
                if view_pos_loc != -1:
                    gl.glUniform3f(view_pos_loc, self.camera_pos.x, self.camera_pos.y, self.camera_pos.z)
                
                # Set up model-view-projection matrices
                # Convert NumPy model matrix to GLM format
                
                model_mat = glm.mat4(*self.model_matrix.flatten())
                
                view = glm.lookAt(self.camera_pos, self.camera_pos + self.camera_front, self.camera_up)
                projection = glm.perspective(glm.radians(45.0), self.width / self.height, 0.1, 1000.0)
                
                # Send matrices to the shader
                model_loc = gl.glGetUniformLocation(self.shader_program, "model")
                view_loc = gl.glGetUniformLocation(self.shader_program, "view")
                proj_loc = gl.glGetUniformLocation(self.shader_program, "projection")
                
                if model_loc != -1:
                    gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, glm.value_ptr(model_mat))
                if view_loc != -1:
                    gl.glUniformMatrix4fv(view_loc, 1, gl.GL_FALSE, glm.value_ptr(view))
                if proj_loc != -1:
                    gl.glUniformMatrix4fv(proj_loc, 1, gl.GL_FALSE, glm.value_ptr(projection))
                
                # Check for combined MVP matrix (used in basic shader)
                mvp_loc = gl.glGetUniformLocation(self.shader_program, "modelViewProj")
                if mvp_loc != -1:
                    # Calculate combined MVP matrix
                    mvp = projection * view * model_mat
                    gl.glUniformMatrix4fv(mvp_loc, 1, gl.GL_FALSE, glm.value_ptr(mvp))
                    
                # Unbind shader program after setting uniforms
                gl.glUseProgram(0)
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Error setting up shader uniforms: {e}")
                gl.glUseProgram(0)
        
        # If we're not in a core profile, use fixed-function pipeline
        if not is_core_profile:
            try:
                # Set up projection matrix
                gl.glMatrixMode(gl.GL_PROJECTION)
                gl.glLoadIdentity()
                glu.gluPerspective(45.0, self.width / self.height, 0.1, 1000.0)
                
                # Set up modelview matrix
                gl.glMatrixMode(gl.GL_MODELVIEW)
                gl.glLoadIdentity()
                
                # Set up view with gluLookAt
                glu.gluLookAt(
                    self.camera_pos.x, self.camera_pos.y, self.camera_pos.z,
                    self.camera_pos.x + self.camera_front.x, 
                    self.camera_pos.y + self.camera_front.y, 
                    self.camera_pos.z + self.camera_front.z,
                    self.camera_up.x, self.camera_up.y, self.camera_up.z
                )
                
                # Apply model matrix (convert from NumPy to a format OpenGL can use)
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
            elif state == glut.GLUT_UP:
                self.left_button_pressed = False
        elif button == glut.GLUT_RIGHT_BUTTON:
            if state == glut.GLUT_DOWN:
                self.right_button_pressed = True
                self.mouse_start_x = x
                self.mouse_start_y = y
            elif state == glut.GLUT_UP:
                self.right_button_pressed = False
    
    def _motion_callback(self, x, y):
        """GLUT mouse motion callback."""
        if self.left_button_pressed:
            # Handle rotation
            dx = x - self.mouse_start_x
            dy = y - self.mouse_start_y
            self.mouse_start_x = x
            self.mouse_start_y = y
            
            # Skip tiny movements to prevent division by zero
            if abs(dx) < 1 and abs(dy) < 1:
                return
                
            # Update rotation angles
            sensitivity = 0.5
            dx *= sensitivity
            dy *= sensitivity
            
            # Create a rotation vector
            vec = linear.GVector((dy, dx, 0))
            veclen = vec.length()
            
            # Only rotate if we have a meaningful rotation angle
            if veclen > 0.001:  # Threshold to avoid tiny rotations
                try:
                    # Safely calculate normalized vector
                    unitvec = linear.GVector(vec.v[0:3] / veclen)
                    
                    # Apply rotation to model matrix
                    rotation_matrix = linear.rotV(unitvec, -veclen).A
                    self.model_matrix = self.model_matrix @ rotation_matrix
                    
                except (ZeroDivisionError, RuntimeWarning, ValueError) as e:
                    if PYOPENGL_VERBOSE:
                        print(f"Viewer: Error computing rotation: {e}")

            # Redraw
            glut.glutPostRedisplay()
            
        elif self.right_button_pressed:
            # Handle translation
            dx = x - self.mouse_start_x
            dy = y - self.mouse_start_y
            self.mouse_start_x = x
            self.mouse_start_y = y
            
            # Skip tiny movements
            if abs(dx) < 1 and abs(dy) < 1:
                return
                
            # Scale the translation based on scene size for better control
            translation_sensitivity = -self.camera_speed * 0.1
            
            # Create a translation matrix
            
            # Apply the translation to the model matrix
            self.model_matrix = self.model_matrix @ linear.translate((
                    dx * translation_sensitivity, -dy * translation_sensitivity, 0)
                ).I.A.transpose()
            
            # Redraw
            glut.glutPostRedisplay()
    
    def _wheel_callback(self, wheel, direction, x, y):
        """GLUT mouse wheel callback."""
        # Zoom in/out by changing camera position along the front vector
        self.camera_pos += self.camera_front * (direction * self.camera_speed * 10.0)
        glut.glutPostRedisplay()
    
    def _keyboard_callback(self, key, x, y):
        """GLUT keyboard callback."""
        # Handle basic keyboard controls
        if key == b'\x1b':  # ESC key
            # Use terminate to properly clean up
            if PYOPENGL_VERBOSE:
                print("Viewer: ESC key pressed, terminating application")
            Viewer.terminate()
        elif key == b'r':
            # Reset view
            self._reset_view()
            glut.glutPostRedisplay()
        elif key == b'a':
            # Toggle antialiasing
            if hasattr(gl, 'GL_MULTISAMPLE') and Viewer._multisample_supported:
                 self.antialiasing_enabled = not self.antialiasing_enabled
                 if self.antialiasing_enabled:
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
        elif key == b'b':
            # Toggle backface culling
            self.backface_culling = not self.backface_culling
            if self.backface_culling:
                gl.glEnable(gl.GL_CULL_FACE)
                gl.glCullFace(gl.GL_BACK)
                if PYOPENGL_VERBOSE:
                    print("Viewer: Backface culling enabled")
            else:
                gl.glDisable(gl.GL_CULL_FACE)
                if PYOPENGL_VERBOSE:
                    print("Viewer: Backface culling disabled")
            glut.glutPostRedisplay()
        elif key == b'w':
            # Toggle wireframe mode
            self.wireframe_mode = not self.wireframe_mode
            if self.wireframe_mode:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                if PYOPENGL_VERBOSE:
                    print("Viewer: Wireframe mode enabled")
            else:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
                if PYOPENGL_VERBOSE:
                    print("Viewer: Wireframe mode disabled")
            glut.glutPostRedisplay()
        elif key == b'z': 
            # Toggle Z-buffer occlusion for wireframes
            self.zbuffer_occlusion = not self.zbuffer_occlusion
            if PYOPENGL_VERBOSE:
                if self.zbuffer_occlusion:
                    print("Viewer: Z-buffer occlusion enabled for wireframes")
                else:
                    print("Viewer: Z-buffer occlusion disabled for wireframes")
            glut.glutPostRedisplay()
        elif key == b'x':
            # Toggle bounding box mode
            self.bounding_box_mode = (self.bounding_box_mode + 1) % 3
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Bounding box mode set to {self.bounding_box_mode}")
            glut.glutPostRedisplay()
        elif key == b'c':
            # Toggle coalesced model mode
            self.toggle_coalesced_mode()
            # Already calls glutPostRedisplay()
        elif key == b'h':
            # Toggle shader-based rendering
            self._toggle_and_diagnose_shader()
            # Already calls glutPostRedisplay()
        elif key == b'd':
            # Print diagnostic information
            self._print_diagnostics()
            # No need to redisplay
        elif key == b'p':
            # Print detailed shader program debug information
            try:
                # Get current program
                current_program = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
                print(f"Current program: {current_program}")
                
                # Debug our shader program
                if self.shader_program:
                    self._print_shader_debug(self.shader_program)
                
                # Debug the special program '3'
                self._print_shader_debug(3)
                
            except Exception as e:
                print(f"Error during shader debugging: {e}")
            # No need to redisplay
        elif key == b's':
            # Save screenshot - only on key down
            try:
                # Generate a default filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                self.save_screenshot(filename)
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"Viewer: Failed to save screenshot: {str(e)}")
            # Don't redisplay after saving screenshot
            
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
            img.save(filename, 'PNG')
            
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Screenshot saved to {filename}")
                
        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Failed to save screenshot: {str(e)}")
            raise

    def _reset_view(self):
        """Reset camera and model transformations to defaults."""
        # Reset model matrix
        self.model_matrix = np.eye(4, dtype=np.float32)
        
        # Reset camera position and orientation
        self._setup_camera()
        
        # Reset mouse rotation tracking
        self.yaw = -90.0
        self.pitch = 0.0

    def _draw_bounding_box(self):
        """Draw the scene bounding box in the current mode (off/wireframe/solid)."""
        if self.bounding_box_mode == 0:
            return
            
        # Store current backface culling state and disable it for the bounding box
        was_culling_enabled = gl.glIsEnabled(gl.GL_CULL_FACE)
        if was_culling_enabled:
            gl.glDisable(gl.GL_CULL_FACE)
            
        # Get bounding box coordinates
        min_x, min_y, min_z = self.bounding_box.min_point
        max_x, max_y, max_z = self.bounding_box.max_point
        
        # Set up transparency for solid mode
        was_blend_enabled = False
        if self.bounding_box_mode == 2:
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
                self.bounding_box_mode = 1
                if PYOPENGL_VERBOSE:
                    print("Viewer: Blending not supported, falling back to wireframe mode")
        
        # Draw the bounding box
        if self.bounding_box_mode == 1:  # Wireframe mode
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
        if self.bounding_box_mode == 2 and not was_blend_enabled:
            try:
                gl.glDisable(gl.GL_BLEND)
            except Exception:
                pass
                
        # Restore backface culling state
        if was_culling_enabled:
            gl.glEnable(gl.GL_CULL_FACE)

    @staticmethod
    def create_colored_test_cube(size: float = 2.0) -> Model:
        """Create a test cube with distinct bright colors for each face."""
        s = size / 2
        vertex_data = []
        
        # Define cube vertices
        vertices = [
            [s, -s, -s], [-s, -s, -s], [-s, s, -s], [s, s, -s],  # 0-3 back face
            [s, -s, s], [-s, -s, s], [-s, s, s], [s, s, s]       # 4-7 front face
        ]
        
        # Define face normals
        normals = [
            [0, 0, -1],  # back - red
            [0, 0, 1],   # front - green
            [0, -1, 0],  # bottom - blue
            [0, 1, 0],   # top - yellow
            [-1, 0, 0],  # left - magenta
            [1, 0, 0]    # right - cyan
        ]
        
        # Bright colors for each face
        colors = [
            [1.0, 0.0, 0.0, 1.0],  # red - back
            [0.0, 1.0, 0.0, 1.0],  # green - front
            [0.0, 0.0, 1.0, 1.0],  # blue - bottom
            [1.0, 1.0, 0.0, 1.0],  # yellow - top
            [1.0, 0.0, 1.0, 1.0],  # magenta - left
            [0.0, 1.0, 1.0, 1.0]   # cyan - right
        ]
        
        # Face definitions (vertices comprising each face)
        faces = [
            [0, 1, 2, 3],  # back
            [4, 7, 6, 5],  # front
            [0, 4, 5, 1],  # bottom
            [3, 2, 6, 7],  # top
            [0, 3, 7, 4],  # left
            [1, 5, 6, 2]   # right
        ]
        
        # Create vertex data for each face
        for face_idx, face in enumerate(faces):
            normal = normals[face_idx]
            color = colors[face_idx]
            
            # Create two triangles per face
            tri1 = [face[0], face[1], face[2]]
            tri2 = [face[0], face[2], face[3]]
            
            for tri in [tri1, tri2]:
                for vertex_idx in tri:
                    # position
                    vertex_data.extend(vertices[vertex_idx])
                    # color
                    vertex_data.extend(color)
                    # normal
                    vertex_data.extend(normal)
        
        return Model(np.array(vertex_data, dtype=np.float32), num_points=36)
    
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
                print(f"Viewer: Initializing GL resources for updated models in window {self.window_id}")
            for model in self.models:
                try:
                    model.initialize_gl_resources()
                except Exception as e:
                    if PYOPENGL_VERBOSE:
                        print(f"Viewer: Error initializing GL resources for an updated model: {e}")
        elif PYOPENGL_VERBOSE:
            print("Viewer: Warning - Cannot initialize updated model GL resources. Window ID mismatch or invalid window.")
            
        # Request redisplay
        if self.window_id:
            glut.glutPostRedisplay()

    def get_view_to_world_matrix(self) -> np.ndarray:
        """Calculate and return the view-to-world transformation matrix."""
        if not HAS_OPENGL:
            return np.eye(4, dtype=np.float32) # Return identity if no OpenGL
        
        try:
            # Calculate the view matrix (world-to-view)
            view_matrix = glm.lookAt(
                self.camera_pos, 
                self.camera_pos + self.camera_front, 
                self.camera_up
            )
            
            # Invert the view matrix to get view-to-world
            view_to_world_matrix = glm.inverse(view_matrix)
            
            # Convert GLM matrix to NumPy array
            # GLM stores matrices column-major, so we need to transpose after reshaping
            np_matrix = np.array(view_to_world_matrix, dtype=np.float32).reshape(4, 4).T
            return np_matrix
            
        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Error calculating view-to-world matrix: {e}")
            return np.eye(4, dtype=np.float32) # Return identity on error

    def get_current_window_dims(self) -> Tuple[int, int]:
        """Get the current dimensions (width, height) of the viewer window."""
        if not HAS_OPENGL or not Viewer._initialized or self.window_id is None:
            return (self.width, self.height) # Return stored dimensions if GLUT not ready
            
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
            return (self.width, self.height) # Fallback to stored dimensions

    # Proposed helper function to be added inside the Viewer class

    def _save_numpy_buffer_to_png(self, buffer: np.ndarray, width: int, height: int, filename: str):
        """Saves a numpy pixel buffer (RGB, uint8) to a PNG file."""
        try:
            # Image needs flipping vertically (OpenGL reads bottom-left)
            image = np.flipud(buffer.reshape((height, width, 3)))

            # Save the image using PIL
            from PIL import Image
            img = Image.fromarray(image)
            img.save(filename, 'PNG')

            if PYOPENGL_VERBOSE:
                print(f"Viewer: Image saved to {filename}")

        except ImportError:
            # Added file=sys.stderr for error messages
            print("Error: PIL (Pillow) library is required to save images. Please install it (`pip install Pillow`).", file=sys.stderr)
            raise # Re-raise to indicate failure
        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"Viewer: Failed to save image buffer: {str(e)}", file=sys.stderr)
            raise # Re-raise to indicate failure

    def offscreen_render(self, filename: str):
        """Renders the model to an offscreen buffer and saves the screen as a PNG image."""
        if not HAS_OPENGL:
            raise RuntimeError("OpenGL is not available.")

        if self.window_id is None or not Viewer._initialized:
             raise RuntimeError("Viewer window is not initialized. Offscreen rendering requires an active OpenGL context.")

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
        original_viewport = None # Initialize to avoid reference before assignment warning

        try:
            # --- FBO Setup ---
            # Check for errors before starting
            gl.glGetError()

            fbo = gl.glGenFramebuffers(1)
            if gl.glGetError() != gl.GL_NO_ERROR or not fbo: raise RuntimeError("Failed to generate Framebuffer Object (FBO).")
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
            if gl.glGetError() != gl.GL_NO_ERROR: raise RuntimeError("Failed to bind FBO.")

            # Color Renderbuffer
            color_rbo = gl.glGenRenderbuffers(1)
            if gl.glGetError() != gl.GL_NO_ERROR or not color_rbo: raise RuntimeError("Failed to generate color Renderbuffer Object (RBO).")
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, color_rbo)
            if gl.glGetError() != gl.GL_NO_ERROR: raise RuntimeError("Failed to bind color RBO.")
            # Use GL_RGB8 for color, GL_DEPTH_COMPONENT24 for depth (common choices)
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGB8, self.width, self.height)
            if gl.glGetError() != gl.GL_NO_ERROR: raise RuntimeError("Failed to allocate color RBO storage.")
            gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, color_rbo)
            if gl.glGetError() != gl.GL_NO_ERROR: raise RuntimeError("Failed to attach color RBO to FBO.")

            # Depth Renderbuffer
            depth_rbo = gl.glGenRenderbuffers(1)
            if gl.glGetError() != gl.GL_NO_ERROR or not depth_rbo: raise RuntimeError("Failed to generate depth RBO.")
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_rbo)
            if gl.glGetError() != gl.GL_NO_ERROR: raise RuntimeError("Failed to bind depth RBO.")
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, self.width, self.height)
            if gl.glGetError() != gl.GL_NO_ERROR: raise RuntimeError("Failed to allocate depth RBO storage.")
            gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_rbo)
            if gl.glGetError() != gl.GL_NO_ERROR: raise RuntimeError("Failed to attach depth RBO to FBO.")

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
                raise RuntimeError(f"Framebuffer is not complete: Status {status_map.get(status, status)}")

            # --- Rendering to FBO ---
            original_viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
            gl.glViewport(0, 0, self.width, self.height)
            if gl.glGetError() != gl.GL_NO_ERROR: print("Warning: Error setting viewport for FBO.", file=sys.stderr)

            # Clear the FBO buffers
            gl.glClearColor(*self.background_color)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            if gl.glGetError() != gl.GL_NO_ERROR: print("Warning: Error clearing FBO buffers.", file=sys.stderr)


            # --- Replicate core rendering logic from _display_callback ---
            # NOTE: This duplicates logic. A refactor of _display_callback is recommended.
            try:
                self._setup_view() # Set up projection and view matrices

                # Determine models to render based on coalesce setting
                if not self.use_coalesced_models:
                    opaque_models = [model for model in self.original_models if not model.has_alpha_lt1]
                    transparent_models = [model for model in self.original_models if model.has_alpha_lt1]
                    # Create temporary coalesced models just for sorting if needed (less efficient)
                    if opaque_models or transparent_models:
                        temp_opaque, temp_transparent = Model.create_coalesced_models(opaque_models + transparent_models)
                        # This approach might be complex, sticking to the coalesced/original logic for now
                        # Reverting to simpler logic based on self.models which ARE already coalesced or not
                        if not self.use_coalesced_models:
                             opaque_models = [model for model in self.models if not model.has_alpha_lt1]
                             transparent_models = [model for model in self.models if model.has_alpha_lt1]
                             # Need to sort transparent models if not coalesced
                             # This requires Z positions which are not readily available without recalculation
                             # Sticking to the assumption that self.models is correct for now
                        else:
                             opaque_models = [self.models[0]] if len(self.models) > 0 else []
                             transparent_models = [self.models[1]] if len(self.models) > 1 else []

                else: # Already using coalesced models
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

                try: # Ensure color material is enabled for fixed-function or basic shaders
                     gl.glEnable(gl.GL_COLOR_MATERIAL)
                     gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
                except Exception: pass # Ignore if not available

                # Render opaque models
                if using_shader:
                    try:
                        gl.glUseProgram(active_program)
                        # Uniforms (lightPos, viewPos) should be set if using the main shader
                        if active_program == self.shader_program: # Check if it's the main shader
                            light_pos_loc = gl.glGetUniformLocation(active_program, "lightPos")
                            view_pos_loc = gl.glGetUniformLocation(active_program, "viewPos")
                            if light_pos_loc != -1:
                                gl.glUniform3f(light_pos_loc, self.camera_pos.x + 5.0, self.camera_pos.y + 5.0, self.camera_pos.z + 10.0)
                            if view_pos_loc != -1:
                                gl.glUniform3f(view_pos_loc, self.camera_pos.x, self.camera_pos.y, self.camera_pos.z)
                    except Exception as e:
                        if PYOPENGL_VERBOSE: print(f"Viewer (offscreen): Error setting shader uniforms: {e}", file=sys.stderr)
                        gl.glUseProgram(0) # Fallback if uniforms fail
                        active_program = 0

                for model in opaque_models:
                    # Pass the active shader program to draw, if any
                    model.draw() # Model.draw now handles using the current program

                # Render transparent models
                if transparent_models:
                     try:
                         gl.glDepthMask(gl.GL_FALSE) # Don't write to depth buffer
                         # Blending is enabled within model.draw if needed
                         for model in transparent_models:
                             model.draw() # Pass active shader
                         gl.glDepthMask(gl.GL_TRUE) # Restore depth writes
                     except Exception as e:
                         if PYOPENGL_VERBOSE: print(f"Viewer (offscreen): Error during transparent rendering: {e}", file=sys.stderr)
                         gl.glDepthMask(gl.GL_TRUE) # Ensure depth mask is restored on error
                         # Fallback rendering without depth mask modification
                         for model in transparent_models: model.draw()


                # Unbind shader if it was used
                if active_program != 0:
                     try: gl.glUseProgram(0)
                     except Exception: pass

                # Draw bounding box if enabled (uses immediate mode)
                self._draw_bounding_box()

                # Draw axes (uses immediate mode)
                self.axes_renderer.draw(self)

            except Exception as render_err:
                 # Handle rendering errors specifically
                 print(f"Viewer (offscreen): Error during scene rendering: {render_err}", file=sys.stderr)
                 # Continue to cleanup if possible, but report the error
                 # We might still have partial results in the buffer


            # --- Read Pixels ---
            # Ensure reading from the correct FBO attachment
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
            if gl.glGetError() != gl.GL_NO_ERROR: print("Warning: Error setting read buffer to FBO attachment.", file=sys.stderr)

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
                 print(f"Warning: Read pixel buffer size ({pixel_data.size}) does not match expected size ({expected_size}). Image may be incorrect.", file=sys.stderr)
                 # Attempt to reshape anyway, might fail if size is wrong
                 # pixel_data = pixel_data[:expected_size] # Truncate/Pad? Risky.

            # --- Save Image ---
            # Check size again before saving
            if pixel_data.size == expected_size:
                 self._save_numpy_buffer_to_png(pixel_data, self.width, self.height, filename)
            else:
                 raise RuntimeError(f"Cannot save image due to incorrect pixel buffer size. Expected {expected_size}, got {pixel_data.size}.")


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
                 gl.glDeleteFramebuffers(1, [fbo]) # Simpler deletion
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


# Helper function to create a viewer with models
def create_viewer_with_models(models, width=800, height=600, title="3D Viewer"):
    """Create and return a viewer with the given models."""
    viewer = Viewer(models, width, height, title)
    return viewer


# Helper function to check if OpenGL is available
def is_opengl_available():
    """Check if OpenGL libraries are available."""
    return HAS_OPENGL

def get_opengl_capabilities(gl_ctx: GLContext):
    """Get a dictionary describing the available OpenGL capabilities."""
    if not HAS_OPENGL:
        return {
            "opengl_available": False,
            "vbo_support": False,
            "shader_support": False,
            "vao_support": False,
            "message": "OpenGL libraries not available. Install PyOpenGL and PyGLM."
        }
    
    
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
    import signal
    import sys
    
    if not HAS_OPENGL:
        print("OpenGL libraries (PyOpenGL and PyGLM) are required for the viewer")
        sys.exit(1)
    
    # Enable verbose mode
    PYOPENGL_VERBOSE = True
    
    # Initialize the OpenGL context
    gl_ctx = GLContext.get_instance()
    gl_ctx.initialize()
    
    # Print OpenGL capabilities
    capabilities = get_opengl_capabilities(gl_ctx)
    print(f"OpenGL capabilities: {capabilities['message']}")
    
    # Create our test models
    triangle = Viewer.create_triangle_model()
    color_cube = Viewer.create_colored_test_cube(2.0)  # Use our special test cube with bright colors
    
    # Create a viewer with both models - using just the color cube for testing colors
    viewer = create_viewer_with_models([color_cube, triangle], title="PyOpenSCAD Viewer Color Test")
    
    # Define a handler for Ctrl+C to properly terminate
    def signal_handler(sig, frame):
        print("\nCtrl+C detected, safely shutting down...")
        Viewer.terminate()
        sys.exit(0)
        
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Press ESC in the viewer window or Ctrl+C in the terminal to exit")
    print("If the cube appears with distinct colors on each face, the color issue is resolved")
    print("If the cube appears black or all one color, the color issue remains")
    
    try:
        # Start the main loop
        print(Viewer.VIEWER_HELP_TEXT)
        viewer.run()
    except KeyboardInterrupt:
        # Handle Ctrl+C
        print("\nSafely shutting down...")
    finally:
        # Ensure proper cleanup
        Viewer.terminate()
    