from dataclasses import InitVar
import sys
import numpy as np
import ctypes # For VAO/VBO offsets if needed by Model class
from typing import Any, List, Tuple, Dict, Callable

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QMessageBox
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Slot, Qt, QPoint, Signal, QTimer
from PySide6.QtGui import QSurfaceFormat, QKeyEvent, QMouseEvent, QWheelEvent
# QOpenGLFunctions provides cross-platform access to GL functions
# but since you're using PyOpenGL's gl, GLU, we'll initialize it but mostly use your imports
# from PySide6.QtGui import QOpenGLFunctions # No longer needed as base class
# from PySide6.QtGui import QOpenGLFunctions_3_3_Compatibility # Trying a more common version

# Your existing imports (slightly modified)
from datatrees import datatree, dtfield, Node # Assuming this is available
import manifold3d as m3d # Assuming this is available

# pythonopenscad.viewer specific imports
# Assuming these paths are correct or your modules are in PYTHONPATH
from pythonopenscad.viewer.bbox import BoundingBox
from pythonopenscad.viewer.bbox_render import BBoxRender
from pythonopenscad.viewer.glctxt import GLContext, PYOPENGL_VERBOSE # PYOPENGL_VERBOSE will be used
from pythonopenscad.viewer.model import Model # CRITICAL: This class definition is not provided but essential
from pythonopenscad.viewer.axes import AxesRenderer, ScreenContext # From axes.py
from pythonopenscad.viewer.shader import Shader, ShaderProgram, SHADER_PROGRAM, BASIC_SHADER_PROGRAM # From shader.py
from pythonopenscad.viewer.basic_models import (
    create_colored_test_cube,
    create_triangle_model,
)


import OpenGL.GL as gl
import OpenGL.GLU as glu
# OpenGL.GLUT will be replaced by PySide6
from pyglm import glm # Retaining pyglm as per original code



# PS_VERTEX_SHADER = Shader(
#     name="vertex_shader",
#     shader_type=gl.GL_VERTEX_SHADER,
#     binding=("aPos", "aColor", "aNormal"),
#     shader_source="""
# #version 120

# attribute vec3 aPos;
# attribute vec4 aColor;
# attribute vec3 aNormal;

# uniform mat4 model;
# uniform mat4 view;
# uniform mat4 projection;

# varying vec3 FragPos;
# varying vec4 VertexColor;
# varying vec3 Normal;
# // varying float dbg_viewZ;

# void main() {
#     vec4 pos_model_space = vec4(aPos, 1.0);
#     vec4 pos_world_space = model * pos_model_space;
#     vec4 pos_view_space  = view * pos_world_space;
#     vec4 pos_clip_space  = projection * pos_view_space;

#     FragPos = vec3(pos_world_space); 
#     Normal = normalize(mat3(model) * aNormal);
#     VertexColor = aColor; 

#     gl_Position = pos_clip_space; // Standard perspective - ACTIVE
#     // gl_Position = vec4(pos_clip_space.xyz, 1.0); // Force W=1.0 - COMMENTED OUT

# }
# """,
# )

# PS_FRAGMENT_SHADER = Shader(
#     name="fragment_shader",
#     shader_type=gl.GL_FRAGMENT_SHADER,
#     shader_source="""
# #version 120

# varying vec3 FragPos;
# varying vec4 VertexColor;
# varying vec3 Normal;

# uniform vec3 lightPos;
# uniform vec3 viewPos;

# void main() {
#     // Ambient - increase to make colors more visible
#     float ambientStrength = 0.5;  // Increased from 0.3
#     vec3 ambient = ambientStrength * VertexColor.rgb;
    
#     // Diffuse - increase strength
#     vec3 norm = normalize(Normal);
#     vec3 lightDir = normalize(lightPos - FragPos);
#     float diff = max(dot(norm, lightDir), 0.0);
#     vec3 diffuse = diff * VertexColor.rgb * 0.8;  // More diffuse influence
    
#     // Specular - keep the same
#     float specularStrength = 0.5;
#     vec3 viewDir = normalize(viewPos - FragPos);
#     vec3 reflectDir = reflect(-lightDir, norm);
#     float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
#     vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
    
#     // Result - ensure colors are visible regardless of lighting
#     vec3 result = ambient + diffuse + specular;
    
#     // Add a minimum brightness to ensure visibility
#     result = max(result, VertexColor.rgb * 0.4);
    
#     // Preserve alpha from vertex color for transparent objects
#     gl_FragColor = vec4(result, VertexColor.a);
# }
# """,
# )

# PS_SHADER_PROGRAM = ShaderProgram(
#     name="shader_program",
#     vertex_shader=PS_VERTEX_SHADER,
#     fragment_shader=PS_FRAGMENT_SHADER,
# )

PS_SHADER_PROGRAM = SHADER_PROGRAM

class PoscGLWidget(QOpenGLWidget):
    """
    PySide6 QOpenGLWidget to replace the GLUT-based Viewer.
    It incorporates most of the logic from your original Viewer class.
    """
    # Signals for UI updates if needed (e.g., status bar messages)
    message_signal = Signal(str)

    def __init__(self,
        models: List[Model] = None,
        use_coalesced_models: bool = True,
        backface_culling: bool = True,
        wireframe_mode: bool = False,
        bounding_box_mode: int = 0,  # 0: off, 1: wireframe, 2: solid
        zbuffer_occlusion: bool = True,
        antialiasing_enabled: bool = True, # Managed by QSurfaceFormat initially
        show_axes: bool = True,
        edge_rotations: bool = False, # From original viewer

        background_color: Tuple[float, float, float, float] = (0.98, 0.98, 0.85, 1.0),
        axes_renderer: AxesRenderer = None,

        projection_mode: str = "perspective",  # 'perspective' or 'orthographic'
        ortho_scale: float = 20.0,  # World-space width for ortho view

        # Camera parameters (from your viewer)
        camera_pos: glm.vec3 = glm.vec3(10.0, -10.0, 10.0),
        camera_front: glm.vec3 = glm.vec3(0.0, 0.0, 1.0),
        camera_up: glm.vec3 = glm.vec3(0.0, 0.0, 1.0),
        camera_target: glm.vec3 = glm.vec3(0.0, 0.0, 0.0),
        camera_speed: float = 0.05, # Will be adjusted based on scene bounds
        model_matrix_np: np.ndarray = np.eye(4, dtype=np.float32),
        
        # Mouse interaction state
        last_mouse_pos: QPoint = None,
        left_button_pressed: bool = False,
        right_button_pressed: bool = False,
        trackball_start_point: glm.vec3 = None, # For trackball rotation

        # Scene bounds
        bounding_box: BoundingBox = None,

        # Shader program (will be initialized in initializeGL)
        active_shader_program_id: Any = None, # ID of the compiled shader program
        use_shaders: bool = True, # To toggle shader usage

        # GLContext (for capability querying mainly)
        gl_ctx: GLContext = None,
        _multisample_supported_by_context: bool = False, # Queried from QSurfaceFormat
        axes_depth_test: bool = True,
        parent: QWidget = None,
    ):
        self.models = models if models else []
        self.use_coalesced_models = use_coalesced_models
        self.backface_culling = backface_culling
        self.wireframe_mode = wireframe_mode
        self.bounding_box_mode = bounding_box_mode
        self.zbuffer_occlusion = zbuffer_occlusion
        self.antialiasing_enabled = antialiasing_enabled
        self.show_axes = show_axes
        self.edge_rotations = edge_rotations
        self.background_color = background_color
        self.axes_renderer = axes_renderer if axes_renderer else AxesRenderer()
        self.projection_mode = projection_mode
        self.ortho_scale = ortho_scale
        self.camera_pos = camera_pos
        self.camera_front = camera_front
        self.camera_up = camera_up
        self.camera_target = camera_target
        self.camera_speed = camera_speed
        self.model_matrix_np = model_matrix_np
        self.last_mouse_pos = last_mouse_pos
        self.left_button_pressed = left_button_pressed
        self.right_button_pressed = right_button_pressed
        self.trackball_start_point = trackball_start_point
        self.bounding_box = bounding_box
        self.active_shader_program_id = active_shader_program_id
        self.use_shaders = use_shaders
        self.gl_ctx = gl_ctx
        self._multisample_supported_by_context = _multisample_supported_by_context
        self.axes_depth_test = axes_depth_test
    
        # Mouse interaction state
        self.last_mouse_pos = last_mouse_pos
        self.left_button_pressed = left_button_pressed
        self.right_button_pressed = right_button_pressed
        self.trackball_start_point = trackball_start_point

        # Scene bounds
        self.bounding_box = bounding_box

        # Shader program (will be initialized in initializeGL)
        self.active_shader_program_id = active_shader_program_id
        self.use_shaders = use_shaders

        # GLContext (for capability querying mainly)
        self.gl_ctx = gl_ctx
        self._multisample_supported_by_context = _multisample_supported_by_context
        self.axes_depth_test = axes_depth_test
        
        self.__post_init__(parent)


    def __post_init__(self, parent: QWidget =None):
        super().__init__(parent)
        
        # Initialize GLContext BEFORE anything else that might need it
        self.gl_ctx = GLContext.get_instance()

        self.original_models = self.models # Store for coalescing logic
        self.models = [] # Initialize as empty list

        self._apply_coalescing()
        self._compute_scene_bounds()
        self._setup_camera_from_bounds()

from dataclasses import InitVar
import sys
import numpy as np
import ctypes # For VAO/VBO offsets if needed by Model class
from typing import Any, List, Tuple, Dict, Callable

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QMessageBox
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Slot, Qt, QPoint, Signal, QTimer
from PySide6.QtGui import QSurfaceFormat, QKeyEvent, QMouseEvent, QWheelEvent
# QOpenGLFunctions provides cross-platform access to GL functions
# but since you're using PyOpenGL's gl, GLU, we'll initialize it but mostly use your imports
# from PySide6.QtGui import QOpenGLFunctions # No longer needed as base class
# from PySide6.QtGui import QOpenGLFunctions_3_3_Compatibility # Trying a more common version

# Your existing imports (slightly modified)
from datatrees import datatree, dtfield, Node # Assuming this is available
import manifold3d as m3d # Assuming this is available

# pythonopenscad.viewer specific imports
# Assuming these paths are correct or your modules are in PYTHONPATH
from pythonopenscad.viewer.bbox import BoundingBox
from pythonopenscad.viewer.bbox_render import BBoxRender
from pythonopenscad.viewer.glctxt import GLContext, PYOPENGL_VERBOSE # PYOPENGL_VERBOSE will be used
from pythonopenscad.viewer.model import Model # CRITICAL: This class definition is not provided but essential
from pythonopenscad.viewer.axes import AxesRenderer, ScreenContext # From axes.py
from pythonopenscad.viewer.shader import Shader, ShaderProgram, SHADER_PROGRAM, BASIC_SHADER_PROGRAM # From shader.py
from pythonopenscad.viewer.basic_models import (
    create_colored_test_cube,
    create_triangle_model,
)


import OpenGL.GL as gl
import OpenGL.GLU as glu
# OpenGL.GLUT will be replaced by PySide6
from pyglm import glm # Retaining pyglm as per original code



# PS_VERTEX_SHADER = Shader(
#     name="vertex_shader",
#     shader_type=gl.GL_VERTEX_SHADER,
#     binding=("aPos", "aColor", "aNormal"),
#     shader_source="""
# #version 120

# attribute vec3 aPos;
# attribute vec4 aColor;
# attribute vec3 aNormal;

# uniform mat4 model;
# uniform mat4 view;
# uniform mat4 projection;

# varying vec3 FragPos;
# varying vec4 VertexColor;
# varying vec3 Normal;
# // varying float dbg_viewZ;

# void main() {
#     vec4 pos_model_space = vec4(aPos, 1.0);
#     vec4 pos_world_space = model * pos_model_space;
#     vec4 pos_view_space  = view * pos_world_space;
#     vec4 pos_clip_space  = projection * pos_view_space;

#     FragPos = vec3(pos_world_space); 
#     Normal = normalize(mat3(model) * aNormal);
#     VertexColor = aColor; 

#     gl_Position = pos_clip_space; // Standard perspective - ACTIVE
#     // gl_Position = vec4(pos_clip_space.xyz, 1.0); // Force W=1.0 - COMMENTED OUT

# }
# """,
# )

# PS_FRAGMENT_SHADER = Shader(
#     name="fragment_shader",
#     shader_type=gl.GL_FRAGMENT_SHADER,
#     shader_source="""
# #version 120

# varying vec3 FragPos;
# varying vec4 VertexColor;
# varying vec3 Normal;

# uniform vec3 lightPos;
# uniform vec3 viewPos;

# void main() {
#     // Ambient - increase to make colors more visible
#     float ambientStrength = 0.5;  // Increased from 0.3
#     vec3 ambient = ambientStrength * VertexColor.rgb;
    
#     // Diffuse - increase strength
#     vec3 norm = normalize(Normal);
#     vec3 lightDir = normalize(lightPos - FragPos);
#     float diff = max(dot(norm, lightDir), 0.0);
#     vec3 diffuse = diff * VertexColor.rgb * 0.8;  // More diffuse influence
    
#     // Specular - keep the same
#     float specularStrength = 0.5;
#     vec3 viewDir = normalize(viewPos - FragPos);
#     vec3 reflectDir = reflect(-lightDir, norm);
#     float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
#     vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
    
#     // Result - ensure colors are visible regardless of lighting
#     vec3 result = ambient + diffuse + specular;
    
#     // Add a minimum brightness to ensure visibility
#     result = max(result, VertexColor.rgb * 0.4);
    
#     // Preserve alpha from vertex color for transparent objects
#     gl_FragColor = vec4(result, VertexColor.a);
# }
# """,
# )

# PS_SHADER_PROGRAM = ShaderProgram(
#     name="shader_program",
#     vertex_shader=PS_VERTEX_SHADER,
#     fragment_shader=PS_FRAGMENT_SHADER,
# )

PS_SHADER_PROGRAM = SHADER_PROGRAM

class PoscGLWidget(QOpenGLWidget):
    """
    PySide6 QOpenGLWidget to replace the GLUT-based Viewer.
    It incorporates most of the logic from your original Viewer class.
    """
    # Signals for UI updates if needed (e.g., status bar messages)
    message_signal = Signal(str)

    def __init__(self,
        models: List[Model] = None,
        use_coalesced_models: bool = True,
        backface_culling: bool = True,
        wireframe_mode: bool = False,
        bounding_box_mode: int = 0,  # 0: off, 1: wireframe, 2: solid
        zbuffer_occlusion: bool = True,
        antialiasing_enabled: bool = True, # Managed by QSurfaceFormat initially
        show_axes: bool = True,
        edge_rotations: bool = False, # From original viewer

        background_color: Tuple[float, float, float, float] = (0.98, 0.98, 0.85, 1.0),
        axes_renderer: AxesRenderer = None,

        projection_mode: str = "perspective",  # 'perspective' or 'orthographic'
        ortho_scale: float = 20.0,  # World-space width for ortho view

        # Camera parameters (from your viewer)
        camera_pos: glm.vec3 = glm.vec3(10.0, -10.0, 10.0),
        camera_front: glm.vec3 = glm.vec3(0.0, 0.0, 1.0),
        camera_up: glm.vec3 = glm.vec3(0.0, 0.0, 1.0),
        camera_target: glm.vec3 = glm.vec3(0.0, 0.0, 0.0),
        camera_speed: float = 0.05, # Will be adjusted based on scene bounds
        model_matrix_np: np.ndarray = np.eye(4, dtype=np.float32),
        
        # Mouse interaction state
        last_mouse_pos: QPoint = None,
        left_button_pressed: bool = False,
        right_button_pressed: bool = False,
        trackball_start_point: glm.vec3 = None, # For trackball rotation

        # Scene bounds
        bounding_box: BoundingBox = None,

        # Shader program (will be initialized in initializeGL)
        active_shader_program_id: Any = None, # ID of the compiled shader program
        use_shaders: bool = True, # To toggle shader usage

        # GLContext (for capability querying mainly)
        gl_ctx: GLContext = None,
        _multisample_supported_by_context: bool = False, # Queried from QSurfaceFormat
        axes_depth_test: bool = True,
        parent: QWidget = None,
    ):
        self.models = models if models else []
        self.use_coalesced_models = use_coalesced_models
        self.backface_culling = backface_culling
        self.wireframe_mode = wireframe_mode
        self.bounding_box_mode = bounding_box_mode
        self.zbuffer_occlusion = zbuffer_occlusion
        self.antialiasing_enabled = antialiasing_enabled
        self.show_axes = show_axes
        self.edge_rotations = edge_rotations
        self.background_color = background_color
        self.axes_renderer = axes_renderer if axes_renderer else AxesRenderer()
        self.projection_mode = projection_mode
        self.ortho_scale = ortho_scale
        self.camera_pos = camera_pos
        self.camera_front = camera_front
        self.camera_up = camera_up
        self.camera_target = camera_target
        self.camera_speed = camera_speed
        self.model_matrix_np = model_matrix_np
        self.last_mouse_pos = last_mouse_pos
        self.left_button_pressed = left_button_pressed
        self.right_button_pressed = right_button_pressed
        self.trackball_start_point = trackball_start_point
        self.bounding_box = bounding_box
        self.active_shader_program_id = active_shader_program_id
        self.use_shaders = use_shaders
        self.gl_ctx = gl_ctx
        self._multisample_supported_by_context = _multisample_supported_by_context
        self.axes_depth_test = axes_depth_test
    
        # Mouse interaction state
        self.last_mouse_pos = last_mouse_pos
        self.left_button_pressed = left_button_pressed
        self.right_button_pressed = right_button_pressed
        self.trackball_start_point = trackball_start_point

        # Scene bounds
        self.bounding_box = bounding_box

        # Shader program (will be initialized in initializeGL)
        self.active_shader_program_id = active_shader_program_id
        self.use_shaders = use_shaders

        # GLContext (for capability querying mainly)
        self.gl_ctx = gl_ctx
        self._multisample_supported_by_context = _multisample_supported_by_context
        self.axes_depth_test = axes_depth_test
        
        self.__post_init__(parent)


    def __post_init__(self, parent: QWidget =None):
        super().__init__(parent)
        # It's good practice to initialize QOpenGLFunctions within initializeGL,
        # but we make it available if methods outside paintGL need it.
        # self.initializeOpenGLFunctions() # Call this in initializeGL

        self.original_models = self.models # Store for coalescing logic
        self.models = [] # Initialize as empty list

        self.gl_ctx = GLContext.get_instance()

        self._apply_coalescing()
        self._compute_scene_bounds()
        self._setup_camera_from_bounds()

        if PYOPENGL_VERBOSE: # Print diagnostics for the calculated camera
            print("PoscGLWidget.__init__: Camera Setup Diagnostics (from _setup_camera_from_bounds):")
            print(f"  BB Min: {self.bounding_box.min_point if self.bounding_box else 'N/A'}") # Added BB info
            print(f"  BB Max: {self.bounding_box.max_point if self.bounding_box else 'N/A'}")
            print(f"  BB Center: {self.bounding_box.center if self.bounding_box else 'N/A'}")
            print(f"  BB Diagonal: {self.bounding_box.diagonal if self.bounding_box else 'N/A'}")
            print(f"  Camera Pos: {self.camera_pos}")
            print(f"  Camera Target: {self.camera_target}")
            print(f"  Camera Front: {self.camera_front}")
            print(f"  Camera Up: {self.camera_up}")
            print(f"  Camera Speed: {self.camera_speed}")
            print(f"  Ortho Scale: {self.ortho_scale}")

        # For PySide6, focus policy needs to be set to receive keyboard events
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        # Timer for Python interpreter to handle signals (e.g., Ctrl+C)
        # GLUT's keep_alive is not needed. PySide6 handles its event loop.
        # However, for console Ctrl+C, a QTimer can help keep Python responsive.
        self.keep_alive_timer = QTimer(self)
        self.keep_alive_timer.timeout.connect(lambda: None) # Dummy slot
        self.keep_alive_timer.start(200) # ms
        
    def num_triangles(self) -> int:
        return sum((m.num_triangles() for m in self.models))

    def _apply_coalescing(self):
        if self.use_coalesced_models and self.original_models:
            opaque_model, transparent_model = Model.create_coalesced_models(self.original_models)
            self.models = []
            if opaque_model and opaque_model.num_points > 0:
                self.models.append(opaque_model)
            if transparent_model and transparent_model.num_points > 0:
                self.models.append(transparent_model)
        else:
            self.models = list(self.original_models) # Make a copy

    def _compute_scene_bounds(self):
        valid_models = [m for m in self.models if m.bounding_box and \
                        not np.all(np.isinf(m.bounding_box.min_point)) and \
                        not np.all(np.isinf(m.bounding_box.max_point))]

        if not valid_models:
            self.bounding_box = BoundingBox() # Reset
            self.bounding_box.min_point = np.array([-0.5, -0.5, -0.5])
            self.bounding_box.max_point = np.array([0.5, 0.5, 0.5])
            return
        
        current_bbox = valid_models[0].bounding_box
        for model in valid_models[1:]:
            current_bbox = current_bbox.union(model.bounding_box)
        self.bounding_box = current_bbox


    def _setup_camera_from_bounds(self):
        """Set up the camera based on the scene bounds (from original Viewer._setup_camera)."""
        if not self.bounding_box:
            self._compute_scene_bounds() # Ensure bounds are computed
            if not self.bounding_box: # Still no bounds, use defaults
                 if PYOPENGL_VERBOSE: print("PoscGLWidget._setup_camera_from_bounds: No valid bounding_box, using default camera.")
                 self.camera_pos = glm.vec3(10.0, -10.0, 10.0)
                 self.camera_target = glm.vec3(0.0, 0.0, 0.0)
                 self.camera_front = glm.normalize(self.camera_target - self.camera_pos)
                 self.camera_up = glm.vec3(0.0, 0.0, 1.0) # Z-up RESTORED
                 self.camera_speed = 0.05
                 return

        center = self.bounding_box.center
        diagonal = self.bounding_box.diagonal

        if PYOPENGL_VERBOSE:
            print(f"PoscGLWidget._setup_camera_from_bounds: Using scene center: {center}, diagonal: {diagonal}")
            if diagonal == 0:
                print("PoscGLWidget._setup_camera_from_bounds: WARNING - Scene diagonal is ZERO.")

        if diagonal == 0: diagonal = 10.0 # Avoid division by zero or too small speed

        center_vec = glm.vec3(center[0], center[1], center[2])
        direction_vec = glm.normalize(glm.vec3(1.0, -1.0, 1.0)) # Default viewing angle

        self.camera_pos = center_vec + direction_vec * diagonal * 1.5
        self.camera_target = center_vec
        self.camera_front = glm.normalize(self.camera_target - self.camera_pos)
        self.camera_up = glm.vec3(0.0, 0.0, 1.0) # Z-up RESTORED
        self.camera_speed = diagonal * 0.05 # Adjust speed based on scene size
        self.ortho_scale = max(1.0, diagonal if diagonal > 0 else 20.0) # Sensible default ortho scale

    # --- QOpenGLWidget Overrides ---
    def initializeGL(self):
        try:
            # Initialize QOpenGLFunctions for the current context
            # This MUST be called before any other GL functions from this class.
            if PYOPENGL_VERBOSE:
                print(f"PoscGLWidget: Context valid before initializeOpenGLFunctions: {self.context().isValid()}")
                print(f"PoscGLWidget: Widget valid before initializeOpenGLFunctions: {self.isValid()}")
                actual_fmt = self.format()
                profile_map = {
                    QSurfaceFormat.NoProfile: "NoProfile",
                    QSurfaceFormat.CoreProfile: "CoreProfile",
                    QSurfaceFormat.CompatibilityProfile: "CompatibilityProfile"
                }
#                print(f"PoscGLWidget: Actual surface format - Profile: {profile_map.get(actual_fmt.profile(), "Unknown")}, Samples: {actual_fmt.samples()}, Version: {actual_fmt.majorVersion()}.{actual_fmt.minorVersion()}")
                
            functions = self.context().functions() # Get functions from QOpenGLWidget's context
            if PYOPENGL_VERBOSE: print(f"PoscGLWidget: Functions object from context: {functions}")

            if functions:
                functions.initializeOpenGLFunctions() # Initialize GL functions via the context's object
                self.gl_functions = functions # Store for potential later use
                if PYOPENGL_VERBOSE: print("PoscGLWidget: Called functions.initializeOpenGLFunctions() successfully.")
            else:
                if PYOPENGL_VERBOSE: print("PoscGLWidget: ERROR - Could not get functions object from context!")
                # Handle error: This would be a critical failure
                QMessageBox.critical(self, "OpenGL Error", "Could not retrieve OpenGL function pointer object from context.")
                QApplication.quit()
                return

            # Initialize your GLContext for capability querying (if it's not auto-init)
            if not self.gl_ctx.is_initialized:
                 # GLContext might try to create a temp GLUT window if not careful.
                 # QOpenGLWidget already provides a context.
                 # We might need to adapt GLContext.initialize() or call a specific part of it.
                 # For now, assume QOpenGLWidget's context is primary.
                 # We can still query GL strings via self.glGetString(...)
                if PYOPENGL_VERBOSE: print("PoscGLWidget: GLContext not pre-initialized by user, relying on QOpenGLWidget's context.")
                # Faking parts of gl_ctx initialization for capability flags
                self.gl_ctx.opengl_version = gl.glGetString(gl.GL_VERSION).decode()
                self.gl_ctx.glsl_version = gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION).decode()
                # Manually set capability flags based on known context features (OpenGL 4.6 Compatibility)
                self.gl_ctx.has_vbo = True
                self.gl_ctx.has_3_3 = True # OpenGL 4.6 implies 3.3 features (for VAOs)
                self.gl_ctx.has_legacy_lighting = True # Compatibility profile should have legacy lighting
                self.gl_ctx.has_legacy_vertex_arrays = True # Compatibility profile should have legacy vertex arrays
                if PYOPENGL_VERBOSE:
                    print(f"PoscGLWidget: Manually set gl_ctx.has_vbo={self.gl_ctx.has_vbo}, gl_ctx.has_3_3={self.gl_ctx.has_3_3}, legacy_lighting={self.gl_ctx.has_legacy_lighting}, legacy_arrays={self.gl_ctx.has_legacy_vertex_arrays}")

                # Manually set has_shader based on GLSL version for now (RE-INSERTED LOGIC)
                try:
                    glsl_major = int(self.gl_ctx.glsl_version.split('.')[0])
                    glsl_minor = int(self.gl_ctx.glsl_version.split('.')[1].split(' ')[0]) # Handle cases like "4.60" or "4.60 NVIDIA"
                    if glsl_major >= 1 and glsl_minor >= 10: # GLSL 1.10 or higher supports shaders
                        self.gl_ctx.has_shader = True
                        if PYOPENGL_VERBOSE: print(f"PoscGLWidget: Manually set self.gl_ctx.has_shader to True based on GLSL version {self.gl_ctx.glsl_version}")
                    else:
                        self.gl_ctx.has_shader = False
                        if PYOPENGL_VERBOSE: print(f"PoscGLWidget: Manually set self.gl_ctx.has_shader to False based on GLSL version {self.gl_ctx.glsl_version}")
                except Exception as e:
                    if PYOPENGL_VERBOSE: print(f"PoscGLWidget: Could not parse GLSL version '{self.gl_ctx.glsl_version}' to set has_shader: {e}")
                    self.gl_ctx.has_shader = False # Default to false on error

            if PYOPENGL_VERBOSE:
                print(f"PoscGLWidget: Initializing OpenGL. Version: {gl.glGetString(gl.GL_VERSION).decode()}")
                print(f"PoscGLWidget: GLSL Version: {gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION).decode()}")
                print(f"PoscGLWidget: Vendor: {gl.glGetString(gl.GL_VENDOR).decode()}")
                print(f"PoscGLWidget: Renderer: {gl.glGetString(gl.GL_RENDERER).decode()}")

            gl.glClearColor(*self.background_color)
            self._setup_gl_state_from_viewer() # Call the GL state setup logic

            # Compile shaders
            self.active_shader_program_id = None
            if PYOPENGL_VERBOSE:
                gl_has_shader_flag = hasattr(self.gl_ctx, 'has_shader') and self.gl_ctx.has_shader
                print(f"PoscGLWidget: Value of self.gl_ctx.has_shader before check (if attr exists): {gl_has_shader_flag}")

            if self.gl_ctx.has_shader:
                compiled_id = PS_SHADER_PROGRAM.compile()
                if compiled_id:
                    self.active_shader_program_id = compiled_id
                else:
                    compiled_id_basic = BASIC_SHADER_PROGRAM.compile()
                    if compiled_id_basic:
                        self.active_shader_program_id = compiled_id_basic
                    else:
                         if PYOPENGL_VERBOSE: 
                             print("PoscGLWidget: All shader compilation attempts failed.")
            else:
                if PYOPENGL_VERBOSE: 
                    print("PoscGLWidget: Shader support not detected by GLContext (self.gl_ctx.has_shader is False).")

            # Initialize GL resources for models
            if PYOPENGL_VERBOSE:
                print("PoscGLWidget: Initializing GL resources for models...")
            for model_idx, model in enumerate(self.models):
                if model and model.data is not None and model.num_points > 0:
                    if PYOPENGL_VERBOSE:
                        print(f"  Model {model_idx}: num_points={model.num_points}, stride={model.stride}")
                        print(f"    pos_offset={model.position_offset}, color_offset={model.color_offset}, normal_offset={model.normal_offset}")
                        print(f"    has_alpha_lt1={model.has_alpha_lt1}")
                        # Print first 2 vertices worth of data (or less if model is smaller)
                        num_floats_to_print = min(model.stride * 2, len(model.data))
                        print(f"    First {num_floats_to_print} floats of data: {model.data[:num_floats_to_print]}")
                    try:
                        model.initialize_gl_resources()
                        if PYOPENGL_VERBOSE:
                             print(f"    Model {model_idx} VAO: {model.vao}, VBO: {model.vbo}")
                    except Exception as e:
                        if PYOPENGL_VERBOSE:
                            print(f"PoscGLWidget: Error initializing GL resources for model {model_idx}: {e}")
                        raise
                elif PYOPENGL_VERBOSE:
                    print(f"  Model {model_idx}: is None or has no data/points.")
            if PYOPENGL_VERBOSE: 
                print("PoscGLWidget: Finished initializing GL resources for models.")
        except Exception as e:
            print(f"FATAL ERROR in initializeGL: {e}")
            QMessageBox.critical(self, "OpenGL Error", f"Failed to initialize OpenGL: {e}")


    def _setup_gl_state_from_viewer(self):
        """Mirrors _setup_gl from your original Viewer, using QOpenGLWidget's context."""
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glCullFace(gl.GL_BACK)
        if self.backface_culling: 
            gl.glEnable(gl.GL_CULL_FACE)
        else: 
            gl.glDisable(gl.GL_CULL_FACE)
        # if PYOPENGL_VERBOSE: print("PoscGLWidget._setup_gl_state_from_viewer: GL_CULL_FACE TEMPORARILY DISABLED FOR DEBUGGING.")
        # gl.glDisable(gl.GL_CULL_FACE) # TEMP DEBUG: Disable culling to check if cube appears

        if self.wireframe_mode: 
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else: 
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        # Antialiasing (MSAA) is typically requested via QSurfaceFormat.
        # We can check if it was granted.
        current_format = self.format()
        self._multisample_supported_by_context = current_format.samples() > 0
        if PYOPENGL_VERBOSE: 
            print(f"PoscGLWidget: QSurfaceFormat samples: {current_format.samples()}")

        if hasattr(gl, "GL_MULTISAMPLE") and self._multisample_supported_by_context:
            if self.antialiasing_enabled: 
                gl.glEnable(gl.GL_MULTISAMPLE)
            else: 
                gl.glDisable(gl.GL_MULTISAMPLE)
        elif self.antialiasing_enabled:
            if PYOPENGL_VERBOSE: 
                print("PoscGLWidget: Antialiasing was enabled but context does not support multisample.")
            self.antialiasing_enabled = False

        # Legacy lighting (mostly for immediate mode / fixed-function compatibility)
        # This is less relevant if solely relying on shaders.
        # Your shaders (GLSL 1.10/1.20) might expect some fixed-function state.
        try:
            gl.glEnable(gl.GL_COLOR_MATERIAL)
            gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
            if self.gl_ctx.has_legacy_lighting: # Check if context even supports it
                gl.glEnable(gl.GL_LIGHTING)
                gl.glEnable(gl.GL_LIGHT0)
                gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, [0.6, 0.6, 0.6, 1.0])
                gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
                gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
                # Light position will be set in paintGL relative to camera
                gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT, [0.4, 0.4, 0.4, 1.0])
        except Exception as e:
            if PYOPENGL_VERBOSE: 
                print(f"PoscGLWidget: Error setting up legacy lighting/material: {e}")


    def resizeGL(self, w: int, h: int):
        if h == 0: 
            h = 1
        gl.glViewport(0, 0, w, h)

    def paintGL(self):
        try:
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            # --- Shader Setup (if using shaders) ---
            current_program_to_use = 0 # Initialize
            if self.use_shaders and self.active_shader_program_id:
                current_program_to_use = self.active_shader_program_id
                gl.glUseProgram(current_program_to_use)

                # Matrix calculations moved here, only if shader will be used
                projection_mat = self._get_projection_matrix_glm()
                view_mat = self._get_view_matrix_glm() # This uses the hardcoded camera_pos/target/up
                model_mat_glm = glm.mat4(*self.model_matrix_np.flatten())

                if PYOPENGL_VERBOSE:
                    print("PoscGLWidget.paintGL: Setting uniforms using glm.value_ptr()")
                    # print(f"  View Matrix for uniform:\n{view_mat}") # Verbose, can be re-enabled
                    # print(f"  Projection Matrix for uniform:\n{projection_mat}") # Verbose
                    
                attrs = (("projection", projection_mat), ("view", view_mat), ("model", model_mat_glm))
                for name, attr in attrs:
                    loc = gl.glGetUniformLocation(current_program_to_use, name)
                    if loc != -1:
                        gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, glm.value_ptr(attr))

                mvp_loc = gl.glGetUniformLocation(current_program_to_use, "modelViewProj") 
                if mvp_loc != -1: 
                    mvp = projection_mat * view_mat * model_mat_glm 
                    gl.glUniformMatrix4fv(mvp_loc, 1, gl.GL_FALSE, glm.value_ptr(mvp)) 
                
                # Check if PS_SHADER_PROGRAM has program_id attribute before comparison
                is_ps_shader = hasattr(PS_SHADER_PROGRAM, 'program_id') and current_program_to_use == PS_SHADER_PROGRAM.program_id
                if is_ps_shader: 
                    light_pos_world = self.camera_pos + glm.vec3(5,5,10) 
                    gl.glUniform3fv(gl.glGetUniformLocation(current_program_to_use, "lightPos"), 1, glm.value_ptr(light_pos_world))
                    gl.glUniform3fv(gl.glGetUniformLocation(current_program_to_use, "viewPos"), 1, glm.value_ptr(self.camera_pos))
            
            # --- 2. Render Models (Opaque then Transparent) ---
            if not self.use_coalesced_models:
                all_models_to_draw = self.original_models
                opaque_models = [m for m in all_models_to_draw if not m.has_alpha_lt1]
                transparent_models = [m for m in all_models_to_draw if m.has_alpha_lt1]
            else:
                opaque_models = [self.models[0]] if len(self.models) > 0 and self.models[0] else []
                transparent_models = [self.models[1]] if len(self.models) > 1 and self.models[1] else []

            # Z-buffer occlusion pass (if enabled for wireframe)
            if self.zbuffer_occlusion and self.wireframe_mode and self.backface_culling:
                gl.glColorMask(gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
                for model in opaque_models + transparent_models:
                    if model:
                        model.draw(use_shaders=self.use_shaders)
                
                gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                gl.glDepthFunc(gl.GL_LEQUAL)
                gl.glEnable(gl.GL_POLYGON_OFFSET_LINE)
                gl.glPolygonOffset(-1.0, -1.0)

                for model in opaque_models + transparent_models:
                     if model: 
                         model.draw(use_shaders=self.use_shaders)

                gl.glDisable(gl.GL_POLYGON_OFFSET_LINE)
                gl.glDepthFunc(gl.GL_LESS)
                if not self.wireframe_mode:
                    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            else:
                # Regular rendering
                for model in opaque_models:
                    if model: 
                        model.draw(use_shaders=self.use_shaders)
                
                if transparent_models:
                    gl.glDepthMask(gl.GL_FALSE)
                    for model in transparent_models:
                        if model: 
                            model.draw(use_shaders=self.use_shaders)
                    gl.glDepthMask(gl.GL_TRUE)

            # Unbind shader program if it was used, and unbind VAO to clean state for 
            # fixed-function.
            if current_program_to_use != 0:
                gl.glUseProgram(0)
                gl.glBindVertexArray(0)
                for i in range(4): # Disable common attribute indices 0, 1, 2, 3
                    try:
                        gl.glDisableVertexAttribArray(i)
                    except Exception: # Silently ignore errors here
                        pass

            # Ensure a clean slate for fixed-function matrices
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()

            # Setup fixed-function matrices for subsequent immediate mode rendering (BBox, Axes)
            projection_mat_ff = self._get_projection_matrix_glm()
            view_mat_ff = self._get_view_matrix_glm()

            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadMatrixf(np.array(glm.transpose(projection_mat_ff), dtype=np.float32))

            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadMatrixf(np.array(glm.transpose(view_mat_ff), dtype=np.float32))
            
            # --- 3. Render Bounding Box ---
            if self.bounding_box_mode > 0 and self.bounding_box:
                bbox_render_instance = BBoxRender(viewer_ref=self) 
                bbox_render_instance.render()

            # --- 4. Render Axes ---
            if self.show_axes:
                depth_test_enabled_before_axes = gl.glIsEnabled(gl.GL_DEPTH_TEST)
                
                if not self.axes_depth_test:
                    gl.glDisable(gl.GL_DEPTH_TEST) # Keep this, axes often need to be drawn on top
                self.axes_renderer.draw(self)
                
                if depth_test_enabled_before_axes:
                    gl.glEnable(gl.GL_DEPTH_TEST)

        except Exception as e:
            print(f"Error in paintGL: {e}")
            import traceback
            traceback.print_exc()


    def _get_projection_matrix_glm(self) -> glm.mat4:
        """Helper to get projection matrix using pyglm."""
        aspect_ratio = self.width() / self.height() if self.height() > 0 else 1.0
        near_plane = 0.1  # Match viewer.py
        far_plane = 1000.0 # Match viewer.py
        if PYOPENGL_VERBOSE: print(f"PoscGLWidget._get_projection_matrix_glm: Using near_plane = {near_plane}, far_plane = {far_plane} (match viewer.py)")

        if self.projection_mode == "perspective":
            return glm.perspective(glm.radians(45.0), aspect_ratio, near_plane, far_plane)
        else:  # orthographic
            if aspect_ratio >= 1.0:
                ortho_width = self.ortho_scale
                ortho_height = self.ortho_scale / aspect_ratio
            else:
                ortho_height = self.ortho_scale
                ortho_width = self.ortho_scale * aspect_ratio
            return glm.ortho(-ortho_width / 2.0, ortho_width / 2.0,
                             -ortho_height / 2.0, ortho_height / 2.0,
                             near_plane, far_plane) # Match viewer.py

    def _get_view_matrix_glm(self) -> glm.mat4:
        """Helper to get view matrix using pyglm."""
        return glm.lookAt(self.camera_pos, self.camera_target, self.camera_up)

    # --- Mouse Event Handlers (Ported from GLUT callbacks) ---
    def mousePressEvent(self, event: QMouseEvent):
        self.last_mouse_pos = event.position().toPoint() # QPointF to QPoint
        if event.button() == Qt.LeftButton:
            self.left_button_pressed = True
            self.trackball_start_point = self._map_to_sphere(self.last_mouse_pos.x(), self.last_mouse_pos.y())
        elif event.button() == Qt.RightButton:
            self.right_button_pressed = True
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.left_button_pressed = False
        elif event.button() == Qt.RightButton:
            self.right_button_pressed = False
        self.last_mouse_pos = None # Reset last position
        self.trackball_start_point = None
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.last_mouse_pos: # Should not happen if press was registered
            return

        current_pos = event.position().toPoint()
        
        if self.left_button_pressed and self.trackball_start_point is not None:
            p1 = self.trackball_start_point
            p2 = self._map_to_sphere(current_pos.x(), current_pos.y())

            if glm.length(p1 - p2) > 0.001:
                try:
                    view_axis = glm.normalize(glm.cross(p1, p2))
                    angle = glm.acos(glm.clamp(glm.dot(p1, p2), -1.0, 1.0)) * 2.0

                    if not (np.isnan(angle) or angle < 1e-6):
                        inv_view_mat = glm.inverse(self._get_view_matrix_glm())
                        world_axis = glm.normalize(glm.vec3(inv_view_mat * glm.vec4(view_axis, 0.0)))
                        world_rotation = glm.rotate(glm.mat4(1.0), -angle, world_axis) # Revert to NEGATIVE ANGLE
                        if PYOPENGL_VERBOSE: print(f"PoscGLWidget.mouseMoveEvent: Trackball angle: {glm.degrees(angle)}, axis (world): {world_axis}")

                        target_to_cam = self.camera_pos - self.camera_target
                        new_target_to_cam = glm.vec3(world_rotation * glm.vec4(target_to_cam, 0.0))
                        self.camera_pos = self.camera_target + new_target_to_cam
                        self.camera_up = glm.normalize(glm.vec3(world_rotation * glm.vec4(self.camera_up, 0.0)))
                        self.camera_front = glm.normalize(self.camera_target - self.camera_pos)
                        
                        self.trackball_start_point = p2 # Update start for continuous rotation
                except Exception as e:
                    if PYOPENGL_VERBOSE: print(f"PoscGLWidget: Trackball rotation error: {e}")
            
        elif self.right_button_pressed:
            delta_x = current_pos.x() - self.last_mouse_pos.x()
            delta_y = current_pos.y() - self.last_mouse_pos.y()
            
            pan_speed = self.camera_speed * 0.1 # Reduced from 0.5
            try:
                camera_right = glm.normalize(glm.cross(self.camera_front, self.camera_up))
            except ValueError: # Handle gimbal lock or parallel vectors
                camera_right = glm.vec3(1,0,0) # Fallback
            
            pan_up = glm.normalize(glm.cross(camera_right, self.camera_front))
            delta = (-camera_right * delta_x * pan_speed) + (pan_up * delta_y * pan_speed)
            self.camera_pos += delta
            self.camera_target += delta

        self.last_mouse_pos = current_pos # Update for next move event if panning
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        # PySide6 wheel delta is usually in multiples of 120
        angle_delta = event.angleDelta().y()
        direction = 0
        if angle_delta > 0: 
            direction = 1  # Zoom in / Scale down ortho
        elif angle_delta < 0: 
            direction = -1 # Zoom out / Scale up ortho

        if self.projection_mode == "perspective":
            dist = glm.length(self.camera_pos - self.camera_target)
            # Ensure camera_front is correctly normalized
            if glm.length(self.camera_front) > 1e-6:
                 normalized_front = glm.normalize(self.camera_front)
                 new_dist = max(0.1, dist - (direction * self.camera_speed * 5.0)) # Adjusted multiplier (was 50.0)
                 self.camera_pos = self.camera_target - normalized_front * new_dist
            else: # Fallback if camera_front is zero (should not happen if target and pos are different)
                 self.camera_pos += glm.vec3(0,0,1) * (direction * self.camera_speed * 25.0)

        else:  # orthographic
            zoom_factor = 1.04
            if direction > 0: self.ortho_scale /= zoom_factor
            else: self.ortho_scale *= zoom_factor
            self.ortho_scale = max(0.1, self.ortho_scale)
            if PYOPENGL_VERBOSE: print(f"PoscGLWidget: Ortho scale set to {self.ortho_scale}")
        self.update()

    def _map_to_sphere(self, x: int, y: int) -> glm.vec3:
        """Maps window coordinates to a point on a virtual unit sphere for trackball."""
        # Normalize x, y to range [-1, 1] with y inverted (PySide6 y grows downwards)
        win_x = (2.0 * x / self.width()) - 1.0
        win_y = 1.0 - (2.0 * y / self.height())

        dist_sq = win_x * win_x + win_y * win_y
        if dist_sq <= 1.0:
            win_z = np.sqrt(1.0 - dist_sq)
        else: # Point is outside; map to the edge
            norm = np.sqrt(dist_sq)
            win_x /= norm
            win_y /= norm
            win_z = 0.0
        return glm.vec3(win_x, win_y, win_z)

    # --- Keyboard Event Handler ---
    def keyPressEvent(self, event: QKeyEvent):
        key_byte = event.text().encode('utf-8') # Get simple key press as byte
        if event.key() == Qt.Key_Escape: key_byte = b"\x1b"
        # Add more mappings for special keys if needed, or check event.key() directly for non-ASCII

        # Lookup in your KEY_BINDINGS (needs to be accessible here)
        # For simplicity, I'll redefine a small handler here.
        # In a full version, you'd adapt your KEY_BINDINGS system.
        
        action = g_key_bindings.get(key_byte)
        if action:
            if PYOPENGL_VERBOSE: print(f"PoscGLWidget: Key '{key_byte.decode()}' pressed.")
            action(self) # Pass self (the widget) as the 'viewer' instance
            self.update() # Trigger repaint
        else:
            super().keyPressEvent(event) # Pass to base class if not handled

    # --- Methods to mirror original Viewer's functionality ---
    def get_current_window_dims(self) -> Tuple[int, int]: # For AxesRenderer
        return (self.width(), self.height())

    def get_projection_mat(self) -> glm.mat4: # For AxesRenderer, BBoxRender
        return self._get_projection_matrix_glm()

    def get_view_mat(self) -> glm.mat4: # For AxesRenderer, BBoxRender
        return self._get_view_matrix_glm()

    def get_model_mat(self) -> glm.mat4: # For AxesRenderer, BBoxRender
        return glm.mat4(*self.model_matrix_np.flatten())
        
    def get_viewport(self) -> List[int]: # For AxesRenderer
        # Ensure GL context is current if there's any doubt, though it should be during paintGL.
        # self.makeCurrent() # Uncomment if context issues arise here
        viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
        # self.doneCurrent() # If makeCurrent was called
        return viewport

    def update_models_pyside(self, new_models: List[Model]):
        """Updates models and re-initializes GL resources."""
        self.makeCurrent() # Ensure GL context is current for resource deletion/creation
        for old_model in self.models:
            if old_model: old_model.delete() # Assuming Model.delete() cleans GL resources

        self.original_models = new_models
        self._apply_coalescing() # This will update self.models
        
        self._compute_scene_bounds()
        self.reset_view_pyside() # Resets camera based on new bounds

        if PYOPENGL_VERBOSE: print("PoscGLWidget: Re-initializing GL resources for updated models...")
        for model in self.models:
            if model:
                try:
                    model.shader_program_id = self.active_shader_program_id
                    model.initialize_gl_resources()
                except Exception as e:
                    if PYOPENGL_VERBOSE: print(f"PoscGLWidget: Error initializing updated model resources: {e}")
        
        self.doneCurrent()
        self.update()


    def reset_view_pyside(self):
        self.model_matrix_np = np.eye(4, dtype=np.float32)
        self._setup_camera_from_bounds() # Recalculates camera based on current (possibly new) bounds
        # self.ortho_scale = 20.0 # Or derive from bounds too
        if PYOPENGL_VERBOSE: print("PoscGLWidget: View reset.")
        self.update()

    def toggle_shader_usage(self):
        self.use_shaders = not self.use_shaders
        if PYOPENGL_VERBOSE: print(f"PoscGLWidget: Use shaders toggled to {self.use_shaders}")
        # Potentially re-check shader validity if turning on
        if self.use_shaders and self.active_shader_program_id:
            # Assuming _check_shader_program is adapted for PySide6 context
            # if not self._check_shader_program_pyside(self.active_shader_program_id):
            # self.use_shaders = False # Fallback if shader is bad
            pass
        self.update()

    def save_screenshot_pyside(self, filename: str):
        try:
            # QOpenGLWidget.grabFramebuffer() returns a QImage
            qimage = self.grabFramebuffer()
            if qimage.save(filename, "PNG"):
                if PYOPENGL_VERBOSE: print(f"PoscGLWidget: Screenshot saved to {filename}")
                self.message_signal.emit(f"Screenshot saved: {filename}")
            else:
                if PYOPENGL_VERBOSE: print(f"PoscGLWidget: Failed to save screenshot (QImage.save failed).")
                self.message_signal.emit(f"Error saving screenshot.")
        except Exception as e:
            if PYOPENGL_VERBOSE: print(f"PoscGLWidget: Exception saving screenshot: {e}")
            self.message_signal.emit(f"Exception saving screenshot: {e}")

    # Add other toggle methods similarly...
    def toggle_backface_culling_pyside(self):
        self.backface_culling = not self.backface_culling
        # GL state is set in _setup_gl_state_from_viewer, which is called before paint or explicitly
        # For immediate effect in paintGL, we just set the flag and self.update()
        if PYOPENGL_VERBOSE: print(f"PoscGLWidget: Backface culling: {self.backface_culling}")
        self._apply_gl_state_changes() # Helper to re-apply relevant GL state
        self.update()

    def toggle_wireframe_mode_pyside(self):
        self.wireframe_mode = not self.wireframe_mode
        if PYOPENGL_VERBOSE: print(f"PoscGLWidget: Wireframe mode: {self.wireframe_mode}")
        self._apply_gl_state_changes()
        self.update()
        
    def _apply_gl_state_changes(self):
        """Apply any pending GL state changes (wireframe, culling, etc.)."""
        # Configure backface culling
        if self.backface_culling:
            gl.glEnable(gl.GL_CULL_FACE)
            gl.glCullFace(gl.GL_BACK)
        else:
            gl.glDisable(gl.GL_CULL_FACE)

        # Configure polygon mode (wireframe or fill)
        if self.wireframe_mode:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    def _render_scene_offscreen(self):
        """Core rendering logic for offscreen rendering.
        
        This method contains the essential rendering steps without Qt-specific
        paintGL overhead, allowing it to be used for FBO-based offscreen rendering.
        """
        # Apply current GL state
        self._apply_gl_state_changes()
        
        # Enable depth testing
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        # Set up view matrices and lighting similar to paintGL
        projection_matrix = self._get_projection_matrix_glm()
        view_matrix = self._get_view_matrix_glm()
        model_matrix = glm.mat4(1.0)  # Identity for now
        
        # Use shader if available
        using_shader = False
        if self.use_shaders and self.gl_ctx and self.gl_ctx.has_shader and self.active_shader_program_id:
            try:
                gl.glUseProgram(self.active_shader_program_id)
                using_shader = True
                
                # Set up shader uniforms
                model_loc = gl.glGetUniformLocation(self.active_shader_program_id, "model")
                view_loc = gl.glGetUniformLocation(self.active_shader_program_id, "view")
                proj_loc = gl.glGetUniformLocation(self.active_shader_program_id, "projection")
                
                if model_loc != -1:
                    gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, glm.value_ptr(model_matrix))
                if view_loc != -1:
                    gl.glUniformMatrix4fv(view_loc, 1, gl.GL_FALSE, glm.value_ptr(view_matrix))
                if proj_loc != -1:
                    gl.glUniformMatrix4fv(proj_loc, 1, gl.GL_FALSE, glm.value_ptr(projection_matrix))
                
                # Set lighting uniforms
                light_pos_loc = gl.glGetUniformLocation(self.active_shader_program_id, "lightPos")
                if light_pos_loc != -1:
                    gl.glUniform3f(light_pos_loc, 
                                 self.camera_pos.x + 5.0, 
                                 self.camera_pos.y + 5.0, 
                                 self.camera_pos.z + 10.0)
                
                view_pos_loc = gl.glGetUniformLocation(self.active_shader_program_id, "viewPos")
                if view_pos_loc != -1:
                    gl.glUniform3f(view_pos_loc, self.camera_pos.x, self.camera_pos.y, self.camera_pos.z)
                    
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"PoscGLWidget: Shader setup failed for offscreen render: {e}")
                using_shader = False
                gl.glUseProgram(0)
        
        # Set up legacy fixed-function pipeline if not using shaders
        if not using_shader:
            # Set up matrices for legacy rendering
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadMatrixf(glm.value_ptr(projection_matrix))
            
            gl.glMatrixMode(gl.GL_MODELVIEW)
            view_model = view_matrix * model_matrix
            gl.glLoadMatrixf(glm.value_ptr(view_model))
            
            # Set up legacy lighting
            if self.gl_ctx and self.gl_ctx.has_legacy_lighting:
                gl.glEnable(gl.GL_LIGHTING)
                gl.glEnable(gl.GL_LIGHT0)
                gl.glEnable(gl.GL_COLOR_MATERIAL)
                gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        
        # Render models
        for model in self.models:
            try:
                model.draw(use_shaders=using_shader)
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"PoscGLWidget: Error rendering model in offscreen mode: {e}")
        
        # Render axes if enabled
        if self.show_axes and self.axes_renderer:
            try:
                # For axes rendering, we need to provide the viewer-like interface
                # The AxesRenderer expects certain methods to be available
                self.axes_renderer.draw(self)
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"PoscGLWidget: Error rendering axes in offscreen mode: {e}")
        
        # Clean up shader state
        if using_shader:
            gl.glUseProgram(0)

    @staticmethod
    def custom_glu_project(obj_x: float, obj_y: float, obj_z: float,
                           modelview_mat: glm.mat4,
                           projection_mat: glm.mat4,
                           viewport: List[int]) -> Tuple[float, float, float]:
        """
        Custom implementation of gluProject.
        Transforms object coordinates to window coordinates.
        """
        obj_vec4 = glm.vec4(obj_x, obj_y, obj_z, 1.0)

        eye_coords = modelview_mat * obj_vec4
        clip_coords = projection_mat * eye_coords

        if clip_coords.w == 0.0:
            if PYOPENGL_VERBOSE:
                print("PoscGLWidget.custom_glu_project: ERROR - clip_coords.w is zero. Projection cannot proceed.")
            raise ValueError("Custom gluProject: clip_coords.w is zero, division by zero.")

        # Perspective division
        ndc_x = clip_coords.x / clip_coords.w
        ndc_y = clip_coords.y / clip_coords.w
        ndc_z = clip_coords.z / clip_coords.w

        # Map to window coordinates
        # viewport = [vx, vy, vw, vh]
        win_x = viewport[0] + viewport[2] * (ndc_x + 1.0) / 2.0
        win_y = viewport[1] + viewport[3] * (ndc_y + 1.0) / 2.0
        # Depth is mapped to [0, 1] range
        win_z = (ndc_z + 1.0) / 2.0

        if PYOPENGL_VERBOSE:
            # print(f"custom_glu_project: obj({obj_x},{obj_y},{obj_z})")
            # print(f"  eye_coords: {eye_coords}")
            # print(f"  clip_coords: {clip_coords} (w={clip_coords.w})")
            # print(f"  ndc_coords: ({ndc_x},{ndc_y},{ndc_z})")
            # print(f"  viewport: {viewport}")
            # print(f"  win_coords: ({win_x},{win_y},{win_z})")
            pass

        return win_x, win_y, win_z

    @staticmethod
    def custom_glu_unproject(win_x: float, win_y: float, win_z: float,
                             modelview_mat: glm.mat4,
                             projection_mat: glm.mat4,
                             viewport: List[int]) -> Tuple[float, float, float]:
        """
        Custom implementation of gluUnProject.
        Transforms window coordinates back to object coordinates.
        """
        # Combined transformation matrix and its inverse
        transform_mat = projection_mat * modelview_mat
        try:
            inv_transform_mat = glm.inverse(transform_mat)
        except Exception as e: # glm.inverse can fail if matrix is singular
            if PYOPENGL_VERBOSE:
                print(f"PoscGLWidget.custom_glu_unproject: ERROR - glm.inverse(projection * modelview) failed: {e}")
            raise ValueError("Custom gluUnProject: Inverse transformation matrix is singular.")

        # Map window coordinates to NDC
        # viewport = [vx, vy, vw, vh]
        ndc_x = (2.0 * (win_x - viewport[0]) / viewport[2]) - 1.0
        ndc_y = (2.0 * (win_y - viewport[1]) / viewport[3]) - 1.0
        ndc_z = (2.0 * win_z) - 1.0  # win_z is in [0, 1]

        ndc_vec4 = glm.vec4(ndc_x, ndc_y, ndc_z, 1.0)
        
        # Transform NDC to object coordinates (or eye coordinates if only P^-1 was used)
        obj_coords_homogeneous = inv_transform_mat * ndc_vec4

        if obj_coords_homogeneous.w == 0.0:
            if PYOPENGL_VERBOSE:
                print("PoscGLWidget.custom_glu_unproject: ERROR - obj_coords_homogeneous.w is zero. Unprojection cannot proceed.")
            raise ValueError("Custom gluUnProject: obj_coords_homogeneous.w is zero, division by zero.")

        # Perspective divide
        obj_x = obj_coords_homogeneous.x / obj_coords_homogeneous.w
        obj_y = obj_coords_homogeneous.y / obj_coords_homogeneous.w
        obj_z = obj_coords_homogeneous.z / obj_coords_homogeneous.w
        
        if PYOPENGL_VERBOSE:
            # print(f"custom_glu_unproject: win({win_x},{win_y},{win_z})")
            # print(f"  ndc_coords: ({ndc_x},{ndc_y},{ndc_z})")
            # print(f"  obj_coords_homogeneous: {obj_coords_homogeneous} (w={obj_coords_homogeneous.w})")
            # print(f"  obj_coords: ({obj_x},{obj_y},{obj_z})")
            pass

        return obj_x, obj_y, obj_z


# --- Key binding functions adapted for PoscGLWidget ---
# Global dictionary for key bindings to avoid class variable issues with GLUT static style
g_key_bindings: Dict[bytes, Callable[[PoscGLWidget], None]] = {}

def pyside_keybinding(key: bytes):
    def decorator(func: Callable[[PoscGLWidget], None]):
        if key in g_key_bindings:
            raise ValueError(f"Key binding {key} already exists.")
        g_key_bindings[key] = func
        return func
    return decorator

@pyside_keybinding(b"\x1b") # Escape
def terminate_app_pyside(widget: PoscGLWidget):
    if PYOPENGL_VERBOSE: print("PoscGLWidget: ESC key pressed, closing window.")
    widget.window().close() # Close the main window

@pyside_keybinding(b"r")
def reset_view_pyside_key(widget: PoscGLWidget):
    widget.reset_view_pyside()

@pyside_keybinding(b"s")
def save_screenshot_pyside_key(widget: PoscGLWidget):
    from PySide6.QtWidgets import QFileDialog
    # Generate a default filename with timestamp
    timestamp = np.datetime_as_string(np.datetime64('now'), unit='s').replace(':','-') # More filename friendly
    default_filename = f"screenshot_{timestamp}.png"
    
    filename, _ = QFileDialog.getSaveFileName(widget, "Save Screenshot", default_filename, "PNG Images (*.png)")
    if filename:
        widget.save_screenshot_pyside(filename)

@pyside_keybinding(b"b")
def toggle_backface_culling_key(widget: PoscGLWidget):
    widget.toggle_backface_culling_pyside()

@pyside_keybinding(b"w")
def toggle_wireframe_mode_key(widget: PoscGLWidget):
    widget.toggle_wireframe_mode_pyside()

@pyside_keybinding(b"h")
def toggle_shader_usage_key(widget: PoscGLWidget):
    widget.toggle_shader_usage()

@pyside_keybinding(b"o")
def toggle_projection_mode_key(widget: PoscGLWidget):
    if widget.projection_mode == "perspective":
        widget.projection_mode = "orthographic"
    else:
        widget.projection_mode = "perspective"
    if PYOPENGL_VERBOSE: print(f"PoscGLWidget: Projection mode: {widget.projection_mode}")
    widget.update()

@pyside_keybinding(b"+")
def toggle_axes_visibility_key(widget: PoscGLWidget):
    widget.show_axes = not widget.show_axes
    if PYOPENGL_VERBOSE: print(f"PoscGLWidget: Show axes: {widget.show_axes}")
    widget.update()

# ... Add more keybindings similarly ...


class MainWindow(QMainWindow):
    def __init__(self, models: List[Model], parent=None):
        super().__init__(parent)
        self.setWindowTitle("PythonOpenSCAD PySide6 Viewer")
        self.setMinimumSize(800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.gl_widget = PoscGLWidget(models, self)
        layout.addWidget(self.gl_widget)
        
        self.gl_widget.message_signal.connect(self.statusBar().showMessage)
        self.statusBar().showMessage("Viewer Ready. Press '?' for help (if implemented).", 2000)

        # TODO: Create menus or toolbars that call the PoscGLWidget's toggle methods
        # Example:
        # file_menu = self.menuBar().addMenu("&File")
        # save_action = file_menu.addAction("&Save Screenshot...")
        # save_action.triggered.connect(lambda: self.gl_widget.save_screenshot_pyside_key(self.gl_widget)) # Bit awkward
        # reset_view_action = self.menuBar().addAction("&Reset View")
        # reset_view_action.triggered.connect(self.gl_widget.reset_view_pyside)
        
        
@datatree
class Viewer():
    """
    PySide6 QOpenGLWidget to replace the GLUT-based Viewer.
    It incorporates most of the logic from your original Viewer class.
    """
    
    VIEWER_HELP_TEXT = """
    Mouse Controls:
     Left button drag: Rotate camera
     Right button drag: Pan camera
     Left+Right drag or Middle drag or Wheel (not on mac):
         Perspective: zoom / Orthographic: scale
    
    Keyboard Controls:
     A - Toggle multisampling antialiasing (MSAA)
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
     T - Toggle axes stipple (dashed negative axes)
     F - Toggle polygon offset for wireframes
     ESC - Close viewer
     = - Zoom in (keyboard alternative to mouse wheel)
     - - Zoom out (keyboard alternative to mouse wheel)
    """

    # --- State variables from your Viewer class ---
    models: List[Model] = dtfield(default_factory=list)
    width: int = 800
    height: int = 600
    title: str = "PythonOpenSCAD PySide6 Viewer"

    use_coalesced_models: bool = True
    backface_culling: bool = True
    wireframe_mode: bool = False
    bounding_box_mode: int = 0  # 0: off, 1: wireframe, 2: solid
    zbuffer_occlusion: bool = True
    antialiasing_enabled: bool = True # Managed by QSurfaceFormat initially
    show_axes: bool = True
    edge_rotations: bool = False # From original viewer

    background_color: Tuple[float, float, float, float] = (0.98, 0.98, 0.85, 1.0)
    
    axes_renderer_node: Node[AxesRenderer] = Node(AxesRenderer, prefix="axes_") # dtfield if using datatree init
    axes_renderer: AxesRenderer = dtfield(self_default=lambda self: self.axes_renderer_node())

    projection_mode: str = "perspective"  # 'perspective' or 'orthographic'
    ortho_scale: float = 20.0  # World-space width for ortho view

    # Camera parameters (from your viewer)
    camera_pos: glm.vec3 = dtfield(default_factory=lambda: glm.vec3(10.0, -10.0, 10.0))
    camera_front: glm.vec3 = dtfield(default_factory=lambda: glm.vec3(0.0, 0.0, 1.0))
    camera_up: glm.vec3 = dtfield(default_factory=lambda: glm.vec3(0.0, 0.0, 1.0))
    camera_target: glm.vec3 = dtfield(default_factory=lambda: glm.vec3(0.0, 0.0, 0.0))
    camera_speed: float = 0.05 # Will be adjusted based on scene bounds
    model_matrix_np: np.ndarray = dtfield(default_factory=lambda: np.eye(4, dtype=np.float32))
    
    # Mouse interaction state
    last_mouse_pos: QPoint = None
    left_button_pressed: bool = False
    right_button_pressed: bool = False
    trackball_start_point: glm.vec3 = None # For trackball rotation

    # Scene bounds
    bounding_box: BoundingBox = None

    # Shader program (will be initialized in initializeGL)
    active_shader_program_id: Any = None # ID of the compiled shader program
    use_shaders: bool = True # To toggle shader usage

    # GLContext (for capability querying mainly)
    gl_ctx: GLContext = None
    _multisample_supported_by_context: bool = False # Queried from QSurfaceFormat
    
    parent: InitVar[QWidget] = None
    
    axes_depth_test: bool = True    

    _app: QApplication = dtfield(default=None, init=False)
    _main_window: QMainWindow = dtfield(default=None, init=False)
    
    def __post_init__(self, parent: QWidget =None):

        app = QApplication(sys.argv)
        self._app = app

        # --- Setup OpenGL Surface Format ---
        # Request a Compatibility Profile to maximize chances for existing GL calls
        # (especially immediate mode in AxesRenderer and GLSL 1.10/1.20 shaders)
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        # fmt.setVersion(3, 3) # Request specific modern version
        # Instead, request CompatibilityProfile for broader support of older GL features
        fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
        # Enable multisampling for antialiasing if desired
        fmt.setSamples(4) # Or more, check system capabilities. 0 to disable.
        QSurfaceFormat.setDefaultFormat(fmt)
        

        if PYOPENGL_VERBOSE: print("Main: Creating MainWindow...")
        main_window = QMainWindow()
        self._main_window = main_window
        main_window.setWindowTitle(self.title)
        main_window.resize(self.width, self.height)  # Use resize() instead of setBaseSize()
        
        central_widget = QWidget()
        main_window.setCentralWidget(central_widget)
        self.main_window = main_window
        
        layout = QVBoxLayout(central_widget)

        self.gl_widget = PoscGLWidget(
            models=self.models,
            use_coalesced_models=self.use_coalesced_models,
            backface_culling=self.backface_culling,
            wireframe_mode=self.wireframe_mode,
            bounding_box_mode=self.bounding_box_mode,
            zbuffer_occlusion=self.zbuffer_occlusion,
            antialiasing_enabled=self.antialiasing_enabled,
            show_axes=self.show_axes,
            edge_rotations=self.edge_rotations,
            background_color=self.background_color,
            axes_renderer=self.axes_renderer,
            projection_mode=self.projection_mode,
            ortho_scale=self.ortho_scale,
            camera_pos=self.camera_pos,
            camera_front=self.camera_front,
            camera_up=self.camera_up,
            camera_target=self.camera_target,
            camera_speed=self.camera_speed,
            model_matrix_np=self.model_matrix_np,
            last_mouse_pos=self.last_mouse_pos,
            left_button_pressed=self.left_button_pressed,
            right_button_pressed=self.right_button_pressed,
            trackball_start_point=self.trackball_start_point,
            bounding_box=self.bounding_box,
            active_shader_program_id=self.active_shader_program_id,
            use_shaders=self.use_shaders,
            gl_ctx=self.gl_ctx,
            axes_depth_test=self.axes_depth_test,
            parent=main_window)
        
        layout.addWidget(self.gl_widget)
        
        self.gl_widget.message_signal.connect(main_window.statusBar().showMessage)
        main_window.statusBar().showMessage(
            "Viewer Ready. Press '?' for help (if implemented).", 2000)
        
    def offscreen_render(self, filename: str):
        """Renders the model to an offscreen buffer and saves as PNG image.
        
        This method attempts to use OpenGL Framebuffer Objects (FBOs) for true offscreen
        rendering, but falls back to other methods when FBOs are not available.
        """
        # Ensure we have an OpenGL context
        if not self.gl_widget:
            raise RuntimeError("No OpenGL widget available for offscreen rendering")
            
        # Make sure the OpenGL context is current
        self.gl_widget.makeCurrent()
        
        try:
            # Check if FBO functions are available
            fbo_available = (
                hasattr(gl, 'glGenFramebuffers') and 
                hasattr(gl, 'glBindFramebuffer') and
                hasattr(gl, 'glFramebufferRenderbuffer') and
                bool(gl.glGenFramebuffers) and
                bool(gl.glBindFramebuffer) and
                bool(gl.glFramebufferRenderbuffer)
            )
            
            if fbo_available:
                if PYOPENGL_VERBOSE:
                    print("PySide Viewer: Using FBO-based offscreen rendering")
                self._offscreen_render_fbo(filename)
            else:
                if PYOPENGL_VERBOSE:
                    print("PySide Viewer: FBOs not available, using fallback method")
                self._offscreen_render_fallback(filename)
                
        except Exception as e:
            if PYOPENGL_VERBOSE:
                print(f"PySide Viewer: FBO rendering failed, trying fallback: {e}")
            # If FBO method fails, try fallback
            try:
                self._offscreen_render_fallback(filename)
            except Exception as fallback_error:
                print(f"All offscreen rendering methods failed: {fallback_error}", file=sys.stderr)
                raise
        finally:
            # Make context available for widget again
            self.gl_widget.doneCurrent()

    def _offscreen_render_fbo(self, filename: str):
        """FBO-based offscreen rendering implementation."""
        fbo = None
        color_rbo = None
        depth_rbo = None
        original_viewport = None
        
        try:
            # Clear any existing OpenGL errors
            gl.glGetError()
            
            # Generate FBO and renderbuffers
            fbo = gl.glGenFramebuffers(1)
            if gl.glGetError() != gl.GL_NO_ERROR or not fbo:
                raise RuntimeError("Failed to generate Framebuffer Object (FBO)")
                
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
            
            # Create color renderbuffer
            color_rbo = gl.glGenRenderbuffers(1)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, color_rbo)
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGB8, self.width, self.height)
            gl.glFramebufferRenderbuffer(
                gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, color_rbo
            )
            
            # Create depth renderbuffer  
            depth_rbo = gl.glGenRenderbuffers(1)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_rbo)
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, self.width, self.height)
            gl.glFramebufferRenderbuffer(
                gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_rbo
            )
            
            # Check FBO completeness
            status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
            if status != gl.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"Framebuffer is not complete: {status}")
                
            # Set viewport for offscreen rendering
            original_viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
            gl.glViewport(0, 0, self.width, self.height)
            
            # Clear and render using the same methods as the widget's paintGL
            gl.glClearColor(*self.background_color)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            
            # Use the widget's rendering logic
            self.gl_widget._render_scene_offscreen()
            
            # Read pixels from the offscreen buffer
            buffer = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            pixel_data = np.frombuffer(buffer, dtype=np.uint8)
            
            # Save the image (flip vertically since OpenGL reads bottom-up)
            image = pixel_data.reshape((self.height, self.width, 3))
            image = np.flipud(image)
            
            from PIL import Image
            img = Image.fromarray(image)
            img.save(filename, "PNG")
            
            if PYOPENGL_VERBOSE:
                print(f"PySide Viewer: FBO offscreen render saved to {filename}")
                
        finally:
            # Cleanup FBO resources
            if fbo is not None:
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
                gl.glDeleteFramebuffers(1, [fbo])
            if color_rbo is not None:
                gl.glDeleteRenderbuffers(1, [color_rbo])
            if depth_rbo is not None:
                gl.glDeleteRenderbuffers(1, [depth_rbo])
            if original_viewport is not None:
                gl.glViewport(*original_viewport)

    def _offscreen_render_fallback(self, filename: str):
        """Fallback offscreen rendering using Qt's grabFramebuffer or back buffer reading."""
        try:
            # Method 1: Try Qt's grabFramebuffer if widget is visible
            if self.gl_widget.isVisible():
                if PYOPENGL_VERBOSE:
                    print("PySide Viewer: Using Qt grabFramebuffer method")
                qimage = self.gl_widget.grabFramebuffer()
                if qimage.save(filename, "PNG"):
                    if PYOPENGL_VERBOSE:
                        print(f"PySide Viewer: Qt grabFramebuffer saved to {filename}")
                    return
                else:
                    raise RuntimeError("Qt grabFramebuffer failed to save image")
            
            # Method 2: Force render to back buffer and read pixels
            if PYOPENGL_VERBOSE:
                print("PySide Viewer: Using back buffer pixel reading method")
            
            # Store original viewport
            original_viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
            
            # Set viewport to desired size
            gl.glViewport(0, 0, self.width, self.height)
            
            # Clear and render
            gl.glClearColor(*self.background_color)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            
            # Use the widget's rendering logic
            self.gl_widget._render_scene_offscreen()
            
            # Force rendering to complete
            gl.glFinish()
            
            # Read from back buffer
            buffer = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            pixel_data = np.frombuffer(buffer, dtype=np.uint8)
            
            # Save the image (flip vertically since OpenGL reads bottom-up)
            image = pixel_data.reshape((self.height, self.width, 3))
            image = np.flipud(image)
            
            from PIL import Image
            img = Image.fromarray(image)
            img.save(filename, "PNG")
            
            # Restore original viewport
            if original_viewport is not None:
                gl.glViewport(*original_viewport)
            
            if PYOPENGL_VERBOSE:
                print(f"PySide Viewer: Back buffer offscreen render saved to {filename}")
                
        except Exception as e:
            # Method 3: Last resort - force widget to be visible temporarily
            if PYOPENGL_VERBOSE:
                print(f"PySide Viewer: Back buffer method failed ({e}), trying forced visibility method")
            
            was_visible = self.gl_widget.isVisible()
            
            try:
                # Temporarily show the widget if it's not visible
                if not was_visible:
                    self.gl_widget.show()
                    self.gl_widget.update()
                    # Process events to ensure rendering occurs
                    from PySide6.QtCore import QCoreApplication
                    QCoreApplication.processEvents()
                
                # Use Qt's grabFramebuffer method
                qimage = self.gl_widget.grabFramebuffer()
                if qimage.save(filename, "PNG"):
                    if PYOPENGL_VERBOSE:
                        print(f"PySide Viewer: Forced visibility method saved to {filename}")
                else:
                    raise RuntimeError("Final fallback method failed to save image")
                    
            finally:
                # Restore original visibility state
                if not was_visible:
                    self.gl_widget.hide()

    def num_triangles(self) -> int:
        return self.gl_widget.num_triangles()

    def run(self):
        self._main_window.show()

        if PYOPENGL_VERBOSE: print("Main: Starting PySide6 event loop...")
        result = self._app.exec()
        sys.exit(result)
    
    
if __name__ == '__main__':
    if PYOPENGL_VERBOSE: print("Main: Creating test models...")
    try:
        triangle = create_triangle_model()
        color_cube = create_colored_test_cube(2.0)
        initial_models = [color_cube, triangle]
    except Exception as e:
        print(f"Error creating models: {e}")
        initial_models = []
        QMessageBox.critical(None, "Model Error", f"Failed to create initial models: {e}")
        sys.exit(1)
            
    viewer = Viewer(models=initial_models)
    viewer.run()