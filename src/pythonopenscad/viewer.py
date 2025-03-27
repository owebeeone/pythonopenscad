"""OpenGL 3D viewer for PyOpenSCAD models."""

import numpy as np
import ctypes
from typing import List, Optional, Tuple, Union, Dict, Callable
from dataclasses import dataclass, field
import warnings
import sys

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

# Check for various OpenGL features
HAS_OPENGL_3_3 = False  # Modern OpenGL with VAOs
HAS_OPENGL_VBO = False  # OpenGL with VBOs but maybe not VAOs
HAS_OPENGL_SHADER = False  # OpenGL with shader support

if HAS_OPENGL:
    try:
        # Check for VBO support
        HAS_OPENGL_VBO = hasattr(gl, 'glGenBuffers') and callable(gl.glGenBuffers) and bool(gl.glGenBuffers)
        # Check for shader support
        HAS_OPENGL_SHADER = (
            hasattr(gl, 'glCreateShader') and callable(gl.glCreateShader) and bool(gl.glCreateShader) and
            hasattr(gl, 'glCreateProgram') and callable(gl.glCreateProgram) and bool(gl.glCreateProgram)
        )
        # Check for VAO support (OpenGL 3.0+)
        HAS_OPENGL_3_3 = HAS_OPENGL_VBO and HAS_OPENGL_SHADER and hasattr(gl, 'glGenVertexArrays') and callable(gl.glGenVertexArrays) and bool(gl.glGenVertexArrays)
        
        # Check for legacy fixed-function pipeline support
        HAS_LEGACY_LIGHTING = hasattr(gl, 'GL_LIGHTING') and hasattr(gl, 'GL_LIGHT0')
        
        # Check for legacy vertex array support
        HAS_LEGACY_VERTEX_ARRAYS = (
            hasattr(gl, 'GL_VERTEX_ARRAY') and
            hasattr(gl, 'glEnableClientState') and
            hasattr(gl, 'glVertexPointer')
        )
    except (AttributeError, TypeError):
        pass

    # Output warnings for missing features
    if not HAS_OPENGL_VBO:
        warnings.warn("OpenGL VBO functions not available. Rendering may not work.")
    if not HAS_OPENGL_SHADER:
        warnings.warn("OpenGL shader functions not available. Using fixed-function pipeline.")
    if not HAS_OPENGL_3_3:
        warnings.warn("OpenGL 3.3+ core profile features not available. Using compatibility mode.")
    if not HAS_LEGACY_LIGHTING and not HAS_OPENGL_SHADER:
        warnings.warn("Neither modern shaders nor legacy lighting available. Rendering will be unlit.")


@dataclass
class BoundingBox:
    """3D bounding box with min and max points."""
    min_point: np.ndarray = field(default_factory=lambda: np.array([float('inf'), float('inf'), float('inf')]))
    max_point: np.ndarray = field(default_factory=lambda: np.array([float('-inf'), float('-inf'), float('-inf')]))

    @property
    def size(self) -> np.ndarray:
        """Get the size of the bounding box as a 3D vector."""
        return self.max_point - self.min_point
    
    @property
    def center(self) -> np.ndarray:
        """Get the center of the bounding box."""
        # Ensure we always return a 3D vector even for an empty bounding box
        if np.all(np.isinf(self.min_point)) or np.all(np.isinf(self.max_point)):
            return np.array([0.0, 0.0, 0.0])
        return (self.max_point + self.min_point) / 2.0
    
    @property
    def diagonal(self) -> float:
        """Get the diagonal length of the bounding box."""
        if np.all(np.isinf(self.min_point)) or np.all(np.isinf(self.max_point)):
            return 1.0  # Return a default value for empty/invalid bounding boxes
        return np.linalg.norm(self.size)
    
    def union(self, other: 'BoundingBox') -> 'BoundingBox':
        """Compute the union of this bounding box with another."""
        # Handle the case where one of the bounding boxes is empty
        if np.all(np.isinf(self.min_point)):
            return other
        if np.all(np.isinf(other.min_point)):
            return self
            
        return BoundingBox(
            min_point=np.minimum(self.min_point, other.min_point),
            max_point=np.maximum(self.max_point, other.max_point)
        )
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is inside the bounding box."""
        if np.all(np.isinf(self.min_point)) or np.all(np.isinf(self.max_point)):
            return False
        return np.all(point >= self.min_point) and np.all(point <= self.max_point)


class Model:
    """3D model with vertex data including positions, colors, and normals."""
    
    def __init__(self, 
                 data: np.ndarray,
                 num_points: Optional[int] = None,
                 position_offset: int = 0,
                 color_offset: int = 3,
                 normal_offset: int = 7,
                 stride: int = 10):
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
        if not HAS_OPENGL:
            raise ImportError("OpenGL libraries (PyOpenGL and PyGLM) are required for the viewer")
        
        self.data = data.astype(np.float32)
        self.num_points = num_points if num_points is not None else len(data) // stride
        self.position_offset = position_offset
        self.color_offset = color_offset
        self.normal_offset = normal_offset
        self.stride = stride
        
        # OpenGL objects
        self.vao = None
        self.vbo = None
        self._init_gl()
        
        # Compute bounding box
        self._compute_bounding_box()
    
    def _init_gl(self):
        """Initialize OpenGL vertex buffer and array objects."""
        # Skip if OpenGL is not available
        if not HAS_OPENGL:
            return
        
        # Skip VBO/VAO initialization if not supported
        if not HAS_OPENGL_VBO:
            return
            
        try:
            # Create VBO
            self.vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, self.data.nbytes, self.data, gl.GL_STATIC_DRAW)
            
            # Use VAO if supported (OpenGL 3.3+)
            if HAS_OPENGL_3_3:
                try:
                    # Create and bind VAO
                    self.vao = gl.glGenVertexArrays(1)
                    gl.glBindVertexArray(self.vao)
                    
                    # Set up vertex attributes
                    # Position attribute
                    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 
                                    self.stride * 4, ctypes.c_void_p(self.position_offset * 4))
                    gl.glEnableVertexAttribArray(0)
                    
                    # Color attribute
                    gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 
                                    self.stride * 4, ctypes.c_void_p(self.color_offset * 4))
                    gl.glEnableVertexAttribArray(1)
                    
                    # Normal attribute
                    gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 
                                    self.stride * 4, ctypes.c_void_p(self.normal_offset * 4))
                    gl.glEnableVertexAttribArray(2)
                    
                    # Unbind VAO
                    gl.glBindVertexArray(0)
                except Exception as e:
                    warnings.warn(f"Failed to initialize VAO: {e}")
                    self.vao = None
            
            # Unbind VBO
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        except Exception as e:
            warnings.warn(f"Failed to initialize VBO: {e}")
            self.vbo = None
    
    def _compute_bounding_box(self):
        """Compute the bounding box of the model."""
        self.bounding_box = BoundingBox()
        
        if self.data is None or len(self.data) == 0 or self.num_points <= 0:
            # Empty model, leave the bounding box with its default values
            return
        
        # Extract positions from the interleaved data
        try:
            # Get position data
            positions = self.data[self.position_offset::self.stride]
            if len(positions) > self.num_points:
                positions = positions[:self.num_points]
            
            # Reshape to (N, 3) if needed
            if len(positions) % 3 == 0:
                positions = positions.reshape(-1, 3)
            
                # Update bounding box if we have valid positions
                if len(positions) > 0:
                    self.bounding_box.min_point = np.min(positions, axis=0)
                    self.bounding_box.max_point = np.max(positions, axis=0)
        except Exception as e:
            warnings.warn(f"Failed to compute bounding box: {e}")
            # Use default bounding box (centered at origin with unit size)
            self.bounding_box.min_point = np.array([-0.5, -0.5, -0.5])
            self.bounding_box.max_point = np.array([0.5, 0.5, 0.5])
    
    def draw(self):
        """Draw the model using OpenGL."""
        if not HAS_OPENGL:
            return
            
        # Use VBO/VAO if available
        if HAS_OPENGL_VBO and self.vbo:
            try:
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
                
                # If we have VAOs, use them
                if HAS_OPENGL_3_3 and self.vao:
                    gl.glBindVertexArray(self.vao)
                    gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.num_points)
                    gl.glBindVertexArray(0)
                elif HAS_LEGACY_VERTEX_ARRAYS:
                    # Legacy mode without VAOs but with VBOs and vertex arrays
                    # Position attribute
                    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
                    gl.glVertexPointer(3, gl.GL_FLOAT, self.stride * 4, ctypes.c_void_p(self.position_offset * 4))
                    
                    # Color attribute (if available)
                    if hasattr(gl, 'GL_COLOR_ARRAY'):
                        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
                        gl.glColorPointer(4, gl.GL_FLOAT, self.stride * 4, ctypes.c_void_p(self.color_offset * 4))
                    
                    # Normal attribute (if available)
                    if hasattr(gl, 'GL_NORMAL_ARRAY'):
                        gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
                        gl.glNormalPointer(gl.GL_FLOAT, self.stride * 4, ctypes.c_void_p(self.normal_offset * 4))
                    
                    # Draw
                    gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.num_points)
                    
                    # Disable arrays
                    gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
                    if hasattr(gl, 'GL_COLOR_ARRAY'):
                        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
                    if hasattr(gl, 'GL_NORMAL_ARRAY'):
                        gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
                else:
                    # Fall back to immediate mode
                    self._draw_immediate_mode()
                
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            except Exception as e:
                warnings.warn(f"Error drawing with VBO: {e}")
                # Fall back to immediate mode
                self._draw_immediate_mode()
        else:
            # Use immediate mode as fallback
            self._draw_immediate_mode()
    
    def _draw_immediate_mode(self):
        """Draw using legacy immediate mode (OpenGL 1.x) as a fallback."""
        try:
            # Try to use immediate mode as last resort
            gl.glBegin(gl.GL_TRIANGLES)
            for i in range(0, len(self.data), self.stride):
                # Get position (3 floats)
                pos = self.data[i + self.position_offset:i + self.position_offset + 3]
                # Get color (4 floats)
                color = self.data[i + self.color_offset:i + self.color_offset + 4]
                # Get normal (3 floats)
                normal = self.data[i + self.normal_offset:i + self.normal_offset + 3]
                
                # Set normal, color, and vertex (might fail in core profiles)
                try:
                    gl.glNormal3fv(normal)
                except Exception:
                    pass  # Skip normal if not supported
                    
                try:
                    gl.glColor4fv(color)
                except Exception:
                    pass  # Skip color if not supported
                    
                gl.glVertex3fv(pos)  # This should always work
            gl.glEnd()
        except Exception as e:
            # Last resort failed
            warnings.warn(f"All rendering methods failed: {e}")
            # Don't raise, just silently fail to render this model
    
    def delete(self):
        """Delete OpenGL resources."""
        if HAS_OPENGL_3_3 and self.vao:
            try:
                gl.glDeleteVertexArrays(1, [self.vao])
            except Exception:
                pass
            self.vao = None
        
        if HAS_OPENGL_VBO and self.vbo:
            try:
                gl.glDeleteBuffers(1, [self.vbo])
            except Exception:
                pass
            self.vbo = None
    
    def __del__(self):
        """Destructor to clean up OpenGL resources."""
        self.delete()


class Viewer:
    """OpenGL viewer for 3D models."""

    # Shader source code
    VERTEX_SHADER = """
    #version 330 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec4 aColor;
    layout(location = 2) in vec3 aNormal;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    out vec3 FragPos;
    out vec4 VertexColor;
    out vec3 Normal;
    
    void main() {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;
        VertexColor = aColor;
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
    """

    FRAGMENT_SHADER = """
    #version 330 core
    in vec3 FragPos;
    in vec4 VertexColor;
    in vec3 Normal;
    
    uniform vec3 lightPos;
    uniform vec3 viewPos;
    
    out vec4 FragColor;
    
    void main() {
        // Ambient
        float ambientStrength = 0.3;
        vec3 ambient = ambientStrength * VertexColor.rgb;
        
        // Diffuse
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * VertexColor.rgb;
        
        // Specular
        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
        
        // Result
        vec3 result = ambient + diffuse + specular;
        FragColor = vec4(result, VertexColor.a);
    }
    """

    # Static window registry to handle GLUT callbacks
    _instances: Dict[int, 'Viewer'] = {}
    _initialized = False
    _next_id = 0

    @classmethod
    def _init_glut(cls):
        """Initialize GLUT if not already initialized."""
        if not cls._initialized:
            # Initialize GLUT
            glut.glutInit()
            glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
            
            # Request an OpenGL 3.3 context if shader support is available
            # otherwise use compatibility profile
            if HAS_OPENGL_SHADER and HAS_OPENGL_3_3:
                try:
                    glut.glutInitContextVersion(3, 3)
                    glut.glutInitContextProfile(glut.GLUT_CORE_PROFILE)
                except (AttributeError, ValueError) as e:
                    warnings.warn(f"Failed to set OpenGL context version: {e}")
            
            cls._initialized = True

    def __init__(self, models: List[Model], width: int = 800, height: int = 600, title: str = "3D Viewer"):
        """
        Initialize the viewer with a list of models.
        
        Args:
            models: List of Model objects to display
            width: Window width
            height: Window height
            title: Window title
        """
        if not HAS_OPENGL:
            raise ImportError("OpenGL libraries (PyOpenGL and PyGLM) are required for the viewer")
        
        self.models = models
        self.width = width
        self.height = height
        self.title = title
        
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
        self.last_mouse_x = width // 2
        self.last_mouse_y = height // 2
        self.first_mouse = True
        self.mouse_button_pressed = False
        self.mouse_start_x = 0
        self.mouse_start_y = 0
        
        # Compute bounding box and set up camera
        self._compute_scene_bounds()
        self._setup_camera()
        
        # OpenGL state
        self.window_id = None
        self.shader_program = None
        
        # Register this instance
        self.instance_id = Viewer._next_id
        Viewer._next_id += 1
        Viewer._instances[self.instance_id] = self
        
        # Create the window and set up OpenGL
        self._create_window()
        self._setup_gl()
    
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
        
        # Register callbacks
        glut.glutDisplayFunc(lambda: self._display_callback())
        glut.glutReshapeFunc(lambda w, h: self._reshape_callback(w, h))
        glut.glutMouseFunc(lambda button, state, x, y: self._mouse_callback(button, state, x, y))
        glut.glutMotionFunc(lambda x, y: self._motion_callback(x, y))
        glut.glutMouseWheelFunc(lambda wheel, direction, x, y: self._wheel_callback(wheel, direction, x, y))
        glut.glutKeyboardFunc(lambda key, x, y: self._keyboard_callback(key, x, y))
    
    def _setup_gl(self):
        """Set up OpenGL state and compile shaders."""
        # Enable depth testing
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        # Compile shaders if supported
        if HAS_OPENGL_SHADER:
            try:
                self._compile_shaders()
                
                # Set up lighting if shader program was created successfully
                if self.shader_program:
                    gl.glUseProgram(self.shader_program)
                    light_pos_loc = gl.glGetUniformLocation(self.shader_program, "lightPos")
                    gl.glUniform3f(light_pos_loc, 
                                self.camera_pos.x + 10.0, 
                                self.camera_pos.y + 10.0, 
                                self.camera_pos.z)
            except Exception as e:
                warnings.warn(f"Failed to compile shaders: {e}")
                self.shader_program = None
        elif HAS_LEGACY_LIGHTING:
            # Set up basic lighting for fixed-function pipeline
            try:
                gl.glEnable(gl.GL_LIGHTING)
                gl.glEnable(gl.GL_LIGHT0)
                gl.glEnable(gl.GL_COLOR_MATERIAL)
                
                # Set up a light
                light_position = [self.camera_pos.x + 10.0, self.camera_pos.y + 10.0, self.camera_pos.z, 1.0]
                gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_position)
                
                # Set ambient and diffuse lighting
                ambient = [0.3, 0.3, 0.3, 1.0]
                diffuse = [0.7, 0.7, 0.7, 1.0]
                gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, ambient)
                gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, diffuse)
            except Exception as e:
                warnings.warn(f"Failed to initialize legacy lighting: {e}")
        else:
            # No lighting available - using unlit rendering
            warnings.warn("No lighting available. Rendering will be unlit.")
    
    def _compile_shaders(self):
        """Compile and link the shader program."""
        if not HAS_OPENGL_SHADER:
            return
            
        # Create vertex shader
        vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vertex_shader, self.VERTEX_SHADER)
        gl.glCompileShader(vertex_shader)
        
        # Check for vertex shader compilation errors
        if not gl.glGetShaderiv(vertex_shader, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(vertex_shader)
            gl.glDeleteShader(vertex_shader)
            raise RuntimeError(f"Vertex shader compilation failed: {error}")
        
        # Create fragment shader
        fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fragment_shader, self.FRAGMENT_SHADER)
        gl.glCompileShader(fragment_shader)
        
        # Check for fragment shader compilation errors
        if not gl.glGetShaderiv(fragment_shader, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(fragment_shader)
            gl.glDeleteShader(vertex_shader)
            gl.glDeleteShader(fragment_shader)
            raise RuntimeError(f"Fragment shader compilation failed: {error}")
        
        # Create and link shader program
        self.shader_program = gl.glCreateProgram()
        gl.glAttachShader(self.shader_program, vertex_shader)
        gl.glAttachShader(self.shader_program, fragment_shader)
        gl.glLinkProgram(self.shader_program)
        
        # Check for linking errors
        if not gl.glGetProgramiv(self.shader_program, gl.GL_LINK_STATUS):
            error = gl.glGetProgramInfoLog(self.shader_program)
            gl.glDeleteShader(vertex_shader)
            gl.glDeleteShader(fragment_shader)
            gl.glDeleteProgram(self.shader_program)
            raise RuntimeError(f"Shader program linking failed: {error}")
        
        # Delete shaders (they're not needed after linking)
        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)
    
    def _display_callback(self):
        """GLUT display callback."""
        # Clear the color and depth buffers
        gl.glClearColor(0.2, 0.2, 0.2, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # Set up view
        self._setup_view()
        
        # Draw all models
        for model in self.models:
            model.draw()
        
        # Swap buffers
        glut.glutSwapBuffers()
    
    def _setup_view(self):
        """Set up the view transformation for rendering."""
        if HAS_OPENGL_SHADER and self.shader_program:
            # Use the shader program for modern pipeline
            gl.glUseProgram(self.shader_program)
            
            # Update view position for specular highlights
            view_pos_loc = gl.glGetUniformLocation(self.shader_program, "viewPos")
            gl.glUniform3f(view_pos_loc, self.camera_pos.x, self.camera_pos.y, self.camera_pos.z)
            
            # Set up model-view-projection matrices
            model = self.model_matrix
            view = glm.lookAt(self.camera_pos, self.camera_pos + self.camera_front, self.camera_up)
            projection = glm.perspective(glm.radians(45.0), self.width / self.height, 0.1, 1000.0)
            
            # Send matrices to the shader
            model_loc = gl.glGetUniformLocation(self.shader_program, "model")
            view_loc = gl.glGetUniformLocation(self.shader_program, "view")
            proj_loc = gl.glGetUniformLocation(self.shader_program, "projection")
            
            gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, glm.value_ptr(model))
            gl.glUniformMatrix4fv(view_loc, 1, gl.GL_FALSE, glm.value_ptr(view))
            gl.glUniformMatrix4fv(proj_loc, 1, gl.GL_FALSE, glm.value_ptr(projection))
        else:
            # Use fixed-function pipeline or core profile with no shaders
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
                gl.glMultMatrixd(self.model_matrix)
            except Exception as e:
                # Core profile with no shaders and no fixed function pipeline
                warnings.warn(f"Failed to set up legacy view matrix: {e}")
    
    def _reshape_callback(self, width, height):
        """GLUT window reshape callback."""
        self.width = width
        self.height = height
        gl.glViewport(0, 0, width, height)
    
    def _mouse_callback(self, button, state, x, y):
        """GLUT mouse button callback."""
        if button == glut.GLUT_LEFT_BUTTON:
            if state == glut.GLUT_DOWN:
                self.mouse_button_pressed = True
                self.mouse_start_x = x
                self.mouse_start_y = y
            elif state == glut.GLUT_UP:
                self.mouse_button_pressed = False
    
    def _motion_callback(self, x, y):
        """GLUT mouse motion callback."""
        if self.mouse_button_pressed:
            dx = x - self.mouse_start_x
            dy = y - self.mouse_start_y
            self.mouse_start_x = x
            self.mouse_start_y = y
            
            # Update rotation angles
            sensitivity = 0.5
            dx *= sensitivity
            dy *= sensitivity
            
            # Update yaw and pitch
            self.yaw += dx
            self.pitch += dy
            
            vec = linear.GVector((dy, dx, 0))
            veclen = vec.length()
            unitvec = vec.N
            
            self.model_matrix = self.model_matrix @ linear.rotV(unitvec, -veclen).A

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
            self.close()
        elif key == b'r':
            # Reset view
            self._reset_view()
            glut.glutPostRedisplay()
    
    def _reset_view(self):
        """Reset camera and model transformations to defaults."""
        # Reset model matrix
        self.model_matrix = np.eye(4, dtype=np.float32)
        
        # Reset camera position and orientation
        self._setup_camera()
        
        # Reset mouse rotation tracking
        self.yaw = -90.0
        self.pitch = 0.0
    
    def close(self):
        """Close the viewer window and clean up resources."""
        # Clean up models
        for model in self.models:
            model.delete()
        
        # Clean up shader program
        if HAS_OPENGL_SHADER and self.shader_program:
            try:
                gl.glDeleteProgram(self.shader_program)
            except Exception:
                pass
            self.shader_program = None
        
        # Remove instance from registry
        if self.instance_id in Viewer._instances:
            del Viewer._instances[self.instance_id]
        
        # Destroy window
        if self.window_id is not None:
            glut.glutDestroyWindow(self.window_id)
            self.window_id = None
    
    def run(self):
        """Start the main rendering loop."""
        glut.glutMainLoop()
    
    @staticmethod
    def create_triangle_model(color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)) -> Model:
        """Create a simple triangle model for testing."""
        # Create a simple colored triangle
        vertex_data = np.array([
            # position (3)     # color (4)                # normal (3)
            -1.5, -1.5, 0.0,   color[0], color[1], color[2], color[3],   0.0, 0.0, 1.0,
            1.5, -1.5, 0.0,    color[0], color[1], color[2], color[3],   0.0, 0.0, 1.0,
            0.0, 1.5, 0.0,     color[0], color[1], color[2], color[3],   0.0, 0.0, 1.0
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
            
            # Create two triangles for each face
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


# Helper function to create a viewer with models
def create_viewer_with_models(models, width=800, height=600, title="3D Viewer"):
    """Create and return a viewer with the given models."""
    viewer = Viewer(models, width, height, title)
    return viewer


# Helper function to check if OpenGL is available
def is_opengl_available():
    """Check if OpenGL libraries are available."""
    return HAS_OPENGL

def get_opengl_capabilities():
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
        "vbo_support": HAS_OPENGL_VBO,
        "shader_support": HAS_OPENGL_SHADER,
        "vao_support": HAS_OPENGL_3_3,
    }
    
    if HAS_OPENGL_3_3:
        capabilities["message"] = "Full modern OpenGL 3.3+ support available."
    elif HAS_OPENGL_SHADER and HAS_OPENGL_VBO:
        capabilities["message"] = "OpenGL with shaders and VBOs available, but no VAO support."
    elif HAS_OPENGL_VBO:
        capabilities["message"] = "OpenGL with VBO support available, but no shader support."
    else:
        capabilities["message"] = "Basic OpenGL available, using legacy immediate mode."
    
    return capabilities


# If this module is run directly, show a simple demo
if __name__ == "__main__":
    if not HAS_OPENGL:
        print("OpenGL libraries (PyOpenGL and PyGLM) are required for the viewer")
        import sys
        sys.exit(1)
    
    # Print OpenGL capabilities
    capabilities = get_opengl_capabilities()
    print(f"OpenGL capabilities: {capabilities['message']}")
    
    # Create a triangle and a cube
    triangle = Viewer.create_triangle_model()
    cube = Viewer.create_cube_model()
    
    # Create a viewer with both models
    viewer = create_viewer_with_models([triangle, cube], title="PyOpenSCAD Viewer Demo")
    
    # Start the main loop
    viewer.run() 