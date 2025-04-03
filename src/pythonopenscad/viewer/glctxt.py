
import sys
from datatrees import datatree, dtfield, Node
from typing import ClassVar, Optional
import numpy as np
import warnings

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

PYOPENGL_VERBOSE = True

@datatree
class GLContext:
    """Singleton class to manage OpenGL context and capabilities."""
    
    # Class variable for the singleton instance
    _instance: ClassVar[Optional['GLContext']] = None
    
    # OpenGL capabilities
    is_initialized: bool = False
    opengl_version: Optional[str] = None
    glsl_version: Optional[str] = None
    has_vbo: bool = False
    has_shader: bool = False
    has_vao: bool = False
    has_legacy_lighting: bool = False
    has_legacy_vertex_arrays: bool = False
    has_3_3: bool = False
    
    # GLUT state tracking
    temp_window_id: Optional[int] = None
    dummy_display_callback = None
    
    def __post_init__(self):
        """Handle post-initialization setup."""
        if not HAS_OPENGL:
            return
    
    @classmethod
    def get_instance(cls) -> 'GLContext':
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = GLContext()
        return cls._instance
    
    @classmethod
    def _dummy_display(cls):
        """Empty display function for the temporary initialization window."""
        if PYOPENGL_VERBOSE:
            current_window = glut.glutGetWindow()
            print(f"GLContext: _dummy_display callback executed for window ID: {current_window}")
        
    def initialize(self):
        """Initialize the OpenGL context and detect capabilities.
        
        This must be called after glutInit() has been executed.
        """
        if not HAS_OPENGL or self.is_initialized:
            return
        
        # Initialize GLUT if needed
        glut.glutInit()
        
        # Check if we already have a temporary window
        if self.temp_window_id is not None:
            # If we already have a temp window, make sure it's current
            try:
                glut.glutSetWindow(self.temp_window_id)
                if PYOPENGL_VERBOSE:
                    print(f"GLContext: Reusing existing temp window with ID: {self.temp_window_id}")
            except Exception:
                # If there was a problem, clear the reference so we create a new one
                self.temp_window_id = None
        
        # Create a temporary window if needed
        if self.temp_window_id is None:
            try:
                display_mode = glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_DEPTH
                glut.glutInitDisplayMode(display_mode)
                glut.glutInitWindowSize(1, 1)  # Minimal window size
                
                # Try to create window off-screen
                glut.glutInitWindowPosition(-9999, -9999)
                self.temp_window_id = glut.glutCreateWindow(b"OpenGL Init")
                if PYOPENGL_VERBOSE:
                    print(f"GLContext: Created temp window with ID: {self.temp_window_id}")
                
                # Store the current window for proper cleanup (in case of nested calls)
                current_window = glut.glutGetWindow()
                if PYOPENGL_VERBOSE:
                    print(f"GLContext: Current window before callback setup: {current_window}")
                
                # Make sure we're operating on the temporary window
                glut.glutSetWindow(self.temp_window_id)
                
                # IMPORTANT: Register a display callback for the temporary window
                # Create a proper method reference to prevent garbage collection issues
                self.dummy_display_callback = GLContext._dummy_display
                glut.glutDisplayFunc(self.dummy_display_callback)
                if PYOPENGL_VERBOSE:
                    print(f"GLContext: Registered display callback for window ID: {self.temp_window_id}")
                
                # Force a redisplay to ensure the callback is executed
                glut.glutPostRedisplay()
            except Exception as e:
                if PYOPENGL_VERBOSE:
                    print(f"GLContext: Error creating temporary window: {e}")
                # Reset state and set fallback capabilities
                self.temp_window_id = None
                self._set_fallback_capabilities()
                self.is_initialized = True
                return
        
        try:
            # Now we have a valid OpenGL context, detect capabilities
            try:
                self.opengl_version = gl.glGetString(gl.GL_VERSION)
                if self.opengl_version is not None:
                    self.opengl_version = self.opengl_version.decode()
                else:
                    self.opengl_version = "Unknown"
            except Exception:
                self.opengl_version = "Unknown"
                    
            try:
                self.glsl_version = gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
                if self.glsl_version is not None:
                    self.glsl_version = self.glsl_version.decode()
                else:
                    self.glsl_version = "Not supported"
            except Exception:
                self.glsl_version = "Not supported"
            
            # Check for feature support
            self._detect_opengl_features()
            
            # Output detailed information if verbose
            if PYOPENGL_VERBOSE:
                warnings.warn(f"OpenGL version: {self.opengl_version}")
                warnings.warn(f"GLSL version: {self.glsl_version}")
                
                if not self.has_vbo:
                    warnings.warn("OpenGL VBO functions not available. Rendering may not work.")
                if not self.has_shader:
                    warnings.warn("OpenGL shader functions not available. Using fixed-function pipeline.")
                if not self.has_vao:
                    warnings.warn("OpenGL 3.3+ core profile features not available. Using compatibility mode.")
                if not self.has_legacy_lighting and not self.has_shader:
                    warnings.warn("Neither modern shaders nor legacy lighting available. Rendering will be unlit.")
        
        finally:
            # We now keep the temporary window around instead of destroying it
            # Only restore the previous window if applicable
            current_window = glut.glutGetWindow()
            if current_window != 0 and current_window != self.temp_window_id:
                if PYOPENGL_VERBOSE:
                    print(f"GLContext: Restoring previous window ID: {current_window}")
                try:
                    glut.glutSetWindow(current_window)
                except Exception as e:
                    if PYOPENGL_VERBOSE:
                        print(f"GLContext: Error restoring window: {e}")
        
        self.is_initialized = True
    
    def _detect_opengl_features(self):
        """Detect available OpenGL features safely."""
        # Start with no capabilities
        self.has_vbo = False
        self.has_shader = False
        self.has_vao = False
        self.has_3_3 = False
        self.has_legacy_lighting = False
        self.has_legacy_vertex_arrays = False
        
        try:
            # Check for errors before testing features
            if gl.glGetError() != gl.GL_NO_ERROR:
                if PYOPENGL_VERBOSE:
                    print("GLContext: OpenGL error state before feature detection, skipping feature tests")
                return
                
            # VBO support
            self.has_vbo = (hasattr(gl, 'glGenBuffers') and 
                           callable(gl.glGenBuffers) and 
                           bool(gl.glGenBuffers))
            
            # Shader support
            self.has_shader = (hasattr(gl, 'glCreateShader') and 
                              callable(gl.glCreateShader) and 
                              bool(gl.glCreateShader) and
                              hasattr(gl, 'glCreateProgram') and 
                              callable(gl.glCreateProgram) and 
                              bool(gl.glCreateProgram))
            
            # VAO support (OpenGL 3.0+)
            self.has_vao = (self.has_vbo and 
                           self.has_shader and 
                           hasattr(gl, 'glGenVertexArrays') and 
                           callable(gl.glGenVertexArrays) and 
                           bool(gl.glGenVertexArrays))
            
            self.has_3_3 = (self.has_vao and 
                            self.has_shader and 
                            hasattr(gl, 'glGenVertexArrays') and 
                            callable(gl.glGenVertexArrays) and 
                            bool(gl.glGenVertexArrays))
            
            # Legacy support
            self.has_legacy_lighting = (hasattr(gl, 'GL_LIGHTING') and 
                                       hasattr(gl, 'GL_LIGHT0'))
            
            self.has_legacy_vertex_arrays = (hasattr(gl, 'GL_VERTEX_ARRAY') and
                                           hasattr(gl, 'glEnableClientState') and
                                           hasattr(gl, 'glVertexPointer'))
            
            # Verify capabilities by actually trying to use them (for VAO)
            if self.has_vao:
                try:
                    vao_id = gl.glGenVertexArrays(1)
                    gl.glBindVertexArray(vao_id)
                    gl.glBindVertexArray(0)
                    gl.glDeleteVertexArrays(1, [vao_id])
                except Exception as e:
                    if PYOPENGL_VERBOSE:
                        print(f"GLContext: VAO test failed: {e}")
                    self.has_vao = False
                    self.has_3_3 = False
            
            # Verify VBO capability
            if self.has_vbo:
                try:
                    vbo_id = gl.glGenBuffers(1)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_id)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
                    gl.glDeleteBuffers(1, [vbo_id])
                except Exception as e:
                    if PYOPENGL_VERBOSE:
                        print(f"GLContext: VBO test failed: {e}")
                    self.has_vbo = False
                    # If VBO fails, VAO will also fail
                    self.has_vao = False
                    self.has_3_3 = False
            
            # Verify legacy lighting if we claim to support it
            if self.has_legacy_lighting:
                try:
                    gl.glEnable(gl.GL_LIGHTING)
                    gl.glDisable(gl.GL_LIGHTING)
                except Exception as e:
                    if PYOPENGL_VERBOSE:
                        print(f"GLContext: Legacy lighting test failed: {e}")
                    self.has_legacy_lighting = False
            
        except (AttributeError, TypeError) as e:
            if PYOPENGL_VERBOSE:
                warnings.warn(f"Error detecting OpenGL capabilities: {e}")
            # Reset all capabilities to be safe
            self._set_fallback_capabilities()
    
    def _set_fallback_capabilities(self):
        """Set fallback capabilities for when detection fails."""
        # Assume nothing works
        self.has_vbo = False
        self.has_shader = False
        self.has_vao = False
        self.has_3_3 = False
        self.has_legacy_lighting = False
        self.has_legacy_vertex_arrays = False
        self.opengl_version = "Unknown (detection failed)"
        self.glsl_version = "Not available (detection failed)"
        
        if PYOPENGL_VERBOSE:
            warnings.warn("Using fallback OpenGL capabilities (minimal feature set)")
            warnings.warn("Only the simplest rendering methods will be available")
    
    def request_context_version(self, major: int, minor: int, core_profile: bool = True):
        """Request a specific OpenGL context version.
        
        This should be called before creating the real window.
        """
        if not HAS_OPENGL:
            return
            
        try:
            glut.glutInitContextVersion(major, minor)
            if core_profile:
                glut.glutInitContextProfile(glut.GLUT_CORE_PROFILE)
            else:
                glut.glutInitContextProfile(glut.GLUT_COMPATIBILITY_PROFILE)
        except (AttributeError, ValueError) as e:
            if PYOPENGL_VERBOSE:
                warnings.warn(f"Failed to set OpenGL context version: {e}")


