from abc import ABC, abstractmethod
import sys
from datatrees.datatrees import datatree, dtfield, Node
from pythonopenscad.viewer.glctxt import PYOPENGL_VERBOSE

import OpenGL.GL as gl

def clear_gl_errors():
    while gl.glGetError() != gl.GL_NO_ERROR:
        pass


def shader_type_to_string(shader_type: int) -> str:
    if shader_type == gl.GL_VERTEX_SHADER:
        return "Vertex"
    elif shader_type == gl.GL_FRAGMENT_SHADER:
        return "Fragment"
    else:
        return "Unknown"


class ErrorLogger(ABC):
    @abstractmethod
    def error(self, message: str):
        pass

    @abstractmethod
    def warn(self, message: str):
        pass

    @abstractmethod
    def info(self, message: str):
        pass


class ConsoleErrorLogger(ErrorLogger):
    def error(self, message: str):
        print(message)

    def warn(self, message: str):
        if PYOPENGL_VERBOSE:
            print(message)

    def info(self, message: str):
        if PYOPENGL_VERBOSE:
            print(message)


@datatree
class Shader:
    name: str
    shader_type: int
    shader_source: str
    binding: tuple[str, ...] = ()

    shader_id: int = dtfield(default=0, init=False)
    program_id: int = dtfield(default=0, init=False)
    is_bound: bool = dtfield(default=False, init=False)

    def compile(self, error_logger: ErrorLogger) -> bool:
        try:
            clear_gl_errors()

            # Create vertex shader
            self.shader_id = gl.glCreateShader(self.shader_type)
            if self.shader_id == 0:
                raise RuntimeError(
                    f"Shader {self.name}: Failed to create "
                    f"{shader_type_to_string(self.shader_type)} shader object"
                )

            gl.glShaderSource(self.shader_id, self.shader_source)
            gl.glCompileShader(self.shader_id)

            # Check for vertex shader compilation errors
            compile_status = gl.glGetShaderiv(self.shader_id, gl.GL_COMPILE_STATUS)
            if not compile_status:
                error = gl.glGetShaderInfoLog(self.shader_id)
                raise RuntimeError(f"Viewer: {self.name} shader compilation failed: {error}")
            return True
        except:
            self.delete()
            raise

    def attach(self, program_id: int):
        self.program_id = program_id
        gl.glAttachShader(program_id, self.shader_id)

    def delete(self):
        if self.shader_id != 0:
            gl.glDeleteShader(self.shader_id)
            self.shader_id = 0
            self.is_bound = False

    def bind_to_program(self):
        if self.binding and not self.is_bound:
            for location, name in enumerate(self.binding):
                gl.glBindAttribLocation(self.program_id, location, name)
            self.is_bound = True


@datatree
class ShaderProgram:
    name: str
    vertex_shader: Shader
    fragment_shader: Shader
    error_logger: ErrorLogger = dtfield(default_factory=ConsoleErrorLogger)

    program_id: int = dtfield(default=0, init=False)
    is_bound: bool = dtfield(default=False, init=False)

    def compile(self) -> int | None:
        clear_gl_errors()

        try:
            self.vertex_shader.compile(self.error_logger)
            self.fragment_shader.compile(self.error_logger)

            # Create and link shader program
            self.program_id = gl.glCreateProgram()
            if self.program_id == 0:
                raise RuntimeError("Failed to create shader program object")

            self.vertex_shader.attach(self.program_id)
            self.fragment_shader.attach(self.program_id)

            # Bind attribute locations for GLSL 1.20 (before linking)
            self.vertex_shader.bind_to_program()
            self.fragment_shader.bind_to_program()

            gl.glLinkProgram(self.program_id)

            # Check for linking errors
            link_status = gl.glGetProgramiv(self.program_id, gl.GL_LINK_STATUS)
            if not link_status:
                error = gl.glGetProgramInfoLog(self.program_id)
                raise RuntimeError(f"Shader program linking failed: {error}")

            # Delete shaders (they're not needed after linking)
            self.vertex_shader.delete()
            self.fragment_shader.delete()

            # Validate the program
            gl.glValidateProgram(self.program_id)
            validate_status = gl.glGetProgramiv(self.program_id, gl.GL_VALIDATE_STATUS)
            if not validate_status:
                error = gl.glGetProgramInfoLog(self.program_id)
                raise RuntimeError(f"Shader program validation failed: {error}")

            self.error_logger.info(
                f"Successfully compiled and linked shader program: {self.program_id}"
            )
            return self.program_id

        except Exception as e:
            self.error_logger.error(f"Shader program {self.name} compilation failed: {e}")
            self.delete()
            return None

    def delete(self):
        self.vertex_shader.delete()
        self.fragment_shader.delete()
        if self.program_id != 0:
            gl.glDeleteProgram(self.program_id)
            self.program_id = 0

    def use(self):
        gl.glUseProgram(self.program_id)

    def unuse(self):
        gl.glUseProgram(0)


VERTEX_SHADER = Shader(
    name="vertex_shader",
    shader_type=gl.GL_VERTEX_SHADER,
    binding=("aPos", "aColor", "aNormal"),
    shader_source="""
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
""",
)

FRAGMENT_SHADER = Shader(
    name="fragment_shader",
    shader_type=gl.GL_FRAGMENT_SHADER,
    shader_source="""
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
""",
)

SHADER_PROGRAM = ShaderProgram(
    name="shader_program",
    vertex_shader=VERTEX_SHADER,
    fragment_shader=FRAGMENT_SHADER,
)


# Basic fallback shader for maximum compatibility
BASIC_VERTEX_SHADER = Shader(
    name="basic_vertex_shader",
    shader_type=gl.GL_VERTEX_SHADER,
    binding=("position", "color"),
    shader_source="""
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
""",
)

BASIC_FRAGMENT_SHADER = Shader(
    name="basic_fragment_shader",
    shader_type=gl.GL_FRAGMENT_SHADER,
    shader_source="""
#version 110

varying vec4 fragColor;

void main() {
    // Use the interpolated color from the vertex shader
    gl_FragColor = fragColor;
}
""",
)

BASIC_SHADER_PROGRAM = ShaderProgram(
    name="basic_shader_program",
    vertex_shader=BASIC_VERTEX_SHADER,
    fragment_shader=BASIC_FRAGMENT_SHADER,
)
