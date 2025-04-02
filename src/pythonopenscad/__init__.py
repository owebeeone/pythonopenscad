from .base import (
    Arg,
    Circle,
    CodeDumper,
    CodeDumperForPython,
    Color,
    ConversionException,
    Cube,
    Cylinder,
    Difference,
    DuplicateNamingOfArgs,
    Fill,
    Hull,
    Import,
    IndentLevelStackEmpty,
    InitializerNotAllowed,
    Intersection,
    InvalidIndentLevel,
    InvalidValue,
    InvalidValueForBool,
    InvalidValueForStr,
    LazyUnion,
    Linear_Extrude,
    Minkowski,
    Mirror,
    Module,
    Multmatrix,
    OSC_FALSE,
    OSC_TRUE,
    Offset,
    OpenScadApiSpecifier,
    OscKeyword,
    ParameterDefinedMoreThanOnce,
    ParameterNotDefined,
    Polygon,
    Polyhedron,
    PoscBase,
    PoscBaseException,
    PoscParentBase,
    Projection,
    RequiredParameterNotProvided,
    Render,
    Resize,
    Rotate,
    Rotate_Extrude,
    Scale,
    Sphere,
    Square,
    Surface,
    Text,
    TooManyParameters,
    Translate,
    Union,
    VECTOR2_FLOAT,
    VECTOR2_FLOAT_DEFAULT_1,
    VECTOR3OR4_FLOAT,
    VECTOR3_BOOL,
    VECTOR3_FLOAT,
    VECTOR3_FLOAT_DEFAULT_1,
    VECTOR4_FLOAT,
    apply_posc_attributes,
    bool_strict,
    circle,
    color,
    copy,
    cube,
    cylinder,
    difference,
    hull,   
    intersection,
    lazy_union,
    linear_extrude,
    list_of,
    minkowski,
    mirror,
    module,
    multmatrix,
    of_set,
    offset,
    one_of,
    polygon,
    polyhedron,
    projection,
    render,
    resize,
    rotate,
    rotate_extrude,
    scale,
    sphere,
    square,
    str_strict,
    surface,
    text,
    translate,
    union
)

from pythonopenscad.modifier import (
    OscModifier,
    DISABLE,
    SHOW_ONLY,
    DEBUG,
    TRANSPARENT,
    BASE_MODIFIERS_SET,
    BASE_MODIFIERS,
    InvalidModifier,
    PoscModifiers,
    PoscRendererBase,
)

from pythonopenscad.m3dapi import (
    M3dRenderer,
    RenderContext,
    RenderContextManifold,
    RenderContextCrossSection,
    Mode
)

# Try to import the viewer module, but don't fail if OpenGL is not available
try:
    from pythonopenscad.viewer import (
        BoundingBox,
        Model,
        Viewer,
        create_viewer_with_models,
        is_opengl_available
    )
    HAS_VIEWER = True
except ImportError:
    HAS_VIEWER = False

__all__ = [
    "Arg",
    "BASE_MODIFIERS",
    "BASE_MODIFIERS_SET",
    "Circle",
    "CodeDumper",
    "CodeDumperForPython",
    "Color",
    "ConversionException",
    "Cube",
    "Cylinder",
    "DEBUG",
    "DISABLE",
    "Difference",
    "DuplicateNamingOfArgs",
    "Fill",
    "Hull",
    "Import",
    "IndentLevelStackEmpty",
    "InitializerNotAllowed",
    "Intersection",
    "InvalidIndentLevel",
    "InvalidModifier",
    "InvalidValue",
    "InvalidValueForBool",
    "InvalidValueForStr",
    "LazyUnion",
    "Linear_Extrude",
    "Minkowski",
    "Mirror",
    "Module",
    "Multmatrix",
    "OSC_FALSE",
    "OSC_TRUE",
    "Offset",
    "OpenScadApiSpecifier",
    "OscKeyword",
    "OscModifier",
    "ParameterDefinedMoreThanOnce",
    "ParameterNotDefined",
    "Polygon",
    "Polyhedron",
    "PoscBase",
    "PoscBaseException",
    "PoscModifiers",
    "PoscParentBase",
    "PoscRendererBase",
    "Projection",
    "RequiredParameterNotProvided",
    "Render",
    "Resize",
    "Rotate",
    "Rotate_Extrude",
    "SHOW_ONLY",
    "Scale",
    "Sphere",
    "Square",
    "Surface",
    "TRANSPARENT",
    "Text",
    "TooManyParameters",
    "Translate",
    "Union",
    "VECTOR2_FLOAT",
    "VECTOR2_FLOAT_DEFAULT_1",
    "VECTOR3OR4_FLOAT",
    "VECTOR3_BOOL",
    "VECTOR3_FLOAT",
    "VECTOR3_FLOAT_DEFAULT_1",
    "VECTOR4_FLOAT",
    "apply_posc_attributes",
    "bool_strict",
    "circle",
    "color",
    "copy",
    "cube",
    "cylinder",
    "difference",
    "hull",   
    "intersection",
    "lazy_union",
    "linear_extrude",
    "list_of",
    "minkowski",
    "mirror",
    "module",
    "multmatrix",
    "of_set",
    "offset",
    "one_of",
    "polygon",
    "polyhedron",
    "projection",
    "render",
    "resize",
    "rotate",
    "rotate_extrude",
    "scale",
    "sphere",
    "square",
    "str_strict",
    "surface",
    "text",
    "translate",
    "union",
    "M3dRenderer",
    "RenderContext",
    "RenderContextManifold",
    "RenderContextCrossSection",
    "Mode"
]

# Add viewer classes if available
if HAS_VIEWER:
    __all__.extend([
        "BoundingBox",
        "Model",
        "Viewer",
        "create_viewer_with_models",
        "is_opengl_available"
    ])
