'''

'''

import copy
from dataclasses import dataclass
from pip._internal import self_outdated_check

from frozendict import frozendict

from ParametricSolid import linear as l
import numpy as np
import pythonopenscad as posc


class BaseException(Exception):
    '''Base exception functionality'''
    def __init__(self, message):
        self.message = message

class DuplicateNameException(BaseException):
    '''Attempting to add a shape with a name that is already used.'''

class UnimplementedRenderException(BaseException):
    '''Attempting to render from a class that has nor implemented render().'''

class IllegalParameterException(BaseException):
    '''Received an unexpected parameter.'''   

class AnchorSpecifierNotFoundException(BaseException):
    '''Requested anchor is not found.'''
    
class IncorrectAnchorArgs(BaseException):
    '''Attempted to call an anchor and it failed.'''
    

def args(*args, **kwds):
    '''Returns a tuple or args and kwds passed to this function.'''
    return (args, kwds)

def args_to_str(args):
    '''Returns a string that represents the arguments passed into args().'''
    positional_bits = ', '.join(repr(v) for v in args[0])
    kwds_bits = ', '.join(f'{k}={v!r}' for k, v in args[1].items())
    return ', '.join((positional_bits, kwds_bits))

def surface_anchor_renderer(maker, anchor_args):
    '''Helper to crate example anchor coordinates on surface of objects.'''
    maker.add_at(AnnotatedCoordinates().solid(args_to_str(anchor_args)).at('origin'),
                 *anchor_args[0], **anchor_args[1])
    
    
def inner_anchor_renderer(maker, anchor_args):
    '''Helper to crate example anchor coordinates inside an object.'''
    maker.add_at(AnnotatedCoordinates().solid(args_to_str(anchor_args)).at('origin'),
                 *anchor_args[0], **anchor_args[1])
    
def surface_args(*args, **kwds):
    '''Defines an instance of an anchor example.'''
    return (surface_anchor_renderer, (args, kwds))

def inner_args(*args, **kwds):
    '''Defines an instance of an anchor example for anchors inside an object.'''
    return (inner_anchor_renderer, (args, kwds))

@dataclass(frozen=True)
class Colour(object):
    value: tuple
    
    def __init__(self, value):
        value = value.value if isinstance(value, Colour) else value
        object.__setattr__(
            self, 'value', tuple(posc.VECTOR3OR4_FLOAT(value)))


@dataclass(frozen=True)
class ModelAttributes(object):
    colour: Colour = None
    fa: float = None
    fs: float = None
    fn: int = None
    disable: bool = None
    show_only: bool = None
    debug: bool = None
    transparent: bool = None
    
    def _merge_of(self, attr, other):
        self_value = getattr(self, attr)
        other_value = getattr(other, attr)
        if self_value == other_value:
            return self_value;
        if other_value == None:
            return self_value
        return other_value
    
    def _diff_of(self, attr, other):
        self_value = getattr(self, attr)
        other_value = getattr(other, attr)
        if self_value == other_value:
            return None
        return other_value

    def merge(self, other):
        '''Returns a copy of self with entries from other replacing self's.'''
        if not other:
            return self
        
        return ModelAttributes(**dict(
            (k, self._merge_of(k, other)) 
            for k in self.__annotations__.keys()))
    
    def diff(self, other):
        '''Returns a new ModelAttributes with the diff of self and other.'''
        if not other:
            return self
        return ModelAttributes(**dict(
            (k, self._diff_of(k, other)) 
            for k in self.__annotations__.keys()))

    
    def _as_non_defaults_dict(self):
        return dict((k, getattr(self, k)) 
                    for k in self.__annotations__.keys() if not getattr(self, k) is None)
    
    def _with(self, fname, value):
        d = self._as_non_defaults_dict()
        d[fname] = value
        return ModelAttributes(**d)
    
    def with_colour(self, colour):
        return self._with('colour', None if colour is None else Colour(colour))
    
    def with_fa(self, fa):
        return self._with('fa', fa)
    
    def with_fs(self, fs):
        return self._with('fs', fs)
    
    def with_fn(self, fn):
        return self._with('fn', fn)
    
    def with_disable(self, disable):
        return self._with('disable', disable)
    
    def with_show_only(self, show_only):
        return self._with('show_only', show_only)
    
    def with_debug(self, debug):
        return self._with('debug', debug)
    
    def with_transparent(self, transparent):
        return self._with('transparent', transparent)
    
    def fill_dict(self, out_dict, field_names=('fn', 'fs', 'fa')):
        for field_name in field_names:
            if field_name in out_dict:
                continue
            value = getattr(self, field_name)
            if value is None:
                continue
            out_dict[field_name] = value
        return out_dict
    
    def to_str(self):
        '''Generates a repr with just the non default values.'''
        return self.__class__.__name__ + '(' + ', '.join(
            f'{k}={v!r}' for k, v in self._as_non_defaults_dict().items()) + ')'
            
    def __str__(self):
        return self.to_str()
    
    def __repr__(self):
        return self.to_str()
    
    
EMPTY_ATTRS = ModelAttributes()


@dataclass(frozen=True)
class ShapeFrame(object):
    name: object  # Hashable
    shape: object  # Shape or Maker
    reference_frame: l.GMatrix
    attributes: ModelAttributes = None

    def inverted(self):
        return ShapeFrame(self.name, self.shape, self.reference_frame.I, self.attributes)
    
    def pre_mul(self, reference_frame):
        return ShapeFrame(
            self.name, self.shape, reference_frame * self.reference_frame, self.attributes)
        
    def post_mul(self, reference_frame):
        return ShapeFrame(
            self.name, self.shape, self.reference_frame * reference_frame, self.attributes)

@dataclass(frozen=True)
class NamedShape(object):
    shape: object  # Shape or Maker
    shape_type: object  # Hashable
    name: object  # Hashable
    attributes: ModelAttributes = None
    
    def at(self, *args, post: l.GMatrix=None, pre: l.GMatrix=None, **kwds):
        '''Creates a shape containing the nominated shape at the reference frame given.
        *args, **kwds: Parameters for the shape given. If none is provided then IDENTITY is used.
        pre: The pre multiplied transform.
        post: The post multiplied transform,
        '''
        
        reference_frame = self.shape.at(*args, **kwds) if args or kwds else l.IDENTITY
        if pre:
            reference_frame = pre * reference_frame
        if post:
            reference_frame = reference_frame * post
        return self.projection(reference_frame)
        
    def projection(self, reference_frame: l.GMatrix):
        return Maker(
            self.shape_type, 
            ShapeFrame(self.name, self.shape, reference_frame), 
            attributes=self.attributes)
        
    def with_attributes(self, attributes):
        return NamedShape(
            shape=self.shape, shape_type=self.shape_type, name=self.name, attributes=attributes)
        
    def get_attributes_or_default(self) :
        attributes = self.attributes
        if not attributes:
            attributes = EMPTY_ATTRS
        return attributes
        
    def colour(self, colour):
        return NamedShape(
            shape=self.shape, shape_type=self.shape_type, name=self.name, 
            attributes=self.get_attributes_or_default().with_colour(colour))
    
    def fa(self, fa):
        return NamedShape(
            shape=self.shape, shape_type=self.shape_type, name=self.name,
            attributes=self.get_attributes_or_default().with_fa(fa))
    
    def fs(self, fs):
        return NamedShape(
            shape=self.shape, shape_type=self.shape_type, name=self.name,
            attributes=self.get_attributes_or_default().with_fs(fs))
    
    def fn(self, fn):
        return NamedShape(
            shape=self.shape, shape_type=self.shape_type, name=self.name,
            attributes=self.get_attributes_or_default().with_fn(fn))
    
    def disable(self, disable):
        return NamedShape(
            shape=self.shape, shape_type=self.shape_type, name=self.name,
            attributes=self.get_attributes_or_default().with_disable(disable))
    
    def show_only(self, show_only):
        return NamedShape(
            shape=self.shape, shape_type=self.shape_type, name=self.name,
            attributes=self.get_attributes_or_default().with_show_only(show_only))
    
    def debug(self, debug):
        return NamedShape(
            shape=self.shape, shape_type=self.shape_type, name=self.name,
            attributes=self.get_attributes_or_default().with_debug(debug))
    
    def transparent(self, transparent):
        return NamedShape(
            shape=self.shape, shape_type=self.shape_type, name=self.name,
            attributes=self.get_attributes_or_default().with_transparent(transparent))


class Shape():
    
    EXAMPLE_ANCHORS=()
    EXAMPLE_SHAPE_ARGS=args()
    
    def __init__(self):
        pass
    
    def copy_if_mutable(self):
        return self
    
    def solid(self, name):
        return NamedShape(self.copy_if_mutable(), ModeShapeFrame.SOLID, name)
    
    def hole(self, name):
        return NamedShape(self.copy_if_mutable(), ModeShapeFrame.HOLE, name)
    
    def cage(self, name):
        return NamedShape(self.copy_if_mutable(), ModeShapeFrame.CAGE, name)
    def as_solid(self, name, reference_frame):
        return Maker(
            ModeShapeFrame.SOLID, ShapeFrame(name, self.copy_if_mutable(), reference_frame))
    
    def as_hole(self, name, reference_frame):
        return Maker(ModeShapeFrame.HOLE, ShapeFrame(name, self.copy_if_mutable(), reference_frame))
    
    def as_cage(self, name, reference_frame):
        return Maker(ModeShapeFrame.CAGE, ShapeFrame(name, self.copy_if_mutable(), reference_frame))
    
    def has_anchor(self, name):
        return name in self.anchors.anchors
    
    def anchor_names(self):
        return tuple(self.anchors.anchors.keys())
    
    def at(self, anchor_name, *args, **kwds):
        spec = self.anchors.get(anchor_name)
        func = spec[0]
        try:
            return func(self, *args, **kwds)
        except TypeError as e:
            raise IncorrectAnchorArgs(
                f'Attempted to call {anchor_name} on {self.__class__.__name__}'
                f' with args={args!r} kwds={kwds!r}') from e
    
    def name(self):
        return self.anchors.name
    
    def render(self, renderer):
        raise UnimplementedRenderException(f'Unimplemented render in {self.name()!r}.')
    
    @classmethod
    def example(cls):
        maker = cls(*cls.EXAMPLE_SHAPE_ARGS[0], **cls.EXAMPLE_SHAPE_ARGS[1]
            ).solid('example').projection(l.IDENTITY)
        for entry in cls.EXAMPLE_ANCHORS:
            entry[0](maker, entry[1])
        return maker

@dataclass()
class _Mode():
    mode : str

@dataclass()
class SolidMode(_Mode):
    def __init__(self):
        super().__init__('solid')
        
    def pick_renderer(self, renderer):
        return renderer.solid()
    

@dataclass()
class HoleMode(_Mode):
    def __init__(self):
        super().__init__('hole')
        
    def pick_renderer(self, renderer):
        return renderer.hole()
    
    
@dataclass()
class CompositeMode(_Mode):
    def __init__(self):
        super().__init__('composite')
        
    def pick_renderer(self, renderer):
        return renderer.hole()
    

@dataclass()
class CageMode(_Mode):
    def __init__(self):
        super().__init__('cage')
        
    def pick_renderer(self, renderer):
        return renderer.null()
    
class Renderer:
    POSC = posc
    
    def solid(self):
        pass
    
    def hole(self):
        pass
    
    def composite(self):
        pass
    
    def null(self):
        pass
    
    
@dataclass(frozen=True)
class ModeShapeFrame():
    SOLID=SolidMode()
    HOLE=HoleMode()
    CAGE=CageMode()
    COMPOSITE=CompositeMode()
    
    mode: _Mode
    shapeframe: ShapeFrame
    attributes: ModelAttributes = None
    
    def inverted(self):
        return ModeShapeFrame(self.mode, 
                              self.shapeframe.inverted(), 
                              attributes=self.attributes)
    
    def pre_mul(self, reference_frame):
        return ModeShapeFrame(self.mode, 
                              self.shapeframe.pre_mul(reference_frame), 
                              attributes=self.attributes)
    
    def post_mul(self, reference_frame):
        return ModeShapeFrame(self.mode, 
                              self.shapeframe.post_mul(reference_frame), 
                              attributes=self.attributes)

    def reference_frame(self):
        return self.shapeframe.reference_frame
    
    def name(self):
        return self.shapeframe.name
    
    def shape(self):
        return self.shapeframe.shape
    
    def colour(self):
        return None if self.attributes is None else self.attributes.colour
    
    def to_str(self):
        parts=(
            repr(self.shape()),
            '.',
            self.mode.mode,
            '(',
            repr(self.name())
            )
        
        attr_parts = ()
        if self.attributes:
            attr_parts = (
                ').attributes(',
                repr(self.attributes)
                )
        projectopm_parts = (
            ').projection(',
            repr(self.reference_frame()),
            ')'
            )
        return ''.join(parts + attr_parts + projectopm_parts)
        

@dataclass
class Maker(Shape):
    reference_shape: ModeShapeFrame
    entries: dict
    
    def __init__(self, mode=None, shape_frame=None, *, copy_of=None, attributes=None):
        if copy_of is None:
            self.reference_shape = ModeShapeFrame(mode, shape_frame, attributes=attributes)
            self.entries = {shape_frame.name: self.reference_shape.inverted()}
        else:
            if mode is None and shape_frame is None and attributes is None:
                self.reference_shape = copy_of.reference_shape
                self.entries = copy.copy(copy_of.entries)
            else:
                raise IllegalParameterException(
                    f'\'copy_of\' named parameter is provided and \'attributes\', \'mode\' or '
                    f'\'shape_frame\' parameters must not be provided but '
                    f'attributes={attributes!r}, mode={mode!r} and shape_frame={shape_frame!r}')
        
    def copy_if_mutable(self):
        return Maker(copy_of=self)
        
    def _add_mode_shape_frame(self, mode_shape_frame):
        # Check for name collision.
        name = mode_shape_frame.shapeframe.name
        previous = self.entries.get(name, None)
        if previous:
            raise DuplicateNameException(
                'Attempted to add %r when it already exists in with mode %r' % (
                    name, 
                    previous.mode.mode))
        self.entries[name] = mode_shape_frame
        return self

    def add(self, maker):
        if not isinstance(maker, Maker):
            raise IllegalParameterException(
                f'Expected a parameter of type {self.__class__.__name__!r} but received an '
                f'object of type {maker.__class__.__name__!r}.')
        
        for entry in maker.entries.values():
            self._add_mode_shape_frame(entry)
        
        return self
    
    def add_at(self, maker, *args, pre=None, post=None, **kwds):
        
        if not isinstance(maker, Maker):
            raise IllegalParameterException(
                f'Expected a parameter of type {self.__class__.__name__!r} but received an '
                f'object of type {maker.__class__.__name__!r}.')
            
        local_frame = self.at(*args, **kwds) if args or kwds else l.IDENTITY
        if pre:
            local_frame = pre * local_frame
        if post:
            local_frame = local_frame * post
        
        for entry in maker.entries.values():
            self._add_mode_shape_frame(entry.pre_mul(local_frame))
            
        return self
        
    def add_shape(self, mode, shape_frame, attributes=None):
        return self._add_mode_shape_frame(ModeShapeFrame(
            mode, shape_frame.inverted(), attributes))
    
    def add_solid(self, shape_frame, attributes=None):
        return self.add_shape(ModeShapeFrame.SOLID, shape_frame, attributes)
    
    def add_hole(self, shape_frame, attributes=None):
        return self.add_shape(ModeShapeFrame.HOLE, shape_frame, attributes)
    
    def add_cage(self, shape_frame, attributes=None):
        return self.add_shape(ModeShapeFrame.CAGE, shape_frame, attributes)
    
    def add_composite(self, shape_frame, attributes=None):
        return self.add_shape(ModeShapeFrame.COMPOSITE, shape_frame, attributes)
    
    def composite(self, name):
        return NamedShape(self.copy_if_mutable(), ModeShapeFrame.COMPOSITE, name)

    def as_composite(self, name, reference_frame, attributes):
        return Maker(
            ModeShapeFrame.COMPOSITE, 
            ShapeFrame(name, self.copy_if_mutable(), reference_frame),
            attributes=attributes)
        
    def has_anchor(self, name):
        ref_shape = self.reference_shape.shapeframe.shape
        if ref_shape.has_anchor(name):
            return True
        return name in self.entries
    
    def anchor_names(self):
        return self.reference_shape.shape().anchor_names() + tuple(self.entries.keys()) 
    
    def at(self, name, *args, **kwds):
        shapeframe = self.reference_shape.shapeframe
        ref_shape = shapeframe.shape
        if ref_shape.has_anchor(name):
            entry = self.entries.get(self.reference_shape.name())
            return entry.reference_frame() * ref_shape.at(name, *args, **kwds)
        entry = self.entries.get(name)
        
        if entry is None:
            raise AnchorSpecifierNotFoundException(
                f'name={name!r} is not an anchor of the reference shape or a named shape. '
                f'Available names are {self.anchor_names()}.')
        
        return entry.reference_frame() * entry.shape().at(*args, **kwds)
            
    def name(self):
        return 'Maker({name!r})'.format(name=self.reference_shape.name())
    
    def to_str(self):
        parts = [self.reference_shape.to_str()]
        for entry in self.entries.values():
            if entry.name() == self.reference_shape.name():
                continue
            parts.append(f'.add(\n    {entry.inverted().to_str()})')
        return ''.join(parts)
    
    def __str__(self):
        return self.to_str()
    
    def __repr__(self):
        return self.to_str()

    def render(self, renderer):
        for v in self.entries.values():
            renderer.push(
                v.mode, v.reference_frame(), v.attributes)
            try:
                v.shape().render(renderer)
            finally:
                renderer.pop()
    
class AnchorSpec():
    def __init__(self, description):
        self.description = description
        

def anchor(description):
    def decorator(func):
        func.anchor_spec = AnchorSpec(description)
        return func
    return decorator

VECTOR3_FLOAT_DEFAULT_1 = l.list_of(
    np.float64, 
    len_min_max=(3, 3), 
    fill_to_min=np.float64(1))


@dataclass(frozen=True)
class Anchors():
    name: str
    anchors: frozendict
        
    def get(self, name):
        return self.anchors.get(name)
    
    
@dataclass()
class AnchorsBuilder():
    name: str
    anchors: dict
    
    def __init__(self, name, anchors={}):
        self.name = name
        self.anchors = dict(anchors)
        
    def add(self, name, func, anchor_spec):
        self.anchors[name] = (func, anchor_spec)
        
    def get(self, name):
        return self.anchors.get(name)
    
    def build(self):
        return Anchors(name=self.name, anchors=frozendict(self.anchors))

def shape(name):
    def decorator(clazz):
        builder = AnchorsBuilder(name)
        for func_name in dir(clazz):
            if func_name.startswith("__"):
                continue
            func = getattr(clazz, func_name)
            if not callable(func):
                continue
            if not hasattr(func, 'anchor_spec'):
                continue
            builder.add(func_name, func, func.anchor_spec)
        clazz.anchors = builder.build()
        return clazz
    return decorator

@shape('box')
@dataclass
class Box(Shape):
    '''Generates rectangular prisms (cubes where l=w=h).'''
    size: l.GVector
    
    # Orientation of the 6 faces.
    ORIENTATION = (
        l.rotX(90),
        l.rotX(90) * l.rotX(90),
        l.rotX(90) * l.rotY(-90),
        l.rotX(90) * l.rotX(180),
        l.rotX(90) * l.rotX(-90),
        l.rotX(90) * l.rotY(90))
    
    COORDINATES_CORNERS = (
        ((), (0,), (0, 2), (2,)),
        ((1,), (0, 1), (0,), ()),
        ((1,), (), (2,), (1, 2)),
        ((1, 2), (0, 1, 2), (0, 1), (1,)),
        ((2,), (0, 2), (0, 1, 2), (1, 2)),
        ((0, ), (0, 1), (0, 1, 2), (0, 2)),
        )
    
    COORDINATES_EDGE_HALVES = tuple(
        tuple([
            tuple([tuple(set(face[i]) ^ set(face[(i + 1) % 4])) for i in range(4)])
            for face in COORDINATES_CORNERS]))
    
    COORDINATES_CORNERS_ZEROS = tuple(
        tuple([
            tuple([tuple(set((0,1,2)) - set(coords)) for coords in face])
            for face in COORDINATES_CORNERS]))
    
    COORDINATES_CENTRES_AXIS = tuple(
        tuple(set((0,1,2)) - set(face[0]) - set(face[2 ])) 
        for face in COORDINATES_CORNERS[0:3])
    
    EXAMPLE_ANCHORS=tuple(
        (surface_args('face_corner', f, c)) for f in (0, 3) for c in range(4)
        ) + tuple(surface_args('face_edge', f, c) for f in (1, 3) for c in range(4)
        ) + tuple(surface_args('face_centre', f) for f in (0, 3)
                                    ) + (inner_args('centre'),)
    EXAMPLE_SHAPE_ARGS=args([20, 30, 40])
    
    def __init__(self, size=[1, 1, 1]):
        self.size = l.GVector(VECTOR3_FLOAT_DEFAULT_1(size))

    def render(self, renderer):
        renderer.add(renderer.model.Cube(self.size.A3))
        return renderer
    
    @anchor('Centre of box oriented same as face 0')
    def centre(self):
        return l.translate(l.GVector(self.size) / 2)
    
    @anchor('Corner of box given face (0-5) and corner (0-3)')
    def face_corner(self, face, corner):
        orientation = self.ORIENTATION[face] * l.rotZ(90 * corner)
        loc = l.GVector(self.size)  # make a copy.
        for i in self.COORDINATES_CORNERS_ZEROS[face][corner]:
            loc[i] = 0.0
        return l.translate(loc) * orientation
    
    @anchor('Edge centre of box given face (0-5) and edge (0-3)')
    def face_edge(self, face, edge):
        orientation = self.ORIENTATION[face] * l.rotZ(90 * edge)
        loc = l.GVector(self.size)  # make a copy.
        half_of = self.COORDINATES_EDGE_HALVES[face][edge]
        zero_of = self.COORDINATES_CORNERS_ZEROS[face][edge]
        for i in range(3):
            if i in half_of:
                loc[i] *= 0.5
            elif i in zero_of:
                loc[i]  = 0.0
        return l.translate(loc) * orientation
        
    @anchor('Centre of face given face (0-5)')
    def face_centre(self, face):
        orientation = self.ORIENTATION[face]
        loc = l.GVector(self.size)  # make a copy.
        keep_value = self.COORDINATES_CENTRES_AXIS[face % 3][0]
        for i in range(3):
            if i == keep_value:
                if face < 3:
                    loc[i] = 0.0
            else:
                loc[i] = loc[i] * 0.5
        return l.translate(loc) * orientation
    


TEXT_DEPTH_MAP={'centre':0.0, 'rear': -0.5, 'front':0.5}

def non_defaults_dict(dataclas_obj, include=None, exclude=()):
    return dict((k, getattr(dataclas_obj, k)) 
                for k in dataclas_obj.__annotations__.keys() 
                if (not k in exclude) and (
                    include is None or k in include) and not getattr(dataclas_obj, k) is None)
    
ARGS_XLATION_TABLE={'fn': '_fn', 'fa': '_fa', 'fs': '_fs'}
def translate_names(out_dict, xlation_table=ARGS_XLATION_TABLE):
    for old_name, new_name in xlation_table.items():
        if old_name in out_dict:
            out_dict[new_name] = out_dict[old_name]
            del out_dict[old_name]
    return out_dict

def fill_params(shape, renderer, attr_names):
    cur_attrs = renderer.get_current_attributes()
    params = cur_attrs.fill_dict(non_defaults_dict(shape), attr_names)
    return translate_names(params)


@shape('text')
@dataclass
class Text(Shape):
    '''Generates 3D text.'''
    text: posc.str_strict=None
    size: float=10.0
    font: posc.str_strict=None
    halign: posc.of_set('left', 'center', 'right')='left'
    valign: posc.of_set('top', 'center', 'baseline' 'bottom')='bottom'
    spacing: float=1.0
    direction: posc.of_set('ltr', 'rtl', 'ttb', 'btt')='ltr'
    language: posc.str_strict=None
    script: posc.str_strict=None
    fn: int=None
    
    
    EXAMPLE_ANCHORS=(surface_args('default', 'rear'),)
    EXAMPLE_SHAPE_ARGS=args('Text Example')

    def render(self, renderer):
        params = fill_params(self, renderer, ('fn',))
        renderer.add(renderer.model.Text(**params))
        return renderer
    
    @anchor('The default position for this text. depth=(rear, centre, front)')
    def default(self, depth='centre'):
        return l.translate([0, 0, TEXT_DEPTH_MAP[depth]])
    

ANGLES_TYPE = l.list_of(l.strict_float, len_min_max=(3, 3), fill_to_min=0.0)
@shape('sphere')
@dataclass
class Sphere(Shape):
    '''Generates a Sphere.'''
    r: float=1.0
    fn: int=None
    fa: float=None
    fs: float=None

    EXAMPLE_ANCHORS=(surface_args('top'),
                     surface_args('base'),
                     inner_args('centre'),
                     surface_args('surface', [90, 30, 45]),
                     surface_args('surface', [-45, 0, 0]),
                     surface_args('surface', [0, 0, 0]),)
    EXAMPLE_SHAPE_ARGS=args(20)
    

    def render(self, renderer):
        params = fill_params(self, renderer, ('fn', 'fa', 'fs'))
        params = translate_names(params)
        renderer.add(renderer.model.Sphere(**params))
        return renderer
    
    @anchor('The base of the cylinder')
    def base(self):
        return l.rotX(180) * l.translate([0, 0, self.r])
    
    @anchor('The top of the cylinder')
    def top(self):
        return l.translate([0, 0, self.r])
    
    @anchor('The centre of the cylinder')
    def centre(self):
        return l.rotX(180)
    
    @anchor('A location on the sphere.')
    def surface(self, degrees: ANGLES_TYPE=ANGLES_TYPE([0, 0, 0]), radians: ANGLES_TYPE=None):
        if radians:
            angle_type = 'radians'
            angles = ANGLES_TYPE(radians)
        else:
            angle_type = 'degrees'
            angles = ANGLES_TYPE(degrees)
        
        return (l.rotY(**{angle_type: angles[2]}) 
             * l.rotX(**{angle_type: angles[1]}) 
             * l.rotZ(**{angle_type: angles[0]})
             * l.translate([self.r, 0, 0])
             * l.ROTV111_120)


CONE_ARGS_XLATION_TABLE={'r_base': 'r1', 'r_top': 'r2'}
@shape('cone')
@dataclass
class Cone(Shape):
    '''Generates cones or horizontal conical slices and cylinders.'''
    h: float=1.0
    r_base: float=1.0
    r_top: float=0.0
    fn: int=None
    fa: float=None
    fs: float=None
    
    EXAMPLE_ANCHORS=(
        surface_args('base'),
        surface_args('top'),
        surface_args('surface', 20, 0),
        surface_args('surface', 10, 45),
        surface_args('surface', 3, 90, tangent=False),
        inner_args('centre'))
    EXAMPLE_SHAPE_ARGS=args(h=50, r_base=30, r_top=5, fn=30)
    
    def __post_init__(self):
        if self.h < 0:
            raise IllegalParameterException(
                f'Parameter h({self.h}) is less than 0.')
        if self.r_base < 0:
            raise IllegalParameterException(
                f'Parameter r_base({self.r_base}) is less than 0.')
        if self.r_top < 0:
            raise IllegalParameterException(
                f'Parameter r_top({self.r_top}) is less than 0.')
        
    def render(self, renderer):
        params = fill_params(self, renderer, ('fn', 'fa', 'fs'))
        params = translate_names(params, CONE_ARGS_XLATION_TABLE)
        renderer.add(renderer.model.Cylinder(r=None, **params))
        return renderer
    
    @anchor('The base of the cylinder')
    def base(self):
        return l.rotX(180)
    
    @anchor('The top of the cylinder')
    def top(self):
        return l.translate([0, 0, self.h])
    
    @anchor('The centre of the cylinder')
    def centre(self):
        return l.translate([0, 0, self.h / 2]) * l.rotX(180)
    
    @anchor('A location on the curved surface.')
    def surface(self, h, degrees=0.0, radians=None, tangent=True):
        if h < 0 or self.h < h:
            raise IllegalParameterException(
                f'Parameter h({h} is not in range [0 {self.h}].')
        r = (h / self.h)
        x = r * self.r_top + (1 - r) * self.r_base
        if tangent:
            m = l.rot_to_V([-1, 0, 0], [self.r_top - self.r_base, 0, self.h]) * l.rotZ(90)
        else:
            m = l.ROTV111_120
        return l.rotZ(degrees=degrees, radians=radians) * l.translate([x, 0, h]) * m


class CompositeShape(Shape):
    '''Provides functionality for composite shapes. Subclasses must set 'maker' in
    the initialization of the class.'''
    
    def render(self, renderer):
        return self.maker.render(renderer)
            
    def copy_if_mutable(self):
        result=copy.copy(self)
        result.maker = Maker(copy_of=self.maker)
        return result

    @anchor('Access to inner elements of this composite shape.')
    def within(self, *args, **kwds):
        return self.maker.at(*args, **kwds)


@shape('arrow')
@dataclass
class Arrow(CompositeShape):
    
    r_stem_top: float=1.0
    r_stem_base: float=None # Defaults to r_stem_top
    l_stem: float=6.0
    l_head: float=3
    r_head_base: float=2
    r_head_top: float=0.0
    fn: int=None
    fa: float=None
    fs: float=None
    
    
    EXAMPLE_ANCHORS=(
        surface_args('base'),
        surface_args('top'),
        surface_args('within', 'stem', 'top'))
    EXAMPLE_SHAPE_ARGS=args(
        r_stem_top=4, r_stem_base=6, l_stem=35, l_head=20, r_head_base=10, fn=30)
    
    
    def __post_init__(self):
        if self.r_stem_base is None:
            self.r_stem_base = self.r_stem_top
            
        f_args = non_defaults_dict(self, include=ARGS_XLATION_TABLE)
        
        head = Cone(h=self.l_head, r_base=self.r_head_base, r_top=self.r_head_top, **f_args)
        stem = Cone(h=self.l_stem, r_base=self.r_stem_base, r_top=self.r_stem_top, **f_args)
        maker = stem.solid('stem').at('base')
        maker.add_at(head.solid('head').at('base', post=l.rotX(180)), 'top')
        self.maker = maker
        
    @anchor('The base of the stem of the object')
    def base(self, *args, **kwds):
        return self.maker.at('stem', 'base', *args, **kwds)
    
    @anchor('The top of the head')
    def top(self, *args, **kwds):
        return self.maker.at('head', 'top', *args, **kwds)
    
    @anchor('Access to inner elements of this shape.')
    def within(self, *args, **kwds):
        return self.maker.at(*args, **kwds)
    
@shape('coordinates_cage')
@dataclass
class CoordinatesCage(Shape):
    base_frame: l.GMatrix=l.IDENTITY

    def render(self, renderer):
        return renderer
    
    @anchor('The untransformed origin.')
    def origin(self):
        return l.IDENTITY
    
    @anchor('x axis orientation')
    def x(self):
        return self.base_frame
    
    @anchor('y axis orientation')
    def y(self):
        return l.ROTV111_120 * self.base_frame
    
    @anchor('z axis orientation')
    def z(self):
        return l.ROTV111_240 * self.base_frame
    

@shape('coordinates')
@dataclass
class Coordinates(CompositeShape):
    
    overlap: float=3.0
    colour_x: Colour=Colour([1, 0, 0])
    colour_y: Colour=Colour([0, 1, 0])
    colour_z: Colour=Colour([0, 0, 1])
    r_stem_top: float=0.75
    r_stem_base: float=None # Defaults to r_stem_top
    l_stem: float=10.0
    l_head: float=3
    r_head_base: float=1.5
    r_head_top: float=0.0
    fn: int=None
    fa: float=None
    fs: float=None
    
    def __post_init__(self):
        if self.r_stem_base is None:
            self.r_stem_base = self.r_stem_top
        exclude=('overlap', 'colour_x', 'colour_y', 'colour_z', )
        arrow = Arrow(**non_defaults_dict(self, exclude=exclude))
        maker = CoordinatesCage().cage('origin').at('origin')
            
        t = l.translate([0, 0, -self.overlap])
        maker .add_at(arrow.solid('x_arrow').colour(self.colour_x).at(
            'base', pre=t * l.rotZ(180)), 'x', pre=l.rotY(-90))
        maker .add_at(arrow.solid('y_arrow').colour(self.colour_y).at(
            'base', pre=t * l.rotZ(180)), 'y', pre=l.rotZ(-90))
        maker .add_at(arrow.solid('z_arrow').colour(self.colour_z).at(
            'base', pre=t * l.rotZ(180)), 'z', pre=l.rotX(-90))
        self.maker = maker
            
    @anchor('The base of the stem of the object')
    def origin(self):
        return l.IDENTITY
    
    @anchor('Access to inner elements of this shape.')
    def within(self, *args, **kwds):
        return self.maker.at(*args, **kwds)
    
    
@shape('annotated_coordinates')
@dataclass
class AnnotatedCoordinates(CompositeShape):
    
    coordinates: Coordinates=Coordinates()
    coord_labels: frozendict=frozendict({'x': 'x', 'y': 'y', 'z': 'z'})
    text_stem_size_ratio:float = 0.3
    coord_label_at: tuple=args(post=l.translate([0, 0, 1]) * l.rotY(-90))
    label: str=None
    label_pos_ratio: l.GVector=l.GVector([0.5, 0.5, 0.5])
    
    def __post_init__(self):
        
        maker = self.coordinates.solid('coords').at('origin')
        if self.coord_labels:
            for k, s in self.coord_labels.items():
                txt = Text(s, size=self.text_stem_size_ratio * self.coordinates.l_stem)
                maker.add_at(txt.solid(k).at('default', 'centre'), 
                             'within', f'{k}_arrow', 'top', 
                             *self.coord_label_at[0], **self.coord_label_at[1])
                
        self.maker = maker
    
    @anchor('The base of the stem of the object')
    def origin(self, *args, **kwds):
        return l.IDENTITY
    

    