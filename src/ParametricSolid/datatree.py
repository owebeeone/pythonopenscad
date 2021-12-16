'''
Created on 8 Dec 2021

@author: gianni

Provides dataclass like functionality across a composition of datatree objects.
'''

from dataclasses import dataclass, field, Field, MISSING
from frozendict import frozendict
import inspect

FIELD_FIELD_NAMES=tuple(inspect.signature(field).parameters.keys())
DATATREE_SENTIENEL_NAME='__datatree_nodes__'
OVERRIDE_FIELD_NAME='override'

class ReservedFieldNameException(Exception):
    f'''The name '{OVERRIDE_FIELD_NAME}' is reserved for use by datatree.'''
    
class NameCollision(Exception):
    '''The requested name is already specified.'''
    
class ExpectedDataclassObject(Exception):
    '''Node requires the given object to be dataclass decorated.'''

class MappedFieldNameNotFound(Exception):
    '''Field name specified is not found in the given class.'''    
    
class DataclassAlreadyApplied(Exception):
    '''The function called in intended to be called before the dataclass decorator.'''
    

def _update_name_map(clz, map, from_name, to_value, description):
    '''Updates the given map but does not allow collision.'''
    if from_name in map:
        raise NameCollision(
            f'{description} {from_name} specified multiple times in {clz.__name__}')
    map[from_name] = to_value
    
        
def _dupes_and_allset(itr):
    '''Returns a tuple containing a set of duplicates and a set of all non 
    duplicated items in itr.'''
    seen = set()
    return set((x for x in itr if x in seen or seen.add(x))), seen

@dataclass
class AnnotationDetails:
    '''A dataclass/annotation pair.'''
    field: object
    anno_type: type
    
    @classmethod
    def from_init_param(cls, name, inspect_params):
        '''Creates an AnnotationDetails from a name and inspect.Signature.parameters'''
        param = inspect_params[name]
        
        anno = param.annotation()
        if anno is inspect._empty:
            anno = object
            
        default = param.default
        
        if default is inspect._empty:
            default = MISSING
        
        this_field = field(default=default)
        return AnnotationDetails(this_field, anno)
        

@dataclass
class Node:
    '''A specifier for a datatree node. This allows the specification of how fields
    from a class initializer is translated from fields in the parent class.'''
    clz: type
    use_defaults: bool
    suffix: str
    prefix: str
    expose_all: bool
    init_signature: tuple
    
    def __init__(self, clz, *expose_spec, use_defaults=True, suffix='', prefix='', expose_all=None):
        self.clz = clz
        self.use_defaults = use_defaults
        self.expose_all = not expose_spec if expose_all is None else expose_all
        self.suffix = suffix
        self.prefix = prefix
        self.init_signature = inspect.signature(clz.__init__)
        
        fields_specified = tuple(f for f in expose_spec if isinstance(f, str))
        maps_specified = tuple(f for f in expose_spec if not isinstance(f, str))
        fields_in_maps = tuple(f for f in (m.values() for m in maps_specified))
        dupes, all_specified = _dupes_and_allset(fields_specified + fields_in_maps) 
        if dupes:
            raise NameCollision(f'Field names have multiple specifiers {dupes:r}')
        
        params = self.init_signature.parameters
        pi = iter(params.keys())
        next(pi)
        init_fields = set(pi)
        if self.expose_all:
            # Add all the fields not already specified.
            all_fields = set(name 
                             for name in init_fields
                             if name != OVERRIDE_FIELD_NAME)
            fields_specified = set(fields_specified).union(all_fields - all_specified)
        
        expose_dict = {}
        expose_rev_dict = {}
        
        # If we have a dataclass decorated class, use the __dataclass_fields__
        # to fill in this class,
        if hasattr(clz, '__dataclass_fields__'):
            for from_id in fields_specified:
                to_id = prefix + from_id + suffix
                if not from_id in init_fields:
                    raise MappedFieldNameNotFound(
                        f'Field name "{from_id}" is not an '
                        f'{clz.__name__}.__init__ parameter name')
                _update_name_map(
                    clz, expose_dict, from_id, to_id, 'Field name')
                anno_detail = AnnotationDetails(
                    clz.__dataclass_fields__[from_id], clz.__annotations__[from_id])
                _update_name_map(
                    clz, expose_rev_dict, to_id, anno_detail, 
                    'Mapped field name')
                
            for map_specified in maps_specified:
                # The dictionary has a set of from:to pairs.
                for from_id, to_id in map_specified.items():
                    if not from_id in init_fields:
                        raise MappedFieldNameNotFound(
                            f'Field name "{from_id}" mapped to "{to_id}" '
                            f'is not an {clz.__name__}.__init__ parameter name')
                    _update_name_map(
                        clz, expose_dict, from_id, to_id, 'Field name')
                    anno_detail = AnnotationDetails(
                        clz.__dataclass_fields__[from_id], clz.__annotations__[from_id])
                    _update_name_map(
                        clz, expose_rev_dict, to_id, anno_detail,
                        'Mapped field name')
        else:  # Not a dataclass type.
            
            for from_id in fields_specified:
                to_id = prefix + from_id + suffix
                if not from_id in init_fields:
                    raise MappedFieldNameNotFound(
                        f'Field name "{from_id}" is not an '
                        f'{clz.__name__}.__init__ parameter name')
                _update_name_map(
                    clz, expose_dict, from_id, to_id, 'Field name')
                anno_detail = AnnotationDetails.from_init_param(from_id, params)
                _update_name_map(
                    clz, expose_rev_dict, to_id, anno_detail, 
                    'Mapped field name')
                
            for map_specified in maps_specified:
                # The dictionary has a set of from:to pairs.
                for from_id, to_id in map_specified.items():
                    if not from_id in init_fields:
                        raise MappedFieldNameNotFound(
                            f'Field name "{from_id}" mapped to "{to_id}" '
                            f'is not an {clz.__name__}.__init__ parameter name')
                    _update_name_map(
                        clz, expose_dict, from_id, to_id, 'Field name')
                    anno_detail = AnnotationDetails.from_init_param(from_id, params)
                    _update_name_map(
                        clz, expose_rev_dict, to_id, anno_detail,
                        'Mapped field name')
            
        self.expose_map = frozendict(expose_dict)
        self.expose_rev_map = frozendict(expose_rev_dict)
        
    def get_rev_map(self):
        return self.expose_rev_map
    
def _make_dataclass_field(field_obj):
    value_map = dict((name, getattr(field_obj, name)) for name in FIELD_FIELD_NAMES)
    return field(**value_map)

def _apply_node_fields(clz):
    '''Adds new fields from Node annotations.'''
    if hasattr(clz, '__dataclass_fields__'):
        raise DataclassAlreadyApplied('The class {clz.__name__) already has dataclass applied.')
    annotations = clz.__annotations__
    new_annos = {}  # New set of annos to build.
    
    # The order in which items are added to the new_annos dictionary is important.
    # Here we maintain the same order of the original with the new exposed fields
    # interspersed between the Node annotated fields.
    nodes = {}
    for name, anno in annotations.items():
        new_annos[name] = anno
        if not hasattr(clz, name):
            continue
        anno_default = getattr(clz, name)
        if isinstance(anno_default, Field):
            anno_default = anno_default.default
        if isinstance(anno_default, Node):
            nodes[name] = anno_default
            rev_map = anno_default.get_rev_map()
            for rev_map_name, anno_detail in rev_map.items():
                if not rev_map_name in new_annos:
                    new_annos[rev_map_name] = anno_detail.anno_type
                    setattr(clz, rev_map_name, _make_dataclass_field(anno_detail.field))
    clz.__annotations__ = new_annos
    setattr(clz, DATATREE_SENTIENEL_NAME, nodes)
    return clz


class BoundNode:
    def __init__(self, parent, name, node, instance_values):
        self.parent = parent
        self.name = name
        self.node = node
        self.instance_values = instance_values
        

    def __call__(self, *args, **kwds):
        # Resolve parameter values.
        # Priority order:
        # 1. Override (if any)
        # 2. Passed in parameters
        # 3. Parent field values
        passed_bind = self.node.init_signature.bind_partial(*args, **kwds).arguments
        clz = self.node.clz
        ovrde = self.parent.override.get_override(self.name)
        if not ovrde is MISSING:
            ovrde_bind = ovrde.bind_signature(self.node.init_signature)
            
            for k, v in passed_bind.items():
                if not k in ovrde_bind:
                    ovrde_bind[k] = v
            
            if ovrde.clazz:
                clz = ovrde.clazz 
        else:
            ovrde_bind = passed_bind
        
        # Pull any values left from the parent.
        for fr, to in self.node.expose_map.items():
            if not fr in ovrde_bind:
                ovrde_bind[fr] = getattr(self.parent, to)
        
        
        return clz(**ovrde_bind)
    
@dataclass
class Exposures:
    items: tuple=None
    
    
    
class Overrides:
    
    def __init__(self, kwds):
        self.kwds = kwds
    
    def get_override(self, name):
        return self.kwds.get(name, MISSING)
    
def override(**kwds):
    
    return Overrides(kwds)


@dataclass
class Args:
    arg: tuple
    kwds: dict
    clazz: type=None
    
    def bind_signature(self, signature):
        return signature.bind_partial(*self.arg, **self.kwds).arguments
    
def args(*arg, clazz=None, **kwds):
    return Args(arg, kwds, clazz=clazz)


def _initialize_node_instances(clz, instance):
    '''Post dataclass initialization binding of nodes to instance.'''
    nodes = getattr(clz, DATATREE_SENTIENEL_NAME)
    
    for name, node in nodes.items():
        # The cur-value may contain args specifically for this node.
        cur_value = getattr(instance, name)
        bound_node = BoundNode(instance, name, node, cur_value)
        setattr(instance, name, bound_node)

# Default values for the dataclass function post Python 3.8.
_POST_38_DEFAULTS=args(match_args=True, kw_only=False, slots=False).kwds

def _process_datatree(clz, init, repr, eq, order, unsafe_hash, frozen,
                   match_args, kw_only, slots):

    if OVERRIDE_FIELD_NAME in clz.__annotations__:
        raise ReservedFieldNameException(
            f'Reserved field name {OVERRIDE_FIELD_NAME} used by class {clz.__name__}')
    clz.__annotations__['override'] = Overrides
    setattr(clz, OVERRIDE_FIELD_NAME, None)
    
    
    if hasattr(clz, '__post_init__'):
        clz.__multi_post_init__ = clz.__post_init__
        
    if hasattr(clz, '__multi_post_init__'):    
        def override_post_init(self):
            _initialize_node_instances(clz, self)
            self.__multi_post_init__()
        clz.__post_init__ = override_post_init
    else:   
        def override_post_init(self):
            _initialize_node_instances(clz, self)
        clz.__post_init__ = override_post_init

    _apply_node_fields(clz)
    
    values_post_38 = args(match_args=match_args, kw_only=kw_only, slots=slots).kwds
    values_post_38_differ = dict(
        ((k, v) for k, v in values_post_38.items() if v != _POST_38_DEFAULTS[k]))
        
    dataclass(clz, init=init, repr=repr, eq=eq, order=order,
              unsafe_hash=unsafe_hash, frozen=frozen, **values_post_38_differ)

    return clz


def datatree(clz=None, /, *, init=True, repr=True, eq=True, order=False,
              unsafe_hash=False, frozen=False, match_args=True,
              kw_only=False, slots=False):
    '''Decroator similar to dataclasses.dataclass providing for relaying
    parameters deeper inside a tree of dataclass objects.
    The __post_tree_init()
    
    '''
    
    def wrap(clz):
        return _process_datatree(clz, init, repr, eq, order, unsafe_hash,
                              frozen, match_args, kw_only, slots)

    # See if we're being called as @datatree or @datatree().
    if clz is None:
        # We're called with parens.
        return wrap

    # We're called as @datatree without parens.
    return wrap(clz)

