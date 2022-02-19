'''PythonOpenScad is a more traditional API layer over OpenScad,
The foundations are similar to that of OpenPyScan but hopefully simpler.

'''

class Arg(object):
    '''Defines an argument and field for PoscBase based APIs.
    '''
    def __init__(self, name, type, default_value, docstring, osc_name=None, required=False):
        self.name = name
        self.osc_name = osc_name if osc_name else name
        self.type = type
        self.default_value = default_value
        self.docstring = docstring
        self.required = required
        

FA_ARG = Arg('_fa', float, None, 'minimum angle (in degrees) of each fragment', osc_name="$fa"),
FS_ARG = Arg('_fs', float, None, 'minimum length of each fragment', osc_name="$fs"),
FN_ARG = Arg('_fn', int, None, 'fixed number of fragments. Values of 3 or more override $fa and $fs', osc_name="$fn"),


class ConversionException(Exception):
    '''Exception for conversion errors.'''
    def __init__(self, message):
        self.message = message


class TooManyParameters(Exception):
    '''Exception when more unnamed parameters are provided than total 
    parameters specified.'''
    def __init__(self, message):
        self.message = message


class ParameterNotDefined(Exception):
    '''Exception when passing a named parameter that has not been provided.'''
    def __init__(self, message):
        self.message = message

class ParameterDefinedMoreThanOnce(Exception):
    '''Exception when passing a named parameter that has already been provided.'''
    def __init__(self, message):
        self.message = message
        
class RequiredParameterNotProvided(Exception):
    '''Exception when a required parameter is not provided.'''
    def __init__(self, message):
        self.message = message

def list_of(type, len_min_max=(3, 3), fill_to_min=None):
    '''Defines a converter for a list.
    type: The type of list elements.
    size_min_max: A tuple of the (min,max) length.
    '''
    def list_converter(value):
        '''Converts provided value as a list of the given type.
        value: The value to be converted
        '''
        if len(value) > size_min_max[1]:
            raise ConversionException(
                'provided length (%d) too large, max is %d' 
                % (len(value), size_min_max[1]))
        
        converted_value = []
        for v in value:
            converted_value.append(type(v))
        if len(value) < size_min_max[0]:
            if fill_to_min is None:
                raise ConversionException(
                    'provided length (%d) too small and fill_to_min is None, min is %d' 
                    % (len(value), size_min_max[0]))
            fill_converted = type(fill_to_min)
            for _ in range(size_min_max[0] - len(value)):
                converted_value.append(fill_converted)
        return converted_value
    return list_converter


class OpenScadApiSpecifier(object):
    '''Contains the specification of an OpenScad primitive. POSC classes should
    '''
    def __init__(self, openscad_name, args):
        '''
        openscad_name: The OpenScad primitive name.
        args: A tuple of Arg()s for each value passed in.
        '''
        self.openscad_name = openscad_name
        self.args = args
        args_map = dict((arg.name, arg) for arg in args)


class PoscBase(object):
    '''Base class for PythonOpenScad public classes.
    
    Class variable OSC_API_SPEC must be set to a OpenScadApiSpecifier() in
    derived classes. This will be used to find all values.
    '''

    def __init__(self, *values, **kwds):
        '''Constructor for all PythonOpenScad based classes.
        '''
        posc_args = self.OSC_API_SPEC.args
        if len(values) > len(posc_args):
            raise TooManyParameters(
                '%d parameters provided but only %d parameters are specified'
                % (len(values), len(posc_args)))
        for i in range(len(values)):
            arg = posc_args[i]
            value = arg.type(values[i])
            setattr(self, arg.name, value)
            
        args_map = self.OSC_API_SPEC.args_map
        for key, in_value in kwds.items:
            arg = args_map.get(key, None)
            if arg is None:
                raise ParameterNotDefined('Undefined parameter "%s" passed' 
                                          % key)
            if hasattr(self, key):
                raise ParameterDefinedMoreThanOnce(
                    'Parameter "%s" is defined at least twice' % key)
            value = arg.type(in_value)
            
        self._check_valid()
    
    def check_valid(self):
        '''Checks that the construction of the object is valid.'''
        self.check_required_parameters()
    
    def check_required_parameters(self):
        '''Checks that required parameters are set and not None.'''
        for arg in self.OSC_API_SPEC.args:
            if arg.required and (getattr(self, arg.name, None) is None):
                raise RequiredParameterNotProvided('"%s" is required and not provided'
                                                   % arg.name)



class Cylinder(PoscBase):
    '''Creates a cylinder about the z axis.
    '''
    OSC_API_SPEC = OpenScadApiSpecifier('cylinder', (
        Arg('h', float, None, 'height of the cylinder or cone'),
        Arg('r', float, None, 'radius of cylinder. r1 = r2 = r'),
        Arg('r1', float, None, 'radius, bottom of cone'),
        Arg('r2', float, None, 'radius, top of cone'),
        Arg('d', float, None, 'diameter of cylinder. r1 = r2 = d / 2'),
        Arg('d1', float, None, 'diameter of bottom of cone. r1 = d1 / 2'),
        Arg('d2', float, None, 'diameter of top of cone. r2 = d2 / 2'),
        Arg('center',  bool, False, 'false (default), z ranges from 0 to h, true z ranges from -h/2 to +h/2'),
        FA_ARG,
        FS_ARG,
        FN_ARG))

