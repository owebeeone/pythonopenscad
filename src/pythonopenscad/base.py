'''PythonOpenScad is API layer over OpenScad scripts for generating OpenScad models.

The design goal for PythonOpenScad is to remain a thin layer for OpenScad scripts
and only provide functionality for producing such scripts. This may include binding
to a CSG library that skips the need to invoke OpenScad altogether but this is not
currently planned but does constrain design choices.

There are 2 other Python libraries that provide more functionality, OpenPyScad and SolidPython
as of writing this seem to have active development.

PythonOpenScad features include:
    * Error checking arguments and providing accurate value defaults.
    * A model API that is compatible with both SolidPython and SolidPython.
    * Python doc which is accurate and links to OpenScad reference documents.

See:
    `PythonOpenScad <http://bitbucket.org/owebeeone/pythonopenscad/src/master/>`
    `OpenScad <http://www.openscad.org/documentation.html>`
    `OpenPyScad <http://github.com/taxpon/openpyscad>`
    `SolidPython <http://github.com/SolidCode/SolidPython>`

License:

Copyright (C) 2020  Gianni Mariani

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
'''

import copy
import sys
from dataclasses import dataclass, field


# Exceptions for dealing with argument checking.
class PoscBaseException(Exception):
    '''Base exception functionality'''
    def __init__(self, message):
        self.message = message


class ConversionException(PoscBaseException):
    '''Exception for conversion errors.'''


class TooManyParameters(PoscBaseException):
    '''Exception when more unnamed parameters are provided than total
    parameters specified.'''


class ParameterNotDefined(PoscBaseException):
    '''Exception when passing a named parameter that has not been provided.'''


class ParameterDefinedMoreThanOnce(PoscBaseException):
    '''Exception when passing a named parameter that has already been provided.'''


class RequiredParameterNotProvided(PoscBaseException):
    '''Exception when a required parameter is not provided.'''


class InvalidModifier(PoscBaseException):
    '''Attempting to add or remove an unknown modifier.'''


class InitializerNotAllowed(PoscBaseException):
    '''An initializer (def __init__) was defined and is not allowed.'''


class InvalidIndentLevel(PoscBaseException):
    '''Indentation level was set to an invalid number.'''


class IndentLevelStackEmpty(PoscBaseException):
    '''Indentation level was set to an invalid number.'''


class InvalidValueForBool(PoscBaseException):
    '''Conversion failure for bool value.'''


class InvalidValueForStr(PoscBaseException):
    '''Conversion failure for str value.'''


class InvalidValue(PoscBaseException):
    '''Invalid value provided..'''


class DuplicateNamingOfArgs(PoscBaseException):
    '''OpenScadApiSpecifier args has names used more than once.'''


class NameCollissionFieldNameReserved(PoscBaseException):
    '''An attempt to define an arg with the same name as a field.'''


class AttemptingToAddNonPoscBaseNode(PoscBaseException):
    '''Attempted to add ad invalid object to child nodes.'''


class Arg(object):
    '''Defines an argument and field for PythonOpenScad PoscBase based APIs.
    '''
    def __init__(self,
                 name,
                 typ,
                 default_value,
                 docstring,
                 required=False,
                 osc_name=None,
                 attr_name=None):
        '''Args:
            name: The pythonopenscad name of this parameter.
            osc_name: The name used by OpenScad. Defaults to name.
            attr_name: Object attribute name.
            typ: The converter for the argument.
            default:_value: Default value for argument (this will be converted by typ).
            docstring: The python doc for the arg.
            required: Throws if the value is not provided.
        '''
        self.name = name
        self.osc_name = osc_name or name
        self.attr_name = attr_name or name
        self.typ = typ
        self.default_value = default_value
        self.docstring = docstring
        self.required = required
        
    def to_dataclass_field(self):
        return field(default=self.default_value)
    
    def annotation(self):
        return (self.name, self.typ)

    def default_value_str(self):
        '''Returns the default value as a string otherwise '' if no default provided.'''
        if self.default_value is None:
            return ''
        try:
            return repr(self.default_value)
        except:
            return ''

    def document(self):
        'Returns formatted documentation for this arg.'
        default_str = self.default_value_str()
        default_str = (' Default ' +
                       default_str) if default_str else default_str
        attribute_name = ('' if self.attr_name == self.name else
                          ' (Attribute name: %s) ' % self.attr_name)
        if self.name == self.osc_name:
            return '%s%s: %s%s' % (self.name, attribute_name, self.docstring,
                                   default_str)
        else:
            return '%s%s (converts to %s): %s %s' % (
                self.name, attribute_name, self.osc_name, self.docstring,
                default_str)


# Some special common Args. These are not consistently documented.
FA_ARG = Arg('_fa',
             float,
             None,
             'minimum angle (in degrees) of each segment',
             osc_name="$fa")
FS_ARG = Arg('_fs',
             float,
             None,
             'minimum length of each segment',
             osc_name="$fs")
FN_ARG = Arg('_fn',
             int,
             None,
             'fixed number of segments. Overrides $fa and $fs',
             osc_name="$fn")


def list_of(typ, len_min_max=(3, 3), fill_to_min=None):
    '''Defines a converter for an iterable to a list of elements of a given type.
    Args:
        typ: The type of list elements.
        len_min_max: A tuple of the (min,max) length, (0, 0) indicates no limits.
        fill_to_min: If the provided list is too short then use this value.
    Returns:
        A function that performs the conversion.
    '''
    description = 'list_of(%s, len_min_max=%r, fill_to_min=%r)' % (
        typ.__name__, len_min_max, fill_to_min)

    def list_converter(value):
        '''Converts provided value as a list of the given type.
        value: The value to be converted
        '''
        converted_value = []
        for v in value:
            if len_min_max[1] and len(converted_value) >= len_min_max[1]:
                raise ConversionException(
                    'provided length too large, max is %d' % len_min_max[1])
            converted_value.append(typ(v))
        if len_min_max[0] and len(value) < len_min_max[0]:
            if fill_to_min is None:
                raise ConversionException(
                    'provided length (%d) too small and fill_to_min is None, min is %d'
                    % (len(converted_value), len_min_max[0]))
            fill_converted = typ(fill_to_min)
            for _ in range(len_min_max[0] - len(converted_value)):
                converted_value.append(fill_converted)
        return converted_value

    list_converter.__name__ = description
    return list_converter


def one_of(typ, *args):
    '''Provides a converter that will iterate over the provided converters until it succeeds.
    Args:
      typ: The first converter argument.
      args: A list of supplemental type argument converters.
    '''
    largs = [typ] + list(args)
    description = 'one_of(%s)' % ', '.join(t.__name__ for t in largs)

    def one_of_converter(value):
        '''Converts a value to one of the list provided to one_of().
        Throws:
          ConversionException if the value failed all conversions.'''
        for atyp in largs:
            try:
                converted_value = atyp(value)
                return converted_value
            except:
                continue
        raise ConversionException('The value %r can\'t convert using %s' %
                                  (value, description))

    one_of_converter.__name__ = description
    return one_of_converter


class OscKeyword(object):
    '''Converts to the given string for allowing True to to true and False to false conversion.
    In the special case of the 'false' keyword, it converts to False on bool cast.
    '''
    def __init__(self, kw):
        self.kw = kw
        self._bool_value = (0, 1)[kw != 'false']

    def __str__(self):
        return self.kw

    def __repr__(self):
        return self.kw

    def __len__(self):
        return self._bool_value


OSC_TRUE = OscKeyword('true')
OSC_FALSE = OscKeyword('false')


def bool_strict(value):
    '''Returns an OscKeyword given bool i.e. 'true' if True else 'false'.
    Args:
        value: A boolean value.
    Throws:
        InvalidValueForBool if the provided value is not a bool.
    '''
    if not isinstance(value, bool):
        raise InvalidValueForBool(
            'expected a bool value but got "%r" of type %s' %
            (value, value.__class__.__name__))
    return OSC_TRUE if value else OSC_FALSE


def str_strict(value):
    '''Returns the given value if it is a str object otherwise raises
    InvalidValueForStr exception.
    Args:
        value: A string value.
    Throws:
        InvalidValueForStr if the provided value is not a str.
    '''
    if not isinstance(value, str) and not isinstance(value, bytes) :
        raise InvalidValueForStr(
            'expected a string value but got "%r" of type %s' %
            (value, value.__class__.__name__))
    return value


def of_set(*args):
    '''Returns a converter function that will throw if the the value to be converted is not
    one of args:
    Args:
        *args: The set of allowed values.
    Throws:
        InvalidValue if the value is not one of the args.
    '''
    allowed_values = set(args)
    description = 'of_set(allowed_values=%r)' % (tuple(allowed_values), )

    def of_set_converter(value):
        if not value in allowed_values:
            raise InvalidValue('%r is not allowed with %s.' %
                               (value, description))
        return value

    of_set_converter.__name__ = description
    return of_set_converter


# The base URL for OpenScad documentation,
OPEN_SCAD_BASE_URL = 'http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/'


class OpenScadApiSpecifier(object):
    '''Contains the specification of an OpenScad primitive.
    '''
    def __init__(self, openscad_name, args, url_base, alt_url_anchor=None):
        '''
        Args:
            openscad_name: The OpenScad primitive name.
            args: A tuple of Arg()s for each value passed in.
            url_base: The base of the document URL for OpenScad documentation.
        '''
        self.openscad_name = openscad_name
        self.args = args
        self.url_base = url_base
        self.alt_url_anchor = alt_url_anchor
        self.args_map = dict((arg.name, arg) for arg in args)

        if len(self.args) != len(self.args_map):
            all_names = [arg.name for arg in self.args]
            dupes = list(
                set([name for name in all_names if all_names.count(name) > 1]))
            raise DuplicateNamingOfArgs('Duplicate parameter names %r' % dupes)

    def generate_class_doc(self):
        '''Generates class level documentation.'''
        lines = [
            '\nConverts to an OpenScad "%s" primitive.' % self.openscad_name
        ]
        if self.url_base:
            anchor = self.openscad_name if self.alt_url_anchor is None else self.alt_url_anchor
            url = OPEN_SCAD_BASE_URL + self.url_base + '#' + anchor
            lines.append("See OpenScad `%s docs <%s>` for more information." %
                         (self.openscad_name, url))

        return '\n'.join(lines)

    def generate_init_doc(self):
        if self.args:
            return 'Args:\n    ' + ('\n    '.join(arg.document()
                                                  for arg in self.args))
        return 'No arguments allowed.'

class PoscMetadataBase(object):
    '''Provides medatadata properties.'''
    
    def getMetadataName(self):
        if not hasattr(self, '_metabase_name'):
            return ''
        return self._metabase_name

    def setMetadataName(self, value):
        self._metabase_name = value
        return self
    

class OscModifier(object):
    '''Defines an OpenScad modifier'''
    def __init__(self, modifier, name):
        self.modifier = modifier
        self.name = name

    def __repr__(self):
        return self.name


DISABLE = OscModifier('*', 'DISABLE')
SHOW_ONLY = OscModifier('!', 'SHOW_ONLY')
DEBUG = OscModifier('#', 'DEBUG')
TRANSPARENT = OscModifier('%', 'TRANSPARENT')
BASE_MODIFIERS = (DISABLE, SHOW_ONLY, DEBUG, TRANSPARENT)
BASE_MODIFIERS_SET = set(BASE_MODIFIERS)


class PoscModifiers(PoscMetadataBase):
    '''Functions to add/remove OpenScad modifiers.

    The add_modifier and remove_modifier functions can be chained as they return self.

    e.g.
    Cylinder() - Cube().add_modifier(SHOW_ONLY, DEBUG).color('#f00')

    Will create a red 1x1x1 cube with the ! and # OpenScad modifiers. The SHOW_ONLY
    modifier will cause the cylinder to not be displayed.
        difference() {
          cylinder(h=1.0, r=1.0, center=false);
          !#cube(size=[1.0, 1.0, 1.0]);
        }

    This API is specified to PythonOpenScad. OpenPyScad and SolidPython use different
    APIs for this feature.
    '''
    def check_is_valid_modifier(self, *modifiers):
        if set(modifiers) - BASE_MODIFIERS_SET:
            raise InvalidModifier(
                '"%r" is not a valid modifier. Muse be one of %r' %
                (modifiers, BASE_MODIFIERS))

    def add_modifier(self, modifier, *args):
        '''Adds one of the model modifiers like DISABLE, SHOW_ONLY, DEBUG or TRANSPARENT.
        Args:
          modifer, *args: The modifier/a being added. Checked for validity.
        '''
        self.check_is_valid_modifier(modifier, *args)
        if not hasattr(self, '_osc_modifier'):
            self._osc_modifier = set((modifier, ))
        self._osc_modifier.update(args + (modifier, ))
        return self

    def remove_modifier(self, modifier, *args):
        '''Removes a modifiers, one of DISABLE, SHOW_ONLY, DEBUG or TRANSPARENT.
        Args:
          modifer, *args: The modifier/s being removed. Checked for validity.
        '''
        self.check_is_valid_modifier(modifier, *args)
        if not hasattr(self, '_osc_modifier'):
            return
        self._osc_modifier.difference_update(args + (modifier, ))
        return self

    def has_modifier(self, modifier):
        '''Checks for presence of a modifier, one of DISABLE, SHOW_ONLY, DEBUG or TRANSPARENT.
        Args:
          modifer: The modifier being inspected. Checked for validity.
        '''
        self.check_is_valid_modifier(modifier)
        if not hasattr(self, '_osc_modifier'):
            return False
        return modifier in self._osc_modifier

    def get_modifiers(self):
        '''Returns the current set of modifiers as an OpenScad equivalent modifier string'''
        if not hasattr(self, '_osc_modifier'):
            return ''
        # Maintains order of modifiers.
        return ''.join(i.modifier for i in BASE_MODIFIERS
                       if i in self._osc_modifier)

    def get_modifiers_repr(self):
        '''Returns the repr() equivalent of the current set or None if none are set.'''
        if not hasattr(self, '_osc_modifier'):
            return None
        if self._osc_modifier:
            return repr(self._osc_modifier)
        return None

    # Deprecated.
    def transparent(self):
        self.add_modifier(TRANSPARENT)
        return self


class StringWriter(object):
    '''A CodeDumper writer that writes to a string. This can API can be implemented for
    file writers or other uses.'''
    def __init__(self):
        self.builder = []

    def get(self):
        '''Returns the contents. This is used by PythonOpenScad code in only the str
        and repr conversions which specifically create a StringWriter writer. Append
        is the only mentod called by the PythonOpenScad renderer.'''
        return '\n'.join(self.builder + [''])

    def append(self, line):
        '''Called by the PythonOpenScad renderer/code_dump to write generated model
        representation. Override this function to implement other output mechanisms.'''
        self.builder.append(line)


class FileWriter(object):
    '''A CodeDumper writer that writes to a file.'''
    def __init__(self, fp):
        self.fp = fp

    def finish(self):
        '''Writes the final components to the output'''
        self.fp.write('\n')

    def append(self, line):
        '''Called by the PythonOpenScad renderer/code_dump to write generated model
        representation. Override this function to implement other output mechanisms.'''
        self.fp.write(line)
        self.fp.write('\n')


class CodeDumper(object):
    '''Helper for pretty printing OpenScad scripts (and other scripts too).'''
    class IndentLevelState:
        '''Indent level state.'''
        def __init__(self, level, is_last):
            self.level = level
            self.is_last = is_last

    def get(self):
        return self.level, self.is_last

    def __init__(self,
                 indent_char=' ',
                 indent_multiple=2,
                 writer=None,
                 str_quotes='"',
                 block_ends=(' {', '}', ';', '//'),
                 target_max_column=100):
        '''
        Args:
           indent_char: the character used to indent.
           indent_multiple: the number of indent_char added per indent level.
           writer: A writer, like StringWriter.
           target_max_column: The max column number where line continuation may be used.
        '''
        self.indent_char = indent_char
        self.indent_multiple = indent_multiple
        self.writer = writer or StringWriter()
        self.str_quotes = str_quotes
        self.block_ends = block_ends
        self.current_indent_level = 0
        self.target_max_column = target_max_column
        self.current_indent_string = ''
        self.is_last = False
        self.indent_level_stack = []

    def check_indent_level(self, level):
        '''Check the adding of the resulting indent level will be in range.
        Args:
           level: The new requested indent level.
        Throws:
           InvalidIndentLevel level would is out of range
        '''
        if level < 0:
            raise InvalidIndentLevel(
                "Requested indent level below zero is not allowed.")

    def push_increase_indent(self, amount=1):
        '''Push an indent level change and increase indent level.
        Args:
           amount: the amount to increase the indent level, Amount can be negative. default 1
        '''
        current_level_state = CodeDumper.IndentLevelState(
            self.current_indent_level, self.is_last)
        try:
            self.set_indent_level(current_level_state.level + amount)
        finally:
            self.indent_level_stack.append(current_level_state)

    def set_indent_level(self, level):
        self.check_indent_level(level)
        self.current_indent_level = level
        self.current_indent_string = (self.indent_char * self.indent_multiple *
                                      self.current_indent_level)

    def pop_indent_level(self):
        '''Pops the indent level stack and sets the indent level to the popped value.'''
        if len(self.indent_level_stack) == 0:
            raise IndentLevelStackEmpty(
                'Empty indent level stack cannot be popped.')
        level_state = self.indent_level_stack.pop()
        self.set_indent_level(level_state.level)
        self.is_last = level_state.is_last

    def set_is_last(self, is_last):
        '''Set this to False if there is another item to be rendered after this one.'''
        self.is_last = is_last

    def get_is_last(self):
        '''Clients that care if a suffix needs adding if it is on the end of the list
        can check this.'''
        return self.is_last

    def add_line(self, line):
        '''Adds the given line as a whole line the output.

        Args:
            line: string to be added.
        '''
        self.writer.append(line)

    def write_line(self, line):
        '''Adds an indented line to the output. This could be used for comments.'''
        self.add_line(self.current_indent_string + line)

    def write_function(self,
                       function_name,
                       params_list,
                       mod_prefix='',
                       mod_suffix='',
                       suffix=';',
                       comment=None):
        '''Dumps a function like lines (may wrap).

        Args:
            function_name: name of function.
            prefix: a string added in front of the function name
            params_list: list of parameters (no commas separating them)
            suffix: A string at the end
        '''
        if comment:
            self.add_line(''.join([self.current_indent_string, comment]))
        strings = [self.current_indent_string, mod_prefix, function_name, '(']
        strings.append(', '.join(params_list))
        strings.append(')')
        strings.append(mod_suffix)
        strings.append(suffix)
        self.add_line(''.join(strings))

    def render_value(self, value):
        '''Returns a string representing the given value.'''
        if isinstance(value, str):
            return self.str_quotes + repr(value)[1:-1] + self.str_quotes
        return repr(value)

    def render_name_value(self, arg, value):
        return '%s=%s' % (arg.osc_name, self.render_value(value))

    def should_add_suffix(self):
        '''Returns true if the suffix should be added. OpenScad is always true.'''
        return True

    def get_modifiers_prefix_suffix(self, obj):
        '''Returns the OpenScad modifiers string.'''
        return (obj.get_modifiers(), '')


class CodeDumperForPython(CodeDumper):
    '''Helper for pretty printing to Python code compatible with SolidPython
    and PythonOpenScad.
    Args:
        Same parameters as CodeDumper but overrides defaults for str_quotes and
        block_ends.
    '''
    def __init__(self, *args, **kwds):
        kwds.setdefault('str_quotes', "'")
        kwds.setdefault('block_ends', (' (', '),', ',', '#'))
        super().__init__(*args, **kwds)
        self.is_last = True

    def render_value(self, value):
        '''Returns a string representing the given value.'''
        if value == OSC_TRUE:
            return 'True'
        elif value == OSC_FALSE:
            return 'False'
        return repr(value)

    def render_name_value(self, arg, value):
        return '%s=%s' % (arg.name, self.render_value(value))

    def should_add_suffix(self):
        '''Returns true if the suffix should be added. Python is true of if not at the
        end..'''
        return not self.is_last

    def get_modifiers_prefix_suffix(self, obj):
        '''Returns a Python mosifiers mutator.'''
        s = obj.get_modifiers_repr()
        if not s:
            return ('', '')
        return ('', '.add_modifier(*%s)' % s)


class PoscBase(PoscModifiers):

    def __post_init__(self):
        for arg in self.OSC_API_SPEC.args:
            value = getattr(self, arg.name)
            if not value is None:
                if arg.name != arg.attr_name:
                    delattr(self, arg.name)
                setattr(self, arg.attr_name, arg.typ(value))
        
        self.init_children()
        # Object should be fully constructed now.
        self.check_valid()

    def init_children(self):
        '''Initalizes objects that contain parents.'''
        # This node has no children.

    def check_valid(self):
        '''Checks that the construction of the object is valid.'''
        self.check_required_parameters()

    def check_required_parameters(self):
        '''Checks that required parameters are set and not None.'''
        for arg in self.OSC_API_SPEC.args:
            if arg.required and (getattr(self, arg.attr_name, None) is None):
                raise RequiredParameterNotProvided(
                    '"%s" is required and not provided' % arg.attr_name)

    def collect_args(self, code_dumper):
        '''Returns a list of arg=value pairs as strings.'''
        posc_args = self.OSC_API_SPEC.args
        result = []
        for arg in posc_args:
            v = getattr(self, arg.attr_name, None)
            if not v is None:
                result.append(code_dumper.render_name_value(arg, v))
        return result

    def can_have_children(self):
        '''This is a childless node, always returns False.'''
        return False

    def has_children(self):
        return False

    def children(self):
        '''This is a childless node, always returns empty tuple.'''
        return ()

    def code_dump(self, code_dumper):
        '''Dump the OpenScad equivalent of this script into the provided dumper.'''
        termial_suffix = (code_dumper.block_ends[2]
                          if code_dumper.should_add_suffix() else '')
        suffix = (code_dumper.block_ends[0]
                  if self.has_children() else termial_suffix)
        function_name = self.OSC_API_SPEC.openscad_name
        params_list = self.collect_args(code_dumper)
        mod_prefix, mod_suffix = code_dumper.get_modifiers_prefix_suffix(self)
        comment = None
        metadataName = self.getMetadataName()
        if metadataName:
            comment = code_dumper.block_ends[3] + ' ' + repr(metadataName)
        code_dumper.write_function(
            function_name, params_list, mod_prefix, mod_suffix, suffix, comment)
        if self.has_children():
            code_dumper.push_increase_indent()
            left = len(self.children())
            for child in self.children():
                left -= 1
                code_dumper.set_is_last(left == 0)
                child.code_dump(code_dumper)
            code_dumper.pop_indent_level()
            code_dumper.write_line(code_dumper.block_ends[1])

    def __str__(self):
        '''Returns the OpenScad equivalent code for this node.'''
        dumper = CodeDumper()
        self.code_dump(dumper)
        return dumper.writer.get()

    def __repr__(self):
        '''Returns the SolidPython equivalent code for this node.'''
        dumper = CodeDumperForPython()
        self.code_dump(dumper)
        return dumper.writer.get()

    def clone(self):
        return copy.deepcopy(self)

    def equals(self, other):
        if not hasattr(other, 'OSC_API_SPEC'):
            return False
        # Since we create other classes with the same OpenScadApiSpecifier, this
        # is the indicator of the object's type.
        if not (self.OSC_API_SPEC is other.OSC_API_SPEC):
            return False
        for arg in self.OSC_API_SPEC.args:
            if getattr(self, arg.attr_name) != getattr(other, arg.attr_name):
                return False
        return self.children() == other.children()

    # OpenPyScad compat functions.
    def dumps(self):
        '''Returns a string of this object's OpenScad script.'''
        dumper = CodeDumper()
        self.code_dump(dumper)
        return dumper.writer.get()

    def dump(self, fp):
        '''Writes this object's OpenScad script to the given file.
        Args:
            fp: The python file object to use.
        '''
        dumper = CodeDumper(writer=FileWriter(fp))
        self.code_dump(dumper)
        dumper.writer.finish()

    def write(self, filename):
        '''Writes the OpenScad script to the given file name.
        Args:
            filename: The filename to create.
        '''
        with open(filename, 'w') as fp:
            self.dump(fp)

    # Documentation for the following functions is generated by the decorator
    # apply_posc_transformation_attributes.

    def translate(self, *args, **kwds):
        return Translate(*args, **kwds)(self)

    def rotate(self, *args, **kwds):
        return Rotate(*args, **kwds)(self)

    def scale(self, *args, **kwds):
        return Scale(*args, **kwds)(self)

    def resize(self, *args, **kwds):
        return Resize(*args, **kwds)(self)

    def mirror(self, *args, **kwds):
        return Mirror(*args, **kwds)(self)

    def color(self, *args, **kwds):
        return Color(*args, **kwds)(self)

    def multmatrix(self, *args, **kwds):
        return Multmatrix(*args, **kwds)(self)

    def offset(self, *args, **kwds):
        return Offset(*args, **kwds)(self)

    def projection(self, *args, **kwds):
        return Projection(*args, **kwds)(self)

    def minkowski(self, *args, **kwds):
        return Minkowski(*args, **kwds)(self)

    def hull(self, *args, **kwds):
        return Hull(*args, **kwds)(self)

    def linear_extrude(self, *args, **kwds):
        return Linear_Extrude(*args, **kwds)(self)

    def rotate_extrude(self, *args, **kwds):
        return Rotate_Extrude(*args, **kwds)(self)

    def __eq__(self, other):
        '''Exact object tree equality. (Not resulting shape equality)'''
        return self.equals(other)

    def __ne__(self, other):
        '''Exact object tree inequality. (Not resulting shape equality)'''
        return not self.equals(other)

    def __add__(self, other):
        '''Union of this with other 3D object. See Union.'''
        return Union()(self, other)

    def __sub__(self, other):
        '''Difference of this with other 3D object. See Difference.'''
        return Difference()(self, other)

    # OpenPyScad compatability
    def __and__(self, other):
        '''Intersect this with other 3D object. See Intersection.'''
        return Intersection()(self, other)

    # SolidPython compatability
    def __mul__(self, other):
        '''Intersect this with other 3D object. See Intersection.'''
        return Intersection()(self, other)


# A decorator for PoscBase classes.
def apply_posc_attributes(clazz):
    '''Decorator that applies an equivalent constructor with it\'s own generated
     docstring. Also adds some SolidPython script compatibility class by providing an alias
     class in the current module.'''
    if clazz.__init__ != PoscBase.__init__:
        raise InitializerNotAllowed('class %s should not define __init__' %
                                    clazz.__name__)
    # Check for name collision.
    for arg in clazz.OSC_API_SPEC.args:
        if hasattr(clazz, arg.attr_name):
            raise NameCollissionFieldNameReserved(
                'There exists an attribute \'%s\' for class %s that collides with an arg.'
                % (arg.name, clazz.__name__))
    annotations = dict((arg.annotation() for arg in clazz.OSC_API_SPEC.args))
    clazz.__annotations__ = annotations
    for arg in clazz.OSC_API_SPEC.args:
        setattr(clazz, arg.name, arg.to_dataclass_field())
    dataclass(repr=False)(clazz)
    clazz.__init__.__doc__ = clazz.OSC_API_SPEC.generate_init_doc()
    strs = []
    if clazz.__doc__:
        strs.append(clazz.__doc__)
    strs.append(clazz.OSC_API_SPEC.generate_class_doc())
    clazz.__doc__ = '\n'.join(strs)
    # Add an alias of this class for the OpenScad name. Be compatible with SolidPython.
    setattr(sys.modules[__name__], clazz.OSC_API_SPEC.openscad_name,
            type(clazz.OSC_API_SPEC.openscad_name, (clazz, ), {}))

    return clazz


class PoscParentBase(PoscBase):
    '''A PoscBase class that has children. All OpenScad forms that have children use this.
    This provides basic child handling functions.'''
    def init_children(self):
        '''Initalizes objects that contain parents.'''
        self._children = []

    def can_have_children(self):
        '''Returns true. This node can have children.'''
        return True

    def has_children(self):
        '''Returns true if the node has children.'''
        return bool(self._children)

    def children(self):
        '''Returns the list of children'''
        return self._children

    def append(self, *children):
        '''Appends the children to this node.
        Args:
          *children: children to append.
          '''
        return self.extend(children)

    def extend(self, children):
        '''Appends the list of children to this node.
        Args:
          children: list of children to append.
          '''
        # Don't add nodes that are not PoscBase nodes.
        for child in children:
            if child is None or not hasattr(child, 'OSC_API_SPEC'):
                raise AttemptingToAddNonPoscBaseNode(
                    'Cannot append object %r as child node' % child)
        self._children.extend(children)
        return self

    # Support Obj(child, ...) constructs like that in OpenScad and SolidPuthon.
    __call__ = append


# A decorator for transformation classes.
def apply_posc_transformation_attributes(clazz):
    '''Does everything that apply_posc_attributes() does but also adds documentation
    to the corresponding PoscBase function of the same name.
    '''
    clazz = apply_posc_attributes(clazz)
    # The base class should have a properly named transform function.
    # If initialization fails here, then add the function.
    transform = getattr(PoscBase, clazz.OSC_API_SPEC.openscad_name)
    transform.__doc__ = clazz.__doc__ + '\n' + clazz.__init__.__doc__
    return clazz


# Often used converters.
VECTOR3_FLOAT = list_of(float, len_min_max=(3, 3), fill_to_min=0.0)
VECTOR3_FLOAT_DEFAULT_1 = list_of(float, len_min_max=(3, 3), fill_to_min=1.0)
VECTOR3OR4_FLOAT = list_of(float, len_min_max=(3, 4), fill_to_min=0.0)
VECTOR4_FLOAT = list_of(float, len_min_max=(4, 4), fill_to_min=0.0)
VECTOR2_FLOAT = list_of(float, len_min_max=(2, 2), fill_to_min=0.0)
VECTOR2_FLOAT_DEFAULT_1 = list_of(float, len_min_max=(2, 2), fill_to_min=1.0)
VECTOR3_BOOL = list_of(bool_strict, fill_to_min=False)

# The set of OpenScad doumentation URL tails.
OPEN_SCAD_URL_TAIL_2D = 'Using_the_2D_Subsystem'
OPEN_SCAD_URL_TAIL_PRIMITIVES = 'Primitive_Solids'
OPEN_SCAD_URL_TAIL_TRANSFORMS = 'Transformations'
OPEN_SCAD_URL_TAIL_CSG = 'CSG_Modelling'
OPEN_SCAD_URL_TAIL_IMPORTING = 'Importing_Geometry'
OPEN_SCAD_URL_TAIL_OTHER = 'Other_Language_Features'


@apply_posc_transformation_attributes
class Translate(PoscParentBase):
    '''Translate child nodes.'''
    OSC_API_SPEC = OpenScadApiSpecifier('translate', (
        Arg('v', VECTOR3_FLOAT, None, '(x,y,z) translation vector.', required=True),),
        OPEN_SCAD_URL_TAIL_TRANSFORMS)


@apply_posc_transformation_attributes
class Rotate(PoscParentBase):
    '''Rotate child nodes. This comes in two forms, the first form
    is a single angle and optional axis of rotation, the second form
    is a vector of angles for three consecutive rotations about the
    z, y and z axis.'''
    OSC_API_SPEC = OpenScadApiSpecifier('rotate', (
        Arg('a', one_of(float, list_of(float, fill_to_min=0.0)), 0,
            'Angle to rotate or vector of angles applied to each axis '
            + 'in sequence.'),
        Arg('v', VECTOR3_FLOAT, None, '(x,y,z) axis of rotation vector.')),
        OPEN_SCAD_URL_TAIL_TRANSFORMS)


@apply_posc_attributes
class Cylinder(PoscBase):
    '''Creates a cylinder or truncated cone about the z axis. Cone needs r1 and r2 or d1 and d2
    provided with different lengths.
    d1 & d2 have precedence over d which have precedence over r1 and r2 have precedence over r.
    Hence setting r is overridden by any other value.
    '''
    OSC_API_SPEC = OpenScadApiSpecifier('cylinder', (
        Arg('h', float, 1.0, 'height of the cylinder or cone.', required=True),
        Arg('r', float, 1.0, 'radius of cylinder. r1 = r2 = r.'),
        Arg('r1', float, None, 'radius, bottom of cone.'),
        Arg('r2', float, None, 'radius, top of cone.'),
        Arg('d', float, None, 'diameter of cylinder. r1 = r2 = d / 2.'),
        Arg('d1', float, None, 'diameter of bottom of cone. r1 = d1 / 2.'),
        Arg('d2', float, None, 'diameter of top of cone. r2 = d2 / 2.'),
        Arg('center', bool_strict, False,
            'z ranges from 0 to h, true z ranges from -h/2 to +h/2.'),
        FA_ARG,
        FS_ARG,
        FN_ARG),
        OPEN_SCAD_URL_TAIL_PRIMITIVES)

    def get_r1(self):
        '''Returns the bottom radius of the cylinder or cone.'''
        if not self.d1 is None:
            return self.d1 / 2

        if not self.r1 is None:
            return self.r1

        if not self.d is None:
            return self.d / 2

        return self.r

    def get_r2(self):
        '''Returns the top radius of the cylinder or cone.'''
        if not self.d2 is None:
            return self.d2 / 2

        if not self.r2 is None:
            return self.r2

        if not self.d is None:
            return self.d / 2

        return self.r

    def check_valid(self):
        '''Checks that the values of cylinder satisfy OpenScad cylinder requirements.'''
        values = (('r1', self.get_r1()), ('r2', self.get_r2()))
        for k, v in values:
            if v is None:
                raise RequiredParameterNotProvided(
                    '"%s" is required and not provided' % k)
        self.check_required_parameters()


@apply_posc_attributes
class Sphere(PoscBase):
    '''Creates a sphere.
    It defaults to a sphere of radius 1. If d is provided it overrides the value of r.
    '''
    OSC_API_SPEC = OpenScadApiSpecifier('sphere', (
        Arg('r', float, 1.0, 'radius of sphere. Ignores d if set.'),
        Arg('d', float, None, 'diameter of sphere.'),
        FA_ARG,
        FS_ARG,
        FN_ARG),
        OPEN_SCAD_URL_TAIL_PRIMITIVES)

    def get_r(self):
        '''Returns the top radius of the cylinder or cone.'''
        if not self.d is None:
            return self.d / 2

        return self.r

    def check_valid(self):
        '''Checks that the construction of cylinder is valid.'''
        if all(x is None for x in [self.r, self.d]):
            raise RequiredParameterNotProvided(
                'Both parameters r and d are None. A value for r or d must be provided.'
            )
        self.check_required_parameters()


@apply_posc_attributes
class Cube(PoscBase):
    '''Creates a cube with it's bottom corner centered at the origin.'''
    OSC_API_SPEC = OpenScadApiSpecifier('cube', (
        Arg('size', one_of(float, VECTOR3_FLOAT_DEFAULT_1), (1, 1, 1),
            'The x, y and z sizes of the cube or rectangular prism', required=True),
        Arg('center', bool_strict, None, 'If true places the center of the cube at the origin.'),),
        OPEN_SCAD_URL_TAIL_PRIMITIVES)


@apply_posc_transformation_attributes
class Scale(PoscParentBase):
    '''Scales the child nodes. scale'''
    OSC_API_SPEC = OpenScadApiSpecifier('scale', (
        Arg('v', one_of(float, VECTOR3_FLOAT_DEFAULT_1), (1, 1, 1), 'The (x,y,z) scale factors.'),),
        OPEN_SCAD_URL_TAIL_TRANSFORMS)


@apply_posc_transformation_attributes
class Resize(PoscParentBase):
    '''Scales the object so the newsize (x,y,z) parameters given. A zero (0.0) scale is ignored
    and that dimension's scale factor is 1.'''
    OSC_API_SPEC = OpenScadApiSpecifier('resize', (
        Arg('newsize', list_of(float, len_min_max=(0, 3)), None,
            'The new (x,y,z) sizes of the resulting object.'),
        Arg('auto', one_of(bool_strict, VECTOR3_BOOL), None,
            'A vector of (x,y,z) booleans to indicate which axes will be resized.'),),
        OPEN_SCAD_URL_TAIL_TRANSFORMS)


@apply_posc_transformation_attributes
class Mirror(PoscParentBase):
    '''Mirrors across a plane defined by the normal v.'''
    OSC_API_SPEC = OpenScadApiSpecifier('mirror', (
        Arg('v', VECTOR3_FLOAT, None, 'The normal of the plane to be mirrored.'),),
        OPEN_SCAD_URL_TAIL_TRANSFORMS)


@apply_posc_transformation_attributes
class Multmatrix(PoscParentBase):
    '''Homogeneous matrix multiply. The provided matrix can both rotate and translate.
    '''
    OSC_API_SPEC = OpenScadApiSpecifier('multmatrix', (
        Arg('m', list_of(VECTOR4_FLOAT, len_min_max=(3, 4)), None,
            '''A 4x4 or 4x3 matrix. The last row must always be [0,0,0,1] and in the
            case of a 4x3 matrix that row is added. The resulting matrix is always 4x4.'''),),
        OPEN_SCAD_URL_TAIL_TRANSFORMS)

    def check_valid(self):
        '''Checks that the construction of cylinder is valid.'''
        if len(self.m) == 3:
            self.m.append([0.0, 0.0, 0.0, 1.0])
        self.check_required_parameters()

    def get_m(self):
        '''Returns the matrix m. The returned value is always a 4x4 matrix.'''
        return self.m


@apply_posc_transformation_attributes
class Color(PoscParentBase):
    '''Apply a color (only supported in OpenScad preview mode). Colors can be a 3 vector
    of values [0.0-1.0] for RGG or additionally a 4 vector if alpha is included for an
    RGBA color. Colors can be specified as #RRGGBB and it's variants.'''
    OSC_API_SPEC = OpenScadApiSpecifier('color', (
        Arg('c', one_of(str_strict, VECTOR3OR4_FLOAT), None,
            'A 3 or 4 color RGB or RGBA vector or a string descriptor of the color.'),
        Arg('alpha', float, None, 'The alpha of the color if not already provided by c.'),),
        OPEN_SCAD_URL_TAIL_TRANSFORMS)

@apply_posc_transformation_attributes
class Offset(PoscParentBase):
    '''Generates a new polygon with the curve offset by the given amount. Negative values
    can be used to shrink paths while positive values enlarge the path.'''
    OSC_API_SPEC = OpenScadApiSpecifier('offset', (
        Arg('r', float, None, 'The radius of the new path when using the radial method.'),
        Arg('delta', float, None, 'The offset of the new path when using the offset method.'),
        Arg('chamfer', bool_strict, False, 'If true will create chamfers at corners.'),
        FA_ARG,
        FS_ARG,
        FN_ARG),
        OPEN_SCAD_URL_TAIL_TRANSFORMS)

    def check_valid(self):
        '''Checks that the construction of cylinder is valid.'''
        if all(x is None for x in [self.r, self.delta]):
            self.r = 1.0
        self.check_required_parameters()


@apply_posc_transformation_attributes
class Projection(PoscParentBase):
    '''Project a 3D object into a 2D surface.'''
    OSC_API_SPEC = OpenScadApiSpecifier('projection', (
        Arg('cut', bool_strict, None,
            'If false, the projection is a "shadow" of the object otherwise it is an intersection.'),),
        OPEN_SCAD_URL_TAIL_2D, '3D_to_2D_Projection')

@apply_posc_transformation_attributes
class Minkowski(PoscParentBase):
    '''Create a Minkowski transformed object.'''
    OSC_API_SPEC = OpenScadApiSpecifier('minkowski', (), OPEN_SCAD_URL_TAIL_TRANSFORMS)

@apply_posc_transformation_attributes
class Hull(PoscParentBase):
    '''Create a hull of two solids.'''
    OSC_API_SPEC = OpenScadApiSpecifier('hull', (), OPEN_SCAD_URL_TAIL_TRANSFORMS)

@apply_posc_transformation_attributes
class Linear_Extrude(PoscParentBase):
    '''Creates an 3D object with a linear extrusion of a 2D shape.'''
    OSC_API_SPEC = OpenScadApiSpecifier('linear_extrude', (
        Arg('height', float, 100, 'The height of the resulting extrusion.'),
        Arg('center', bool_strict, None,
            'If true, the final object\'s height center point is placed at z=0.'),
        Arg('convexity', int, None, 'A convexity value used for preview mode to aid rendering.'),
        Arg('twist', float, None,
            'If provided the object is rotated about the z axis by this total angle'),
        Arg('slices', int, None, 'The number of slices to be applied in the resulting extrusion.'),
        Arg('scale', one_of(float, VECTOR2_FLOAT), None,
            'A scale factor to applied to the children incrementally per extrusion layer.',
             attr_name='scale_'),
        FN_ARG),
        OPEN_SCAD_URL_TAIL_2D, 'Linear_Extrude')


@apply_posc_transformation_attributes
class Rotate_Extrude(PoscParentBase):
    '''Creates an 3D object with a rotating extrusion of a 2D shape.'''
    OSC_API_SPEC = OpenScadApiSpecifier('rotate_extrude', (
        Arg('angle', float, 360, 'The total angle to extrude.'),
        Arg('convexity', int, None, 'A convexity value used for preview mode to aid rendering.'),
        FA_ARG,
        FS_ARG,
        FN_ARG),
        OPEN_SCAD_URL_TAIL_2D, 'Rotate_Extrude')


@apply_posc_attributes
class Circle(PoscBase):
    '''Creates a 2D circle shape.
    Note that if d is provided it has precedence over r if provided.
    '''
    OSC_API_SPEC = OpenScadApiSpecifier('circle', (
        Arg('r', float, 1, 'The radius of the generated circle.'),
        Arg('d', float, None, 'The diameter of the circle, overrides r.'),
        FA_ARG,
        FS_ARG,
        FN_ARG),
        OPEN_SCAD_URL_TAIL_2D)

    def get_r(self):
        '''Returns the top radius of the circle.'''
        if not self.d is None:
            return self.d / 2
        return self.r

    def check_valid(self):
        '''Checks that the parameters of circle satisfy OpenScad circle requirements.'''
        if not self.d is None:
            self.r = None
        if self.get_r() is None:
            raise RequiredParameterNotProvided('r or d is required and not provided')
        self.check_required_parameters()


@apply_posc_attributes
class Square(PoscBase):
    '''Creates a 2D square shape'''
    OSC_API_SPEC = OpenScadApiSpecifier('square', (
        Arg('size', one_of(float, VECTOR2_FLOAT_DEFAULT_1), 1,
            'The square size, if a 2 vector (x,y) is provided a rectangle is generated.'),
        Arg('center', bool_strict, None,
            'If true the resulting shape is centered otherwise a corner is at the origin.'),),
        OPEN_SCAD_URL_TAIL_2D)


@apply_posc_attributes
class Polygon(PoscBase):
    '''Creates a polygon 2D shape (with optional holes).
    If paths is not provided, one is constructed by generating a sequence 0,,N-1 where N
    is the number of points provided.
    '''
    OSC_API_SPEC = OpenScadApiSpecifier('polygon', (
        Arg('points', list_of(VECTOR2_FLOAT, len_min_max=(None, None)), None,
            'A collection of (x,y) points to be indexed in paths.'),
        Arg('paths', list_of(list_of(int, len_min_max=(None, None)), len_min_max=(None, None)),
            None,
            'A list of paths which are a list of indexes into the points collection.'),
        Arg('convexity', int, None, 'A convexity value used for preview mode to aid rendering.'),),
        OPEN_SCAD_URL_TAIL_2D)

@apply_posc_attributes
class Text(PoscBase):
    '''Creates a 2D shape from a text string with a given font. A 2D shape consisting of
    an outline for each glyph in the string.'''
    OSC_API_SPEC = OpenScadApiSpecifier('text', (
        Arg('text', str_strict, None, 'The string used to generate the'),
        Arg('size', float, None, ''),
        Arg('font', str_strict, None, ''),
        Arg('halign', of_set('left', 'center', 'right'), None, ''),
        Arg('valign', of_set('top', 'center', 'baseline', 'bottom'), None, ''),
        Arg('spacing', float, None, ''),
        Arg('direction', of_set('ltr', 'rtl', 'ttb', 'btt'), None, ''),
        Arg('language', str_strict, None, ''),
        Arg('script', str_strict, None, ''),
        FA_ARG,
        FS_ARG,
        FN_ARG),
        'Text', '')


@apply_posc_attributes
class Polyhedron(PoscBase):
    '''Creates an arbitrary polyhedron 3D object.
    Note: triangles is deprecated.'''
    OSC_API_SPEC = OpenScadApiSpecifier('polyhedron', (
        Arg('points', list_of(VECTOR3_FLOAT, len_min_max=(None, None)), None,
            'A list of 3D points. The index to these points are used in faces or triangles.'),
        Arg('triangles', list_of(list_of(int, len_min_max=(3, 3)), len_min_max=(None, None)), None,
            'A list of triangles. Each triangle is 3 indexes into the points list.'),
        Arg('faces', list_of(list_of(int, len_min_max=(3, None)), len_min_max=(None, None)), None,
            'A list of faces. Each face is a minimum of 3 indexes into the points list'),
        Arg('convexity', int, 10, 'A convexity value used for preview mode to aid rendering.'),),
        OPEN_SCAD_URL_TAIL_PRIMITIVES)

@apply_posc_attributes
class Union(PoscParentBase):
    '''Unifies a set of 3D objects into a single object by performing a union of all the space
    contained by all the shapes.'''
    OSC_API_SPEC = OpenScadApiSpecifier('union', (), OPEN_SCAD_URL_TAIL_CSG)


@apply_posc_attributes
class Difference(PoscParentBase):
    '''Creates a 3D object by removing the space of the 3D objects following the first
    object provided from the first object.'''
    OSC_API_SPEC = OpenScadApiSpecifier('difference', (), OPEN_SCAD_URL_TAIL_CSG)


@apply_posc_attributes
class Intersection(PoscParentBase):
    '''Creates a 3D object by finding the common space contained in all the provided
    3D objects.'''
    OSC_API_SPEC = OpenScadApiSpecifier('intersection', (), OPEN_SCAD_URL_TAIL_CSG)


@apply_posc_attributes
class Import(PoscBase):
    '''Import a file as 3D or 2D shapes.
    SVG and DXF files generate 2D shapes.
    STL, OFF, AMF and 3MF files generate 3D shapes.'''
    OSC_API_SPEC = OpenScadApiSpecifier('import', (
        Arg('file', str_strict, None,
            'The filename to import. Relative path names are relative to the script location.'),
        Arg('convexity', int, None, 'A convexity value used for preview mode to aid rendering.'),
        Arg('layer', str, None,
            'When importing a DXF file, this will select the layer to be imported.'),),
        OPEN_SCAD_URL_TAIL_IMPORTING)


@apply_posc_attributes
class Surface(PoscBase):
    '''Import a file as a height map. This can be a image file or a text file.'''
    OSC_API_SPEC = OpenScadApiSpecifier('surface', (
        Arg('file', str_strict, None, 'File name used to load the height map.'),
        Arg('center', bool_strict, None,
            'If true the resulting shape is centered otherwise a corner is at the origin.'),
        Arg('invert', bool_strict, None,
            'If the file is an image, a value of true will invert the height data.'),
        Arg('convexity', int, None, 'A convexity value used for preview mode to aid rendering.'),),
        OPEN_SCAD_URL_TAIL_OTHER, 'Surface')

