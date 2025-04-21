"""PythonOpenScad is a thin layer API for generating OpenSCAD scripts.

PythonOpenScad aims to remain a minimal layer for generating OpenScad scripts while providing:
* Type checking and conversion of arguments with accurate error messages
* Support for both OpenPyScad and SolidPython style APIs
* Comprehensive documentation with links to OpenSCAD reference docs
* Module functionality for reducing code duplication

The primary client for PythonOpenScad is anchorSCAD, which provides higher-level 
functionality for building complex models.

See:
    `PythonOpenScad <https://github.com/owebeeone/pythonopenscad>` (this)
    `OpenSCAD <http://www.openscad.org/documentation.html>`
    `OpenPyScad <http://github.com/taxpon/openpyscad>`
    `SolidPython <http://github.com/SolidCode/SolidPython>`
    `anchorscad <https://github.com/owebeeone/anchorscad>`

License:

Copyright (C) 2025 Gianni Mariani

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
"""

import copy
from numbers import Integral
import sys
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Tuple
from pythonopenscad.m3dapi import M3dRenderer, RenderContext, RenderContextCrossSection
from pythonopenscad.modifier import (
    PoscBaseException,
    PoscRendererBase,
    get_fragments_from_fn_fa_fs
)


class ConversionException(PoscBaseException):
    """Exception for conversion errors."""


class TooManyParameters(PoscBaseException):
    """Exception when more unnamed parameters are provided than total
    parameters specified."""


class ParameterNotDefined(PoscBaseException):
    """Exception when passing a named parameter that has not been provided."""


class ParameterDefinedMoreThanOnce(PoscBaseException):
    """Exception when passing a named parameter that has already been provided."""


class RequiredParameterNotProvided(PoscBaseException):
    """Exception when a required parameter is not provided."""


class InitializerNotAllowed(PoscBaseException):
    """An initializer (def __init__) was defined and is not allowed."""


class InvalidIndentLevel(PoscBaseException):
    """Indentation level was set to an invalid number."""


class IndentLevelStackEmpty(PoscBaseException):
    """Indentation level was set to an invalid number."""


class InvalidValueForBool(PoscBaseException):
    """Conversion failure for bool value."""


class InvalidValueForStr(PoscBaseException):
    """Conversion failure for str value."""


class InvalidValue(PoscBaseException):
    """Invalid value provided.."""


class DuplicateNamingOfArgs(PoscBaseException):
    """OpenScadApiSpecifier args has names used more than once."""


class NameCollissionFieldNameReserved(PoscBaseException):
    """An attempt to define an arg with the same name as a field."""


class AttemptingToAddNonPoscBaseNode(PoscBaseException):
    """Attempted to add ad invalid object to child nodes."""


class Arg(object):
    """Defines an argument and field for PythonOpenScad PoscBase based APIs."""

    def __init__(
        self,
        name,
        typ,
        default_value,
        docstring,
        required=False,
        osc_name=None,
        attr_name=None,
        init=True,
        compare=True,
    ):
        """Args:
        name: The pythonopenscad name of this parameter.
        osc_name: The name used by OpenScad. Defaults to name.
        attr_name: Object attribute name.
        typ: The converter for the argument.
        default:_value: Default value for argument (this will be converted by typ).
        docstring: The python doc for the arg.
        required: Throws if the value is not provided.
        init: If True then the arg is added to the __init__ function.
        compare: If True then the arg is compared in the equals function.
        """
        self.name = name
        self.osc_name = osc_name or name
        self.attr_name = attr_name or name
        self.typ = typ
        self.default_value = default_value
        self.docstring = docstring
        self.required = required
        self.init = init
        self.compare = compare

    def to_dataclass_field(self):
        kwds = dict()
        if not self.required:
            kwds['default'] = self.default_value
        if not self.init:
            kwds['init'] = False
        if not self.compare:
            kwds['compare'] = False
        return field(**kwds)

    def annotation(self):
        return (self.name, self.typ)

    def default_value_str(self):
        """Returns the default value as a string otherwise '' if no default provided."""
        if self.default_value is None:
            return ''
        try:
            return repr(self.default_value)
        except:  # noqa: E722
            return ''

    def document(self):
        "Returns formatted documentation for this arg."
        default_str = self.default_value_str()
        default_str = (' Default ' + default_str) if default_str else default_str
        attribute_name = (
            '' if self.attr_name == self.name else ' (Attribute name: %s) ' % self.attr_name
        )
        if self.name == self.osc_name:
            return '%s%s: %s%s' % (self.name, attribute_name, self.docstring, default_str)
        else:
            return '%s%s (converts to %s): %s %s' % (
                self.name,
                attribute_name,
                self.osc_name,
                self.docstring,
                default_str,
            )


# Some special common Args. These are not consistently documented.
FA_ARG = Arg('_fa', float, None, 'minimum angle (in degrees) of each segment', osc_name='$fa')
FS_ARG = Arg('_fs', float, None, 'minimum length of each segment', osc_name='$fs')
FN_ARG = Arg('_fn', int, None, 'fixed number of segments. Overrides $fa and $fs', osc_name='$fn')


@dataclass(frozen=True)
class _ConverterWrapper:
    func: object

    def __repr__(self):
        return self.func.__name__

    def __str__(self):
        return self.func.__name__

    def __call__(self, v):
        return self.func(v)

    @property
    def __name__(self):
        return self.func.__name__


def _as_converter(arg=None):
    if isinstance(arg, str):

        def decorator(f):
            f.__name__ = arg
            return _ConverterWrapper(f)

        return decorator
    return _ConverterWrapper(arg)


def list_of(typ, len_min_max=(3, 3), fill_to_min=None):
    """Defines a converter for an iterable to a list of elements of a given type.
    Args:
        typ: The type of list elements.
        len_min_max: A tuple of the (min,max) length, (0, 0) indicates no limits.
        fill_to_min: If the provided list is too short then use this value.
    Returns:
        A function that performs the conversion.
    """
    description = 'list_of(%s, len_min_max=%r, fill_to_min=%r)' % (
        typ.__name__,
        len_min_max,
        fill_to_min,
    )

    @_as_converter(description)
    def list_converter(value):
        """Converts provided value as a list of the given type.
        value: The value to be converted
        """
        converted_value = []
        for v in value:
            if len_min_max[1] and len(converted_value) >= len_min_max[1]:
                raise ConversionException('provided length too large, max is %d' % len_min_max[1])
            converted_value.append(typ(v))
        if len_min_max[0] and len(value) < len_min_max[0]:
            if fill_to_min is None:
                raise ConversionException(
                    'provided length (%d) too small and fill_to_min is None, min is %d'
                    % (len(converted_value), len_min_max[0])
                )
            fill_converted = typ(fill_to_min)
            for _ in range(len_min_max[0] - len(converted_value)):
                converted_value.append(fill_converted)
        return converted_value

    return list_converter


def one_of(typ, *args):
    """Provides a converter that will iterate over the provided converters until it succeeds.
    Args:
      typ: The first converter argument.
      args: A list of supplemental type argument converters.
    """
    largs = [typ] + list(args)
    description = 'one_of(%s)' % ', '.join(t.__name__ for t in largs)

    @_as_converter(description)
    def one_of_converter(value):
        """Converts a value to one of the list provided to one_of().
        Throws:
          ConversionException if the value failed all conversions."""
        for atyp in largs:
            try:
                converted_value = atyp(value)
                return converted_value
            except:  # noqa: E722
                continue
        raise ConversionException("The value %r can't convert using %s" % (value, description))

    return one_of_converter


class OscKeyword(object):
    """Converts to the given string for allowing True to to true and False to false conversion.
    In the special case of the 'false' keyword, it converts to False on bool cast.
    """

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


@_as_converter
def bool_strict(value):
    """Returns an OscKeyword given bool i.e. 'true' if True else 'false'.
    Args:
        value: A boolean value.
    Throws:
        InvalidValueForBool if the provided value is not a bool.
    """
    if not isinstance(value, bool):
        raise InvalidValueForBool(
            'expected a bool value but got "%r" of type %s' % (value, value.__class__.__name__)
        )
    return OSC_TRUE if value else OSC_FALSE


@_as_converter
def str_strict(value):
    """Returns the given value if it is a str object otherwise raises
    InvalidValueForStr exception.
    Args:
        value: A string value.
    Throws:
        InvalidValueForStr if the provided value is not a str.
    """
    if not isinstance(value, str) and not isinstance(value, bytes):
        raise InvalidValueForStr(
            'expected a string value but got "%r" of type %s' % (value, value.__class__.__name__)
        )
    return value

@_as_converter
def int_strict(value):
    """Returns the given value if it is a Number object otherwise raises
    InvalidValueForStr exception.
    Args:
        value: A string value.
    Throws:
        ValueError if the provided value is not a str.
    """
    if not isinstance(value, Integral):
        raise ValueError(
            f'expected an integer value but got "{value!r}" of type {value.__class__.__name__}')
    return int(value)


def of_set(*args):
    """Returns a converter function that will throw if the the value to be converted is not
    one of args:
    Args:
        *args: The set of allowed values.
    Throws:
        InvalidValue if the value is not one of the args.
    """
    allowed_values = set(args)
    description = 'of_set(allowed_values=%r)' % (tuple(allowed_values),)

    @_as_converter(description)
    def of_set_converter(value):
        if value not in allowed_values:
            raise InvalidValue('%r is not allowed with %s.' % (value, description))
        return value

    return of_set_converter


# The base URL for OpenScad documentation,
OPEN_SCAD_BASE_URL = 'http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/'


class OpenScadApiSpecifier(object):
    """Contains the specification of an OpenScad primitive."""

    def __init__(self, openscad_name, args, url_base, alt_url_anchor=None):
        """
        Args:
            openscad_name: The OpenScad primitive name.
            args: A tuple of Arg()s for each value passed in.
            url_base: The base of the document URL for OpenScad documentation.
        """
        self.openscad_name = openscad_name
        self.args = args
        self.url_base = url_base
        self.alt_url_anchor = alt_url_anchor
        self.args_map = dict((arg.name, arg) for arg in args)

        if len(self.args) != len(self.args_map):
            all_names = [arg.name for arg in self.args]
            dupes = list(set([name for name in all_names if all_names.count(name) > 1]))
            raise DuplicateNamingOfArgs('Duplicate parameter names %r' % dupes)

    def generate_class_doc(self):
        """Generates class level documentation."""
        lines = ['\nConverts to an OpenScad "%s" primitive.' % self.openscad_name]
        if self.url_base:
            anchor = self.openscad_name if self.alt_url_anchor is None else self.alt_url_anchor
            url = OPEN_SCAD_BASE_URL + self.url_base + '#' + anchor
            lines.append(
                'See OpenScad `%s docs <%s>` for more information.' % (self.openscad_name, url)
            )

        return '\n'.join(lines)

    def generate_init_doc(self):
        if self.args:
            return 'Args:\n    ' + ('\n    '.join(arg.document() for arg in self.args))
        return 'No arguments allowed.'


class StringWriter(object):
    """A CodeDumper writer that writes to a string. This can API can be implemented for
    file writers or other uses."""

    def __init__(self):
        self._builder = []

    def get(self):
        """Returns the contents. This is used by PythonOpenScad code in only the str
        and repr conversions which specifically create a StringWriter writer. Append
        is the only mentod called by the PythonOpenScad renderer."""
        return '\n'.join(self._builder + [''])

    def append(self, line):
        """Called by the PythonOpenScad renderer/code_dump to write generated model
        representation. Override this function to implement other output mechanisms."""
        self._builder.append(line)


class FileWriter(object):
    """A CodeDumper writer that writes to a file."""

    def __init__(self, fp):
        self.fp = fp

    def finish(self):
        """Writes the final components to the output"""
        self.fp.write('\n')

    def append(self, line):
        """Called by the PythonOpenScad renderer/code_dump to write generated model
        representation. Override this function to implement other output mechanisms."""
        self.fp.write(line)
        self.fp.write('\n')


class CodeDumper(object):
    """Helper for pretty printing OpenScad scripts (and other scripts too)."""

    class IndentLevelState:
        """Indent level state."""

        def __init__(self, level, is_last):
            self.level = level
            self.is_last = is_last

    DUMPS_OPENSCAD = True

    def get(self):
        return self.level, self.is_last

    def __init__(
        self,
        indent_char=' ',
        indent_multiple=2,
        writer=None,
        str_quotes='"',
        block_ends=(' {', '}', ';', '//', 'module', '', ''),
        target_max_column=100,
    ):
        """
        Args:
           indent_char: the character used to indent.
           indent_multiple: the number of indent_char added per indent level.
           writer: A writer, like StringWriter.
           target_max_column: The max column number where line continuation may be used.
           variables: A list of variables to be defined at the top of the script.
        """
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
        self.modules_dict = dict()
        self.modules_num = defaultdict(int)

    def check_indent_level(self, level):
        """Check the adding of the resulting indent level will be in range.
        Args:
           level: The new requested indent level.
        Throws:
           InvalidIndentLevel level would is out of range
        """
        if level < 0:
            raise InvalidIndentLevel('Requested indent level below zero is not allowed.')

    def push_increase_indent(self, amount=1):
        """Push an indent level change and increase indent level.
        Args:
           amount: the amount to increase the indent level, Amount can be negative. default 1
        """
        current_level_state = CodeDumper.IndentLevelState(self.current_indent_level, self.is_last)
        try:
            self.set_indent_level(current_level_state.level + amount)
        finally:
            self.indent_level_stack.append(current_level_state)

    def set_indent_level(self, level):
        self.check_indent_level(level)
        self.current_indent_level = level
        self.current_indent_string = (
            self.indent_char * self.indent_multiple * self.current_indent_level
        )

    def pop_indent_level(self):
        """Pops the indent level stack and sets the indent level to the popped value."""
        if len(self.indent_level_stack) == 0:
            raise IndentLevelStackEmpty('Empty indent level stack cannot be popped.')
        level_state = self.indent_level_stack.pop()
        self.set_indent_level(level_state.level)
        self.is_last = level_state.is_last

    def set_is_last(self, is_last):
        """Set this to False if there is another item to be rendered after this one."""
        self.is_last = is_last

    def get_is_last(self):
        """Clients that care if a suffix needs adding if it is on the end of the list
        can check this."""
        return self.is_last

    def add_line(self, line):
        """Adds the given line as a whole line the output.

        Args:
            line: string to be added.
        """
        self.writer.append(line)

    def write_line(self, line):
        """Adds an indented line to the output. This could be used for comments."""
        self.add_line(self.current_indent_string + line)

    def write_function(
        self, function_name, params_list, mod_prefix='', mod_suffix='', suffix=';', comment=None
    ):
        """Dumps a function like lines (may wrap).

        Args:
            function_name: name of function.
            prefix: a string added in front of the function name
            params_list: list of parameters (no commas separating them)
            suffix: A string at the end
        """
        if comment:
            self.add_line(''.join([self.current_indent_string, comment]))
        strings = [self.current_indent_string, mod_prefix, function_name, '(']
        strings.append(', '.join(params_list))
        strings.append(')')
        strings.append(mod_suffix)
        strings.append(suffix)
        self.add_line(''.join(strings))

    def render_value(self, value):
        """Returns a string representing the given value."""
        if isinstance(value, str):
            return self.str_quotes + repr(value)[1:-1] + self.str_quotes
        return repr(value)

    def render_name_value(self, arg, value):
        return '%s=%s' % (arg.osc_name, self.render_value(value))

    def should_add_suffix(self):
        """Returns true if the suffix should be added. OpenScad is always true."""
        return True

    def get_modifiers_prefix_suffix(self, obj):
        """Returns the OpenScad modifiers string."""
        return (obj.get_modifiers(), '')

    def add_modules(self, modules: List['Module']):
        """Adds the modules to the output. If a module name is already in use then
        a new name is generated."""

        for module in modules:
            while module.get_name() in self.modules_dict:
                other_module = self.modules_dict[module.name]
                if other_module != module:
                    # Two different modules with the same name. Change the name of the new one.
                    newnum = 1 + self.modules_num[module.name]
                    self.modules_num[module.name] = newnum
                    # In theoru this name could a manuallu created name, so we need to
                    # continue to increment until we find a unique name.
                    module.gen_name = f'{module.name}_{newnum}'
                else:
                    break

            self.modules_dict[module.get_name()] = module

    def reset_modules(self, modules):
        """Resets the module names."""
        for module in modules:
            module.gen_name = None

    def dump_modules(self):
        """Writes the modules to the output."""
        start_comment = self.block_ends[3]
        if self.modules_dict:
            self.add_line('')
            self.add_line(f'{start_comment} Modules.')
        module_names = [k for k in self.modules_dict.keys()]
        module_names.sort()
        for module_name in module_names:
            self.render_modules(self.modules_dict[module_name])

    def render_modules(self, module):
        """Returns a string representing the given variable."""
        start_func = self.block_ends[0]
        end_func = self.block_ends[1]
        start_comment = self.block_ends[3]
        name = module.get_name()
        metadataName = module.getMetadataName()
        self.add_line('')
        if metadataName:
            comment = start_comment + ' ' + repr(metadataName)
            self.add_line(comment)

        end_func_decl = self.block_ends[6]
        return_func = self.block_ends[5]

        define_module = self.block_ends[4]
        self.add_line(f'{define_module} {name}(){end_func_decl}{return_func}{start_func}')
        self.push_increase_indent()
        module.code_dump_contained(self)
        self.pop_indent_level()
        self.add_line(f'{end_func} {start_comment} end module {name}')


class CodeDumperForPython(CodeDumper):
    """Helper for pretty printing to Python code compatible with SolidPython
    and PythonOpenScad.
    Args:
        Same parameters as CodeDumper but overrides defaults for str_quotes and
        block_ends.
    """

    DUMPS_OPENSCAD = False

    def __init__(self, *args, **kwds):
        kwds.setdefault('str_quotes', "'")
        kwds.setdefault('block_ends', (' (', '),', ',', '#', 'def', ' return', ':'))
        super().__init__(*args, **kwds)
        self.is_last = True

    def render_value(self, value):
        """Returns a string representing the given value."""
        if value == OSC_TRUE:
            return 'True'
        elif value == OSC_FALSE:
            return 'False'
        return repr(value)

    def render_name_value(self, arg, value):
        return '%s=%s' % (arg.name, self.render_value(value))

    def should_add_suffix(self):
        """Returns true if the suffix should be added. Python is true of if not at the
        end.."""
        return not self.is_last

    def get_modifiers_prefix_suffix(self, obj):
        """Returns a Python mosifiers mutator."""
        s = obj.get_modifiers_repr()
        if not s:
            return ('', '')
        return ('', '.add_modifier(*%s)' % s)


class PoscBase(PoscRendererBase):
    DUMP_CONTAINER = True
    DUMP_MODULE = False

    def __post_init__(self):
        for arg in self.OSC_API_SPEC.args:
            value = getattr(self, arg.name)
            is_different_name = arg.name != arg.attr_name
            if is_different_name:
                delattr(self, arg.name)
            if value is not None:
                setattr(self, arg.attr_name, arg.typ(value))
            elif is_different_name:
                setattr(self, arg.attr_name, None)

        self.init_children()
        # Object should be fully constructed now.
        self.check_valid()

    def init_children(self):
        """Initalizes objects that contain parents."""
        # This node has no children.

    def check_valid(self):
        """Checks that the construction of the object is valid."""
        self.check_required_parameters()

    def check_required_parameters(self):
        """Checks that required parameters are set and not None."""
        for arg in self.OSC_API_SPEC.args:
            if arg.required and (getattr(self, arg.attr_name, None) is None):
                raise RequiredParameterNotProvided(
                    '"%s" is required and not provided' % arg.attr_name
                )

    def collect_args(self, code_dumper):
        """Returns a list of arg=value pairs as strings."""
        posc_args = self.OSC_API_SPEC.args
        result = []
        for arg in posc_args:
            v = getattr(self, arg.attr_name, None)
            if v is not None:
                result.append(code_dumper.render_name_value(arg, v))
        return result

    def has_children(self):
        return False

    def children(self):
        """This is a childless node, always returns empty tuple."""
        return ()

    def code_dump_scad(self, code_dumper: CodeDumper):
        """Dump the OpenScad equivalent of this script into the provided dumper."""
        termial_suffix = code_dumper.block_ends[2] if code_dumper.should_add_suffix() else ''
        suffix = code_dumper.block_ends[0] if self.has_children() else termial_suffix
        function_name = self.OSC_API_SPEC.openscad_name
        params_list = self.collect_args(code_dumper)
        mod_prefix, mod_suffix = code_dumper.get_modifiers_prefix_suffix(self)
        comment = None
        metadataName = self.getMetadataName()
        if metadataName:
            comment = code_dumper.block_ends[3] + ' ' + repr(metadataName)

        code_dumper.write_function(
            function_name, params_list, mod_prefix, mod_suffix, suffix, comment
        )
        if self.has_children():
            code_dumper.push_increase_indent()
            left = len(self.children())
            for child in self.children():
                left -= 1
                code_dumper.set_is_last(left == 0)
                child.code_dump(code_dumper)
            code_dumper.pop_indent_level()
            code_dumper.write_line(code_dumper.block_ends[1])

    def code_dump_contained(self, code_dumper: CodeDumper):
        code_dumper.write_line(
            code_dumper.block_ends[3] + ' Start: ' + self.OSC_API_SPEC.openscad_name
        )
        for child in self.children():
            child.code_dump(code_dumper)
        code_dumper.write_line(
            code_dumper.block_ends[3] + ' End: ' + self.OSC_API_SPEC.openscad_name
        )

    def code_dump(self, code_dumper: CodeDumper):
        if self.DUMP_CONTAINER or not code_dumper.DUMPS_OPENSCAD:
            self.code_dump_scad(code_dumper)
        else:
            # Must be a LazyUnion or Module dumping to OpenScad. Dump the children directly
            # to invoke the "lazy" union behavior.
            self.code_dump_contained(code_dumper)

    def get_modules(self):
        return ()

    def dump_with_code_dumper(self, code_dumper: CodeDumper):
        """Returns the OpenScad equivalent code for this node."""
        code_dumper.add_modules(self.get_modules())
        self.code_dump(code_dumper)
        code_dumper.dump_modules()
        # Reset the module names sp that these can be reused.
        code_dumper.reset_modules(self.get_modules())
        return code_dumper

    def __str__(self):
        """Returns the OpenScad equivalent code for this node."""
        return self.dump_with_code_dumper(CodeDumper()).writer.get()

    def __repr__(self):
        """Returns the SolidPython equivalent code for this node."""
        return self.dump_with_code_dumper(CodeDumperForPython()).writer.get()

    def clone(self):
        return copy.deepcopy(self)

    def equals(self, other):
        if not hasattr(other, 'OSC_API_SPEC'):
            return False
        # Since we create other classes with the same OpenScadApiSpecifier, this
        # is the indicator of the object's type.
        if self.OSC_API_SPEC is not other.OSC_API_SPEC:
            return False
        for arg in self.OSC_API_SPEC.args:
            if getattr(self, arg.attr_name) != getattr(other, arg.attr_name):
                return False
        return self.children() == other.children()

    # OpenPyScad compat functions.
    def dumps(self):
        """Returns a string of this object's OpenScad script."""
        return self.dump_with_code_dumper(CodeDumper()).writer.get()

    def dump(self, fp):
        """Writes this object's OpenScad script to the given file.
        Args:
            fp: The python file object to use.
        """
        self.dump_with_code_dumper(CodeDumper(writer=FileWriter(fp))).writer.finish()

    def write(self, filename, encoding='utf-8'):
        """Writes the OpenScad script to the given file name.
        Args:
            filename: The filename to create.
        """
        with open(filename, 'w', encoding=encoding) as fp:
            self.dump(fp)
            
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        assert False, "Not implemented"

    def module(self, name):
        """Returns a variable that references this object."""
        return Module(name)(self)

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
    
    def render(self, *args, **kwds):
        return Render(*args, **kwds)(self)

    def rotate_extrude(self, *args, **kwds):
        return Rotate_Extrude(*args, **kwds)(self)

    def fill(self, *args, **kwds):
        return Fill(*args, **kwds)(self)

    def __eq__(self, other):
        """Exact object tree equality. (Not resulting shape equality)"""
        return self.equals(other)

    def __ne__(self, other):
        """Exact object tree inequality. (Not resulting shape equality)"""
        return not self.equals(other)

    def __add__(self, other):
        """Union of this with other 3D object. See Union."""
        return Union()(self, other)

    def __sub__(self, other):
        """Difference of this with other 3D object. See Difference."""
        return Difference()(self, other)

    # OpenPyScad compatability
    def __and__(self, other):
        """Intersect this with other 3D object. See Intersection."""
        return Intersection()(self, other)

    # SolidPython compatability
    def __mul__(self, other):
        """Intersect this with other 3D object. See Intersection."""
        return Intersection()(self, other)


# A decorator for PoscBase classes.
def apply_posc_attributes(clazz):
    """Decorator that applies an equivalent constructor with it\'s own generated
    docstring. Also adds some SolidPython script compatibility class by providing an alias
    class in the current module."""
    if clazz.__init__ != PoscBase.__init__:
        raise InitializerNotAllowed('class %s should not define __init__' % clazz.__name__)
    # Check for name collision.
    args: Tuple[Arg] = clazz.OSC_API_SPEC.args
    for arg in args:
        if hasattr(clazz, arg.attr_name):
            raise NameCollissionFieldNameReserved(
                "There exists an attribute '%s' for class %s that collides with an arg."
                % (arg.name, clazz.__name__)
            )
    annotations = dict((arg.annotation() for arg in clazz.OSC_API_SPEC.args))
    clazz.__annotations__ = annotations
    for arg in args:
        setattr(clazz, arg.name, arg.to_dataclass_field())
    dataclass(repr=False)(clazz)
    clazz.__init__.__doc__ = clazz.OSC_API_SPEC.generate_init_doc()
    strs = []
    if clazz.__doc__:
        strs.append(clazz.__doc__)
    strs.append(clazz.OSC_API_SPEC.generate_class_doc())
    clazz.__doc__ = '\n'.join(strs)
    # Add an alias of this class for the OpenScad name. Be compatible with SolidPython.
    other_class = type(clazz.OSC_API_SPEC.openscad_name, (clazz,), {})
    other_class.__doc__ = clazz.__doc__  # For some reason __doc__ needs setting.
    setattr(sys.modules[__name__], clazz.OSC_API_SPEC.openscad_name, other_class)

    return clazz


class PoscParentBase(PoscBase):
    """A PoscBase class that has children. All OpenScad forms that have children use this.
    This provides basic child handling functions."""

    def init_children(self):
        """Initalizes objects for parents."""
        self._children = []
        self._modules = []

    def can_have_children(self):
        """Returns true. This node can have children."""
        return True

    def has_children(self):
        """Returns true if the node has children."""
        return bool(self._children)

    def children(self) -> list[PoscBase]:
        """Returns the list of children"""
        return self._children

    def get_modules(self):
        """Returns the list of modules."""
        return self._modules

    def append(self, *children):
        """Appends the children to this node.
        Args:
          *children: children to append.
        """
        return self.extend(children)

    def extend(self, children):
        """Appends the list of children to this node.
        Args:
          children: list of children to append.
        """
        # Don't add nodes that are not PoscBase nodes and collect modules.
        for child in children:
            if child is None or not hasattr(child, 'OSC_API_SPEC'):
                raise AttemptingToAddNonPoscBaseNode(
                    'Cannot append object %r as child node' % child
                )
            if child.DUMP_MODULE:
                self._modules.append(child)
            self._modules.extend(child.get_modules())
        self._children.extend(children)
        return self

    # Support Obj(child, ...) constructs like that in OpenScad and SolidPuthon.
    __call__ = append


# A decorator for transformation classes.
def apply_posc_transformation_attributes(clazz):
    """Does everything that apply_posc_attributes() does but also adds documentation
    to the corresponding PoscBase function of the same name.
    """
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
    """Translate child nodes."""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'translate',
        (Arg('v', VECTOR3_FLOAT, None, '(x,y,z) translation vector.', required=True),),
        OPEN_SCAD_URL_TAIL_TRANSFORMS,
    )
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.translate(self, self.v)


@apply_posc_transformation_attributes
class Rotate(PoscParentBase):
    """Rotate child nodes. This comes in two forms, the first form
    is a single angle and optional axis of rotation, the second form
    is a vector of angles for three consecutive rotations about the
    z, y and z axis."""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'rotate',
        (
            Arg(
                'a',
                one_of(float, list_of(float, fill_to_min=0.0)),
                0,
                'Angle to rotate or vector of angles applied to each axis ' + 'in sequence.',
            ),
            Arg('v', VECTOR3_FLOAT, None, '(x,y,z) axis of rotation vector.'),
        ),
        OPEN_SCAD_URL_TAIL_TRANSFORMS,
    )
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.rotate(self, self.a, self.v)


@apply_posc_attributes
class Cylinder(PoscBase):
    """Creates a cylinder or truncated cone about the z axis. Cone needs r1 and r2 or d1 and d2
    provided with different lengths.
    d1 & d2 have precedence over d which have precedence over r1 and r2 have precedence over r.
    Hence setting r is overridden by any other value.
    """

    OSC_API_SPEC = OpenScadApiSpecifier(
        'cylinder',
        (
            Arg('h', float, 1.0, 'height of the cylinder or cone.', required=True),
            Arg('r', float, 1.0, 'radius of cylinder. r1 = r2 = r.'),
            Arg('r1', float, None, 'radius, bottom of cone.'),
            Arg('r2', float, None, 'radius, top of cone.'),
            Arg('d', float, None, 'diameter of cylinder. r1 = r2 = d / 2.'),
            Arg('d1', float, None, 'diameter of bottom of cone. r1 = d1 / 2.'),
            Arg('d2', float, None, 'diameter of top of cone. r2 = d2 / 2.'),
            Arg(
                'center',
                bool_strict,
                False,
                'z ranges from 0 to h, true z ranges from -h/2 to +h/2.',
            ),
            FA_ARG,
            FS_ARG,
            FN_ARG,
        ),
        OPEN_SCAD_URL_TAIL_PRIMITIVES,
    )

    def get_r1(self):
        """Returns the bottom radius of the cylinder or cone."""
        if self.d1 is not None:
            return self.d1 / 2

        if self.r1 is not None:
            return self.r1

        if self.d is not None:
            return self.d / 2

        return self.r

    def get_r2(self):
        """Returns the top radius of the cylinder or cone."""
        if self.d2 is not None:
            return self.d2 / 2

        if self.r2 is not None:
            return self.r2

        if self.d is not None:
            return self.d / 2

        return self.r

    def check_valid(self):
        """Checks that the values of cylinder satisfy OpenScad cylinder requirements."""
        values = (('r1', self.get_r1()), ('r2', self.get_r2()))
        for k, v in values:
            if v is None:
                raise RequiredParameterNotProvided('"%s" is required and not provided' % k)
        self.check_required_parameters()
        
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        centre = True if self.center else False
        return renderer.cylinder(
            self, self.h, self.get_r1(), self.get_r2(), 
            get_fragments_from_fn_fa_fs(self.get_r1(), self._fn, self._fa, self._fs),
            centre)


@apply_posc_attributes
class Sphere(PoscBase):
    """Creates a sphere.
    It defaults to a sphere of radius 1. If d is provided it overrides the value of r.
    """

    OSC_API_SPEC = OpenScadApiSpecifier(
        'sphere',
        (
            Arg('r', float, 1.0, 'radius of sphere. Ignores d if set.'),
            Arg('d', float, None, 'diameter of sphere.'),
            FA_ARG,
            FS_ARG,
            FN_ARG,
        ),
        OPEN_SCAD_URL_TAIL_PRIMITIVES,
    )

    def get_r(self):
        """Returns the top radius of the cylinder or cone."""
        if self.d is not None:
            return self.d / 2

        return self.r

    def check_valid(self):
        """Checks that the construction of cylinder is valid."""
        if all(x is None for x in [self.r, self.d]):
            raise RequiredParameterNotProvided(
                'Both parameters r and d are None. A value for r or d must be provided.'
            )
        self.check_required_parameters()
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.sphere(self, self.r, get_fragments_from_fn_fa_fs(self.r, self._fn, self._fa, self._fs))


@apply_posc_attributes
class Cube(PoscBase):
    """Creates a cube with it's bottom corner centered at the origin."""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'cube',
        (
            Arg(
                'size',
                one_of(float, VECTOR3_FLOAT_DEFAULT_1),
                (1, 1, 1),
                'The x, y and z sizes of the cube or rectangular prism',
                required=True,
            ),
            Arg(
                'center', bool_strict, None, 'If true places the center of the cube at the origin.'
            ),
        ),
        OPEN_SCAD_URL_TAIL_PRIMITIVES,
    )
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.cube(self, self.size, True if self.center else False)


@apply_posc_transformation_attributes
class Scale(PoscParentBase):
    """Scales the child nodes. scale"""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'scale',
        (
            Arg(
                'v', one_of(float, VECTOR3_FLOAT_DEFAULT_1), (1, 1, 1), 'The (x,y,z) scale factors.'
            ),
        ),
        OPEN_SCAD_URL_TAIL_TRANSFORMS,
    )
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.scale(self, self.v)


@apply_posc_transformation_attributes
class Resize(PoscParentBase):
    """Scales the object so the newsize (x,y,z) parameters given. A zero (0.0) scale is ignored
    and that dimension's scale factor is 1."""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'resize',
        (
            Arg(
                'newsize',
                list_of(float, len_min_max=(0, 3)),
                None,
                'The new (x,y,z) sizes of the resulting object.',
            ),
            Arg(
                'auto',
                one_of(bool_strict, VECTOR3_BOOL),
                None,
                'A vector of (x,y,z) booleans to indicate which axes will be resized.',
            ),
        ),
        OPEN_SCAD_URL_TAIL_TRANSFORMS,
    )
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.resize(self, self.newsize, self.auto)


@apply_posc_transformation_attributes
class Mirror(PoscParentBase):
    """Mirrors across a plane defined by the normal v."""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'mirror',
        (Arg('v', VECTOR3_FLOAT, None, 'The normal of the plane to be mirrored.'),),
        OPEN_SCAD_URL_TAIL_TRANSFORMS,
    )
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.mirror(self, self.v)


@apply_posc_transformation_attributes
class Multmatrix(PoscParentBase):
    """Homogeneous matrix multiply. The provided matrix can both rotate and translate."""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'multmatrix',
        (
            Arg(
                'm',
                list_of(VECTOR4_FLOAT, len_min_max=(3, 4)),
                None,
                """A 4x4 or 4x3 matrix. The last row must always be [0,0,0,1] and in the
            case of a 4x3 matrix that row is added. The resulting matrix is always 4x4.""",
            ),
        ),
        OPEN_SCAD_URL_TAIL_TRANSFORMS,
    )

    def check_valid(self):
        """Checks that the construction of cylinder is valid."""
        if len(self.m) == 3:
            self.m.append([0.0, 0.0, 0.0, 1.0])
        self.check_required_parameters()

    def get_m(self):
        """Returns the matrix m. The returned value is always a 4x4 matrix."""
        return self.m
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.multmatrix(self, self.m)


@apply_posc_transformation_attributes
class Color(PoscParentBase):
    """Apply a color (only supported in OpenScad preview mode). Colors can be a 3 vector
    of values [0.0-1.0] for RGB or additionally a 4 vector if alpha is included for an
    RGBA color. Colors can be specified as #RRGGBB and it's variants."""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'color',
        (
            Arg(
                'c',
                one_of(str_strict, VECTOR3OR4_FLOAT),
                None,
                'A 3 or 4 color RGB or RGBA vector or a string descriptor of the color.',
            ),
            Arg('alpha', float, None, 'The alpha of the color if not already provided by c.'),
        ),
        OPEN_SCAD_URL_TAIL_TRANSFORMS,
    )
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.color(self, self.c, self.alpha)


@apply_posc_transformation_attributes
class Offset(PoscParentBase):
    """Generates a new polygon with the curve offset by the given amount. Negative values
    can be used to shrink paths while positive values enlarge the path."""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'offset',
        (
            Arg('r', float, None, 'The radius of the new path when using the radial method.'),
            Arg('delta', float, None, 'The offset of the new path when using the offset method.'),
            Arg('chamfer', bool_strict, False, 'If true will create chamfers at corners.'),
            FA_ARG,
            FS_ARG,
            FN_ARG,
        ),
        OPEN_SCAD_URL_TAIL_TRANSFORMS,
    )

    def check_valid(self):
        """Checks that the construction of cylinder is valid."""
        if all(x is None for x in [self.r, self.delta]):
            self.r = 1.0
        self.check_required_parameters()
        
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.offset(self, self.r, self.delta, self.chamfer, self._fn, self._fa, self._fs)


@apply_posc_transformation_attributes
class Projection(PoscParentBase):
    """Project a 3D object into a 2D surface."""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'projection',
        (
            Arg(
                'cut',
                bool_strict,
                None,
                'If false, the projection is a "shadow" of the object otherwise it is an intersection.',
            ),
        ),
        OPEN_SCAD_URL_TAIL_2D,
        '3D_to_2D_Projection',
    )
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.projection(self, self.cut)
    
@apply_posc_transformation_attributes
class Render(PoscParentBase):
    """Forces the generation of a mesh even in preview mode."""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'render',
        (
            Arg(
                'convexity',
                int_strict,
                10,
                'A convexity value used for optimization of rendering.',
            ),
        ),
        OPEN_SCAD_URL_TAIL_OTHER,
        'render',
    )
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.render(self, self.convexity)


@apply_posc_transformation_attributes
class Minkowski(PoscParentBase):
    """Create a Minkowski transformed object."""

    OSC_API_SPEC = OpenScadApiSpecifier('minkowski', (), OPEN_SCAD_URL_TAIL_TRANSFORMS)
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.minkowski(self)


@apply_posc_transformation_attributes
class Hull(PoscParentBase):
    """Create a hull of two solids."""

    OSC_API_SPEC = OpenScadApiSpecifier('hull', (), OPEN_SCAD_URL_TAIL_TRANSFORMS)
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.hull(self)


@apply_posc_transformation_attributes
class Linear_Extrude(PoscParentBase):
    """Creates an 3D object with a linear extrusion of a 2D shape."""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'linear_extrude',
        (
            Arg('height', float, 100, 'The height of the resulting extrusion.'),
            Arg(
                'center',
                bool_strict,
                None,
                "If true, the final object's height center point is placed at z=0.",
            ),
            Arg(
                'convexity', int, None, 'A convexity value used for preview mode to aid rendering.'
            ),
            Arg(
                'twist',
                float,
                None,
                'If provided the object is rotated about the z axis by this total angle',
            ),
            Arg(
                'slices',
                int,
                None,
                'The number of slices to be applied in the resulting extrusion.',
            ),
            Arg(
                'scale',
                one_of(float, VECTOR2_FLOAT),
                None,
                'A scale factor to applied to the children incrementally per extrusion layer.',
                attr_name='scale_',
            ),
            FN_ARG,
        ),
        OPEN_SCAD_URL_TAIL_2D,
        'Linear_Extrude',
    )
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.linear_extrude(self, self.height, self.center, self.convexity, 
                                       self.twist, self.slices, self.scale_, self._fn)


@apply_posc_transformation_attributes
class Rotate_Extrude(PoscParentBase):
    """Creates an 3D object with a rotating extrusion of a 2D shape."""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'rotate_extrude',
        (
            Arg('angle', float, 360, 'The total angle to extrude.'),
            Arg(
                'convexity', int, None, 'A convexity value used for preview mode to aid rendering.'
            ),
            FA_ARG,
            FS_ARG,
            FN_ARG,
        ),
        OPEN_SCAD_URL_TAIL_2D,
        'Rotate_Extrude',
    )
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:

        return renderer.rotate_extrude(self, 
                                       self.angle, 
                                       self.convexity, 
                                       self._fn, 
                                       self._fa, 
                                       self._fs)


@apply_posc_attributes
class Circle(PoscBase):
    """Creates a 2D circle shape.
    Note that if d is provided it has precedence over r if provided.
    """

    OSC_API_SPEC = OpenScadApiSpecifier(
        'circle',
        (
            Arg('r', float, 1, 'The radius of the generated circle.'),
            Arg('d', float, None, 'The diameter of the circle, overrides r.'),
            FA_ARG,
            FS_ARG,
            FN_ARG,
        ),
        OPEN_SCAD_URL_TAIL_2D,
    )

    def get_r(self):
        """Returns the top radius of the circle."""
        if self.d is not None:
            return self.d / 2
        return self.r

    def check_valid(self):
        """Checks that the parameters of circle satisfy OpenScad circle requirements."""
        if self.d is not None:
            self.r = None
        if self.get_r() is None:
            raise RequiredParameterNotProvided('r or d is required and not provided')
        self.check_required_parameters()
        
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        if self.d is None:
            r = self.r
        else:
            r = self.d / 2  
        fn = get_fragments_from_fn_fa_fs(r, self._fn, self._fa, self._fs)
        return renderer.circle(self, r, fn)


@apply_posc_attributes
class Square(PoscBase):
    """Creates a 2D square shape"""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'square',
        (
            Arg(
                'size',
                one_of(float, VECTOR2_FLOAT_DEFAULT_1),
                1,
                'The square size, if a 2 vector (x,y) is provided a rectangle is generated.',
            ),
            Arg(
                'center',
                bool_strict,
                None,
                'If true the resulting shape is centered otherwise a corner is at the origin.',
            ),
        ),
        OPEN_SCAD_URL_TAIL_2D,
    )

    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.square(self, self.size, self.center)


@apply_posc_attributes
class Polygon(PoscBase):
    """Creates a polygon 2D shape (with optional holes).
    If paths is not provided, one is constructed by generating a sequence 0,,N-1 where N
    is the number of points provided.
    """

    OSC_API_SPEC = OpenScadApiSpecifier(
        'polygon',
        (
            Arg(
                'points',
                list_of(VECTOR2_FLOAT, len_min_max=(None, None)),
                None,
                'A collection of (x,y) points to be indexed in paths.',
            ),
            Arg(
                'paths',
                list_of(list_of(int, len_min_max=(None, None)), len_min_max=(None, None)),
                None,
                'A list of paths which are a list of indexes into the points collection.',
            ),
            Arg(
                'convexity', int, None, 'A convexity value used for preview mode to aid rendering.'
            ),
        ),
        OPEN_SCAD_URL_TAIL_2D,
        'Polygons',
    )
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.polygon(self, self.points, self.paths, self.convexity)


@apply_posc_attributes
class Text(PoscBase):
    """Creates a 2D shape from a text string with a given font. A 2D shape consisting of
    an outline for each glyph in the string."""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'text',
        (
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
            FN_ARG,
        ),
        'Text',
        '',
    )
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        direction = self.direction if self.direction else "ltr"
        size = self.size if self.size else 1.0
        font = self.font if self.font else "Liberation Sans"
        halign = self.halign if self.halign else "left"
        valign = self.valign if self.valign else "baseline"
        spacing = self.spacing if self.spacing else 1.0
        language = self.language if self.language else "en"
        script = self.script if self.script else "latin"
        
        return renderer.text(
            self,
            self.text, 
            size, 
            font, 
            halign, 
            valign, 
            spacing, 
            direction,
            language, 
            script, 
            self._fa, 
            self._fs, 
            self._fn)


@apply_posc_attributes
class Polyhedron(PoscBase):
    """Creates an arbitrary polyhedron 3D object.
    Note: triangles is deprecated."""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'polyhedron',
        (
            Arg(
                'points',
                list_of(VECTOR3_FLOAT, len_min_max=(None, None)),
                None,
                'A list of 3D points. The index to these points are used in faces or triangles.',
            ),
            Arg(
                'triangles',
                list_of(list_of(int, len_min_max=(3, 3)), len_min_max=(None, None)),
                None,
                'A list of triangles. Each triangle is 3 indexes into the points list.',
            ),
            Arg(
                'faces',
                list_of(list_of(int, len_min_max=(3, None)), len_min_max=(None, None)),
                None,
                'A list of faces. Each face is a minimum of 3 indexes into the points list',
            ),
            Arg('convexity', int, 10, 'A convexity value used for preview mode to aid rendering.'),
        ),
        OPEN_SCAD_URL_TAIL_PRIMITIVES,
    )

    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.polyhedron(
            self, verts=self.points, faces=self.faces, triangles=self.triangles)

@apply_posc_attributes
class Union(PoscParentBase):
    """Unifies a set of 3D objects into a single object by performing a union of all the space
    contained by all the shapes."""

    OSC_API_SPEC = OpenScadApiSpecifier('union', (), OPEN_SCAD_URL_TAIL_CSG)

    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.union(self)

@apply_posc_attributes
class LazyUnion(PoscParentBase):
    """An implicit union for the top level node. This allows the top level nodes to be rendered
    separeately if the model is exported as a 3mf file."""

    DUMP_CONTAINER = False  # When rendering to OpenScad, don't render the container.
    OSC_API_SPEC = OpenScadApiSpecifier('lazy_union', (), OPEN_SCAD_URL_TAIL_CSG)
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.union(self)


@apply_posc_attributes
class Module(PoscParentBase):
    """This will be rendered as a module in the OpenScad script. It will be placed at the
    end of the script. This is useful for defining shapes that are used multiple times."""

    DUMP_CONTAINER = False
    DUMP_MODULE = True  # Indicate this is a variable.
    OSC_API_SPEC = OpenScadApiSpecifier(
        'module',
        (
            Arg(
                'name',
                str_strict,
                None,
                required=True,
                docstring='The filename to import. Relative path names are relative to the script location.',
            ),
            Arg(
                'gen_name',
                str_strict,
                None,
                init=False,
                compare=False,
                docstring='The generated name to avoid name collision.',
            ),
        ),
        OPEN_SCAD_URL_TAIL_CSG,
    )

    def get_name(self):
        """Returns the name of the module."""
        return self.gen_name if self.gen_name else self.name

    def code_dump(self, code_dumper: CodeDumper):
        code_dumper.write_line(f'{self.get_name()}();')

    def code_dump_contained(self, code_dumper: CodeDumper):
        for child in self.children():
            child.code_dump(code_dumper)

    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.union(self)

@apply_posc_attributes
class Difference(PoscParentBase):
    """Creates a 3D object by removing the space of the 3D objects following the first
    object provided from the first object."""

    OSC_API_SPEC = OpenScadApiSpecifier('difference', (), OPEN_SCAD_URL_TAIL_CSG)

    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.difference(self)
    
@apply_posc_attributes
class Intersection(PoscParentBase):
    """Creates a 3D object by finding the common space contained in all the provided
    3D objects."""

    OSC_API_SPEC = OpenScadApiSpecifier('intersection', (), OPEN_SCAD_URL_TAIL_CSG)
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.intersection(self)


@apply_posc_attributes
class Import(PoscBase):
    """Import a file as 3D or 2D shapes.
    SVG and DXF files generate 2D shapes.
    STL, OFF, AMF and 3MF files generate 3D shapes."""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'import',
        (
            Arg(
                'file',
                str_strict,
                None,
                'The filename to import. Relative path names are relative to the script location.',
            ),
            Arg(
                'convexity', int, None, 'A convexity value used for preview mode to aid rendering.'
            ),
            Arg(
                'layer',
                str,
                None,
                'When importing a DXF file, this will select the layer to be imported.',
            ),
        ),
        OPEN_SCAD_URL_TAIL_IMPORTING,
    )
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.import_file(self, self.file, self.layer, self.convexity)

@apply_posc_attributes
class Surface(PoscBase):
    """Import a file as a height map. This can be a image file or a text file."""

    OSC_API_SPEC = OpenScadApiSpecifier(
        'surface',
        (
            Arg('file', str_strict, None, 'File name used to load the height map.'),
            Arg(
                'center',
                bool_strict,
                None,
                'If true the resulting shape is centered otherwise a corner is at the origin.',
            ),
            Arg(
                'invert',
                bool_strict,
                None,
                'If the file is an image, a value of true will invert the height data.',
            ),
            Arg(
                'convexity', int, None, 'A convexity value used for preview mode to aid rendering.'
            ),
        ),
        OPEN_SCAD_URL_TAIL_OTHER,
        'Surface',
    )
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.surface(self, self.file, self.center, self.invert, self.convexity)


@apply_posc_attributes
class Fill(PoscParentBase):
    """Removes holes from polygons without changing the outline. For convex polygons
    the result is identical to hull()."""

    OSC_API_SPEC = OpenScadApiSpecifier('fill', (), OPEN_SCAD_URL_TAIL_TRANSFORMS)
    
    def renderObj(self, renderer: M3dRenderer) -> RenderContext:
        return renderer.fill(self)
