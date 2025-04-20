"""
OpenScad modifiers.
"""

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, repr=False)
class OscModifier(object):
    """Defines an OpenScad modifier
    
    see: https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Modifier_Characters
    """
    modifier: str = field(compare=True)
    name: str = field(compare=False)

    def __repr__(self):
        return self.name


DISABLE = OscModifier('*', 'DISABLE') # Ignore this subtree
SHOW_ONLY = OscModifier('!', 'SHOW_ONLY') # Ignore the rest of the tree
DEBUG = OscModifier('#', 'DEBUG') # Highlight the object
TRANSPARENT = OscModifier('%', 'TRANSPARENT')  # Background modifier
BASE_MODIFIERS = (DISABLE, SHOW_ONLY, DEBUG, TRANSPARENT)
BASE_MODIFIERS_SET = set(BASE_MODIFIERS)


# Exceptions for dealing with argument checking.
class PoscBaseException(Exception):
    """Base exception functionality"""


class InvalidModifier(PoscBaseException):
    """Attempting to add or remove an unknown modifier."""
    
class NotParentException(PoscBaseException):
    """Attempting to get children of a non-parent."""


class PoscMetadataBase(object):
    """Provides medatadata properties. i.e. optional application specific 
    properties. The metabase_name is printed in comments in the output file
    while the descriptor is used to store application specific data."""

    def getMetadataName(self) -> str:
        if not hasattr(self, '_metabase_name'):
            return ''
        return self._metabase_name

    def setMetadataName(self, value: str):
        self._metabase_name = value
        return self
        
    def getDescriptor(self) -> Any:
        if not hasattr(self, '_metabase_descriptor'):
            return None
        return self._metabase_descriptor

    def setDescriptor(self, value: Any):
        self._metabase_descriptor = value
        return self


class PoscModifiers(PoscMetadataBase):
    """Functions to add/remove OpenScad modifiers.

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
    """

    def check_is_valid_modifier(self, *modifiers):
        if set(modifiers) - BASE_MODIFIERS_SET:
            raise InvalidModifier(
                '"%r" is not a valid modifier. Muse be one of %r' % (modifiers, BASE_MODIFIERS)
            )

    def add_modifier(self, modifier, *args):
        """Adds one of the model modifiers like DISABLE, SHOW_ONLY, DEBUG or TRANSPARENT.
        Args:
          modifer, *args: The modifier/a being added. Checked for validity.
        """
        self.check_is_valid_modifier(modifier, *args)
        if not hasattr(self, '_osc_modifier'):
            self._osc_modifier = set((modifier,))
        self._osc_modifier.update(args + (modifier,))
        return self

    def remove_modifier(self, modifier, *args):
        """Removes a modifiers, one of DISABLE, SHOW_ONLY, DEBUG or TRANSPARENT.
        Args:
          modifer, *args: The modifier/s being removed. Checked for validity.
        """
        self.check_is_valid_modifier(modifier, *args)
        if not hasattr(self, '_osc_modifier'):
            return
        self._osc_modifier.difference_update(args + (modifier,))
        return self

    def has_modifier(self, modifier):
        """Checks for presence of a modifier, one of DISABLE, SHOW_ONLY, DEBUG or TRANSPARENT.
        Args:
          modifer: The modifier being inspected. Checked for validity.
        """
        self.check_is_valid_modifier(modifier)
        if not hasattr(self, '_osc_modifier'):
            return False
        return modifier in self._osc_modifier

    def get_modifiers(self):
        """Returns the current set of modifiers as an OpenScad equivalent modifier string"""
        if not hasattr(self, '_osc_modifier'):
            return ''
        # Maintains order of modifiers.
        return ''.join(i.modifier for i in BASE_MODIFIERS if i in self._osc_modifier)

    def get_modifiers_repr(self):
        """Returns the repr() equivalent of the current set or None if none are set."""
        if not hasattr(self, '_osc_modifier'):
            return None
        if self._osc_modifier:
            return repr(self._osc_modifier)
        return None

    # Deprecated.
    def transparent(self):
        self.add_modifier(TRANSPARENT)
        return self
    

class UidGen:
    """Basic id generator class"""
    curid: int = 1
    
    def genid(self):
        self.curid += 1
        return str(self.curid)
_UIDGENNY = UidGen()

@dataclass
class PoscRendererBase(PoscModifiers):
    """Base class for renderer interfaces."""
    
    @property
    def uid(self) -> str:
        if not hasattr(self, "_uid"):
            self._uid = _UIDGENNY.genid()
        return self._uid
    
    def children(self) -> list["PoscRendererBase"]:
        # This should be implemented in PoscParentBase. Illegal to call on
        # non-parent types.
        raise NotParentException("get_children is not implemented")
    
    def renderObj(self, renderer: "RendererBase") -> "RenderContextBase":
        # This should be implemented in each of the leaf classes.
        raise NotImplementedError("renderObj is not implemented")
    
    def can_have_children(self) -> bool:
        """This is a childless node, always returns False."""
        return False
    
class RenderContextBase:
    """Base class for render context interfaces."""
    
class RendererBase:
    """Base class for renderer interfaces."""

    def renderChildren(self, posc_obj: PoscRendererBase) -> list[RenderContextBase]:
        # This should be implemented in each of the renderer classes.   
        return [child.renderObj(self) for child in posc_obj.children()]



def get_fragments_from_fn_fa_fs(r: float, fn: int | None, fa: float | None, fs: float | None) -> int:
    # From openscad/src/utils/calc.cpp
    # int Calc::get_fragments_from_r(double r, double fn, double fs, double fa)
    # {
    #   // FIXME: It would be better to refuse to create an object. Let's do more strict error handling
    #   // in future versions of OpenSCAD
    #   if (r < GRID_FINE || std::isinf(fn) || std::isnan(fn)) return 3;
    #   if (fn > 0.0) return static_cast<int>(fn >= 3 ? fn : 3);
    #   return static_cast<int>(ceil(fmax(fmin(360.0 / fa, r * 2 * M_PI / fs), 5)));
    # }
    GRID_FINE = 0.00000095367431640625
    if r < GRID_FINE or fn is not None and (math.isinf(fn) or math.isnan(fn)):
        return 3
    if fn is not None and fn > 0:
        return max(fn, 3)
    if fa is None:
        fa = 1
    if fs is None:
        fs = 1
    return max(5, math.ceil(max(min(360.0 / fa, r * 2 * math.pi / fs), 5)))