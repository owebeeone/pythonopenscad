"""
OpenScad modifiers.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OscModifier(object):
    """Defines an OpenScad modifier
    
    see: https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Modifier_Characters
    """
    modifier: str = field(repr=False, compare=True)
    name: str = field(repr=True, compare=False)


DISABLE = OscModifier('*', 'DISABLE') # Ignore this subtree
SHOW_ONLY = OscModifier('!', 'SHOW_ONLY') # Ignore the rest of the tree
DEBUG = OscModifier('#', 'DEBUG') # Highlight the object
TRANSPARENT = OscModifier('%', 'TRANSPARENT')  # Background modifier
BASE_MODIFIERS = (DISABLE, SHOW_ONLY, DEBUG, TRANSPARENT)
BASE_MODIFIERS_SET = set(BASE_MODIFIERS)