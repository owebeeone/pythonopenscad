"""
text_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Text` model to
generate the following OpenSCAD code:

```openscad
text(text="POSC", size=10.0, font="Liberation Sans:style=Bold", halign="center");

```

Creates a 2D shape from a text string with a given font. A 2D shape consisting of
an outline for each glyph in the string.

Converts to an OpenScad "text" primitive.
See OpenScad `text docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Text#>` for more information.
Args:
    text: The string used to generate the
    size: 
    font: 
    halign: 
    valign: 
    spacing: 
    direction: 
    language: 
    script: 
    _fa (converts to $fa): minimum angle (in degrees) of each segment 
    _fs (converts to $fs): minimum length of each segment 
    _fn (converts to $fn): fixed number of segments. Overrides $fa and $fs 
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_2d.text_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_2d.text_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_2d.text_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Text
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Text(text="POSC", size=10, font="Liberation Sans:style=Bold", halign="center")

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)