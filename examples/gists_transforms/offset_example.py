"""
offset_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Offset` model to
generate the following OpenSCAD code:

```openscad
difference() {
  offset(delta=2.0, chamfer=false) {
    square(size=10.0);
  }
  square(size=10.0);
}

```

Generates a new polygon with the curve offset by the given amount. Negative values
can be used to shrink paths while positive values enlarge the path.

Converts to an OpenScad "offset" primitive.
See OpenScad `offset docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Transformations#offset>` for more information.
Args:
    r: The radius of the new path when using the radial method.
    delta: The offset of the new path when using the offset method.
    chamfer: If true will create chamfers at corners. Default False
    _fa (converts to $fa): minimum angle (in degrees) of each segment 
    _fs (converts to $fs): minimum length of each segment 
    _fn (converts to $fn): fixed number of segments. Overrides $fa and $fs 
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_transforms.offset_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_transforms.offset_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_transforms.offset_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Offset, Square
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Offset(delta=2)(Square(10)) - Square(10)

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)