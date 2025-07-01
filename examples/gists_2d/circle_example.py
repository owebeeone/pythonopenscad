"""
circle_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Circle` model to
generate the following OpenSCAD code:

```openscad
circle(d=10.0, $fn=64);

```

Creates a 2D circle shape.
Note that if d is provided it has precedence over r if provided.


Converts to an OpenScad "circle" primitive.
See OpenScad `circle docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Using_the_2D_Subsystem#circle>` for more information.
Args:
    r: The radius of the generated circle. Default 1
    d: The diameter of the circle, overrides r.
    _fa (converts to $fa): minimum angle (in degrees) of each segment 
    _fs (converts to $fs): minimum length of each segment 
    _fn (converts to $fn): fixed number of segments. Overrides $fa and $fs 
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_2d.circle_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_2d.circle_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_2d.circle_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Circle
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Circle(d=10, _fn=64)

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)