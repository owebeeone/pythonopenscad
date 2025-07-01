"""
rotate_extrude_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Rotate_Extrude` model to
generate the following OpenSCAD code:

```openscad
rotate_extrude(angle=270.0, $fn=128) {
  translate(v=[5.0, 0.0, 0.0]) {
    circle(r=2.0);
  }
}

```

Creates an 3D object with a rotating extrusion of a 2D shape.

Converts to an OpenScad "rotate_extrude" primitive.
See OpenScad `rotate_extrude docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Using_the_2D_Subsystem#Rotate_Extrude>` for more information.
Args:
    angle: The total angle to extrude. Default 360
    convexity: A convexity value used for preview mode to aid rendering.
    _fa (converts to $fa): minimum angle (in degrees) of each segment 
    _fs (converts to $fs): minimum length of each segment 
    _fn (converts to $fn): fixed number of segments. Overrides $fa and $fs 
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_transforms.rotate_extrude_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_transforms.rotate_extrude_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_transforms.rotate_extrude_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Circle, Rotate_Extrude, Translate
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Rotate_Extrude(angle=270, _fn=128)(Translate([5,0,0])(Circle(r=2)))

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)