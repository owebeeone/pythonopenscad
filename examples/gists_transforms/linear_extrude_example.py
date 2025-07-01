"""
linear_extrude_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Linear_Extrude` model to
generate the following OpenSCAD code:

```openscad
linear_extrude(height=5.0, center=true, twist=90.0, scale=0.5) {
  square(size=10.0);
}

```

Creates an 3D object with a linear extrusion of a 2D shape.

Converts to an OpenScad "linear_extrude" primitive.
See OpenScad `linear_extrude docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Using_the_2D_Subsystem#linear_extrude>` for more information.
Args:
    height: The height of the resulting extrusion. Default 100
    center: If true, the final object's height center point is placed at z=0.
    convexity: A convexity value used for preview mode to aid rendering.
    twist: If provided the object is rotated about the z axis by this total angle
    slices: The number of slices to be applied in the resulting extrusion.
    scale (Attribute name: scale_) : A scale factor to applied to the children incrementally per extrusion layer.
    _fn (converts to $fn): fixed number of segments. Overrides $fa and $fs 
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_transforms.linear_extrude_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_transforms.linear_extrude_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_transforms.linear_extrude_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Linear_Extrude, Square
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Linear_Extrude(height=5, center=True, scale=0.5, twist=90)(Square(10))

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)