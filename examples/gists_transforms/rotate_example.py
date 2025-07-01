"""
rotate_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Rotate` model to
generate the following OpenSCAD code:

```openscad
rotate(a=[45.0, 45.0, 0.0]) {
  cube(size=[10.0, 15.0, 5.0]);
}

```

Rotate child nodes. This comes in two forms, the first form
is a single angle and optional axis of rotation, the second form
is a vector of angles for three consecutive rotations about the
z, y and z axis.

Converts to an OpenScad "rotate" primitive.
See OpenScad `rotate docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Transformations#rotate>` for more information.
Args:
    a: Angle to rotate or vector of angles applied to each axis in sequence. Default 0
    v: (x,y,z) axis of rotation vector.
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_transforms.rotate_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_transforms.rotate_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_transforms.rotate_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Cube, Rotate
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Rotate([45, 45, 0])(Cube(size=[10, 15, 5]))

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)