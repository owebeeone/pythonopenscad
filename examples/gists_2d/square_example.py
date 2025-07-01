"""
square_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Square` model to
generate the following OpenSCAD code:

```openscad
square(size=[15.0, 10.0], center=true);

```

Creates a 2D square shape

Converts to an OpenScad "square" primitive.
See OpenScad `square docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Using_the_2D_Subsystem#square>` for more information.
Args:
    size: The square size, if a 2 vector (x,y) is provided a rectangle is generated. Default 1
    center: If true the resulting shape is centered otherwise a corner is at the origin.
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_2d.square_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_2d.square_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_2d.square_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Square
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Square(size=[15, 10], center=True)

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)