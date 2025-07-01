"""
cube_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Cube` model to
generate the following OpenSCAD code:

```openscad
cube(size=[10.0, 15.0, 5.0], center=true);

```

Creates a cube with it's bottom corner centered at the origin.

Converts to an OpenScad "cube" primitive.
See OpenScad `cube docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Primitive_Solids#cube>` for more information.
Args:
    size: The x, y and z sizes of the cube or rectangular prism Default (1, 1, 1)
    center: If true places the center of the cube at the origin.
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_3d.cube_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_3d.cube_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_3d.cube_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Cube
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Cube(size=[10, 15, 5], center=True)

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)