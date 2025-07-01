"""
multmatrix_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Multmatrix` model to
generate the following OpenSCAD code:

```openscad
multmatrix(m=[[1.0, 0.5, 0.0, 5.0], [0.0, 1.0, 0.5, 10.0], [0.5, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
  cube(size=10.0);
}

```

Homogeneous matrix multiply. The provided matrix can both rotate and translate.

Converts to an OpenScad "multmatrix" primitive.
See OpenScad `multmatrix docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Transformations#multmatrix>` for more information.
Args:
    m: A 4x4 or 4x3 matrix. The last row must always be [0,0,0,1] and in the
            case of a 4x3 matrix that row is added. The resulting matrix is always 4x4.
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_transforms.multmatrix_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_transforms.multmatrix_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_transforms.multmatrix_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Cube, Multmatrix
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Multmatrix(m=[
    [1, 0.5, 0, 5],
    [0, 1, 0.5, 10],
    [0.5, 0, 1, 0],
    [0, 0, 0, 1]
])(Cube(size=10))

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)