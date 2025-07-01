"""
resize_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Resize` model to
generate the following OpenSCAD code:

```openscad
resize(newsize=[30.0, 10.0, 5.0]) {
  sphere(r=5.0);
}

```

Scales the object so the newsize (x,y,z) parameters given. A zero (0.0) scale is ignored
and that dimension's scale factor is 1.

Converts to an OpenScad "resize" primitive.
See OpenScad `resize docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Transformations#resize>` for more information.
Args:
    newsize: The new (x,y,z) sizes of the resulting object.
    auto: A vector of (x,y,z) booleans to indicate which axes will be resized.
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_transforms.resize_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_transforms.resize_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_transforms.resize_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Resize, Sphere
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Resize(newsize=[30, 10, 5])(Sphere(r=5))

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)