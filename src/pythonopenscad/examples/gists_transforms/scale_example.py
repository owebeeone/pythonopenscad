"""
scale_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Scale` model to
generate the following OpenSCAD code:

```openscad
scale(v=[1.5, 1.0, 0.5]) {
  sphere(r=10.0);
}

```

Scales the child nodes. scale

Converts to an OpenScad "scale" primitive.
See OpenScad `scale docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Transformations#scale>` for more information.
Args:
    v: The (x,y,z) scale factors. Default (1, 1, 1)
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_transforms.scale_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_transforms.scale_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_transforms.scale_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Scale, Sphere
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Scale([1.5, 1, 0.5])(Sphere(r=10))

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)