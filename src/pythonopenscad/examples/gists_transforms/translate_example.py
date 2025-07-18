"""
translate_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Translate` model to
generate the following OpenSCAD code:

```openscad
translate(v=[10.0, -10.0, 5.0]) {
  cube(size=5.0, center=true);
}

```

Translate child nodes.

Converts to an OpenScad "translate" primitive.
See OpenScad `translate docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Transformations#translate>` for more information.
Args:
    v: (x,y,z) translation vector.
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_transforms.translate_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_transforms.translate_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_transforms.translate_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Cube, Translate
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Translate([10, -10, 5])(Cube(size=5, center=True))

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)