"""
minkowski_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Minkowski` model to
generate the following OpenSCAD code:

```openscad
minkowski() {
  square(size=[10.0, 2.0]);
  circle(r=2.0);
}

```

Create a Minkowski transformed object.

Converts to an OpenScad "minkowski" primitive.
See OpenScad `minkowski docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Transformations#minkowski>` for more information.
No arguments allowed.
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_csg.minkowski_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_csg.minkowski_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_csg.minkowski_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Circle, Minkowski, Square
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Minkowski()(Square([10, 2]), Circle(r=2))

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)