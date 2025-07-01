"""
union_operator.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Union` model to
generate the following OpenSCAD code:

```openscad
union() {
  color(c="red") {
    cube(size=10.0);
  }
  color(c="blue") {
    translate(v=[5.0, 5.0, 5.0]) {
      sphere(r=7.0);
    }
  }
}

```

Unifies a set of 3D objects into a single object by performing a union of all the space
contained by all the shapes.

Converts to an OpenScad "union" primitive.
See OpenScad `union docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/CSG_Modelling#union>` for more information.
No arguments allowed.
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_csg.union_operator --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_csg.union_operator --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_csg.union_operator --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Color, Cube, Sphere, Translate
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Color('red')(Cube(10)) + Color('blue')(Translate([5,5,5])(Sphere(r=7)))

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)