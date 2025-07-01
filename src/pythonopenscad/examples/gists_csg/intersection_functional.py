"""
intersection_functional.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Intersection` model to
generate the following OpenSCAD code:

```openscad
intersection() {
  color(c="red") {
    cube(size=10.0, center=true);
  }
  color(c="blue") {
    sphere(r=7.0);
  }
}

```

Creates a 3D object by finding the common space contained in all the provided
3D objects.

Converts to an OpenScad "intersection" primitive.
See OpenScad `intersection docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/CSG_Modelling#intersection>` for more information.
No arguments allowed.
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_csg.intersection_functional --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_csg.intersection_functional --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_csg.intersection_functional --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Color, Cube, Intersection, Sphere
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Intersection()(
    Color("red")(Cube(10, center=True)),
    Color("blue")(Sphere(r=7))
)

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)