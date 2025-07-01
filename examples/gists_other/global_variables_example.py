"""
global_variables_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This shows how to set the $fn global variable.

```openscad
$fn = 50;

union() {
  sphere(r=10.0);
  cylinder(h=12.0, r=4.0, center=false);
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
  python -m pythonopenscad.examples.gists_other.global_variables_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_other.global_variables_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_other.global_variables_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Cylinder, POSC_GLOBALS, Sphere
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
POSC_GLOBALS._fn = 50
MODEL = Sphere(r=10) + Cylinder(h=12, r=4)

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)