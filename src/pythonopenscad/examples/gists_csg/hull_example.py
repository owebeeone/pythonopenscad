"""
hull_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Hull` model to
generate the following OpenSCAD code:

```openscad
hull() {
  color(c="red") {
    translate(v=[-10.0, 0.0, 0.0]) {
      sphere(r=5.0);
    }
  }
  color(c="blue") {
    translate(v=[10.0, 0.0, 0.0]) {
      cube(size=[1.0, 1.0, 15.0], center=true);
    }
  }
}

```

Create a hull of two solids.

Converts to an OpenScad "hull" primitive.
See OpenScad `hull docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Transformations#hull>` for more information.
No arguments allowed.
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_csg.hull_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_csg.hull_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_csg.hull_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Color, Cube, Hull, Sphere, Translate
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Hull()(
    Color("red")(Translate([-10,0,0])(Sphere(r=5))),
    Color("blue")(Translate([10,0,0])(Cube([1,1,15], center=True)))
)

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)