"""
modifier_combinations_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This shows how to use the debug (#) and background (%) modifiers. This will highlight the sphere in the viewer and make it the background.

```openscad
union() {
  color(c="limegreen") {
    cube(size=10.0);
  }
  #%color(c="violet") {
    sphere(r=7.0);
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
  python -m pythonopenscad.examples.gists_other.modifier_combinations_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_other.modifier_combinations_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_other.modifier_combinations_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import BACKGROUND, Color, Cube, DEBUG, Sphere
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Color("limegreen")(Cube(10)) + Color("violet")(Sphere(r=7)).add_modifier(DEBUG, BACKGROUND)

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)