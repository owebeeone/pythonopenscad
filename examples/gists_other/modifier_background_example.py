"""
modifier_background_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This shows how to use the background (%) modifier. This will be shown in the viewer but discarded when generating the mesh files.

```openscad
difference() {
  color(c="limegreen") {
    cube(size=10.0);
  }
  %color(c="violet") {
    sphere(r=7.0);
  }
}

```

Creates a 3D object by removing the space of the 3D objects following the first
object provided from the first object.

Converts to an OpenScad "difference" primitive.
See OpenScad `difference docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/CSG_Modelling#difference>` for more information.
No arguments allowed.
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_other.modifier_background_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_other.modifier_background_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_other.modifier_background_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import BACKGROUND, Color, Cube, Sphere
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Color("limegreen")(Cube(10)) - Color("violet")(Sphere(r=7)).add_modifier(BACKGROUND)

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)