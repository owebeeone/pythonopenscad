"""
projection_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Projection` model to
generate the following OpenSCAD code:

```openscad
projection(cut=true) {
  union() {
    cube(size=7.0);
    translate(v=[0.0, 0.0, 2.5]) {
      sphere(r=5.0);
    }
  }
}

```

Project a 3D object into a 2D surface.

Converts to an OpenScad "projection" primitive.
See OpenScad `projection docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Using_the_2D_Subsystem#3D_to_2D_Projection>` for more information.
Args:
    cut: If false, the projection is a "shadow" of the object otherwise it is an intersection.
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_transforms.projection_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_transforms.projection_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_transforms.projection_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Cube, Projection, Sphere, Translate
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Projection(cut=True)(Cube(7) + Translate([0,0,2.5])(Sphere(r=5)))

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)