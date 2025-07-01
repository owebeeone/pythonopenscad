"""
sphere_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Sphere` model to
generate the following OpenSCAD code:

```openscad
sphere(r=10.0, $fn=128);

```

Creates a sphere.
It defaults to a sphere of radius 1. If d is provided it overrides the value of r.


Converts to an OpenScad "sphere" primitive.
See OpenScad `sphere docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Primitive_Solids#sphere>` for more information.
Args:
    r: radius of sphere. Ignores d if set.
    d: diameter of sphere.
    _fa (converts to $fa): minimum angle (in degrees) of each segment 
    _fs (converts to $fs): minimum length of each segment 
    _fn (converts to $fn): fixed number of segments. Overrides $fa and $fs 
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_3d.sphere_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_3d.sphere_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_3d.sphere_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Sphere
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Sphere(r=10, _fn=128)

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)