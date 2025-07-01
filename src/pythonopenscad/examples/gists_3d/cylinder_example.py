"""
cylinder_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Cylinder` model to
generate the following OpenSCAD code:

```openscad
cylinder(h=20.0, r1=5.0, r2=10.0, center=true, $fn=128);

```

Creates a cylinder or truncated cone about the z axis. Cone needs r1 and r2 or d1 and d2
provided with different lengths.
d1 & d2 have precedence over d which have precedence over r1 and r2 have precedence over r.
Hence setting r is overridden by any other value.


Converts to an OpenScad "cylinder" primitive.
See OpenScad `cylinder docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Primitive_Solids#cylinder>` for more information.
Args:
    h: height of the cylinder or cone. Default 1.0
    r: radius of cylinder. r1 = r2 = r.
    r1: radius, bottom of cone.
    r2: radius, top of cone.
    d: diameter of cylinder. r1 = r2 = d / 2.
    d1: diameter of bottom of cone. r1 = d1 / 2.
    d2: diameter of top of cone. r2 = d2 / 2.
    center: z ranges from 0 to h, true z ranges from -h/2 to +h/2. Default False
    _fa (converts to $fa): minimum angle (in degrees) of each segment 
    _fs (converts to $fs): minimum length of each segment 
    _fn (converts to $fn): fixed number of segments. Overrides $fa and $fs 
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_3d.cylinder_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_3d.cylinder_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_3d.cylinder_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Cylinder
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Cylinder(h=20, r1=5, r2=10, center=True, _fn=128)

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)