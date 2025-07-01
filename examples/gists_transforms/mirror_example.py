"""
mirror_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Mirror` model to
generate the following OpenSCAD code:

```openscad
union() {
  mirror(v=[1.0, 1.0, 0.0]) {
    translate(v=[10.0, 0.0, 0.0]) {
      color(c="green") {
        cube(size=5.0);
      }
    }
  }
  // 'not mirrored'
  translate(v=[10.0, 0.0, 0.0]) {
    color(c="red") {
      cube(size=5.0);
    }
  }
}

```

Mirrors across a plane defined by the normal v.

Converts to an OpenScad "mirror" primitive.
See OpenScad `mirror docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Transformations#mirror>` for more information.
Args:
    v: The normal of the plane to be mirrored.
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_transforms.mirror_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_transforms.mirror_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_transforms.mirror_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Color, Cube, Mirror, Translate
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Mirror([1, 1, 0])(Translate([10, 0, 0])(Color('green')(Cube(size=5)))) \
    + Translate([10, 0, 0])(Color('red')(Cube(size=5))).setMetadataName("not mirrored")

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)