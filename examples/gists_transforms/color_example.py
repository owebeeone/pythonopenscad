"""
color_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Color` model to
generate the following OpenSCAD code:

```openscad
color(c="green") {
  sphere(r=10.0);
}

```

Apply a color (only supported in OpenScad preview mode). Colors can be a 3 vector
of values [0.0-1.0] for RGB or additionally a 4 vector if alpha is included for an
RGBA color. Colors can be specified as #RRGGBB and it's variants.

Converts to an OpenScad "color" primitive.
See OpenScad `color docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Transformations#color>` for more information.
Args:
    c: A 3 or 4 color RGB or RGBA vector or a string descriptor of the color.
    alpha: The alpha of the color if not already provided by c.
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_transforms.color_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_transforms.color_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_transforms.color_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Color, Sphere
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Color("green")(Sphere(r=10))

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)