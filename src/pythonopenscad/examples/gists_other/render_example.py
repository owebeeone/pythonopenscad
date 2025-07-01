"""
render_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Render` model to
generate the following OpenSCAD code:

```openscad
render(convexity=10) {
  difference() {
    cube(size=10.0);
    sphere(r=6.0);
  }
}

```

Forces the generation of a mesh even in preview mode.

Converts to an OpenScad "render" primitive.
See OpenScad `render docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Other_Language_Features#render>` for more information.
Args:
    convexity: A convexity value used for optimization of rendering. Default 10
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_other.render_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_other.render_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_other.render_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Cube, Difference, Render, Sphere
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Render(convexity=10)(Difference()(Cube(10), Sphere(r=6)))

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)