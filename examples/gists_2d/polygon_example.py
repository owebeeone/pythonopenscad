"""
polygon_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Polygon` model to
generate the following OpenSCAD code:

```openscad
polygon(points=[[0.0, 0.0], [0.0, 10.0], [10.0, 10.0]]);

```

Creates a polygon 2D shape (with optional holes).
If paths is not provided, one is constructed by generating a sequence 0,,N-1 where N
is the number of points provided.


Converts to an OpenScad "polygon" primitive.
See OpenScad `polygon docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Using_the_2D_Subsystem#Polygons>` for more information.
Args:
    points: A collection of (x,y) points to be indexed in paths.
    paths: A list of paths which are a list of indexes into the points collection.
    convexity: A convexity value used for preview mode to aid rendering.
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_2d.polygon_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_2d.polygon_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_2d.polygon_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Polygon
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Polygon(points=[[0,0], [0,10], [10,10]])

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)