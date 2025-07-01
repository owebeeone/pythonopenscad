"""
polyhedron_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

This script demonstrates the creation of a `Polyhedron` model to
generate the following OpenSCAD code:

```openscad
polyhedron(points=[[10.0, 10.0, 0.0], [10.0, -10.0, 0.0], [-10.0, -10.0, 0.0], [-10.0, 10.0, 0.0], [0.0, 0.0, 10.0]], faces=[[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [3, 2, 1, 0]], convexity=10);

```

Creates an arbitrary polyhedron 3D object.
Note: triangles is deprecated.

Converts to an OpenScad "polyhedron" primitive.
See OpenScad `polyhedron docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Primitive_Solids#polyhedron>` for more information.
Args:
    points: A list of 3D points. The index to these points are used in faces or triangles.
    triangles: A list of triangles. Each triangle is 3 indexes into the points list.
    faces: A list of faces. Each face is a minimum of 3 indexes into the points list
    convexity: A convexity value used for preview mode to aid rendering. Default 10
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_3d.polyhedron_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_3d.polyhedron_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_3d.polyhedron_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Polyhedron
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = Polyhedron(
    points=[[10,10,0],[10,-10,0],[-10,-10,0],[-10,10,0], [0,0,10]],
    faces=[[0,1,4],[1,2,4],[2,3,4],[3,0,4], [3,2,1,0]]
)

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)