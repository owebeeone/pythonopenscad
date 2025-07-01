"""
modules_example.py: A PythonOpenScad example gist.

Note: This file was automatically generated.

Compactify using Module types.

Module in pythonopenscad is basically a named union. The generated code is
split into OpenSCAD modules.


```openscad
// Start: lazy_union
my_module();
my_other_module();
// End: lazy_union

// Modules.

module my_module() {
  sphere(r=10.0);
  cylinder(h=12.0, r=4.0, center=false);
} // end module my_module

module my_other_module() {
  translate(v=[10.0, 0.0, 0.0]) {
    cube(size=10.0);
    my_module();
  }
} // end module my_other_module

```

An implicit union for the top level node. This allows the top level nodes to be rendered
separeately if the model is exported as a 3mf file.

Converts to an OpenScad "lazy_union" primitive.
See OpenScad `lazy_union docs <http://en.wikibooks.org/wiki/OpenSCAD_User_Manual/CSG_Modelling#lazy_union>` for more information.
No arguments allowed.
---
How to run this example:

- To view in the interactive viewer:
  python -m pythonopenscad.examples.gists_other.modules_example --view

- To generate a .scad file (the default action for this script):
  python -m pythonopenscad.examples.gists_other.modules_example --no-view --scad

- To generate a .stl file:
  python -m pythonopenscad.examples.gists_other.modules_example --stl --no-view --no-scad
---

"""

# 1. Import the necessary components from the library.
from pythonopenscad import Cube, Cylinder, LazyUnion, Module, Sphere, Translate
from pythonopenscad.posc_main import posc_main

# 2. Create an instance of the model.
MODEL = LazyUnion()(
    Module("my_module")(Sphere(r=10), Cylinder(h=12, r=4)),
    Module("my_other_module")(
        Translate([10, 0, 0])(Cube(10),
        Module("my_module")(Sphere(r=10), Cylinder(h=12, r=4)))
    )
)

# Notice the openscad version of this code only shows up once:
#     Module("my_module")(Sphere(r=10), Cylinder(h=12, r=4))
# This allows for more compact code and less duplicated processing.

# 3. Use the `posc_main` utility to process the model.
if __name__ == "__main__":
    posc_main([MODEL], default_view=True, default_scad=True)