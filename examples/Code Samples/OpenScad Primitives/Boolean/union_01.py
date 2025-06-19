#   [2025 Jun 19 - tomwwolf]
import sys, os

# extract filename
dirSCAD = os.path.dirname(sys.argv[0])
filenameSCAD = dirSCAD + '/' + os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.scad'
print("output SCAD file: ", filenameSCAD)
"""
    Prototype to generate the following OpenSCAD code

    union() {
        cylinder(h=4.0, r=1.0, center=true, $fn=100);
        rotate(a=[90.0, 0.0, 0.0]) {
            cylinder(h=4.0, r=0.9, center=true, $fn=100);
        }
        cylinder(h=5.0, r=0.4, center=true, $fn=100);
    }

    reference: https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/CSG_Modelling#union

"""
# pythonopenscad code
from pythonopenscad import *

# create a simple shape
shape1 = Cylinder(h = 4, r=1, center = True, _fn=100)
shape2 = Rotate([90,0,0])(Cylinder(h = 4, r=0.9, center = True, _fn=100))
shape3 = Cylinder(h = 5, r=0.4, center = True, _fn=100)
myShape = Union()(shape1,shape2,shape3)

# Save to OpenSCAD file
myShape.write(filenameSCAD)