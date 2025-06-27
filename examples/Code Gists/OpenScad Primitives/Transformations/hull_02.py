#   [2025 Jun 19 - tomwwolf]
import sys, os

# extract filename
dirSCAD = os.path.dirname(sys.argv[0])
filenameSCAD = dirSCAD + '/' + os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.scad'
print("output SCAD file: ", filenameSCAD)
"""
    Prototype to generate the following OpenSCAD code

    hull() {
        sphere(r=5.0);
        translate(v=[10.0, 10.0, 0.0]) {
            sphere(r=2.0);
        }
    }

    reference: https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Transformations#hull

"""
# pythonopenscad code
from pythonopenscad import *

# create a simple shape
shape1 = Sphere(r=5)
shape2 = Translate([10,10,0])(Sphere(r=2))
myShape = Hull()(shape1,shape2)
# Save to OpenSCAD file
myShape.write(filenameSCAD)