#   [2025 Jun 21 - tomwwolf]
import sys, os

# extract filename
dirSCAD = os.path.dirname(sys.argv[0])
filenameSCAD = dirSCAD + '/' + os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.scad'
print("output SCAD file: ", filenameSCAD)
"""
    Prototype to generate the following OpenSCAD code

    difference() {
        offset(r=7.0, chamfer=false) {
            square(size=10.0, center=true);
        }
        offset(r=5.0, chamfer=false) {
            square(size=10.0, center=true);
        }
    }

    https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Transformations#offset

"""
# pythonopenscad code
from pythonopenscad import *

# create a simple shape
shape1 = Offset(r=5)(Square(size=10, center=True))
shape2 = Offset(r=7)(Square(size=10, center=True))
myShape = shape2 - shape1
# Save to OpenSCAD file
myShape.write(filenameSCAD)