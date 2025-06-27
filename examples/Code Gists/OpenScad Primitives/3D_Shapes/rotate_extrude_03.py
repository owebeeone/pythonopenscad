#   [2025 Jun 19 - tomwwolf]
import sys, os

# extract filename
dirSCAD = os.path.dirname(sys.argv[0])
filenameSCAD = dirSCAD + '/' + os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.scad'
print("output SCAD file: ", filenameSCAD)
"""
    Prototype to generate the following OpenSCAD code

    rotate_extrude(angle=120.0, convexity=10) {
        square(size=5.0);
    }

    reference: https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Using_the_2D_Subsystem#rotate_extrude

"""
# pythonopenscad code
from pythonopenscad import *

# create a simple shape
myShape = Rotate_Extrude(angle=120,convexity=10)(Square(size=5))
# Save to OpenSCAD file
myShape.write(filenameSCAD)