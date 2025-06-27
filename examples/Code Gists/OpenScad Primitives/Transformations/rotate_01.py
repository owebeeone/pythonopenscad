#   [2025 Jun 21 - tomwwolf]
import sys, os

# extract filename
dirSCAD = os.path.dirname(sys.argv[0])
filenameSCAD = dirSCAD + '/' + os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.scad'
print("output SCAD file: ", filenameSCAD)
"""
    Prototype to generate the following OpenSCAD code

    rotate(a=[45.0, 45.0, 45.0]) {
        cube(size=5.0, center=true);
    }

    https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Transformations#rotate

"""
# pythonopenscad code
from pythonopenscad import *

# create a simple shape
myShape = Rotate([45,45,45])(Cube(size=5,center=True))
# Save to OpenSCAD file
myShape.write(filenameSCAD)