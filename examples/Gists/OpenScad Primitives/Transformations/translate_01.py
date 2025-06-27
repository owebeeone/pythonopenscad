#   [2025 Jun 19 - tomwwolf]
import sys, os

# extract filename
dirSCAD = os.path.dirname(sys.argv[0])
filenameSCAD = dirSCAD + '/' + os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.scad'
print("output SCAD file: ", filenameSCAD)
"""
    Prototype to generate the following OpenSCAD code

    translate(v=[5.0, 5.0, 0.0]) {
        circle(r=5.0);
    }

    https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Transformations#translate

"""
# pythonopenscad code
from pythonopenscad import *

# create a simple shape
myShape = Translate([5,5,0])(Circle(r=5))
# Save to OpenSCAD file
myShape.write(filenameSCAD)