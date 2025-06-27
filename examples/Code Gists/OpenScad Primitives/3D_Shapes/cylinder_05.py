#   [2025 Jun 17 - tomwwolf]
import sys, os

# extract filename
dirSCAD = os.path.dirname(sys.argv[0])
filenameSCAD = dirSCAD + '/' + os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.scad'
print("output SCAD file: ", filenameSCAD)
"""
    Prototype to generate the following OpenSCAD code

        cylinder(h=10.0, d1=3.0, d2=5.0, center=false);
        --- this does not appear correct---

    reference: https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Primitive_Solids#cylinder

"""
# pythonopenscad code
from pythonopenscad import *

# create a simple shape
myShape = Cylinder(h=10,d1=3,d2=5)
# Save to OpenSCAD file
myShape.write(filenameSCAD)