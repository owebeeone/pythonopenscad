#   [2025 Jun 17 - tomwwolf]
import sys, os

# extract filename
dirSCAD = os.path.dirname(sys.argv[0])
filenameSCAD = dirSCAD + '/' + os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.scad'
print("output SCAD file: ", filenameSCAD)
"""
    Prototype to generate the following OpenSCAD code

        cylinder(h=10.0, r1=2.0, r2=3.0, center=false);

    reference: https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Primitive_Solids#cylinder

"""
# pythonopenscad code
from pythonopenscad import *

# create a simple shape
myShape = Cylinder(h=10,r1=2,r2=3)
# Save to OpenSCAD file
myShape.write(filenameSCAD)