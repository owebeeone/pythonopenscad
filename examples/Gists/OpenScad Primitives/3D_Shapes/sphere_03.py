#   [2025 Jun 18 - tomwwolf]
import sys, os

# extract filename
dirSCAD = os.path.dirname(sys.argv[0])
filenameSCAD = dirSCAD + '/' + os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.scad'
print("output SCAD file: ", filenameSCAD)
"""
    Prototype to generate the following OpenSCAD code

        sphere(d=5.0, $fa=12.0, $fs=2.0, $fn=100);

    reference: https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Primitive_Solids#sphere

"""
# pythonopenscad code
from pythonopenscad import *

# create a simple shape
myShape = Sphere(d=5, _fn = 100, _fa = 12, _fs = 2)
# Save to OpenSCAD file
myShape.write(filenameSCAD)