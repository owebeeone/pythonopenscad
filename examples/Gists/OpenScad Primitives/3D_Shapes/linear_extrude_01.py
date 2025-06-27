#   [2025 Jun 18 - tomwwolf]
import sys, os

# extract filename
dirSCAD = os.path.dirname(sys.argv[0])
filenameSCAD = dirSCAD + '/' + os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.scad'
print("output SCAD file: ", filenameSCAD)
"""
    Prototype to generate the following OpenSCAD code

    linear_extrude(height=5.0) {
        polygon(points=[[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]]);
    }

    reference: https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Using_the_2D_Subsystem#linear_extrude

"""
# python scad code
from pythonopenscad import *

# create a simple shape
pts = ([0, 0], [1, 1], [-1, 2])
myShape = Polygon(pts).linear_extrude(height=5)
# Save to OpenSCAD file
myShape.write(filenameSCAD)
