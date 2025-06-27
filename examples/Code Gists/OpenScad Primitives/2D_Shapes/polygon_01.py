#   [2025 Jun 17 - tomwwolf]
import sys, os
# extract filename
dirSCAD = os.path.dirname(sys.argv[0])
filenameSCAD = dirSCAD + '/' + os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.scad'
print("output SCAD file: ", filenameSCAD)
"""
    Prototype to generate the following OpenSCAD code

        polygon(points=[[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]]);
    
    reference: https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Using_the_2D_Subsystem#polygon

"""
# python scad code
from pythonopenscad import *
# create a simple shape
pts = ([0,0],[1,1],[-1,2])
myShape = Polygon(pts)
# Save to OpenSCAD file
myShape.write(filenameSCAD)
