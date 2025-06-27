#   [2025 Jun 17 - tomwwolf]
import sys, os
# extract filename
dirSCAD = os.path.dirname(sys.argv[0])
filenameSCAD = dirSCAD + '/' + os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.scad'
print("output SCAD file: ", filenameSCAD)
"""
    Prototype to generate the following OpenSCAD code

        square(size=5);
        
    reference: https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Using_the_2D_Subsystem#square
        
"""
# python scad code
from pythonopenscad import *
# create a simple shape
myShape = Square(size=5)
# Save to OpenSCAD file
myShape.write(filenameSCAD)
