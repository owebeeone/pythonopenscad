#   [2025 Jun 17 - tomwwolf]
import sys, os
# extract filename
dirSCAD = os.path.dirname(sys.argv[0])
filenameSCAD = dirSCAD + '/' + os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.scad'
print("output SCAD file: ", filenameSCAD)
"""
    Prototype to generate the following OpenSCAD code

        square(size=[5.0, 10.0], center=true);
        
    reference: https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Using_the_2D_Subsystem#polygon
        
"""
# python scad code
from pythonopenscad import *
# create a simple shape
myShape = Square(size=[5,10],center=True)
# Save to OpenSCAD file
myShape.write(filenameSCAD)
