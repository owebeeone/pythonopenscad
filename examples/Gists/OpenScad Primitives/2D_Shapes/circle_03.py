#   [2025 Jun 17 - tomwwolf]
import sys, os
# extract filename
dirSCAD = os.path.dirname(sys.argv[0])
filenameSCAD = dirSCAD + '/' + os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.scad'
print("output SCAD file: ", filenameSCAD)
"""
    Prototype to generate the following OpenSCAD code

        circle(r=5.0, $fn=6);
        
    reference: https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Using_the_2D_Subsystem#circle
        
"""
# pythonopenscad code
from pythonopenscad import *
# create a simple shape
myShape = Circle(r=5, _fn=6)
# Save to OpenSCAD file
myShape.write(filenameSCAD)