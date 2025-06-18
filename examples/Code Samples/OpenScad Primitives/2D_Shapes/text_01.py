#   [2025 Jun 17 - tomwwolf]
import sys, os
# extract filename
dirSCAD = os.path.dirname(sys.argv[0])
filenameSCAD = dirSCAD + '/' + os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.scad'
print("output SCAD file: ", filenameSCAD)
"""
    Prototype to generate the following OpenSCAD code

        text(text="hello world!", size=10.0);
        
    reference: https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Text
        
"""
# python scad code
from pythonopenscad import *
# create a simple shape
myString = 'hello world!'
myShape = Text(myString,size=10)
# Save to OpenSCAD file
myShape.write(filenameSCAD)
