#   [2025 Jun 19 - tomwwolf]
import sys, os

# extract filename
dirSCAD = os.path.dirname(sys.argv[0])
filenameSCAD = dirSCAD + '/' + os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.scad'
print("output SCAD file: ", filenameSCAD)
"""
    Prototype to generate the following OpenSCAD code

        polyhedron(points=[[10.0, 10.0, 0.0], [10.0, -10.0, 0.0], [-10.0, -10.0, 0.0], [-10.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]], faces=[[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [1, 0, 3], [2, 1, 3]], convexity=10);


    https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Primitive_Solids#polyhedron

"""
# pythonopenscad code
from pythonopenscad import *

# create a simple shape
points=[[10,10,0],[10,-10,0],[-10,-10,0],[-10,10,0],[0,0,10]]
faces=[[0,1,4],[1,2,4],[2,3,4],[3,0,4],[1,0,3],[2,1,3]]
myShape = Polyhedron(points=points,faces=faces)
# Save to OpenSCAD file
myShape.write(filenameSCAD)