'''
Created on 31 Dec 2020

@author: gianni
'''


from dataclasses import dataclass
import sys
import unittest

from frozendict import frozendict

from ParametricSolid import linear as l
from ParametricSolid.core import Box, Colour, Text, Cone, Arrow, Coordinates, \
    Sphere, AnnotatedCoordinates
from ParametricSolid.renderer import render
import numpy as np
import pythonopenscad as posc

COLOURS=[
    Colour([1, 0, 0]),
    Colour([0, 1, 0]),
    Colour([0, 0, 1]),
    Colour([1, 1, 0]),
    Colour([1, 0, 1]),
    Colour([0, 1, 1]),
    ]

def pick_colour(face, corner):
    return COLOURS[(face * 17 + corner) % len(COLOURS)]

class CoreTest(unittest.TestCase):

    def write(self, maker, test):
        obj = render(maker)
        filename = f'test_{test}.scad'
        obj.write(filename)
        print(f'written scad file: {filename}')

    def testSimple(self):
        b1 = Box([1, 1, 1])
        maker = b1.solid('outer').at('centre', post=l.rotZ(45))
        self.write(maker, 'Simple')
        
    def testAddAt(self):
        
        maker = Box([3, 4, 6]).solid('block').at('face_corner', 0, 0)
        for corner in range(4):
            maker.add_at(
                Box([1, 1, 1])
                .solid(f'corner_{corner}')
                .colour(pick_colour(0, corner))
                .at('centre'), 
                'face_corner', 0, corner)
            maker.add_at(
                Box([1, 1, 1])
                .solid(f'corner_{corner + 4}')
                .colour(pick_colour(3, corner))
                .at('centre'),
                'face_corner', 3, corner)
        for face in range(6):
            maker.add_at(
                Box([1, 1, 1])
                .solid(f'face_{face}')
                .colour(pick_colour(face, 10))
                .at('centre'), 
                'face_centre', face)
       
        self.write(maker, 'AddAt')
    
    def makeDiffBox(self, face, corner):    
        b1 = Box([2, 2.7, 3.5])
        b2 = Box([1, 1, 1])
        maker = b1.solid('outer').at('face_corner', face, corner)
        maker.add(
                b2.hole('etch1').at('centre')
            ).add(
                b2.hole('etch2').at('centre', post=l.rotZ(45)))
            
        return maker
    
    def makeAt(self, face, corner, tv, ref=''):
        return self.makeDiffBox(
            face, corner).solid(f'at_{ref}{face}_{corner}').colour(
            pick_colour(face, corner)).at(
            'outer', 'face_corner', face, corner, post=l.translate(tv))
    
    def testDiff(self):
        cage = Box([1, 1, 1])
        maker = cage.cage('reference').at()
        
        offsetx = l.GVector([-4, 0, 0])
        offsety = l.GVector([0, -5, 0])
        maker.add(self.makeAt(0, 0, l.GVector([0, 9, 0]), ref='r'))
        
        for i in range(4):
            for j in range(3):
                test_face = j
                offsetx = l.GVector([-4 * (1.1 * (i + 1)), 0, -5 * j])
                offsety = l.GVector([0, -5, 0])
                maker.add(self.makeAt(test_face, i, offsetx))
                maker.add(self.makeAt(test_face + 3, i, offsetx + offsety))
        
        self.write(maker, 'Diff')
        
    def testExamples(self):
        self.write(Box.example(), 'BoxExample')
        self.write(Text.example(), 'TextExample')
        self.write(Sphere.example(), 'SphereExample')
        self.write(Cone.example(), 'ConeExample')
        self.write(Arrow.example(), 'ArroeExample')
        self.write(Coordinates.example(), 'CoordinatesExample')
        self.write(AnnotatedCoordinates.example(), 'AnnotatedCoordinatesExample')
        
        
    def testColour(self):
        b1 = Box([2, 3, 4])
        redthing1 = b1.solid('thing1').colour([1,0,0]).at()
        self.write(redthing1, 'Colour')
        
    def testText(self):
        t1 = Text('Hello')
        self.write(
            t1.solid('default_text').fn(20).at('default', 'rear').add(
                t1.solid('scaled_text').fn(20)
                .at('default', 'rear', 
                    post=l.translate([0, 20, 0]) * l.scale([1, 1, 1/10]))
            ), 'Text')
        
        
    def testCone(self):
        c1 = Cone(h=40, r_base=22, r_top=4)
        maker = c1.solid('cone1').fn(40).colour([.1,1,0]).at('top')
        maker.add_at(
            Cone(h=40, r_base=10, r_top=5).solid('cone2').colour([0,1,0])
               .at('top', pre=l.translate([0, 0, -5]), post=l.rotX(180)),
            'surface', 20, 45)
        self.write(maker, 'Cone')
        
        
    def testArrow(self):
        a1 = Arrow(l_stem=10)
        maker = a1.solid('x').fn(40).colour([.1,1,0]).at('base')
        self.write(maker, 'Arrow')
        
    def testCoordinates(self):
        coordinates = Coordinates(fn=20)
        maker = coordinates.solid('coordinates').at('origin')
        self.write(maker, 'Coordinates')

    def testAnnotated(self):
        coordinates = AnnotatedCoordinates(Coordinates())
        maker = coordinates.solid('coordinates').fn(27).at('origin')
        self.write(maker, 'AnnotatedCoordinates')
        
    def testBoxStack(self):
        box1 = Box([3, 3, 3])
        box2 = Box([6, 6, 6])
        box3 = Box([9, 9, 9])
        
        maker = box1.solid('box1').colour(Colour([0, 0, 1])).at('face_centre', 0)
        maker.add_at(box2.solid('box2').colour(Colour([0, 1, 0])).at('face_centre', 0), 
                     'box1', 'face_centre', 0, post=l.rotX(180))
        
        maker.add_at(box3.solid('box3').colour(Colour([1, 0, 0])).at('face_corner', 0, 2), 
                     'box2', 'face_centre', 3, post=l.rotX(180))
        self.write(maker, 'BoxStack')
        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
