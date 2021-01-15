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
    Sphere, AnnotatedCoordinates, at_spec, lazy_shape, args, CoordinatesCage
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
    
    def setUp(self):
        self.points = []

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
        
    
    def testAddBetween(self):
        shape1 = Box([30, 30, 30])
        #shape2 = Sphere(50)
        shape2 = Box([50, 50, 50])
        maker = shape1.solid('inner').transparent(True).at('centre'
                     ).add(shape2.solid('outer').transparent(True).at('centre'))
        
        #outer_here = at_spec('outer', 'surface', [0, 0, 90])
        outer_here = at_spec('outer', 'face_corner', 5, 1)
        maker.add_between(
            at_spec('inner', 'face_edge', 2, 3, 0),
            outer_here,
            lazy_shape(Cone, 'h', other_args=args(r_base=5)).solid('in_between').colour([1, 0, 0]),
            at_spec('top'),
            at_spec('base'),
            align_axis=l.X_AXIS,
            align_plane=l.Z_AXIS
            )
        
        print(maker)
         
        for i in range(6):
            maker.add_at(Text(str(i)).solid(('face_text', 'inner', i)).colour([0,1,0]).at(), 'face_centre', i)
              
        maker.add_at(Text('Here').solid('here_text').at(), 
                     *outer_here.args_positional, **outer_here.args_named)
        
        self.write(maker, 'addBetween')
    
    def plot_point(self, v, name):
        colour = COLOURS[len(self.points) % len(COLOURS)]
        self.points.append((v, name, colour))
        
    def render_points(self, maker):
        scale = 30
        for v, name, colour in self.points:
            pos_frame = l.translate(v * scale)
            maker.add(Sphere().solid(name).colour(colour).projection(pos_frame.I))
            maker.add(Text(name).solid(('text', name)).colour(colour).projection(l.rotZ(-45) * pos_frame.I))
        
    def testRotAlign(self):
        maker = CoordinatesCage().cage('origin').at('origin')
        
        maker.add(Coordinates().solid('coordinates').at('origin'))
        
        preserve_axis = l.GVector([1, 1, 1]).N
        
        preserve_frame =  l.rotV(preserve_axis, 45) * l.rot_to_V(l.Y_AXIS, preserve_axis)
        
        align_preserve_axis = preserve_frame * l.X_AXIS
        
        o_align_pres_axis = preserve_axis.cross3D(align_preserve_axis).cross3D(preserve_axis)
        
        #o_align_pres_axis = -preserve_axis.cross3D(preserve_axis.cross3D(align_preserve_axis)).N
        
        self.plot_point(preserve_axis, 'preserve_axis')
        self.plot_point(o_align_pres_axis, 'o_align_pres_axis')

        plane_axis = l.Z_AXIS
        
        self.plot_point(plane_axis, 'plane_axis')
        
        to_plane = l.rotToPlane(preserve_axis, plane_axis)
        on_plane = to_plane * preserve_axis
        self.plot_point(on_plane, 'on_plane')
        
        t1_o_align_pres_axis = to_plane * o_align_pres_axis
        self.plot_point(t1_o_align_pres_axis, 't1_o_align_pres_axis')
        
        tx = l.rotToPlane(t1_o_align_pres_axis, plane_axis)
        on_plane_align_pres_axis = tx * align_preserve_axis
        self.plot_point(on_plane_align_pres_axis, 'on_plane_align_pres_axis')
        
        result = to_plane.I * tx * to_plane
        
        new_align_preserve_axis = result * align_preserve_axis
        self.plot_point(new_align_preserve_axis, 'new_align_preserve_axis')
        
        new_preserve_axis = result * preserve_axis
        self.plot_point(new_preserve_axis, '  new_preserve_axis')
    
        self.render_points(maker)

        self.write(maker, 'rotAlign')

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
