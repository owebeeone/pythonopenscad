'''
Created on 8 Jan 2021

@author: gianni
'''

from dataclasses import dataclass
from unittest import TestCase

import ParametricSolid.core as core
import ParametricSolid.extrude as extrude
import ParametricSolid.linear as l
from ParametricSolid.renderer import render
from ParametricSolid.test_tools import iterable_assert
import numpy as np


@dataclass
class TestMetaData:
    fn: int=10


class ExtrudeTest(TestCase):
    
    def write(self, maker, test):
        obj = render(maker)
        filename = f'test_{test}.scad'
        obj.write(filename)
        print(f'written scad file: {filename}')


    def testBezierExtents2D(self):
        b = extrude.CubicSpline([[0, 0], [1, -1], [1, 1], [0, 1]])
        extents = b.extents()
        expects = [[0.0, 0.75], [-0.28, 1.0]]
        iterable_assert(self.assertAlmostEqual, b.extents(), [[0., -0.28], [ 0.75, 1]])
        minima = b.curve_maxima_minima_t()
        self.assertAlmostEqual(b.normal2d(minima[0][0])[1], 1, 5)
        
    def testPathGenerator(self):
        builder = extrude.PathBuilder()
        builder.move([0, 0], 'start').spline([[1, -1], [1, 1], [0, 1]], 'curve')
        
        path = builder.build()
        points = path.points(TestMetaData())
        
        expected = np.array([
            [ 0.   ,  0.   ],
            [ 0.27 , -0.215],
            [ 0.48 , -0.28 ],
            [ 0.63 , -0.225],
            [ 0.72 , -0.08 ],
            [ 0.75 ,  0.125],
            [ 0.72 ,  0.36 ],
            [ 0.63 ,  0.595],
            [ 0.48 ,  0.8  ],
            [ 0.27 ,  0.945],
            [ 0.   ,  1.   ]])
        
        iterable_assert(self.assertAlmostEqual, expected, points)
        
        node = path.get_node('curve')
        assert not node is None
        
        iterable_assert(self.assertAlmostEqual, node.extents(), [[0., -0.28], [ 0.75, 1]])
        
    def testDirection(self):
        builder = extrude.PathBuilder()
        builder.move([0, 0], 'start'
                     ).line([1, 0], 'line'
                     ).spline([[2, 0], [2.5, 4], [3, 3]], 'curve')

        path = builder.build()
        points = path.points(TestMetaData())
        
        expected = np.array([
            [0.    , 0.    ],
            [1.    , 0.    ],
            [1.2855, 0.111 ],
            [1.544 , 0.408 ],
            [1.7785, 0.837 ],
            [1.992 , 1.344 ],
            [2.1875, 1.875 ],
            [2.368 , 2.376 ],
            [2.5365, 2.793 ],
            [2.696 , 3.072 ],
            [2.8495, 3.159 ],
            [3.    , 3.    ]])
        
        iterable_assert(self.assertAlmostEqual, expected, points)
        
        curve = path.get_node('curve')
        line = path.get_node('line')
        
        iterable_assert(self.assertAlmostEqual, 
                        line.direction_normalized(1),  
                        curve.direction_normalized(0))
        
        iterable_assert(self.assertAlmostEqual, 
                        [[0, 0], [3, 3.16049383]],  
                        path.extents())
       
    def testPreviousDirection(self):
        builder = extrude.PathBuilder()
        builder.move([0, 0], 'start'
                     ).line([1, 0], 'line'
                     ).spline([[2.5, 4], [3, 3]], 'curve', cv_len=(0.5,))

        path = builder.build()
        points = path.points(TestMetaData())
        
        iterable_assert(self.assertAlmostEqual, 
                        path.get_node('curve').points,
                        np.array([[1.5, 0. ],
                                  [2.5, 4. ],
                                  [3. , 3. ]]))
        expected = np.array([
            [0.   , 0.   ],
            [1.   , 0.   ],
            [1.164, 0.111],
            [1.352, 0.408],
            [1.558, 0.837],
            [1.776, 1.344],
            [2.   , 1.875],
            [2.224, 2.376],
            [2.442, 2.793],
            [2.648, 3.072],
            [2.836, 3.159],
            [3.   , 3.   ]])
        
        iterable_assert(self.assertAlmostEqual, expected, points)
        

    def testPreviousDirectionFirstAngle(self):
        builder = extrude.PathBuilder()

        builder.move([0, 0], 'start'
                     ).line([1, 0], 'line'
                     ).spline([[2.5, 4], [3, 3]], 'curve', cv_len=(0.5,), degrees=(90,))

        path = builder.build()
        points = path.points(TestMetaData())
        
        iterable_assert(self.assertAlmostEqual, 
                        path.get_node('curve').points,
                        np.array([[1, 0.5 ],
                                  [2.5, 4. ],
                                  [3. , 3. ]]))
        expected = np.array([
            [0.    , 0.    ],
            [1.    , 0.    ],
            [1.0425, 0.2325],
            [1.16  , 0.6   ],
            [1.3375, 1.0575],
            [1.56  , 1.56  ],
            [1.8125, 2.0625],
            [2.08  , 2.52  ],
            [2.3475, 2.8875],
            [2.6   , 3.12  ],
            [2.8225, 3.1725],
            [3.    , 3.    ]])
        
        iterable_assert(self.assertAlmostEqual, expected, points)

    def testPreviousDirectionSecondAngle(self):
        builder = extrude.PathBuilder()
        builder.move([0, 0], 'start'
                     ).line([1, 0], 'line'
                     ).spline([[2.5, 4], [3, 3]], 'curve', cv_len=(0.5,), degrees=(None, -90))

        path = builder.build()
        
        
        iterable_assert(self.assertAlmostEqual, 
                        path.get_node('curve').points,
                        np.array([[1.5, 0. ],
                                  [4. , 3.5],
                                  [3. , 3. ]]))

    def testPreviousDirectionSecondRelative(self):
        builder = extrude.PathBuilder()
        builder.move([0, 0], 'start'
                     ).line([1, 0], 'line'
                     ).spline([[2.5, 4], [3, 3]], 'curve', cv_len=(0.5,), rel_len=0.5)

        path = builder.build()
        
        iterable_assert(self.assertAlmostEqual, 
                        path.get_node('curve').points,
                        np.array([[1.90138782, 0.        ],
                                  [2.19377423, 4.61245155],
                                  [3.        , 3.        ]]))
        

    def testPolygonMultiplePoltygons(self):
        builder = extrude.PathBuilder(multi=True)
        builder.move([0, 0], 'start'
                     ).line([1, 0], 'line'
                     ).spline([[2.5, 4], [3, 3]], 'curve', cv_len=(0.5,), rel_len=0.5
                     ).move([1, 0]
                     ).line([2, 0]
                     ).line([2, 1]
                     ).line([1, 2])

        path = builder.build()

        iterable_assert(self.assertAlmostEqual, path.polygons(TestMetaData()),
                        (([[0.        , 0.        ],
                           [1.        , 0.        ],
                           [1.25326914, 0.12753619],
                           [1.47673525, 0.46679535],
                           [1.67713536, 0.95275334],
                           [1.86120651, 1.52038605],
                           [2.03568577, 2.10466933],
                           [2.20731016, 2.64057907],
                           [2.38281673, 3.06309113],
                           [2.56894253, 3.3071814 ],
                           [2.77242461, 3.30782573],
                           [3.        , 3.        ],
                           [1.        , 0.        ],
                           [2.        , 0.        ],
                           [2.        , 1.        ],
                           [1.        , 2.        ]]), 
                           ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 
                            (13, 14, 15))))
        
        # Try rotating the path.
        new_path = path.transform(l.rotZ(90))
        
        iterable_assert(self.assertAlmostEqual, new_path.polygons(TestMetaData()),
                        (([[ 0.        ,  0.        ],
                           [ 0.        ,  1.        ],
                           [-0.12753619,  1.25326914],
                           [-0.46679535,  1.47673525],
                           [-0.95275334,  1.67713536],
                           [-1.52038605,  1.86120651],
                           [-2.10466933,  2.03568577],
                           [-2.64057907,  2.20731016],
                           [-3.06309113,  2.38281673],
                           [-3.3071814 ,  2.56894253],
                           [-3.30782573,  2.77242461],
                           [-3.        ,  3.        ],
                           [ 0.        ,  1.        ],
                           [ 0.        ,  2.        ],
                           [-1.        ,  2.        ],
                           [-2.        ,  1.        ]]), 
                           ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 
                            (13, 14, 15))))
    

    def makeTestObject(self, scale=1):
        return extrude.LinearExtrude(
            extrude.PathBuilder()
                .move([0, 0])
                .line([100 * scale, 0], 'linear')
                .spline([[150 * scale, 100 * scale], [20 * scale, 100 * scale]],
                         name='curve', cv_len=(0.5,0.4), degrees=(90,), rel_len=0.8)
                .line([0, 100 * scale], 'linear2')
                .line([0, 0], 'linear3')
                .build(),
            h=40,
            fn=30,
            twist=90,
            scale=(1, 0.3)
            )
    

    def testLinearExtrude(self):
        le = self.makeTestObject()
        iterable_assert(self.assertAlmostEqual, le.at('linear', 0.5).A, 
                        [[ 1.,  0.,  0., 50.],
                         [ 0.,  0., -1.,  0.],
                         [ 0.,  1.,  0.,  0.],
                         [ 0.,  0.,  0.,  1.]])
        
    def test_find_a_b_c_from_point_tangent(self):
        l, p, t = extrude.find_a_b_c_from_point_tangent([10, 0], [-20, 20])
        expect_0 = l[0] * p[0] + l[1] * p[1] - l[2]
        self.assertAlmostEqual(expect_0, 0)
        k = 1
        expect_0 = l[0] * (p[0] + k * t[0]) + l[1] * (p[1] + k * t[1]) - l[2]
        self.assertAlmostEqual(expect_0, 0)
        k = -1
        expect_0 = l[0] * (p[0] + k * t[0]) + l[1] * (p[1] + k * t[1]) - l[2]
        self.assertAlmostEqual(expect_0, 0)
        
    def test_solve_circle_tangent_point(self):
        c, r = extrude.solve_circle_tangent_point([10, 0], [20, 20], [10, 10])
        iterable_assert(self.assertAlmostEqual, c, [5, 5]) 
        self.assertAlmostEqual(r, 7.0710678118654755)
        
        c, r = extrude.solve_circle_tangent_point([10, 0], [20, 20], [0, 10])
        iterable_assert(self.assertAlmostEqual, c, [5, 5]) 
        self.assertAlmostEqual(r, 7.0710678118654755)

        c, r = extrude.solve_circle_tangent_point([10, 0], [10, 10], [0, 0])
        iterable_assert(self.assertAlmostEqual, c, [5, 5]) 
        self.assertAlmostEqual(r, 7.0710678118654755)
        
    def makeArcLinearTestObject(self, scale=1):
        return extrude.LinearExtrude(
            extrude.PathBuilder()
                .move([0, 0])
                .line([100 * scale, 0], 'linear')
                .arc_tangent_point([20 * scale, 100 * scale],
                         name='arc', degrees=90)
                .line([0, 100 * scale], 'linear2')
                .line([0, 0], 'linear3')
                .build(),
            h=40,
            fn=30,
            twist=0,
            scale=(1, 1)
            )
        
    def testArcLinearTestObject(self):
        le = self.makeArcLinearTestObject()
        self.write(le, 'ArcLinear')
        iterable_assert(self.assertAlmostEqual, le.at('linear', 0.5).A, 
                        [[ 1.,  0.,  0., 50.],
                         [ 0.,  0., -1.,  0.],
                         [ 0.,  1.,  0.,  0.],
                         [ 0.,  0.,  0.,  1.]])
        
        
    def makeArcArcExtrudeTestObject(self, scale=1):
        return extrude.RotateExtrude(
            extrude.PathBuilder()
                .move([0, 0])
                .line([100 * scale, 0], 'linear')
                .arc_tangent_point([20 * scale, 100 * scale],
                         name='arc', degrees=90)
                .line([0, 100 * scale], 'linear2')
                .line([0, 0], 'linear3')
                .build(),
            degrees = 90
            )
        
    def testArcArcExtrudeTestObject(self):
        le = self.makeArcArcExtrudeTestObject()
        self.write(le, 'ArcRotate')
        iterable_assert(self.assertAlmostEqual, le.at('linear', 0.5).A, 
                        [[ 0.,  1.,  0., 50.],
                         [ 1.,  0.,  0.,  0.],
                         [ 0.,  0., -1.,  0.],
                         [ 0.,  0.,  0.,  1.]])
        
        
    def testArcTangentPoint(self):
        SCALE=0.8
        
        path = (extrude.PathBuilder()
            .move([0, 0])
            .line([100 * SCALE, 0], 'linear')
            .arc_tangent_point([0 * SCALE, 100 * SCALE], name='curve', degrees=90)
            .line([0, 100 * SCALE], 'linear2')
            .line([0, 0], 'linear3')
            .build())
        
        iterable_assert(self.assertAlmostEqual, path.polygons(TestMetaData()),
                        ([[ 0.        ,  0.        ],
                           [80.        ,  0.        ],
                           [79.01506725, 12.5147572 ],
                           [76.0845213 , 24.72135955],
                           [71.28052194, 36.31923998],
                           [64.72135955, 47.02282018],
                           [56.56854249, 56.56854249],
                           [47.02282018, 64.72135955],
                           [36.31923998, 71.28052194],
                           [24.72135955, 76.0845213 ],
                           [12.5147572 , 79.01506725],
                           [ 0.        , 80.        ],
                           [ 0.        , 80.        ],
                           [ 0.        ,  0.        ]],))
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
