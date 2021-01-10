'''
Created on 8 Jan 2021

@author: gianni
'''
from dataclasses import dataclass
from unittest import TestCase

import ParametricSolid.extrude as extrude
import numpy as np


@dataclass
class TestMetaData:
    fn: int=10

def is_iterable(v):
    try:
        gen = (d for d in v)
        return (gen,)
    except:
        pass
    return ()

@dataclass
class AssertionException(Exception):
    depth: tuple
    ex: object
    
class IterableAssert(Exception):
    '''Exception in iterable_assert'''


def _iterable_assert(expect_fun, va, vb, depth=()):
    ii_va = is_iterable(va)
    ii_vb = is_iterable(vb)

    try:
        both_true = ii_va and ii_vb
    except BaseException as ex:
        raise AssertionException(depth, ex)
    
    if both_true:
        try:
            assert len(va) == len(vb)
            for i, evab in enumerate(zip(va, vb)):
                eva, evb = evab
                _iterable_assert(expect_fun, eva, evb, depth + (i,))
        except AssertionException:
            raise
        except BaseException as ex:
            raise AssertionException(depth, ex)
    else:
        try:
            assert not ii_va and not ii_vb
            expect_fun(va, vb)
        except AssertionException:
            raise
        except (BaseException, AssertionError) as ex:
            raise AssertionException(depth, ex)

def iterable_assert(expect_fun, va, vb):
    try:
        _iterable_assert(expect_fun, va, vb)
    except AssertionException as e:
        msg = f'depth={e.depth!r}\nva={va!r}\nvb={vb!r}\n{e.ex}\n'
        raise IterableAssert(msg)


class Test(TestCase):


    def testBezierExtents2D(self):
        b = extrude.CubicSpline([[0, 0], [1, -1], [1, 1], [0, 1]])
        extents = b.extents()
        expects = [[0.0, 0.75], [-0.28, 1.0]]
        iterable_assert(self.assertAlmostEqual, b.extents(), [[0., -0.28], [ 0.75, 1]])
        minima = b.cuve_maxima_minima_t()
        self.assertAlmostEqual(b.normal2d(minima[0][0])[1], -1, 5)
        
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
                        np.array([[1.125    , 0.       ],
                                  [2.7763932, 3.4472136],
                                  [3.       , 3.       ]]))
        

    def testPolygonParams(self):
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
                           [1.08033762, 0.09607477],
                           [1.23453375, 0.35493251],
                           [1.44486332, 0.73252337],
                           [1.69360124, 1.18479752],
                           [1.96302245, 1.6677051 ],
                           [2.23540186, 2.13719627],
                           [2.4930144 , 2.5492212 ],
                           [2.71813499, 2.85973002],
                           [2.89303855, 3.0246729 ],
                           [3.        , 3.        ],
                           [1.        , 0.        ],
                           [2.        , 0.        ],
                           [2.        , 1.        ],
                           [1.        , 2.        ]]), 
                          ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 
                           (13, 14, 15))))

                
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
