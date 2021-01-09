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
    exception: object
    
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
        except (BaseException, AssertionError) as ex:
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
        msg = f'depth={e.depth!r}\nva={va!r}\nvb{vb!r}'
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
        path = builder.build(TestMetaData())
        expected = np.array([[ 0.   ,  0.   ],
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
        
        iterable_assert(self.assertAlmostEqual, expected, path.points)
        
        node = path.get_node('curve')
        assert not node is None
        
        iterable_assert(self.assertAlmostEqual, node.extents(), [[0., -0.28], [ 0.75, 1]])
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()