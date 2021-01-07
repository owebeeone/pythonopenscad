'''
Created on 17 May 2020

@author: gianni
'''
import unittest
from ParametricSolid import renderer as r
from ParametricSolid import posc_impl



class RendererTest(unittest.TestCase):


    def testName(self):
        impl = posc_impl.PoscImpl()
        renderer = r.RenderContext(impl)
        
        box1 = renderer.box([2,3,4])
        box2 = renderer.box([1,2,3])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
