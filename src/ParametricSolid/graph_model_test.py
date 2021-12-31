'''
Created on 31 Dec 2021

@author: gianni
'''
import unittest
from ParametricSolid.graph_model import DirectedGraph


class Test(unittest.TestCase):


    def test_simple_graph(self):
        dg = DirectedGraph()
        
        na = dg.new_node('a')
        nb = dg.new_node('b')
        nc = dg.new_node('c')
        nd = dg.new_node('d')
        
        dg.add_edge(na, nb)
        dg.add_edge(na, nc)
        dg.add_edge(nb, nd)
        dg.add_edge(nc, nd)
        
        self.assertEquals(dg.dump('foo'), 
                'digraph foo {\n'
                '    a_1 [label="a"];\n'
                '    b_2 [label="b"];\n'
                '    c_3 [label="c"];\n'
                '    d_4 [label="d"];\n'
                '    a_1 -> b_2;\n'
                '    a_1 -> c_3;\n'
                '    b_2 -> d_4;\n'
                '    c_3 -> d_4;\n'
                '}\n')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()