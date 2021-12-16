'''
Created on 8 Dec 2021

@author: gianni
'''

import unittest
from ParametricSolid.datatree import datatree, args, override, Node
from dataclasses import dataclass

@datatree
class LeafType1():
    
    leaf_a: float=1
    leaf_b: float=2
    
@datatree
class LeafType2():
    
    leaf_a: float=10
    leaf_b: float=20

@datatree
class LeafType3():
    
    leaf_a: float=11
    leaf_b: float=22
    
@datatree
class LeafType4():
    
    leaf_a: float=111
    leaf_b: float=222
    
# Check for support of non dataclass/datatree types.
class LeafType5():
    
    def __init__(self, a=1111, b=2222):
        self.a = a
        self.b = b
        
    def __eq__(self, other):
        if isinstance(other, LeafType5):
            return self.a == other.a and self.b == other.b
        return False

@datatree
class Overridable:
    
    leaf_a: float=53  # Overrides the default value for all classes.
    
    # Only the nominated fields are mapped and leaf1_b
    leaf1: Node=Node(LeafType1, 'leaf_a', {'leaf_b': 'leaf1_b'})
    leaf1a: Node=Node(LeafType1, 'leaf_a', {'leaf_b': 'leaf1a_b'})
   
    leaf2: Node=Node(LeafType2) # All fields are mapped.
    leaf3: Node=Node(LeafType3, {}) # No fields are mapped.
    leaf4: Node=Node(LeafType4, use_defaults=False) # Default values are mapped from this.
    leaf5: Node=Node(LeafType5)
    
    def __post_init__(self):
        
        self.l1 = self.leaf1(leaf_a=99)
        self.l1a = self.leaf1a()
        self.l2 = self.leaf2()
        self.l3 = self.leaf3()
        self.l4 = self.leaf4()
        self.l5 = self.leaf5(b=3333)
        

OVERRIDER1=Overridable(
    leaf_a=3, 
    leaf1_b=44,
    override=override(
        leaf1=args(leaf_b=7)),
    )

class Test(unittest.TestCase):

    def test_l1(self):
        self.assertEqual(OVERRIDER1.l1, LeafType1(leaf_a=99, leaf_b=7))
        
    def test_l1a(self):
        self.assertEqual(OVERRIDER1.l1a, LeafType1(leaf_a=3, leaf_b=2))
        
    def test_l2(self):
        self.assertEqual(OVERRIDER1.l2, LeafType2(leaf_a=3, leaf_b=20))
        
    def test_l3(self):
        self.assertEqual(OVERRIDER1.l3, LeafType3(leaf_a=11, leaf_b=22))
        
    def test_l4(self):
        self.assertEqual(OVERRIDER1.l4, LeafType4(leaf_a=3, leaf_b=20))
        
    def test_l5(self):
        self.assertEqual(OVERRIDER1.l5, LeafType5(a=1111, b=3333))
        
    def test_inherits(self):
        
        @dataclass
        class A:
            a: int = 1
            
        @datatree
        class B(A):
            b: int = 2
            
        ab = B(10, 20)
        self.assertEqual(ab.a, 10)
        self.assertEqual(ab.b, 20)
        
    def test_name_map(self):
        
        @datatree
        class A:
            anode: Node=Node(LeafType1, {'leaf_a': 'aa'}, expose_all=True)
            
            def __post_init__(self):
                self.lt1 = self.anode()
            
        ao = A()
        self.assertEqual(ao.lt1, LeafType1())
        self.assertEqual(ao.aa, LeafType1().leaf_a)
        self.assertEqual(ao.leaf_b, LeafType1().leaf_b)
        
    def test_ignore_default_bad(self):
        
        try:
            @datatree
            class A:
                anode: Node=Node(LeafType1, use_defaults=False)
            self.fail("Expected field order issue.")
        except TypeError:
            pass
        
    def test_ignore_default(self):
        @datatree
        class A:
            anode: Node=Node(LeafType1)
            leaf_a: float=51
            leaf_b: float
            
        lt1 = A().anode()
        
        self.assertEqual(lt1, LeafType1(51, 2))
      
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()