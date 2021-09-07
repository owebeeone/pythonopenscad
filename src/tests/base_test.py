'''
Basic set of tests.

'''
import unittest
import pythonopenscad as base


class Test(unittest.TestCase):

    # Check for duplicate names.
    def make_broken_duplicate_arg_named_specifier(self):
        base.OpenScadApiSpecifier('translate', (
            base.Arg('v',
                     base.VECTOR3_FLOAT,
                     None,
                     '(x,y,z) translation vector.',
                     required=True),
            base.Arg('v',
                     base.VECTOR3_FLOAT,
                     None,
                     '(x,y,z) translation vector.',
                     required=True),
        ), 'url')

    def testDuplicateNameSpecifier(self):
        self.assertRaisesRegex(base.DuplicateNamingOfArgs,
                               'Duplicate parameter names \\[\'v\'\\]',
                               self.make_broken_duplicate_arg_named_specifier)

    def testLinear_ExtrudeConstruction(self):
        obj = base.Linear_Extrude(scale=10)
        self.assertEqual(repr(obj),
                         'linear_extrude(height=100.0, scale=10.0)\n')

    def testCylinderConstruction(self):
        obj = base.Cylinder(10, 11)
        self.assertEqual(obj.h, 10, "Expected h 10")
        self.assertEqual(obj.r, 11, "Expected r 11")
        self.assertEqual(obj.get_r1(), 11, "Expected computed actual radius.")

        obj = base.Cylinder(10, 11, d=10)
        self.assertEqual(obj.h, 10, "Expected h 10")
        self.assertEqual(obj.r, 11, "Expected r 11")
        self.assertEqual(obj.get_r1(), 5, "Expected computed actual radius.")

    def testPoscModifiers(self):
        pm = base.PoscModifiers()
        self.assertFalse(pm.get_modifiers())
        self.assertRaisesRegex(base.InvalidModifier, '.*x.*not a valid.*',
                               pm.add_modifier, 'x')
        pm.add_modifier(base.DEBUG)
        self.assertEqual(pm.get_modifiers(), '#')
        pm.add_modifier(base.SHOW_ONLY)
        self.assertEqual(pm.get_modifiers(), '!#')
        pm.remove_modifier(base.DEBUG)
        self.assertEqual(pm.get_modifiers(), '!')

        obj = base.Cylinder(10, 11)
        obj.add_modifier(base.SHOW_ONLY)

    def testDocument_init_decorator_compains_about_init(self):
        class Z(base.PoscBase):
            OSC_API_SPEC = base.OpenScadApiSpecifier(
                'zzz', (base.Arg('v', int, None, 'some value'), ), 'url')

            def __init__(self):
                pass

        self.assertRaisesRegex(base.InitializerNotAllowed,
                               'class Z should not define __init__',
                               base.apply_posc_attributes, Z)

    def testtestDocument_init_decorator(self):
        @base.apply_posc_attributes
        class Z(base.PoscBase):
            OSC_API_SPEC = base.OpenScadApiSpecifier(
                'zzz', (base.Arg('v', int, None, 'some value'), ), 'url')

        self.assertRegex(Z.__doc__, 'zzz',
                         'init_decorator failed to add docstring.')

    def testCodeDumper(self):
        cd = base.CodeDumper()
        self.assertRaisesRegex(base.IndentLevelStackEmpty,
                               'Empty indent level stack cannot be popped\.',
                               cd.pop_indent_level)
        line = 'A line\n'
        cd.write_line(line[:-1])
        self.assertEquals(cd.writer.get(), line)

        cd = base.CodeDumper()
        cd.push_increase_indent()
        cd.write_line(line[:-1])
        self.assertEquals(cd.writer.get(), '  ' + line)

        cd.pop_indent_level()
        cd.write_line(line[:-1])
        self.assertEquals(cd.writer.get(), '  ' + line + line)

        self.assertRaisesRegex(base.IndentLevelStackEmpty,
                               'Empty indent level stack cannot be popped\.',
                               cd.pop_indent_level)

    def testCodeDumper_Function(self):
        cd = base.CodeDumper()
        cd.write_function('fname', ['a=1', 'b=2'])
        expected = 'fname(a=1, b=2)'
        self.assertEquals(cd.writer.get(), expected + ';\n')

        cd = base.CodeDumper()
        cd.push_increase_indent()
        cd.write_function('fname', ['a=1', 'b=2'])
        self.assertEquals(cd.writer.get(), '  ' + expected + ';\n')

        cd = base.CodeDumper()
        cd.push_increase_indent()
        cd.write_function('fname', ['a=1', 'b=2'], mod_prefix='!*')
        self.assertEquals(cd.writer.get(), '  !*' + expected + ';\n')

        cd = base.CodeDumper()
        cd.push_increase_indent()
        cd.write_function('fname', ['a=1', 'b=2'], mod_prefix='!*', suffix='')
        self.assertEquals(cd.writer.get(), '  !*' + expected + '\n')

    def testDumper(self):
        obj = base.Cylinder(10, 11)
        obj.add_modifier(base.DEBUG)
        cd = base.CodeDumper()
        obj.code_dump(cd)
        self.assertEquals(cd.writer.get(),
                          '#cylinder(h=10.0, r=11.0, center=false);\n')

    def testTranslate(self):
        obj = base.Translate((10, ))
        self.assertEquals(str(obj), 'translate(v=[10.0, 0.0, 0.0]);\n')

    def testTranslateCylinder(self):
        obj = base.Cylinder(10, 11).translate((10, ))
        self.assertEquals(
            str(obj),
            'translate(v=[10.0, 0.0, 0.0]) {\n  cylinder(h=10.0, r=11.0, center=false);\n}\n'
        )

    def testPassingByName(self):
        obj = base.Cylinder(h=10, r=11)

    def test_list_of(self):
        v = base.list_of(base.list_of(int, fill_to_min=0),
                         fill_to_min=[1, 1, 1])([[0]])
        self.assertEquals(repr(v), '[[0, 0, 0], [1, 1, 1], [1, 1, 1]]')

    def testRotateA_AX(self):
        obj = base.Rotate(10)
        self.assertEquals(str(obj), 'rotate(a=10.0);\n')
        obj = base.Rotate(10, [1, 1, 1])
        self.assertEquals(str(obj), 'rotate(a=10.0, v=[1.0, 1.0, 1.0]);\n')

    def testRotateA3(self):
        obj = base.Rotate([10])
        self.assertEquals(str(obj), 'rotate(a=[10.0, 0.0, 0.0]);\n')

    def test_List_of(self):
        converter = base.list_of(int, len_min_max=(None, None))
        self.assertEquals(converter([]), [])
        self.assertEquals(converter([1.0]), [1])
        self.assertEquals(converter([1.0] * 100), [1] * 100)

    def test_of_set(self):
        converter = base.of_set('a', 'b')
        self.assertRaisesRegex(base.InvalidValue,
                               '\'c\' is not allowed with .*', converter, 'c')
        self.assertEquals(converter('a'), 'a')
        self.assertEquals(converter('b'), 'b')

    def test_osc_true_false(self):
        self.assertFalse(base.OSC_FALSE)
        self.assertTrue(base.OSC_TRUE)

    def test_offset(self):
        self.assertEqual(base.Offset(r=3.0).r, 3.0)
        self.assertEqual(base.Offset().r, 1.0)

    def testModifiers(self):
        obj = base.Cylinder()
        self.assertFalse(obj.has_modifier(base.DEBUG))
        obj = base.Cylinder().add_modifier(base.DEBUG, base.TRANSPARENT)
        self.assertEquals(obj.get_modifiers(), '#%')
        obj.remove_modifier(base.DEBUG)
        self.assertEquals(obj.get_modifiers(), '%')
        self.assertEquals(str(obj), '%cylinder(h=1.0, r=1.0, center=false);\n')
        self.assertEquals(
            repr(obj),
            'cylinder(h=1.0, r=1.0, center=False).add_modifier(*{TRANSPARENT})\n'
        )
        self.assertFalse(obj.has_modifier(base.DEBUG))
        self.assertTrue(obj.has_modifier(base.TRANSPARENT))
        
    def testMetadataName(self):
        obj = base.Sphere()
        self.assertEquals(str(obj), 'sphere(r=1.0);\n')
        obj.setMetadataName("a_name")
        self.assertEquals(str(obj), "// 'a_name'\nsphere(r=1.0);\n")
        obj.setMetadataName(('a', 'tuple'))
        self.assertEquals(str(obj), "// ('a', 'tuple')\nsphere(r=1.0);\n")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
