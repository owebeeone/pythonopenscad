'''
Basic set of PythonOpenScad tests.

'''

import unittest
import pythonopenscad as base


class Test(unittest.TestCase):
    # Check for duplicate names.
    def make_broken_duplicate_arg_named_specifier(self):
        base.OpenScadApiSpecifier(
            'translate',
            (
                base.Arg(
                    'v', base.VECTOR3_FLOAT, None, '(x,y,z) translation vector.', required=True
                ),
                base.Arg(
                    'v', base.VECTOR3_FLOAT, None, '(x,y,z) translation vector.', required=True
                ),
            ),
            'url',
        )

    def testDuplicateNameSpecifier(self):
        self.assertRaisesRegex(
            base.DuplicateNamingOfArgs,
            'Duplicate parameter names \\[\'v\'\\]',
            self.make_broken_duplicate_arg_named_specifier,
        )

    def testLinear_ExtrudeConstruction(self):
        obj = base.Linear_Extrude(scale=10)
        self.assertEqual(repr(obj), 'linear_extrude(height=100.0, scale=10.0)\n')

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
        self.assertRaisesRegex(base.InvalidModifier, '.*x.*not a valid.*', pm.add_modifier, 'x')
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
                'zzz', (base.Arg('v', int, None, 'some value'),), 'url'
            )

            def __init__(self):
                pass

        self.assertRaisesRegex(
            base.InitializerNotAllowed,
            'class Z should not define __init__',
            base.apply_posc_attributes,
            Z,
        )

    def testtestDocument_init_decorator(self):
        @base.apply_posc_attributes
        class Z(base.PoscBase):
            OSC_API_SPEC = base.OpenScadApiSpecifier(
                'zzz', (base.Arg('v', int, None, 'some value'),), 'url'
            )

        self.assertRegex(Z.__doc__, 'zzz', 'init_decorator failed to add docstring.')

    def testCodeDumper(self):
        cd = base.CodeDumper()
        self.assertRaisesRegex(
            base.IndentLevelStackEmpty,
            'Empty indent level stack cannot be popped.',
            cd.pop_indent_level,
        )
        line = 'A line\n'
        cd.write_line(line[:-1])
        self.assertEqual(cd.writer.get(), line)

        cd = base.CodeDumper()
        cd.push_increase_indent()
        cd.write_line(line[:-1])
        self.assertEqual(cd.writer.get(), '  ' + line)

        cd.pop_indent_level()
        cd.write_line(line[:-1])
        self.assertEqual(cd.writer.get(), '  ' + line + line)

        self.assertRaisesRegex(
            base.IndentLevelStackEmpty,
            'Empty indent level stack cannot be popped.',
            cd.pop_indent_level,
        )

    def testCodeDumper_Function(self):
        cd = base.CodeDumper()
        cd.write_function('fname', ['a=1', 'b=2'])
        expected = 'fname(a=1, b=2)'
        self.assertEqual(cd.writer.get(), expected + ';\n')

        cd = base.CodeDumper()
        cd.push_increase_indent()
        cd.write_function('fname', ['a=1', 'b=2'])
        self.assertEqual(cd.writer.get(), '  ' + expected + ';\n')

        cd = base.CodeDumper()
        cd.push_increase_indent()
        cd.write_function('fname', ['a=1', 'b=2'], mod_prefix='!*')
        self.assertEqual(cd.writer.get(), '  !*' + expected + ';\n')

        cd = base.CodeDumper()
        cd.push_increase_indent()
        cd.write_function('fname', ['a=1', 'b=2'], mod_prefix='!*', suffix='')
        self.assertEqual(cd.writer.get(), '  !*' + expected + '\n')

    def testDumper(self):
        obj = base.Cylinder(10, 11)
        obj.add_modifier(base.DEBUG)
        cd = base.CodeDumper()
        obj.code_dump(cd)
        self.assertEqual(cd.writer.get(), '#cylinder(h=10.0, r=11.0, center=false);\n')

    def testTranslate(self):
        obj = base.Translate((10,))
        self.assertEqual(str(obj), 'translate(v=[10.0, 0.0, 0.0]);\n')

    def testTranslateCylinder(self):
        obj = base.Cylinder(10, 11).translate((10,))
        self.assertEqual(
            str(obj),
            'translate(v=[10.0, 0.0, 0.0]) {\n  cylinder(h=10.0, r=11.0, center=false);\n}\n',
        )

    def testPassingByName(self):
        obj = base.Cylinder(h=10, r=11)

    def test_list_of(self):
        v = base.list_of(base.list_of(int, fill_to_min=0), fill_to_min=[1, 1, 1])([[0]])
        self.assertEqual(repr(v), '[[0, 0, 0], [1, 1, 1], [1, 1, 1]]')

    def testRotateA_AX(self):
        obj = base.Rotate(10)
        self.assertEqual(str(obj), 'rotate(a=10.0);\n')
        obj = base.Rotate(10, [1, 1, 1])
        self.assertEqual(str(obj), 'rotate(a=10.0, v=[1.0, 1.0, 1.0]);\n')

    def testRotateA3(self):
        obj = base.Rotate([10])
        self.assertEqual(str(obj), 'rotate(a=[10.0, 0.0, 0.0]);\n')

    def test_List_of(self):
        converter = base.list_of(int, len_min_max=(None, None))
        self.assertEqual(converter([]), [])
        self.assertEqual(converter([1.0]), [1])
        self.assertEqual(converter([1.0] * 100), [1] * 100)

    def test_of_set(self):
        converter = base.of_set('a', 'b')
        self.assertRaisesRegex(base.InvalidValue, '\'c\' is not allowed with .*', converter, 'c')
        self.assertEqual(converter('a'), 'a')
        self.assertEqual(converter('b'), 'b')

    def test_osc_true_false(self):
        self.assertFalse(base.OSC_FALSE)
        self.assertTrue(base.OSC_TRUE)

    def test_offset(self):
        self.assertEqual(base.Offset(r=3.0).r, 3.0)
        self.assertEqual(base.Offset().r, 1.0)

    def testModifiers(self):
        obj = base.Cylinder(h=1)
        self.assertFalse(obj.has_modifier(base.DEBUG))
        obj = base.Cylinder(h=1).add_modifier(base.DEBUG, base.TRANSPARENT)
        self.assertEqual(obj.get_modifiers(), '#%')
        obj.remove_modifier(base.DEBUG)
        self.assertEqual(obj.get_modifiers(), '%')
        self.assertEqual(str(obj), '%cylinder(h=1.0, r=1.0, center=false);\n')
        self.assertEqual(
            repr(obj), 'cylinder(h=1.0, r=1.0, center=False).add_modifier(*{TRANSPARENT})\n'
        )
        self.assertFalse(obj.has_modifier(base.DEBUG))
        self.assertTrue(obj.has_modifier(base.TRANSPARENT))

    def testMetadataName(self):
        obj = base.Sphere()
        self.assertEqual(str(obj), 'sphere(r=1.0);\n')
        obj.setMetadataName("a_name")
        self.assertEqual(str(obj), "// 'a_name'\nsphere(r=1.0);\n")
        obj.setMetadataName(('a', 'tuple'))
        self.assertEqual(str(obj), "// ('a', 'tuple')\nsphere(r=1.0);\n")

    def testFill(self):
        obj1 = base.Circle(r=10)
        obj2 = base.Circle(r=5)

        result = base.difference()(obj1, obj2).fill()

        self.assertEqual(
            str(result),
            '\n'.join(
                (
                    'fill() {',
                    '  difference() {',
                    '    circle(r=10.0);',
                    '    circle(r=5.0);',
                    '  }',
                    '}\n',
                )
            ),
        )

    def testLazyUnion(self):
        obj1 = base.Circle(r=10)
        obj2 = base.Circle(r=5)

        result = base.lazy_union()(obj1, obj2)
        result.setMetadataName("a_name")

        self.assertEqual(
            repr(result), '''# 'a_name'\nlazy_union() (\n  circle(r=10.0),\n  circle(r=5.0)\n),\n'''
        )

        self.assertEqual(
            str(result),
            '\n'.join(
                (
                    '// Start: lazy_union',
                    'circle(r=10.0);',
                    'circle(r=5.0);',
                    '// End: lazy_union\n',
                )
            ),
        )

    def testModules(self):
        obj1 = base.module('obj1')(base.Circle(r=10))
        obj1.setMetadataName("obj 1")
        obj2 = base.module('obj2')(base.Circle(r=5))
        obj2.setMetadataName("obj 2")

        result = base.lazy_union()(obj1, obj2)
        result.setMetadataName("a_name")

        # Debug helper - uncomment to print the result.
        # print('str: ')
        # print(self.dump_str(str(result)))
        # print('repr: ')
        # print(self.dump_str(repr(result)))

        self.assertEqual(
            str(result),
            '\n'.join(
                [
                    '// Start: lazy_union',
                    'obj1();',
                    'obj2();',
                    '// End: lazy_union',
                    '',
                    '// Modules.',
                    '',
                    "// 'obj 1'",
                    'module obj1() {',
                    '  circle(r=10.0);',
                    '} // end module obj1',
                    '',
                    "// 'obj 2'",
                    'module obj2() {',
                    '  circle(r=5.0);',
                    '} // end module obj2',
                    '',
                ]
            ),
        )

        self.assertEqual(
            repr(result),
            '\n'.join(
                [
                    "# 'a_name'",
                    'lazy_union() (',
                    '  obj1();',
                    '  obj2();',
                    '),',
                    '',
                    '# Modules.',
                    '',
                    "# 'obj 1'",
                    'def obj1(): return (',
                    '  circle(r=10.0)',
                    '), # end module obj1',
                    '',
                    "# 'obj 2'",
                    'def obj2(): return (',
                    '  circle(r=5.0)',
                    '), # end module obj2',
                    '',
                ]
            ),
        )

    def testModules_nameCollision(self):
        obj1 = base.module('colliding_name')(base.Circle(r=10))
        obj1.setMetadataName("obj 1")
        obj2 = base.module('colliding_name')(base.Circle(r=5))
        obj2.setMetadataName("obj 2")

        result = base.lazy_union()(obj1, obj2)
        result.setMetadataName("a_name")

        # Debug helper - uncomment to print the result.
        # print('str: ')
        # print(self.dump_str(str(result)))
        # print('repr: ')
        # print(self.dump_str(repr(result)))

        self.assertEqual(
            str(result),
            '\n'.join(
                [
                    '// Start: lazy_union',
                    'colliding_name();',
                    'colliding_name_1();',
                    '// End: lazy_union',
                    '',
                    '// Modules.',
                    '',
                    "// 'obj 1'",
                    'module colliding_name() {',
                    '  circle(r=10.0);',
                    '} // end module colliding_name',
                    '',
                    "// 'obj 2'",
                    'module colliding_name_1() {',
                    '  circle(r=5.0);',
                    '} // end module colliding_name_1',
                    '',
                ]
            ),
        )

    def testModules_nameCollision_elided(self):
        obj1 = base.module('colliding_name')(base.Circle(r=10))
        obj1.setMetadataName("obj 1")
        obj2 = base.module('colliding_name')(base.Circle(r=10))
        obj2.setMetadataName("obj 2")

        result = base.lazy_union()(obj1, obj2)
        result.setMetadataName("a_name")

        # Debug helper - uncomment to print the result.
        # print('str: ')
        # print(self.dump_str(str(result)))
        # print('repr: ')
        # print(self.dump_str(repr(result)))

        self.assertEqual(
            str(result),
            '\n'.join(
                [
                    '// Start: lazy_union',
                    'colliding_name();',
                    'colliding_name();',
                    '// End: lazy_union',
                    '',
                    '// Modules.',
                    '',
                    "// 'obj 2'",
                    'module colliding_name() {',
                    '  circle(r=10.0);',
                    '} // end module colliding_name',
                    '',
                ]
            ),
        )

    def dump_str(self, s):
        return '[\n' + ',\n'.join([repr(l) for l in s.split('\n')]) + ']'


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
