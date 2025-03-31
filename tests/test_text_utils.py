import unittest
from pythonopenscad.text_utils import CubicSpline, QuadraticSpline, extentsof
from anchorscad_lib.test_tools import iterable_assert
import numpy as np


class TestTextUtils(unittest.TestCase):
    CUBIC_POINTS1 = [
        (0, -1),
        (1, -1),
        (1, 1),
        (0, 1),
    ]
    QUADRATIC_POINTS1 = [
        (0, -1),
        (1, 0),
        (0, 1),
    ]

    def test_cubic_spline(self):
        cubic = CubicSpline(self.CUBIC_POINTS1)
        iterable_assert(self.assertAlmostEqual, cubic.evaluate(0), (0, -1))
        iterable_assert(self.assertAlmostEqual, cubic.evaluate(1), (0, 1))
        iterable_assert(self.assertAlmostEqual, cubic.evaluate(0.5), (0.75, 0))

        iterable_assert(
            self.assertAlmostEqual, cubic.evaluate([0, 0.5, 1]), ((0, -1), (0.75, 0), (0, 1))
        )

    def test_cubic_azimuth_t(self):
        cubic = CubicSpline(self.CUBIC_POINTS1)
        iterable_assert(self.assertAlmostEqual, cubic.azimuth_t(0), (0, 1))
        iterable_assert(self.assertAlmostEqual, cubic.azimuth_t(90), (0.5,))
        iterable_assert(self.assertAlmostEqual, cubic.azimuth_t(45), (0.19098300,))

    def test_cubic_derivative(self):
        cubic = CubicSpline(self.CUBIC_POINTS1)
        iterable_assert(self.assertAlmostEqual, cubic.derivative(0), (-3, 0))
        iterable_assert(self.assertAlmostEqual, cubic.derivative(1), (3, 0))
        iterable_assert(self.assertAlmostEqual, cubic.derivative(0.5), (0, -3))

        iterable_assert(
            self.assertAlmostEqual, cubic.derivative([0, 0.5, 1]), ((-3, 0), (0, -3), (3, 0))
        )

    def test_cubic_normal2d(self):
        cubic = CubicSpline(self.CUBIC_POINTS1)
        iterable_assert(self.assertAlmostEqual, cubic.normal2d(0), (0, 1))
        iterable_assert(self.assertAlmostEqual, cubic.normal2d(1), (0, -1))
        iterable_assert(self.assertAlmostEqual, cubic.normal2d(0.5), (-1, 0))

        iterable_assert(
            self.assertAlmostEqual, cubic.normal2d([0, 0.5, 1]), ((0, 1), (-1, 0), (0, -1))
        )

    def test_quadratic_spline(self):
        quadratic = QuadraticSpline(self.QUADRATIC_POINTS1)
        iterable_assert(self.assertAlmostEqual, quadratic.evaluate(0), (0, -1))
        iterable_assert(self.assertAlmostEqual, quadratic.evaluate(1), (0, 1))
        iterable_assert(self.assertAlmostEqual, quadratic.evaluate(0.5), (0.5, 0))

        iterable_assert(
            self.assertAlmostEqual, quadratic.evaluate([0, 0.5, 1]), ((0, -1), (0.5, 0), (0, 1))
        )

    def test_quadratic_azimuth_t(self):
        quadratic = QuadraticSpline(self.QUADRATIC_POINTS1)
        iterable_assert(self.assertAlmostEqual, quadratic.azimuth_t(45), (0.5,))
        iterable_assert(self.assertAlmostEqual, quadratic.azimuth_t(90), (1,))

    def test_quadratic_derivative(self):
        quadratic = QuadraticSpline(self.QUADRATIC_POINTS1)
        iterable_assert(self.assertAlmostEqual, quadratic.derivative(0), (2, 2))
        iterable_assert(self.assertAlmostEqual, quadratic.derivative(1), (-2, 2))
        iterable_assert(self.assertAlmostEqual, quadratic.derivative(0.5), (0, 2))

        iterable_assert(
            self.assertAlmostEqual, quadratic.derivative([0, 0.5, 1]), ((2, 2), (0, 2), (-2, 2))
        )

    def test_quadratic_normal2d(self):
        quadratic = QuadraticSpline(self.QUADRATIC_POINTS1)
        sqrt2 = 1 / np.sqrt(2)
        iterable_assert(self.assertAlmostEqual, quadratic.normal2d(0), (sqrt2, -sqrt2))
        iterable_assert(self.assertAlmostEqual, quadratic.normal2d(1), (sqrt2, sqrt2))
        iterable_assert(self.assertAlmostEqual, quadratic.normal2d(0.5), (1, 0))

        iterable_assert(
            self.assertAlmostEqual,
            quadratic.normal2d([0, 0.5, 1]),
            ((sqrt2, -sqrt2), (1, 0), (sqrt2, sqrt2)),
        )

    def test_cubic_extentsof(self):
        cubic = CubicSpline(self.CUBIC_POINTS1)
        iterable_assert(self.assertAlmostEqual, cubic.extents(), [[0.0, -1.0], [0.75, 1.0]])

    def test_quadratic_extentsof(self):
        quadratic = QuadraticSpline(self.QUADRATIC_POINTS1)
        iterable_assert(self.assertAlmostEqual, quadratic.extents(), [[0.0, -1.0], [0.5, 1.0]])


if __name__ == "__main__":
    unittest.main()
