#!/usr/bin/env python3

import unittest
from helpers.layout import Region, RegionError, divide_interval
from sympy import Point


class TestRegionInit(unittest.TestCase):
    def test_invalid(self):
        with self.assertRaises(RegionError):
            Region(width=100, height=100, line_height=42,
                   sub_regions=[Region()])

    def test_missing_parameter(self):
        with self.assertRaises(RegionError):
            Region(width=100, height=100)

    def test_missing_direciton(self):
        with self.assertRaises(RegionError):
            Region(1.0, 1.0,
                   sub_regions=[Region(1.0, 1.0, line_height=20)])

    def test_valid(self):
        r = Region(100, 100, line_height=42)
        self.assertIsInstance(r, Region)

        r = Region(100, 100, direction='vertical',
                   sub_regions=[Region(100, 100, line_height=314)])
        self.assertIsInstance(r, Region)


class TestRegionIteration(unittest.TestCase):
    def test_simple(self):
        r = Region(10, 10, line_height=5, padding=1)
        r.fit_to_region((Point(0, 0), Point(10, 10)))
        l = list(iter(r))
        self.assertEqual(l, [(Point(1, 1), Point(9, 5))])

    def test_simple_division(self):
        r = Region(1.0, 1.0, direction='horizontal',
                   sub_regions=[
                       Region(0.5, 1.0, line_height=5),
                       Region(0.5, 1.0, line_height=5)])
        r.fit_to_region((Point(0, 0), Point(9, 9)))
        self.assertEqual(list(iter(r)),
                         [(Point(0, 0), Point(4, 4)),
                          (Point(0, 5), Point(4, 9)),
                          (Point(5, 0), Point(9, 4)),
                          (Point(5, 5), Point(9, 9))])

    def test_advanced_division(self):
        r = Region(1.0, 1.0, direction='horizontal',
                   sub_regions=[
                       Region(0.5, 1.0, line_height=10),
                       Region(0.5, 1.0, direction='vertical',
                              sub_regions=[
                                  Region(1.0, 0.5, line_height=2),
                                  Region(1.0, 0.5, line_height=4)])])
        r.fit_to_region((Point(0, 0), Point(9, 9)))
        self.assertEqual(list(iter(r)),
                         [(Point(0, 0), Point(4, 9)),
                          (Point(5, 0), Point(9, 1)),
                          (Point(5, 2), Point(9, 3)),
                          (Point(5, 5), Point(9, 8))])

    def test_bad_region_sizes(self):
        r = Region(1.0, 1.0, padding=2, direction='horizontal',
                   sub_regions=[
                       Region(0.5, 1.0, padding=1, line_height=26),
                       Region(0.5, 1.0, direction='vertical',
                              sub_regions=[
                                  Region(40, 40, line_height=20),
                                  Region(40, 60, line_height=20)])])
        r.fit_to_region((Point(0, 0), Point(50, 80)))
        self.assertEqual(list(iter(r)), [(Point(3, 3), Point(24, 28)),
                                         (Point(3, 29), Point(24, 54)),
                                         (Point(26, 2), Point(48, 21)),
                                         (Point(26, 22), Point(48, 41)),
                                         (Point(26, 42), Point(48, 61))])


class TestDivideInterval(unittest.TestCase):
    def setUp(self):
        self.lower = 0
        self.upper = 99

    def test_aggregated(self):
        test_cases = {
            'easy': {
                'in': [20, 30, 40],
                'out': [(0, 19), (20, 49), (50, 89)],
            },
            'percentage_ok': {
                'in': [0.5, 0.4],
                'out': [(0, 49), (50, 89)],
            },
            'percentage_bigger': {
                'in': [0.2, 0.8, 0.3],
                'out': [(0, 19), (20, 99), (99, 99)],
            },
            'one_too_big': {
                'in': [0.2, 0.3, 0.6],
                'out': [(0, 19), (20, 49), (50, 99)],
            },
            'mixed': {
                'in': [50, 0.2, 20, 0.2],
                'out': [(0, 49), (50, 69), (70, 89), (90, 99)],
            },
            'fill_in_rest': {
                'in': [1.0, 42, 31, 27],
                'out': [(0, 99), (99, 99), (99, 99), (99, 99)],
            },
        }

        for (k, v) in test_cases.items():
            with self.subTest(msg=k):
                r = divide_interval(self.lower, self.upper, v['in'])
                self.assertEqual(r, v['out'])

    def test_empty_interval(self):
        r = divide_interval(42, 42, [1.0, 20])
        self.assertEqual(r, [(42, 42), (42, 42)])


if __name__ == '__main__':
    unittest.main()
