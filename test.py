#!/usr/bin/env python

from compress import find_plane, split_range_into_quad
import unittest


class TestCompression(unittest.TestCase):

    def test_find_plane(self):

        d = [[3 * x + 2 * y + 5 for x in range(4)] for y in range(4)]

        out = [[0 for x in range(4)] for y in range(4)]

        a, b, c, error = find_plane(d, out, 0, 0, 4)
        print a, b, c, error

        self.assertEqual(error, 0)

        self.assertEqual(a, 2)
        self.assertEqual(b, 3)
        self.assertEqual(c, 5)

    def test_spit_range_into_quad(self):

        ranges = split_range_into_quad(8, 8, 4)

        self.assertEqual(((8, 8, 2),
                          (8, 10, 2),
                          (10, 8, 2),
                          (10, 10, 2)),
            ranges)


if __name__ == '__main__':
    unittest.main()
#
