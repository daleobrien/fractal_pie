#!/usr/bin/env python

import png
from itertools import izip_longest
import numpy as np


def find_plane(data,
               output,
               x_offset=0,
               y_offset=0,
               n=0):
    '''
    For a given sub section of the image, find a plane which best fits the
    pixel data.  E.g. Least mean square,  pixel = a y + b x + c

    '''

    # assumes square data, but the solution in the paper is generic
    #http://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf
    m = n - 1

    # these simplifications assume the sub-image is square
    sn = n * n
    sxx = m * (2 * n - 1) * sn / 6
    syx = m * m * sn / 4
    sx = m * sn / 2
    sn = n * n

    sxz = 0
    syz = 0
    sz = 0

    for i, row in enumerate(data[x_offset:x_offset + n]):
        for j, z in enumerate(row[y_offset:y_offset + n]):
            sxz += i * z
            syz += j * z
            sz += z

    A = ((sxx, syx, sx), (syx, sxx, sx), (sx, sx, sn))
    B = (sxz, syz, sz)

    a, b, c = [int(_x + 0.499999) for _x in np.linalg.lstsq(A, B)[0]]

    error = 0
    for i, row in enumerate(data[x_offset:x_offset + n]):
        for j, z in enumerate(row[y_offset:y_offset + n]):

            new_z = min(max(int(a * i + b * j + c), 0), 255)
            output[i + x_offset][j + y_offset] = new_z

            e = (new_z - z)
            error += (e * e)

    error /= (n * n)

    return (a, b, c, error)


def split_range_into_quad(x, y, n):
    '''for now, assume source image is always square'''
    m = int((n + 1.0) / 2.0)
    return ((x, y, m),
            (x, y + m, m),
            (x + m, y, m),
            (x + m, y + m, m))


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def check_quadrant(input_image, output, x, y, n, tree, parameters, max_error):

    a, b, c, error = find_plane(input_image, output, x, y, n)

    if error > max_error:
        tree.append(1)

        for _x, _y, _n in split_range_into_quad(x, y, n):
            check_quadrant(input_image,
                           output, _x, _y, _n, tree,
                           parameters, max_error)

    tree.append(0)
    [parameters.append(p) for p in (a, b, c)]


def compress(input_image_name, output_image_name):
    print 'processing ', input_image_name

    # Load image
    grey = []
    f = open(input_image_name, 'r')
    r = png.Reader(f)
    (x, y, data, details) = r.read()
    planes = details['planes']

    # get grey data
    for row in data:
        # make new image, but make it grey
        # might be a better way of making it grey too
        if planes == 3:
            grey.append([(r + g + b) / 3 for r, g, b in grouper(3, row)])
        elif planes == 4:
            grey.append([(r + g + b) / 3 for r, g, b, _ in grouper(4, row)])

    f.close()

    x, y, n = 0, 0, len(grey)
    output = [[0 for _ in range(n)] for _ in range(n)]

    tree = []
    parameters = []

    check_quadrant(grey, output, x, y, n, tree, parameters, 32)

    approximate_compressed_size = len(tree) / 8 + len(parameters) + 2

    print 'maybe need around %d bytes to store compressed image (greyscale)' %\
        approximate_compressed_size

    # write it out
    f = open(output_image_name, 'wb')
    w = png.Writer(512, 512, greyscale=True)
    w.write(f, output)
    f.close()


if __name__ == "__main__":

    compress('lena.png', 'output_lena.png')

#
