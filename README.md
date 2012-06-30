fractal_pie
===========

Just playing around with image compression.

The program fits a plane, described by pixel=ax+by+c, through the image data, and if the total error is too great from the raw image, the image is broken up into 4 sub-images and the process it repeated one each subimage.

   python compress.py

will take this file;

![Lena](/daleobrien/fractal_pie/raw/master/lena.png)

and produce this file;

![Grey](/daleobrien/fractal_pie/raw/master/output_lena.png)
