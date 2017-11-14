fractal_pie
===========

Just playing around with image compression.

The program fits a plane, described by pixel=ax+by+c, through the image data. If the total error is too great with respect to the raw image, the image is broken up into 4 sub-images and the process is repeated on each subimage.

Running,

   python compress.py

will take this file,

![Lena](https://raw.githubusercontent.com/daleobrien/fractal_pie/master/lena.png)

and produce this file,

![Grey](https://raw.githubusercontent.com/daleobrien/fractal_pie/raw/master/output_lena.png)

That is all.
