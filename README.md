fractal_pie
===========

Just playing around with image compression. Written in Rust.

The program fits a plane, described by `pixel = a*i + b*j + c` (where `i` and
`j` are the row and column offsets within a sub-image), through the image data
by least squares. If the mean-squared error is too great with respect to the
raw image, the image is broken up into 4 sub-images and the process is repeated
on each one. The resulting quadtree of planes is the compressed representation.

Running,

    cargo run --release

will take this file,

![Lena](https://raw.githubusercontent.com/daleobrien/fractal_pie/master/lena.png)

and produce this file,
```
  processing  lena.png
  maybe need around 128035 bytes to store compressed image (greyscale)
  compression ratio: 2.05:1 (262144 bytes raw -> 128035 bytes)
```
![Grey](https://raw.githubusercontent.com/daleobrien/fractal_pie/master/output_lena.png)

Input and output paths can be overridden:

    cargo run --release -- input.png output.png

The input must be a square PNG; it is converted to 8-bit greyscale first.

## Why it is fast

The original Python prototype took roughly 39 seconds for the 512x512 Lena
image. This port does the same work in about 10 milliseconds — the arithmetic
is identical (the output image and the size estimate match byte for byte), but
the per-pixel loops run as native code and the quadtree is traversed in
parallel with [rayon](https://docs.rs/rayon).

## Tests

    cargo test --release

That is all.
