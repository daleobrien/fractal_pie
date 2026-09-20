fractal_pie
===========

Just playing around with image compression. Written in Rust.

The program fits a plane, described by `pixel = a*i + b*j + c` (where `i` and
`j` are the row and column offsets within a sub-image), through the image data
by least squares. If the mean-squared error is too great with respect to the
raw image, the image is broken up into 4 sub-images and the process is repeated
on each one. The resulting quadtree of planes is the compressed representation.

## Colour

Colour images are converted to YCbCr and encoded as 4:2:0. The luma plane is
fitted at full resolution with the same error bound as the greyscale image
(`MAX_ERROR = 32`). The chroma planes (`Cb`, `Cr`) are averaged over 2x2
blocks, quartering their size, and fitted with a looser bound
(`CHROMA_MAX_ERROR = 64`): we do not have to fit chroma as tightly, since the
eye is far less sensitive to colour detail than to brightness.

Running,

    cargo run --release

will take this file,

![Lena](https://raw.githubusercontent.com/daleobrien/fractal_pie/master/lena.png)

and produce this file,
```
  processing  lena.png
  maybe need around 131172 bytes to store compressed image (colour)
  compression ratio: 6.00:1 (786432 bytes raw -> 131172 bytes)
```
![Colour](https://raw.githubusercontent.com/daleobrien/fractal_pie/master/output_lena.png)

Input and output paths can be overridden, and both error bounds are optional
flags:

    cargo run --release -- [input.png] [output.png] [--max-error N] [--chroma-max-error N]

`--max-error` is the luma (and greyscale) bound; `--chroma-max-error` is the
chroma bound. A larger bound tolerates a coarser fit, so fewer regions are
subdivided and the file gets smaller. For the 512x512 Lena image:

| luma | chroma | bytes  | ratio  |
| ---- | ------ | ------ | ------ |
| 32   | 16     | 160462 | 4.90:1 |
| 32   | 32     | 141335 | 5.56:1 |
| 32   | 64     | 131172 | 6.00:1 |
| 32   | 128    | 127246 | 6.18:1 |
| 32   | 256    | 126261 | 6.23:1 |
| 64   | 64     | 80849  | 9.73:1 |
| 16   | 64     | 216403 | 3.63:1 |

The luma plane dominates: once the chroma bound reaches about 128 the chroma
planes cost next to nothing.

The input must be a square PNG. Already-greyscale inputs skip the YCbCr step
and are encoded directly:

```
  processing  grey.png
  maybe need around 128035 bytes to store compressed image (greyscale)
  compression ratio: 2.05:1 (262144 bytes raw -> 128035 bytes)
```

## Why it is fast

The original Python prototype took roughly 39 seconds for the 512x512 Lena
image. This port does the same work in about 10 milliseconds — the arithmetic
is identical (the output image and the size estimate match byte for byte), but
the per-pixel loops run as native code and the quadtree is traversed in
parallel with [rayon](https://docs.rs/rayon).

## Tests

    cargo test --release

That is all.
