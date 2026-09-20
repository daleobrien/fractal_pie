Fractal Pie
===========

Just playing around with image compression. Written in Rust.

The program fits a plane, described by `pixel = a*i + b*j + c` (where `i` and
`j` are the row and column offsets within a region), through the image data by
least squares. If the mean-squared error is too great with respect to the raw
image, the region is subdivided and the process is repeated on each child: a
region close to square is cut into four quadrants, while a long thin one is cut
in two along its longer side. Keeping the children near-square this way lets a
single tree cover an image of any dimensions. The resulting tree of planes is
the compressed representation, and it is written to a `.pie` file with an
adaptive range coder.

![Lena](https://raw.githubusercontent.com/daleobrien/fractal_pie/master/lena.png)

## Building and running

There are two binaries. `pie-encode` compresses an image into a `.pie` file:

    cargo run --release --bin pie-encode -- lena.png lena.pie

    processing  lena.png (  786432 bytes)
    wrote       lena.pie (   57369 bytes, colour (7.3%, 1.75 bpp))

`pie-decode` turns it back into an image (here a PNG):

    cargo run --release --bin pie-decode -- lena.pie decoded_lena.png

    decoding    lena.pie
    wrote       decoded_lena.png (512x512 colour)

![Colour](https://raw.githubusercontent.com/daleobrien/fractal_pie/master/output_lena.png)

The decoded image is bit-for-bit identical to the reference render above.

Input and output paths can be overridden, and both error bounds are optional
flags on the encoder:

    pie-encode [input] [output.pie] [--max-error N] [--chroma-max-error N]
    pie-decode [input.pie] [output]

`--max-error` is the luma (and greyscale) bound; `--chroma-max-error` is the
chroma bound. A larger bound tolerates a coarser fit, so fewer regions are
subdivided and the file gets smaller. For the 512x512 Lena image:

| luma | chroma | `.pie` | bpp  |
| ---- | ------ | ------ | ---- |
| 32   | 16     | 67326  | 2.05 |
| 32   | 32     | 61062  | 1.86 |
| 32   | 64     | 57369  | 1.75 |
| 32   | 128    | 55824  | 1.70 |
| 32   | 256    | 55463  | 1.69 |
| 64   | 64     | 38501  | 1.17 |
| 16   | 64     | 85925  | 2.62 |

The luma plane dominates: once the chroma bound reaches about 128 the chroma
planes cost next to nothing.

### Image formats

The format is chosen from the file extension, in both directions:

| extension         | format    | read | write |
| ----------------- | --------- | ---- | ----- |
| `.png`            | PNG       | yes  | yes   |
| `.jpg` / `.jpeg`  | JPEG      | yes  | yes   |
| `.tif` / `.tiff`  | TIFF      | yes  | yes   |
| `.bmp`            | BMP       | yes  | yes   |
| `.gif`            | GIF       | yes  | yes   |
| `.webp`           | WebP      | yes  | yes   |
| `.jp2`            | JPEG 2000 | yes  | yes   |
| `.j2k` / `.j2c`   | JPEG 2000 | yes  | yes   |
| `.heic` / `.heif` | HEIC      | yes  | no    |

So `pie-decode lena.pie lena.jpg` writes a JPEG and `pie-encode photo.tiff
photo.pie` reads a TIFF. An unrecognised extension is an error rather than a
guess. Any listed input can be decoded to any listed output, except that HEIC
is read-only.

Most formats go through [the `image` crate](https://docs.rs/image), which
normalises bit depth and palettes. JPEG 2000 uses
[`oxideav-jpeg2000`](https://docs.rs/oxideav-jpeg2000): a `.jp2` output is a
JP2 container and a `.j2k` / `.j2c` output a bare codestream, both with the
reversible 5-3 kernel, so they round-trip losslessly. HEIC reads through
[`heic-rs`](https://docs.rs/heic-rs); that crate is a decoder, so writing a
`.heic` is refused with an explanation. All three are pure Rust under
permissive licences — nothing here links a C library or takes on AGPL/GPL
terms.

JPEG is lossy and GIF is palettised, so those outputs only approximate the
`.pie` payload; PNG, TIFF, BMP, WebP and JPEG 2000 round-trip losslessly.

### Aspect ratio

A near-square region is still cut into four quadrants; a region whose sides
differ by more than 2:1 is instead cut in two along its longer side. Both
choices follow from the region's dimensions alone, so the decoder reproduces
them for free and no image has to be padded to a square. The extra two-way
nodes cost very little, even at extreme ratios:

| image   | `.pie` | bpp  |
| ------- | ------ | ---- |
| 512x512 | 57369  | 1.75 |
| 512x200 | 27457  | 2.15 |
| 200x512 | 27198  | 2.12 |
| 1024x64 | 13983  | 1.71 |

The non-square figures are crops/rescales of Lena, all at the default bounds.

Any image dimensions are supported. Already-greyscale inputs skip the YCbCr
step and are encoded directly:

    processing  grey.png (  262144 bytes)
    wrote       grey.pie (   55487 bytes, greyscale (21.2%, 1.69 bpp))

## Colour

Colour images are converted to YCbCr and encoded as 4:2:0. The luma plane is
fitted at full resolution with the same error bound as the greyscale image
(`MAX_ERROR = 32`). The chroma planes (`Cb`, `Cr`) are averaged over 2x2
blocks, quartering their size, and fitted with a looser bound
(`CHROMA_MAX_ERROR = 64`): we do not have to fit chroma as tightly, since the
eye is far less sensitive to colour detail than to brightness.

## Where the compression comes from

A naive encoding of the quadtree would spend one bit per tree entry and three
bytes per leaf, which puts it at roughly 4 bpp for colour.
The `.pie` file does much better (1.75 bpp instead of 4 bpp) by coding every
decision against a probability that adapts as the image is processed:

* **Range coding.** Every bit goes through a binary range coder whose models
  start at 1/2 and drift towards whatever actually happens, so a decision that
  is almost always the same costs far less than one bit.
* **Tree structure.** Each internal node codes one split bit, conditioned on
  the node's depth, on whether its previous sibling split, and on how finely
  the region directly above it was split (detail is spatially clustered, so a
  finely split neighbour is a strong hint). A 1x1 node can never split, so its
  split bit is implied and never coded.
* **Predictive constants.** A leaf's `c` is the plane's value at its top-left
  pixel, so it is predicted from the already-decoded pixels above and beside
  that corner and only the residual is coded. This is what makes smooth regions
  essentially free.
* **Adaptive integer coding.** The plane's gradients `a` and `b` are near zero
  over flat regions, so each integer is ZigZag-mapped and coded with an
  adaptive Elias-gamma binarisation (a unary bit-length, then the mantissa).

Averaged over the planes, the result is about 1.4 bytes per leaf, against the
3 bytes per leaf in the naive estimate.

Things that were tried and did **not** earn their keep, measured on Lena:

* Predicting Cr's split decisions from Cb's. The two chroma planes are
  decorrelated by the YCbCr transform, and the "is this region detailed?"
  signal is already carried by the same-plane spatial context above. As a
  split context it was a wash (57496 with it, versus a 57498 baseline), and a
  lean 2-state version was worse still (57521). As a value predictor for Cr's
  constant term it made the file 27 bytes *bigger* instead of smaller.
* JPEG-LS's median edge detector for the `c` predictor. Because `c` is a
  least-squares intercept (already smoothed over the leaf), a plain mean of
  the neighbouring pixels beats an edge-preserving predictor.

## The `.pie` format

    offset  size  field
    0       4     magic "PIE1"
    4       1     mode: 0 = greyscale, 1 = colour (4:2:0 YCbCr)
    5       4     width, little-endian u32
    9       4     height, little-endian u32
    13      8     luma max error as f64 bits, little-endian (metadata)
    21      8     chroma max error as f64 bits, little-endian (metadata)
    29      ..    range-coded tree payload

The payload is three range-coded bit streams (luma, then Cb and Cr for colour),
concatenated. The error bounds are stored only so a file describes how it was
made; the decoder does not need them. The decoder renders each leaf into a
reconstruction buffer as it decodes it, which is the same buffer the encoder's
predictor used, so the two stay in lockstep.

## Why it is fast

The original Python prototype took roughly 39 seconds for the 512x512 Lena
image. This port does the same work in about 10 milliseconds — the arithmetic
is identical (the reference render matches byte for byte), but the per-pixel
loops run as native code and the quadtree is traversed in parallel with
[rayon](https://docs.rs/rayon). Encoding the whole `.pie` takes about 10 ms and
decoding is faster still.

## Tests

    cargo test --release

That is all.
