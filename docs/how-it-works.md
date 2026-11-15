# How it works

[Back to the README](../README.md)

## The tree of planes

The program fits a plane, described by `pixel = a*i + b*j + c` (where `i` and
`j` are the row and column offsets within a region), through the image data by
least squares. If the mean-squared error is too great with respect to the raw
image, the region is subdivided and the process is repeated on each child: a
region close to square is cut into four quadrants, while a long thin one is cut
in two along its longer side. Keeping the children near-square this way lets a
single tree cover an image of any dimensions. The resulting tree of planes is
the compressed representation, and it is written to a `.pie` file with an
adaptive range coder.

![Lena](../lena.png)

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

## Why it is fast

The original Python prototype took roughly 39 seconds for the 512x512 Lena
image. This port does the same work in about 10 milliseconds — the arithmetic
is identical (the reference render matches byte for byte), but the per-pixel
loops run as native code and the quadtree is traversed in parallel with
[rayon](https://docs.rs/rayon). Encoding the whole `.pie` takes about 10 ms and
decoding is faster still.
