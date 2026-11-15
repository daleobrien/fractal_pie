<h1 align="center">Fractal Pie</h1>

<p align="center">Image compression in Rust, from a quadtree of least-squares planes.</p>

<p align="center">
  <a href="#quick-start"><b>Quick start</b></a> &nbsp;·&nbsp;
  <a href="#documents">Documents</a> &nbsp;·&nbsp;
  <a href="#how-it-works">How it works</a> &nbsp;·&nbsp;
  <a href="docs/format.md">The format</a>
</p>

<p align="center">
  <img src="docs/cover.png" alt="An image and its reconstruction from a tree of planes" width="100%">
</p>

Just playing around with image compression. Fractal Pie covers an image with
flat planes, splitting wherever a plane fits badly, and range-codes the tree
into a `.pie` file. The 512×512 colour image above takes 57,369 bytes
(1.75 bpp), encodes in about 10 ms where the original Python prototype took
39 seconds, and decodes to exactly the reference render.

## Quick start

```bash
cargo build --release
./target/release/pie-encode lena.png lena.pie       # 57,369 bytes, 1.75 bpp
./target/release/pie-decode lena.pie decoded.png
cargo test --release
```

`--max-error N` and `--chroma-max-error N` trade quality for size. It reads and
writes PNG, JPEG, TIFF, BMP, GIF, WebP and JPEG 2000, and reads HEIC; see
[Encoding and decoding](docs/usage.md).

## Documents

| Document | What's in it |
| -------- | ------------ |
| [Encoding and decoding](docs/usage.md) | The two programs, the error bounds and their effect on size, image formats, aspect ratios and the tests |
| [How it works](docs/how-it-works.md) | The tree of planes, colour, where the compression comes from, what didn't help, and why it is fast |
| [The `.pie` format](docs/format.md) | The file layout |

## How it works

A plane, `pixel = a*i + b*j + c`, is fitted through each region by least
squares. Where it fits too badly the region is split, into four if it is near
square or two along its longer side, so a single tree covers an image of any
shape. Colour images are coded as YCbCr 4:2:0, the chroma with a looser bound.
Every split decision and plane is then range-coded against adaptive
probabilities, with each plane's constant predicted from the pixels already
decoded. [How it works](docs/how-it-works.md) has the details.

---

<p align="center"><sub>Part of <a href="https://daleobrien.github.io/#images">Images</a> &nbsp;·&nbsp; more at <a href="https://daleobrien.github.io">daleobrien.github.io</a></sub></p>
