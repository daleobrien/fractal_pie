# The `.pie` format

[Back to the README](../README.md)

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
