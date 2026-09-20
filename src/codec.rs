//! The `.pie` container: serialise a compressed quadtree to bytes and back.
//!
//! # Format
//!
//! ```text
//! offset  size  field
//! 0       4     magic "PIE1"
//! 4       1     mode: 0 = greyscale, 1 = colour (4:2:0 YCbCr)
//! 5       4     width, little-endian u32
//! 9       4     height, little-endian u32
//! 13      8     luma max error as f64 bits, little-endian (metadata)
//! 21      8     chroma max error as f64 bits, little-endian (metadata)
//! 29      ..    range-coded quadtree payload
//! ```
//!
//! The payload is one range-coded bit stream per plane (luma, and for colour
//! the Cb and Cr planes), concatenated: each plane is decoded until its tree
//! is exhausted, and each uses its own fresh set of probability models.
//!
//! # Tree coding
//!
//! Nodes are visited in the same depth-first order [`walk`] produces its
//! leaves in. Every internal node codes one *split* bit (0 = leaf, 1 = split);
//! a 1x1 node can never split, so its split bit is implied and not coded. A
//! leaf then codes its plane `pixel = a*i + b*j + c` as three integers.
//!
//! Two tricks keep those integers small:
//!
//! * The constant `c` is *predicted* from the pixels already reconstructed
//!   around the leaf's top-left corner, and only the signed residual is coded.
//!   Since `c` is the plane's value at that corner, this is simply a
//!   smooth-image predictor.
//! * `a` and `b` are near zero in flat regions, and are coded with an adaptive
//!   Elias-gamma binarisation: a unary bit-length prefix, then the mantissa.
//!
//! The decoder paints each leaf it decodes into the same reconstruction buffer
//! the encoder used, so the predictor sees identical pixels on both sides.

use std::error::Error;

use crate::range::{BitModel, Decoder, Encoder};
use crate::{
    combine_to_rgb, split_range_into_quad, square_edge, walk, Colour, Grey, Image, Leaf, Options,
    Plane,
};

/// File magic; the trailing digit is the format version.
const MAGIC: [u8; 4] = *b"PIE1";
const MODE_GREY: u8 = 0;
const MODE_COLOUR: u8 = 1;
/// magic + mode + width + height + max_error + chroma_max_error.
const HEADER_LEN: usize = 4 + 1 + 4 + 4 + 8 + 8;
/// Largest bit-length an integer may use (it has to fit a `u64`).
const MAX_BITS: usize = 64;
/// Number of tree depths that get their own split-probability model.
const MAX_DEPTH: usize = 24;

/// A compressed image, plus the numbers needed to report on it.
pub struct Encoded {
    /// The complete `.pie` file: header followed by the coded payload.
    pub bytes: Vec<u8>,
    /// Size of the equivalent raw bitmap, in bytes.
    pub raw: u64,
    /// The old "one bit of tree, three bytes per leaf" size estimate.
    pub estimate: u64,
    /// Whether the source was colour.
    pub colour: bool,
}

/// Metadata read back out of a `.pie` header.
#[derive(Clone, Copy, Debug)]
pub struct PieInfo {
    pub colour: bool,
    pub width: u32,
    pub height: u32,
    pub max_error: f64,
    pub chroma_max_error: f64,
}

/// Adaptive models for coding one signed integer, using an Elias-gamma
/// binarisation: a unary bit-length followed by the mantissa.
struct IntModels {
    /// `exp[k]` codes "is there a (k+1)th bit?" in the unary prefix.
    exp: [BitModel; MAX_BITS + 2],
    /// `mant[k]` codes the (k+1)th mantissa bit counting from the top.
    mant: [BitModel; MAX_BITS],
}

impl IntModels {
    fn new() -> Self {
        IntModels {
            exp: [BitModel::new(); MAX_BITS + 2],
            mant: [BitModel::new(); MAX_BITS],
        }
    }

    fn encode(&mut self, enc: &mut Encoder, value: u64) {
        let bits = if value == 0 {
            0
        } else {
            64 - value.leading_zeros() as usize
        };

        // Unary prefix: `bits` ones, then a terminating zero.
        for k in 0..bits {
            enc.encode_bit(&mut self.exp[k], 1);
        }
        enc.encode_bit(&mut self.exp[bits], 0);
        if bits < 2 {
            return;
        }

        // Mantissa below the implicit leading one, most significant first.
        for pos in 0..bits - 1 {
            let bit = ((value >> (bits - 2 - pos)) & 1) as u32;
            enc.encode_bit(&mut self.mant[pos], bit);
        }
    }

    fn decode(&mut self, dec: &mut Decoder) -> u64 {
        let mut bits = 0usize;
        // `bits == MAX_BITS` is a full-width value; its terminator is coded
        // with `exp[MAX_BITS]`, so the loop runs one past that.
        while bits <= MAX_BITS {
            if dec.decode_bit(&mut self.exp[bits]) == 0 {
                break;
            }
            bits += 1;
        }
        if bits == 0 {
            return 0;
        }
        let bits = bits.min(MAX_BITS);

        let mut value: u64 = 1;
        for pos in 0..bits - 1 {
            let bit = dec.decode_bit(&mut self.mant[pos]) as u64;
            value = (value << 1) | bit;
        }
        value
    }

    fn encode_signed(&mut self, enc: &mut Encoder, value: i64) {
        self.encode(enc, zigzag(value));
    }

    fn decode_signed(&mut self, dec: &mut Decoder) -> i64 {
        unzigzag(self.decode(dec))
    }
}

/// Models for one plane's quadtree.
struct PlaneModels {
    /// Split decision, indexed by `depth * 2 + previous_sibling_split`.
    split: [BitModel; MAX_DEPTH * 2],
    a: IntModels,
    b: IntModels,
    c: IntModels,
}

impl PlaneModels {
    fn new() -> Self {
        PlaneModels {
            split: [BitModel::new(); MAX_DEPTH * 2],
            a: IntModels::new(),
            b: IntModels::new(),
            c: IntModels::new(),
        }
    }

    #[inline]
    fn split_index(&self, depth: usize, prev_split: bool) -> usize {
        depth.min(MAX_DEPTH - 1) * 2 + prev_split as usize
    }
}

/// A partially reconstructed plane, painted leaf by leaf as it is coded.
struct Recon {
    w: usize,
    buf: Vec<u8>,
    mask: Vec<bool>,
}

impl Recon {
    fn new(w: usize, h: usize) -> Self {
        Recon {
            w,
            buf: vec![0u8; w * h],
            mask: vec![false; w * h],
        }
    }

    /// Predict a leaf's constant term from the reconstructed pixels above-left,
    /// above, above-right and left of its top-left corner.
    ///
    /// `c` is the plane's value at `(x, y)`, so this is a smooth-image
    /// predictor: neighbouring pixels are a good guess for it. Because `c` is a
    /// least-squares intercept (already smoothed over the whole leaf), a plain
    /// mean beats an edge-preserving predictor like JPEG-LS's MED here.
    fn predict(&self, x: usize, y: usize) -> i64 {
        let mut sum: i64 = 0;
        let mut count: i64 = 0;

        if x > 0 {
            let row = (x - 1) * self.w;
            if y > 0 && self.mask[row + y - 1] {
                sum += self.buf[row + y - 1] as i64;
                count += 1;
            }
            if self.mask[row + y] {
                sum += self.buf[row + y] as i64;
                count += 1;
            }
            if y + 1 < self.w && self.mask[row + y + 1] {
                sum += self.buf[row + y + 1] as i64;
                count += 1;
            }
        }
        if y > 0 {
            let idx = x * self.w + y - 1;
            if self.mask[idx] {
                sum += self.buf[idx] as i64;
                count += 1;
            }
        }

        if count == 0 {
            128
        } else {
            (sum + count / 2) / count
        }
    }

    /// Paint a leaf's plane over its sub-image, exactly as [`crate::render`]
    /// would.
    fn paint(&mut self, plane: &Plane, x: usize, y: usize, n: usize) {
        for i in 0..n {
            let base = (x + i) * self.w + y;
            let ai = plane.a * i as i64 + plane.c;
            for j in 0..n {
                self.buf[base + j] = (ai + plane.b * j as i64).clamp(0, 255) as u8;
                self.mask[base + j] = true;
            }
        }
    }
}

/// Map a signed value onto an unsigned one so small magnitudes stay small:
/// `0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, ...`.
#[inline]
fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

#[inline]
fn unzigzag(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

/// Encode one plane's leaves. `idx` walks the DFS-ordered leaf list.
#[allow(clippy::too_many_arguments)]
fn encode_node(
    enc: &mut Encoder,
    models: &mut PlaneModels,
    recon: &mut Recon,
    leaves: &[Leaf],
    idx: &mut usize,
    x: usize,
    y: usize,
    n: usize,
    depth: usize,
    prev_split: bool,
) -> bool {
    let leaf = leaves[*idx];
    let is_leaf = leaf.x == x && leaf.y == y && leaf.n == n;

    if n > 1 {
        let ctx = models.split_index(depth, prev_split);
        enc.encode_bit(&mut models.split[ctx], if is_leaf { 0 } else { 1 });
    }

    if is_leaf {
        *idx += 1;
        let pred = recon.predict(x, y);
        models.a.encode_signed(enc, leaf.plane.a);
        models.b.encode_signed(enc, leaf.plane.b);
        models.c.encode_signed(enc, leaf.plane.c - pred);
        recon.paint(&leaf.plane, x, y, n);
        true
    } else {
        let quads = split_range_into_quad(x, y, n);
        let mut prev_child_leaf = false;
        for (j, &(qx, qy, qn)) in quads.iter().enumerate() {
            let prev = j > 0 && !prev_child_leaf;
            prev_child_leaf =
                encode_node(enc, models, recon, leaves, idx, qx, qy, qn, depth + 1, prev);
        }
        false
    }
}

/// Decode one plane's tree, returning the reconstructed plane.
#[allow(clippy::too_many_arguments)]
fn decode_node(
    dec: &mut Decoder,
    models: &mut PlaneModels,
    recon: &mut Recon,
    x: usize,
    y: usize,
    n: usize,
    depth: usize,
    prev_split: bool,
) -> bool {
    let is_leaf = if n <= 1 {
        true
    } else {
        let ctx = models.split_index(depth, prev_split);
        dec.decode_bit(&mut models.split[ctx]) == 0
    };

    if is_leaf {
        let a = models.a.decode_signed(dec);
        let b = models.b.decode_signed(dec);
        let pred = recon.predict(x, y);
        let c = models.c.decode_signed(dec) + pred;
        recon.paint(&Plane { a, b, c }, x, y, n);
        true
    } else {
        let quads = split_range_into_quad(x, y, n);
        let mut prev_child_leaf = false;
        for (j, &(qx, qy, qn)) in quads.iter().enumerate() {
            let prev = j > 0 && !prev_child_leaf;
            prev_child_leaf = decode_node(dec, models, recon, qx, qy, qn, depth + 1, prev);
        }
        false
    }
}

/// Encode one plane of `n x n` tiles, appending to the shared coder.
fn encode_plane(enc: &mut Encoder, leaves: &[Leaf], w: usize, h: usize, n: usize) {
    let mut models = PlaneModels::new();
    let mut recon = Recon::new(w, h);
    let mut idx = 0usize;
    encode_node(
        enc,
        &mut models,
        &mut recon,
        leaves,
        &mut idx,
        0,
        0,
        n,
        0,
        false,
    );
}

/// Decode one plane of `n x n` tiles from the shared coder.
fn decode_plane(dec: &mut Decoder, w: usize, h: usize, n: usize) -> Vec<u8> {
    let mut models = PlaneModels::new();
    let mut recon = Recon::new(w, h);
    decode_node(dec, &mut models, &mut recon, 0, 0, n, 0, false);
    recon.buf
}

fn write_header(buf: &mut Vec<u8>, mode: u8, width: u32, height: u32, options: Options) {
    buf.extend_from_slice(&MAGIC);
    buf.push(mode);
    buf.extend_from_slice(&width.to_le_bytes());
    buf.extend_from_slice(&height.to_le_bytes());
    buf.extend_from_slice(&options.max_error.to_le_bytes());
    buf.extend_from_slice(&options.chroma_max_error.to_le_bytes());
}

fn read_header(data: &[u8]) -> Result<(PieInfo, &[u8]), Box<dyn Error>> {
    if data.len() < HEADER_LEN {
        return Err("file is too short to be a .pie file".into());
    }
    if data[0..4] != MAGIC {
        return Err("not a .pie file (bad magic)".into());
    }
    let mode = data[4];
    if mode != MODE_GREY && mode != MODE_COLOUR {
        return Err(format!("unknown .pie mode {mode}").into());
    }
    let width = u32::from_le_bytes(data[5..9].try_into().unwrap());
    let height = u32::from_le_bytes(data[9..13].try_into().unwrap());
    if width == 0 || height == 0 {
        return Err("degenerate .pie dimensions".into());
    }
    let max_error = f64::from_le_bytes(data[13..21].try_into().unwrap());
    let chroma_max_error = f64::from_le_bytes(data[21..29].try_into().unwrap());

    let info = PieInfo {
        colour: mode == MODE_COLOUR,
        width,
        height,
        max_error,
        chroma_max_error,
    };
    Ok((info, &data[HEADER_LEN..]))
}

/// Compress an already-decoded [`Image`] into a `.pie` file image.
pub fn encode_to_vec(image: &Image, options: Options) -> Result<Encoded, Box<dyn Error>> {
    let mut enc = Encoder::new();
    let (mode, width, height, raw, estimate);

    match image {
        Image::Grey(img) => {
            let n = square_edge(img.width, img.height)?;
            let tree = walk(&img.data, img.width, 0, 0, n, options.max_error);
            estimate = tree.tree_len / 8 + tree.params_len + 2;

            encode_plane(&mut enc, &tree.leaves, img.width, img.height, n);

            mode = MODE_GREY;
            width = img.width as u32;
            height = img.height as u32;
            raw = (img.width * img.height) as u64;
        }
        Image::Colour(img) => {
            let n = square_edge(img.width, img.height)?;
            let (cw, ch) = img.chroma_dims();

            // Luma gets the tight bound; the quarter-size chroma planes get
            // the looser one.
            let y_tree = walk(&img.y, img.width, 0, 0, n, options.max_error);
            let cb_tree = walk(&img.cb, cw, 0, 0, cw, options.chroma_max_error);
            let cr_tree = walk(&img.cr, cw, 0, 0, cw, options.chroma_max_error);
            estimate = (y_tree.tree_len + cb_tree.tree_len + cr_tree.tree_len) / 8
                + (y_tree.params_len + cb_tree.params_len + cr_tree.params_len)
                + 2;

            encode_plane(&mut enc, &y_tree.leaves, img.width, img.height, n);
            encode_plane(&mut enc, &cb_tree.leaves, cw, ch, cw);
            encode_plane(&mut enc, &cr_tree.leaves, cw, ch, cw);

            mode = MODE_COLOUR;
            width = img.width as u32;
            height = img.height as u32;
            raw = (img.width * img.height * 3) as u64;
        }
    }

    let payload = enc.finish();
    let mut bytes = Vec::with_capacity(HEADER_LEN + payload.len());
    write_header(&mut bytes, mode, width, height, options);
    bytes.extend_from_slice(&payload);

    Ok(Encoded {
        bytes,
        raw,
        estimate,
        colour: mode == MODE_COLOUR,
    })
}

/// Decode a `.pie` image into reconstructed planes.
pub fn decode_to_image(data: &[u8]) -> Result<(Image, PieInfo), Box<dyn Error>> {
    let (info, payload) = read_header(data)?;
    let width = info.width as usize;
    let height = info.height as usize;
    if width != height {
        return Err(format!("image must be square, got {width}x{height}").into());
    }

    let mut dec = Decoder::new(payload);
    if info.colour {
        let cw = width.div_ceil(2);
        let ch = height.div_ceil(2);
        let y = decode_plane(&mut dec, width, height, width);
        let cb = decode_plane(&mut dec, cw, ch, cw);
        let cr = decode_plane(&mut dec, cw, ch, cw);
        let colour = Colour {
            width,
            height,
            y,
            cb,
            cr,
        };
        Ok((Image::Colour(colour), info))
    } else {
        let plane = decode_plane(&mut dec, width, height, width);
        let grey = Grey {
            width,
            height,
            data: plane,
        };
        Ok((Image::Grey(grey), info))
    }
}

/// Read a PNG, compress it to a `.pie` file, and print the resulting sizes.
pub fn encode_file(input: &str, output: &str, options: Options) -> Result<(), Box<dyn Error>> {
    println!("processing  {input}");

    let image = crate::read_image(input)?;
    let encoded = encode_to_vec(&image, options)?;
    std::fs::write(output, &encoded.bytes)?;

    let size = encoded.bytes.len() as u64;
    let kind = if encoded.colour {
        "colour"
    } else {
        "greyscale"
    };
    println!("wrote       {output} ({size} bytes, {kind})");
    let pct = 100.0 * size as f64 / encoded.estimate.max(1) as f64;
    println!(
        "quadtree estimate {} bytes -> {} bytes ({pct:.1}% of estimate)",
        encoded.estimate, size
    );
    println!(
        "compression ratio: {:.2}:1 ({} bytes raw -> {size} bytes)",
        encoded.raw as f64 / size as f64,
        encoded.raw
    );
    Ok(())
}

/// Read a `.pie` file, decode it, and write the reconstructed PNG.
pub fn decode_file(input: &str, output: &str) -> Result<(), Box<dyn Error>> {
    println!("decoding    {input}");

    let data = std::fs::read(input)?;
    let (image, info) = decode_to_image(&data)?;

    match image {
        Image::Grey(g) => {
            crate::write_greyscale(output, g.width as u32, g.height as u32, &g.data)?;
        }
        Image::Colour(c) => {
            let rgb = combine_to_rgb(c.width, c.height, &c.y, &c.cb, &c.cr);
            crate::write_rgb(output, c.width as u32, c.height as u32, &rgb)?;
        }
    }

    let kind = if info.colour { "colour" } else { "greyscale" };
    println!(
        "wrote       {output} ({}x{} {kind})",
        info.width, info.height
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{render, Grey, MAX_ERROR};

    fn grey_image(width: usize, height: usize, f: impl Fn(usize, usize) -> u8) -> Grey {
        let mut data = vec![0u8; width * height];
        for r in 0..height {
            for c in 0..width {
                data[r * width + c] = f(r, c);
            }
        }
        Grey {
            width,
            height,
            data,
        }
    }

    #[test]
    fn zigzag_round_trips() {
        for v in [-1000i64, -2, -1, 0, 1, 2, 1000, i64::MAX, i64::MIN] {
            assert_eq!(unzigzag(zigzag(v)), v);
        }
    }

    #[test]
    fn header_round_trips() {
        let mut bytes = Vec::new();
        write_header(&mut bytes, MODE_COLOUR, 512, 256, Options::default());
        let (info, payload) = read_header(&bytes).unwrap();
        assert!(info.colour);
        assert_eq!(info.width, 512);
        assert_eq!(info.height, 256);
        assert_eq!(info.max_error, Options::default().max_error);
        assert!(payload.is_empty());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = Vec::new();
        write_header(&mut bytes, MODE_GREY, 8, 8, Options::default());
        bytes[0] = b'X';
        assert!(read_header(&bytes).is_err());
    }

    #[test]
    fn single_pixel_round_trips() {
        let image = Image::Grey(grey_image(1, 1, |_, _| 137));
        let encoded = encode_to_vec(&image, Options::default()).unwrap();
        let (decoded, info) = decode_to_image(&encoded.bytes).unwrap();
        assert_eq!((info.width, info.height), (1, 1));
        match decoded {
            Image::Grey(g) => assert_eq!(g.data, vec![137]),
            _ => panic!("expected greyscale"),
        }
    }

    #[test]
    fn grey_round_trip_matches_render() {
        // A smooth ramp plus a noisy block, so the tree both splits and prunes.
        let image = Image::Grey(grey_image(64, 64, |r, c| {
            let ramp = (r * 2 + c * 3) as u8;
            if (16..32).contains(&r) && (16..32).contains(&c) {
                ramp.wrapping_add(97)
            } else {
                ramp
            }
        }));

        let encoded = encode_to_vec(&image, Options::default()).unwrap();
        let (decoded, _) = decode_to_image(&encoded.bytes).unwrap();

        let expected = match &image {
            Image::Grey(g) => {
                let tree = walk(&g.data, g.width, 0, 0, g.width, MAX_ERROR);
                render(g.width, g.height, &tree.leaves)
            }
            _ => unreachable!(),
        };

        match decoded {
            Image::Grey(g) => assert_eq!(g.data, expected),
            _ => panic!("expected greyscale"),
        }
    }

    #[test]
    fn colour_round_trip_matches_render() {
        let width: usize = 32;
        let height: usize = 32;
        let cw = width.div_ceil(2);
        let ch = height.div_ceil(2);

        let y: Vec<u8> = (0..width * height).map(|i| ((i * 5) % 256) as u8).collect();
        let cb: Vec<u8> = (0..cw * ch).map(|i| ((i * 11 + 40) % 256) as u8).collect();
        let cr: Vec<u8> = (0..cw * ch).map(|i| ((i * 17 + 90) % 256) as u8).collect();

        let image = Image::Colour(Colour {
            width,
            height,
            y: y.clone(),
            cb: cb.clone(),
            cr: cr.clone(),
        });

        let encoded = encode_to_vec(&image, Options::default()).unwrap();
        let (decoded, info) = decode_to_image(&encoded.bytes).unwrap();
        assert!(info.colour);

        // Luma uses MAX_ERROR; chroma uses CHROMA_MAX_ERROR.
        let expected_y = {
            let tree = walk(&y, width, 0, 0, width, MAX_ERROR);
            render(width, height, &tree.leaves)
        };
        let expected_cb = {
            let tree = walk(&cb, cw, 0, 0, cw, crate::CHROMA_MAX_ERROR);
            render(cw, ch, &tree.leaves)
        };
        let expected_cr = {
            let tree = walk(&cr, cw, 0, 0, cw, crate::CHROMA_MAX_ERROR);
            render(cw, ch, &tree.leaves)
        };

        match decoded {
            Image::Colour(c) => {
                assert_eq!(c.y, expected_y);
                assert_eq!(c.cb, expected_cb);
                assert_eq!(c.cr, expected_cr);
            }
            _ => panic!("expected colour"),
        }
    }

    #[test]
    fn entropy_coding_beats_the_naive_estimate() {
        // A natural-ish image: smooth gradients with structure. The adaptive
        // coder should beat "1 bit per tree entry + 3 bytes per leaf".
        let image = Image::Grey(grey_image(128, 128, |r, c| {
            let base = 40 + (r as i32 / 4) + (c as i32 / 2);
            let ripple = if (r / 8 + c / 8) % 2 == 0 { 6 } else { -6 };
            (base + ripple).clamp(0, 255) as u8
        }));

        let encoded = encode_to_vec(&image, Options::default()).unwrap();
        assert!(
            (encoded.bytes.len() as u64) < encoded.estimate,
            "coded {} bytes did not beat estimate {} bytes",
            encoded.bytes.len(),
            encoded.estimate
        );
    }
}
