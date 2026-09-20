//! Fractal image compression by recursively fitting planes to a tree.
//!
//! For each rectangular region we fit the plane `pixel = a*i + b*j + c` by
//! least squares. If the mean-squared error of that fit exceeds a threshold
//! the region is subdivided and the process repeats on each child: a region
//! close to square splits into four quadrants, while an elongated one splits
//! in two along its longer side. Keeping every child near-square lets a single
//! tree cover an image of any dimensions (see [`split_rect`]).
//!
//! Colour images are converted to YCbCr and encoded as 4:2:0. The luma plane
//! is fitted at full resolution; the chroma planes (`Cb`, `Cr`) are averaged
//! over 2x2 blocks, quartering their size, and fitted with a looser error
//! bound, since the eye is far less sensitive to colour detail than to
//! brightness.
//!
//! This is a Rust port of the original Python prototype (`compress.py`). For
//! greyscale input it deliberately reproduces the original's arithmetic
//! (including its integer rounding and its quadtree bookkeeping) so that the
//! rendered image and the reported compressed-size estimate match.

use std::error::Error;
use std::path::Path;

use rayon::prelude::*;

pub mod cli;
mod codec;
mod range;

pub use codec::{decode_file, decode_to_image, encode_file, encode_to_vec, Encoded, PieInfo};

/// Maximum mean-squared error tolerated by a fitted plane before the
/// sub-image is subdivided. Matches the original `max_error` of 32. Used for
/// the luma (brightness) plane.
pub const MAX_ERROR: f64 = 32.0;

/// Maximum mean-squared error for the chroma planes. This is deliberately
/// looser than [`MAX_ERROR`]: we do not have to fit chroma as tightly, so the
/// chroma planes subdivide less often and cost far fewer leaves.
pub const CHROMA_MAX_ERROR: f64 = 64.0;

/// Below this longer-side length the tree is walked serially. Larger regions
/// are split across threads; the fan-out is at most 4-way, so this keeps the
/// number of rayon tasks modest.
const PARALLEL_MIN_EDGE: usize = 128;

/// A fitted plane `pixel = a*i + b*j + c` with integer coefficients, where
/// `i` is the row offset and `j` the column offset within the sub-image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plane {
    pub a: i64,
    pub b: i64,
    pub c: i64,
}

/// The result of fitting a plane to one sub-image.
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    pub plane: Plane,
    /// Mean-squared error between the clamped plane and the source pixels.
    pub error: f64,
}

/// An axis-aligned region of a plane: `h` rows by `w` columns, with its
/// top-left pixel at `(x, y)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub h: usize,
    pub w: usize,
}

impl Rect {
    /// The rectangle covering a whole `width` x `height` plane.
    pub fn whole(width: usize, height: usize) -> Rect {
        Rect {
            x: 0,
            y: 0,
            h: height,
            w: width,
        }
    }

    /// Number of pixels in the region.
    #[inline]
    pub fn area(&self) -> usize {
        self.h * self.w
    }

    /// A region can only be subdivided if it spans more than one pixel; a
    /// single pixel is always a leaf.
    #[inline]
    pub fn can_split(&self) -> bool {
        self.h > 1 || self.w > 1
    }
}

/// The children of a subdivided [`Rect`].
///
/// A near-square region is cut into four quadrants, as in a classic quadtree.
/// An elongated region is instead cut in two along its longer side, which keeps
/// every child's aspect ratio within a factor of two and stops a thin strip
/// from forcing the tree to resolve the short dimension everywhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Children {
    Two([Rect; 2]),
    Four([Rect; 4]),
}

impl Children {
    #[inline]
    pub fn as_slice(&self) -> &[Rect] {
        match self {
            Children::Two(rs) => rs,
            Children::Four(rs) => rs,
        }
    }
}

/// An 8-bit greyscale image in row-major order.
pub struct Grey {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// An 8-bit colour image in 4:2:0 YCbCr: the luma plane `y` is `width` x
/// `height`, while the chroma planes `cb`/`cr` are each `ceil(width/2)` x
/// `ceil(height/2)`.
pub struct Colour {
    pub width: usize,
    pub height: usize,
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
}

/// A decoded image, tagged with whether it should be compressed as greyscale
/// or as colour.
pub enum Image {
    Grey(Grey),
    Colour(Colour),
}

/// One leaf of the tree: a rectangular region described by a single plane.
#[derive(Clone, Copy, Debug)]
pub struct Leaf {
    pub rect: Rect,
    pub plane: Plane,
}

/// The tree produced for one region, flattened into its leaves plus the counts
/// used to estimate the compressed size.
pub(crate) struct Subtree {
    pub(crate) leaves: Vec<Leaf>,
    /// Number of tree nodes; the naive estimate codes one split bit per node.
    pub(crate) tree_len: u64,
    /// Number of plane parameters; the naive estimate spends three bytes per
    /// leaf.
    pub(crate) params_len: u64,
}

#[inline]
fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Solve the 3x3 system `a * x = b` by Cramer's rule.
#[inline]
fn solve3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> [f64; 3] {
    let d = det3(a);
    let mut a1 = *a;
    let mut a2 = *a;
    let mut a3 = *a;
    for r in 0..3 {
        a1[r][0] = b[r];
        a2[r][1] = b[r];
        a3[r][2] = b[r];
    }
    [det3(&a1) / d, det3(&a2) / d, det3(&a3) / d]
}

/// Round as the original Python does: `int(v + 0.499999)` truncates toward
/// zero, as does `as i64`.
#[inline]
fn round(v: f64) -> i64 {
    (v + 0.499999) as i64
}

/// Solve the 2x2 system `a * x = b` by Cramer's rule.
#[inline]
fn solve2(a: [[f64; 2]; 2], b: [f64; 2]) -> [f64; 2] {
    let d = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    [
        (b[0] * a[1][1] - a[0][1] * b[1]) / d,
        (a[0][0] * b[1] - b[0] * a[1][0]) / d,
    ]
}

/// Fit the plane `pixel = a*i + b*j + c` to the region `rect` of `image`
/// (row-major, `width` pixels per row), using ordinary least squares.
///
/// The normal-equations matrix depends only on the region's dimensions, which
/// lets the coordinate sums be computed in closed form rather than re-summing
/// them per pixel. When the region is a single row (or column) the matching
/// gradient is unidentifiable, so it is pinned to zero and the remaining
/// coefficients are solved from the 2x2 system — matching numpy's minimum-norm
/// `lstsq`.
pub fn find_plane(image: &[u8], width: usize, rect: Rect) -> Fit {
    let (h, w) = (rect.h, rect.w);

    // A single pixel has a singular normal-equations matrix (the row and
    // column terms are all zero). The minimum-norm solution is simply the
    // pixel value at the origin.
    if h * w == 1 {
        let z = image[rect.x * width + rect.y] as i64;
        return Fit {
            plane: Plane { a: 0, b: 0, c: z },
            error: 0.0,
        };
    }

    let (hi, wi) = (h as i64, w as i64);
    let sn = (hi * wi) as f64;

    // Sums of the coordinate terms over the h x w grid, computed exactly in
    // integers.
    let sum_i = hi * (hi - 1) / 2;
    let sum_j = wi * (wi - 1) / 2;
    let sum_i2 = (hi - 1) * hi * (2 * hi - 1) / 6;
    let sum_j2 = (wi - 1) * wi * (2 * wi - 1) / 6;

    let sxx = (sum_i2 * wi) as f64;
    let syy = (sum_j2 * hi) as f64;
    let sxy = (sum_i * sum_j) as f64;
    let sx = (sum_i * wi) as f64;
    let sy = (sum_j * hi) as f64;

    let mut sxz: i64 = 0;
    let mut syz: i64 = 0;
    let mut sz: i64 = 0;
    for i in 0..hi {
        let base = (rect.x + i as usize) * width + rect.y;
        let mut row_sum: i64 = 0;
        let mut row_jz: i64 = 0;
        for j in 0..wi {
            let z = image[base + j as usize] as i64;
            row_sum += z;
            row_jz += j * z;
        }
        sxz += i * row_sum;
        syz += row_jz;
        sz += row_sum;
    }

    let (a, b, c) = if h == 1 {
        let sol = solve2([[syy, sy], [sy, sn]], [syz as f64, sz as f64]);
        (0, round(sol[0]), round(sol[1]))
    } else if w == 1 {
        let sol = solve2([[sxx, sx], [sx, sn]], [sxz as f64, sz as f64]);
        (round(sol[0]), 0, round(sol[1]))
    } else {
        let a_mat = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, sn]];
        let b_vec = [sxz as f64, syz as f64, sz as f64];
        let sol = solve3(&a_mat, &b_vec);
        (round(sol[0]), round(sol[1]), round(sol[2]))
    };

    let mut err: i64 = 0;
    for i in 0..hi {
        let base = (rect.x + i as usize) * width + rect.y;
        let ai = a * i + c;
        for j in 0..wi {
            let z = image[base + j as usize] as i64;
            let new_z = (ai + b * j).clamp(0, 255);
            let e = new_z - z;
            err += e * e;
        }
    }

    Fit {
        plane: Plane { a, b, c },
        error: err as f64 / sn,
    }
}

/// Subdivide a rectangle, keeping its children's aspect ratios bounded.
///
/// When one side is at least twice the other the region is bisected along its
/// longer side, giving two children; otherwise it is cut into four quadrants.
/// The choice follows from the region's dimensions alone, so the decoder
/// derives the same subdivision with no extra bits. Children are ordered
/// top-left, top-right, bottom-left, bottom-right — the depth-first order the
/// tree is coded in.
pub fn split_rect(rect: Rect) -> Children {
    debug_assert!(rect.can_split());

    if rect.w >= 2 * rect.h {
        // Wide and thin: cut vertically.
        let m = rect.w.div_ceil(2);
        Children::Two([
            Rect { w: m, ..rect },
            Rect {
                y: rect.y + m,
                w: rect.w - m,
                ..rect
            },
        ])
    } else if rect.h >= 2 * rect.w {
        // Tall and thin: cut horizontally.
        let m = rect.h.div_ceil(2);
        Children::Two([
            Rect { h: m, ..rect },
            Rect {
                x: rect.x + m,
                h: rect.h - m,
                ..rect
            },
        ])
    } else {
        let mh = rect.h.div_ceil(2);
        let mw = rect.w.div_ceil(2);
        Children::Four([
            Rect {
                h: mh,
                w: mw,
                ..rect
            },
            Rect {
                y: rect.y + mw,
                h: mh,
                w: rect.w - mw,
                ..rect
            },
            Rect {
                x: rect.x + mh,
                h: rect.h - mh,
                w: mw,
                ..rect
            },
            Rect {
                x: rect.x + mh,
                y: rect.y + mw,
                h: rect.h - mh,
                w: rect.w - mw,
            },
        ])
    }
}

pub(crate) fn walk(image: &[u8], width: usize, rect: Rect, max_error: f64) -> Subtree {
    let fit = find_plane(image, width, rect);

    if fit.error > max_error && rect.can_split() {
        let children = split_rect(rect);

        // Each node costs one split bit; only leaves carry plane parameters.
        let mut subtree = Subtree {
            leaves: Vec::new(),
            tree_len: 1,
            params_len: 0,
        };

        let absorb = |child: Subtree, subtree: &mut Subtree| {
            subtree.tree_len += child.tree_len;
            subtree.params_len += child.params_len;
            subtree.leaves.extend(child.leaves);
        };

        let kids = children.as_slice();
        if rect.h.max(rect.w) >= PARALLEL_MIN_EDGE {
            let walked: Vec<Subtree> = kids
                .par_iter()
                .map(|&child| walk(image, width, child, max_error))
                .collect();
            for child in walked {
                absorb(child, &mut subtree);
            }
        } else {
            for &child in kids {
                absorb(walk(image, width, child, max_error), &mut subtree);
            }
        }

        subtree
    } else {
        Subtree {
            leaves: vec![Leaf {
                rect,
                plane: fit.plane,
            }],
            tree_len: 1,
            params_len: 3,
        }
    }
}

/// Render the tree: every leaf paints its plane over its own region.
/// Leaves tile the whole image, so no parent region is needed.
pub fn render(width: usize, height: usize, leaves: &[Leaf]) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    for leaf in leaves {
        let r = leaf.rect;
        for i in 0..r.h {
            let base = (r.x + i) * width + r.y;
            let ai = leaf.plane.a * i as i64 + leaf.plane.c;
            for j in 0..r.w {
                out[base + j] = (ai + leaf.plane.b * j as i64).clamp(0, 255) as u8;
            }
        }
    }
    out
}

#[inline]
fn clamp_u8(v: f64) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// Convert one RGB pixel to `(Y, Cb, Cr)` using the BT.601 (JPEG) coefficients.
#[inline]
fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let (r, g, b) = (r as f64, g as f64, b as f64);
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b;
    let cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;
    (clamp_u8(y), clamp_u8(cb), clamp_u8(cr))
}

/// Convert one `(Y, Cb, Cr)` triple back to RGB.
#[inline]
fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let (y, cb, cr) = (y as f64, cb as f64 - 128.0, cr as f64 - 128.0);
    let r = y + 1.402 * cr;
    let g = y - 0.344_136 * cb - 0.714_136 * cr;
    let b = y + 1.772 * cb;
    (clamp_u8(r), clamp_u8(g), clamp_u8(b))
}

#[inline]
fn average_chroma(sum: &[u32], count: &[u32]) -> Vec<u8> {
    sum.iter()
        .zip(count)
        .map(|(&s, &n)| (s as f64 / n as f64).round() as u8)
        .collect()
}

impl Colour {
    /// Convert an RGB8 buffer to 4:2:0 YCbCr, averaging each chroma sample
    /// over its 2x2 block of pixels.
    pub fn from_rgb(rgb: &[u8], width: usize, height: usize) -> Colour {
        let cw = width.div_ceil(2);
        let ch = height.div_ceil(2);

        let mut y = vec![0u8; width * height];
        let mut cb_sum = vec![0u32; cw * ch];
        let mut cr_sum = vec![0u32; cw * ch];
        let mut count = vec![0u32; cw * ch];

        for r in 0..height {
            for c in 0..width {
                let p = (r * width + c) * 3;
                let (sy, scb, scr) = rgb_to_ycbcr(rgb[p], rgb[p + 1], rgb[p + 2]);
                y[r * width + c] = sy;

                let idx = (r / 2) * cw + c / 2;
                cb_sum[idx] += scb as u32;
                cr_sum[idx] += scr as u32;
                count[idx] += 1;
            }
        }

        Colour {
            width,
            height,
            y,
            cb: average_chroma(&cb_sum, &count),
            cr: average_chroma(&cr_sum, &count),
        }
    }

    /// Dimensions of the (subsampled) chroma planes.
    pub fn chroma_dims(&self) -> (usize, usize) {
        (self.width.div_ceil(2), self.height.div_ceil(2))
    }
}

/// Recombine full-resolution Y/Cb/Cr planes into an RGB8 buffer. The chroma
/// planes are upsampled with nearest-neighbour: each chroma sample paints a
/// 2x2 block, undoing the 4:2:0 subsampling.
pub(crate) fn combine_to_rgb(
    width: usize,
    height: usize,
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
) -> Vec<u8> {
    let cw = width.div_ceil(2);
    let mut rgb = vec![0u8; width * height * 3];
    for r in 0..height {
        for c in 0..width {
            let ci = (r / 2) * cw + c / 2;
            let (rr, gg, bb) = ycbcr_to_rgb(y[r * width + c], cb[ci], cr[ci]);
            let p = (r * width + c) * 3;
            rgb[p] = rr;
            rgb[p + 1] = gg;
            rgb[p + 2] = bb;
        }
    }
    rgb
}

/// Read an image in any supported format, converting it to 8-bit greyscale or
/// 4:2:0 YCbCr depending on whether it carries colour.
///
/// The format is chosen from the file extension; see
/// [`SUPPORTED_EXTENSIONS`] for the list, and [`read_greyscale`] to force a
/// greyscale result.
pub fn read_image(path: &str) -> Result<Image, Box<dyn Error>> {
    let image = decode_image(path)?;
    let width = image.width() as usize;
    let height = image.height() as usize;
    if image.color().has_color() {
        let rgb = image.to_rgb8();
        Ok(Image::Colour(Colour::from_rgb(rgb.as_raw(), width, height)))
    } else {
        Ok(Image::Grey(Grey {
            width,
            height,
            data: image.to_luma8().into_raw(),
        }))
    }
}

/// Read an image in any supported format and convert it to 8-bit greyscale.
///
/// RGB pixels are averaged as `(r + g + b) / 3`, matching the original, and
/// any alpha channel is discarded.
pub fn read_greyscale(path: &str) -> Result<Grey, Box<dyn Error>> {
    let image = decode_image(path)?;
    let width = image.width() as usize;
    let height = image.height() as usize;
    let data = if image.color().has_color() {
        rgb_to_grey(image.to_rgb8().as_raw(), 3)
    } else {
        image.to_luma8().into_raw()
    };

    Ok(Grey {
        width,
        height,
        data,
    })
}

/// The image codecs we know, chosen by file extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Codec {
    /// A format the `image` crate decodes and encodes end to end.
    Image(image::ImageFormat),
    /// JPEG 2000 in a JP2 container (`.jp2`).
    Jp2,
    /// A bare JPEG 2000 codestream (`.j2k`, `.j2c`).
    J2k,
    /// HEIC/HEIF (`.heic`, `.heif`). Decode only: no encoder is linked.
    Heic,
}

/// The image extensions we can read, for help and error text.
pub const SUPPORTED_EXTENSIONS: &str =
    "png, jpg/jpeg, tif/tiff, bmp, gif, webp, jp2, j2k/j2c, heic/heif";

/// The extensions we can read but not write.
pub const DECODE_ONLY_EXTENSIONS: &str = "heic/heif";

/// Map a path's extension to a codec, or explain why we cannot.
///
/// Most formats are handled by the `image` crate; JPEG 2000 and HEIC have
/// their own pure-Rust codecs. Adding a format means adding an arm here and,
/// for an `image` format, enabling its feature in `Cargo.toml`.
fn codec_for(path: &str) -> Result<Codec, Box<dyn Error>> {
    let extension = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    match extension.as_deref() {
        Some("png") => Ok(Codec::Image(image::ImageFormat::Png)),
        Some("jpg" | "jpeg" | "jpe") => Ok(Codec::Image(image::ImageFormat::Jpeg)),
        Some("tif" | "tiff") => Ok(Codec::Image(image::ImageFormat::Tiff)),
        Some("bmp") => Ok(Codec::Image(image::ImageFormat::Bmp)),
        Some("gif") => Ok(Codec::Image(image::ImageFormat::Gif)),
        Some("webp") => Ok(Codec::Image(image::ImageFormat::WebP)),
        Some("jp2") => Ok(Codec::Jp2),
        Some("j2k" | "j2c") => Ok(Codec::J2k),
        Some("heic" | "heif") => Ok(Codec::Heic),
        Some(other) => Err(format!(
            "unsupported image extension '.{other}' (supported: {SUPPORTED_EXTENSIONS})"
        )
        .into()),
        None => Err(format!(
            "cannot tell the image format of '{path}' from its extension \
             (supported: {SUPPORTED_EXTENSIONS})"
        )
        .into()),
    }
}

/// Decode a file into an 8-bit-per-channel image, choosing the decoder from
/// the path's extension.
fn decode_image(path: &str) -> Result<image::DynamicImage, Box<dyn Error>> {
    match codec_for(path)? {
        Codec::Image(format) => {
            let mut reader = image::ImageReader::open(path)?;
            reader.set_format(format);
            Ok(reader.decode()?)
        }
        Codec::Jp2 | Codec::J2k => decode_jpeg2000(path),
        Codec::Heic => decode_heic(path),
    }
}

/// Decode a JPEG 2000 codestream or JP2 file into an image.
///
/// `oxideav-jpeg2000` hands back each component as a separate plane, so this
/// re-interleaves them. Only the shape `pie` can also write is accepted: one
/// or three unsigned components of 8 bits, all at full resolution.
fn decode_jpeg2000(path: &str) -> Result<image::DynamicImage, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    let decoded = if oxideav_jpeg2000::looks_like_jp2(&bytes) {
        oxideav_jpeg2000::jp2::decode_jp2(&bytes)?
    } else {
        oxideav_jpeg2000::decode_j2k(&bytes)?
    };

    let (width, height) = (decoded.width, decoded.height);
    let components = decoded.components.len();
    let plain = decoded
        .components
        .iter()
        .all(|c| !c.is_signed && c.precision_bits <= 8 && c.width == width && c.height == height);
    if !plain || !(components == 1 || components == 3) {
        return Err(format!(
            "unsupported JPEG 2000 layout in '{path}': need 1 or 3 unsigned 8-bit \
             components at full resolution"
        )
        .into());
    }

    let pixels = (width as usize) * (height as usize);
    let mut samples = vec![0u8; pixels * components];
    for (channel, component) in decoded.components.iter().enumerate() {
        for (i, &sample) in component.samples.iter().enumerate() {
            samples[i * components + channel] = sample.clamp(0, 255) as u8;
        }
    }
    from_samples(path, width, height, components, samples)
}

/// Decode a HEIC/HEIF file into an image.
///
/// `heic-rs` has no encoder, so this is the only direction HEIC supports.
fn decode_heic(path: &str) -> Result<image::DynamicImage, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    let decoded = heic_rs::decode(&bytes, &heic_rs::DecodeOptions::default())?;
    match decoded.layout {
        heic_rs::PixelLayout::Rgb8 => {
            from_samples(path, decoded.width, decoded.height, 3, decoded.data)
        }
        other => {
            Err(format!("HEIC decoder returned {other:?} rather than RGB8 for '{path}'").into())
        }
    }
}

/// Build an image from interleaved 8-bit samples of one or three channels.
fn from_samples(
    path: &str,
    width: u32,
    height: u32,
    components: usize,
    samples: Vec<u8>,
) -> Result<image::DynamicImage, Box<dyn Error>> {
    match components {
        1 => Ok(image::DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(width, height, samples)
                .ok_or_else(|| format!("'{path}': greyscale buffer size mismatch"))?,
        )),
        3 => Ok(image::DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(width, height, samples)
                .ok_or_else(|| format!("'{path}': RGB buffer size mismatch"))?,
        )),
        n => Err(format!("'{path}': {n} components is not supported").into()),
    }
}

fn rgb_to_grey(data: &[u8], stride: usize) -> Vec<u8> {
    data.chunks_exact(stride)
        .map(|p| ((p[0] as u16 + p[1] as u16 + p[2] as u16) / 3) as u8)
        .collect()
}

/// Write an 8-bit greyscale image, choosing the encoder from the path's
/// extension.
pub fn write_greyscale(
    path: &str,
    width: u32,
    height: u32,
    data: &[u8],
) -> Result<(), Box<dyn Error>> {
    let buffer = image::GrayImage::from_raw(width, height, data.to_vec())
        .ok_or_else(|| format!("greyscale buffer is not {width}x{height} bytes"))?;
    write_image(path, &image::DynamicImage::ImageLuma8(buffer))
}

/// Write an 8-bit RGB image, choosing the encoder from the path's extension.
pub fn write_rgb(path: &str, width: u32, height: u32, data: &[u8]) -> Result<(), Box<dyn Error>> {
    let buffer = image::RgbImage::from_raw(width, height, data.to_vec())
        .ok_or_else(|| format!("RGB buffer is not {width}x{height} pixels"))?;
    write_image(path, &image::DynamicImage::ImageRgb8(buffer))
}

/// Encode an image to `path`, choosing the codec from its extension.
fn write_image(path: &str, image: &image::DynamicImage) -> Result<(), Box<dyn Error>> {
    match codec_for(path)? {
        Codec::Image(format) => {
            image.save_with_format(path, format)?;
            Ok(())
        }
        Codec::Jp2 => {
            std::fs::write(path, encode_jpeg2000(image, true)?)?;
            Ok(())
        }
        Codec::J2k => {
            std::fs::write(path, encode_jpeg2000(image, false)?)?;
            Ok(())
        }
        Codec::Heic => Err(format!(
            "cannot write '{path}': HEIC/HEIF is decode-only (no encoder is \
             linked); choose another output extension"
        )
        .into()),
    }
}

/// Encode an image as JPEG 2000, wrapped in a JP2 container when `jp2` is
/// set and as a bare codestream otherwise.
///
/// The codestream is the reversible 5-3 one, so the pixels survive a decode
/// unchanged.
fn encode_jpeg2000(image: &image::DynamicImage, jp2: bool) -> Result<Vec<u8>, Box<dyn Error>> {
    let (width, height) = (image.width(), image.height());
    let (samples, components) = if image.color().has_color() {
        (image.to_rgb8().into_raw(), 3usize)
    } else {
        (image.to_luma8().into_raw(), 1usize)
    };

    let codestream = oxideav_jpeg2000::encode_jpeg2000(&samples, width, height)?;
    if !jp2 {
        return Ok(codestream);
    }
    let options = oxideav_jpeg2000::jp2::Jp2WriteOptions::for_components(components);
    Ok(oxideav_jpeg2000::jp2::write_jp2(&codestream, &options)?)
}

/// Tunable error bounds for the tree fits.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Bound for the luma plane of a colour image, and for any greyscale
    /// input.
    pub max_error: f64,
    /// Bound for the chroma planes of a colour image. Looser than `max_error`
    /// by default: chroma can be fitted less tightly.
    pub chroma_max_error: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            max_error: MAX_ERROR,
            chroma_max_error: CHROMA_MAX_ERROR,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_plane_recovers_exact_plane() {
        // d[r][c] = 2*r + 3*c + 5  =>  a = 2, b = 3, c = 5.
        let mut image = [0u8; 16];
        for r in 0..4usize {
            for c in 0..4usize {
                image[r * 4 + c] = (2 * r + 3 * c + 5) as u8;
            }
        }

        let fit = find_plane(&image, 4, Rect::whole(4, 4));
        assert_eq!(fit.plane, Plane { a: 2, b: 3, c: 5 });
        assert_eq!(fit.error, 0.0);
    }

    #[test]
    fn square_region_splits_into_quadrants() {
        assert_eq!(
            split_rect(Rect { x: 8, y: 8, h: 4, w: 4 }),
            Children::Four([
                Rect { x: 8, y: 8, h: 2, w: 2 },
                Rect { x: 8, y: 10, h: 2, w: 2 },
                Rect { x: 10, y: 8, h: 2, w: 2 },
                Rect { x: 10, y: 10, h: 2, w: 2 },
            ])
        );
    }

    #[test]
    fn elongated_region_bisects_its_longer_side() {
        // Wide: cut vertically into two.
        assert_eq!(
            split_rect(Rect { x: 0, y: 0, h: 2, w: 8 }),
            Children::Two([
                Rect { x: 0, y: 0, h: 2, w: 4 },
                Rect { x: 0, y: 4, h: 2, w: 4 },
            ])
        );
        // Tall: cut horizontally into two.
        assert_eq!(
            split_rect(Rect { x: 0, y: 0, h: 8, w: 2 }),
            Children::Two([
                Rect { x: 0, y: 0, h: 4, w: 2 },
                Rect { x: 4, y: 0, h: 4, w: 2 },
            ])
        );
    }

    #[test]
    fn splits_tile_their_parent_exactly() {
        for h in 1..12usize {
            for w in 1..12usize {
                let rect = Rect { x: 3, y: 5, h, w };
                if !rect.can_split() {
                    continue;
                }
                let children = split_rect(rect);
                let mut area = 0;
                for c in children.as_slice() {
                    assert!(c.h > 0 && c.w > 0, "empty child of {rect:?}");
                    assert!(c.x >= rect.x && c.y >= rect.y, "child {c:?} escapes {rect:?}");
                    assert!(c.x + c.h <= rect.x + rect.h && c.y + c.w <= rect.y + rect.w);
                    area += c.area();
                }
                assert_eq!(area, rect.area(), "children of {rect:?} do not tile it");
            }
        }
    }

    #[test]
    fn single_pixel_is_a_leaf() {
        let image = [123u8];
        let fit = find_plane(&image, 1, Rect::whole(1, 1));
        assert_eq!(fit.plane, Plane { a: 0, b: 0, c: 123 });
        assert_eq!(fit.error, 0.0);
    }

    #[test]
    fn ycbcr_round_trips_within_rounding_error() {
        for &(r, g, b) in &[
            (0, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (123, 45, 200),
        ] {
            let (y, cb, cr) = rgb_to_ycbcr(r, g, b);
            let (r2, g2, b2) = ycbcr_to_rgb(y, cb, cr);
            assert!((r2 as i32 - r as i32).abs() <= 2, "r {r} -> {r2}");
            assert!((g2 as i32 - g as i32).abs() <= 2, "g {g} -> {g2}");
            assert!((b2 as i32 - b as i32).abs() <= 2, "b {b} -> {b2}");
        }
    }

    #[test]
    fn subsampling_averages_each_2x2_block() {
        // One chroma sample covers all four pixels, so Cb/Cr must be their
        // average while luma keeps full resolution.
        let rgb = [
            255, 0, 0, 0, 255, 0, //
            0, 0, 255, 255, 255, 0,
        ];
        let image = Colour::from_rgb(&rgb, 2, 2);
        assert_eq!((image.y.len(), image.cb.len(), image.cr.len()), (4, 1, 1));

        let samples: Vec<(u8, u8, u8)> = (0..4)
            .map(|p| rgb_to_ycbcr(rgb[p * 3], rgb[p * 3 + 1], rgb[p * 3 + 2]))
            .collect();
        let cb_avg = (samples.iter().map(|s| s.1 as u32).sum::<u32>() as f64 / 4.0).round() as u8;
        let cr_avg = (samples.iter().map(|s| s.2 as u32).sum::<u32>() as f64 / 4.0).round() as u8;
        assert_eq!(image.cb[0], cb_avg);
        assert_eq!(image.cr[0], cr_avg);
    }

    #[test]
    fn odd_sized_chroma_rounds_up() {
        let rgb = vec![0u8; 3 * 3 * 3];
        let image = Colour::from_rgb(&rgb, 3, 3);
        assert_eq!(image.chroma_dims(), (2, 2));
        assert_eq!(image.cb.len(), 4);
    }

    #[test]
    fn default_bounds_fit_chroma_less_tightly() {
        let options = Options::default();
        assert_eq!(options.max_error, MAX_ERROR);
        assert_eq!(options.chroma_max_error, CHROMA_MAX_ERROR);
        assert!(options.chroma_max_error > options.max_error);
    }

    #[test]
    fn codec_is_chosen_from_the_extension() {
        for (path, expected) in [
            ("lena.PNG", Codec::Image(image::ImageFormat::Png)),
            ("lena.jpg", Codec::Image(image::ImageFormat::Jpeg)),
            ("lena.JPEG", Codec::Image(image::ImageFormat::Jpeg)),
            ("lena.jpe", Codec::Image(image::ImageFormat::Jpeg)),
            ("lena.tif", Codec::Image(image::ImageFormat::Tiff)),
            ("lena.TIFF", Codec::Image(image::ImageFormat::Tiff)),
            ("lena.bmp", Codec::Image(image::ImageFormat::Bmp)),
            ("lena.gif", Codec::Image(image::ImageFormat::Gif)),
            ("lena.webp", Codec::Image(image::ImageFormat::WebP)),
            ("lena.jp2", Codec::Jp2),
            ("lena.j2k", Codec::J2k),
            ("lena.J2C", Codec::J2k),
            ("lena.heic", Codec::Heic),
            ("lena.HEIF", Codec::Heic),
        ] {
            assert_eq!(codec_for(path).unwrap(), expected, "for {path}");
        }
    }

    #[test]
    fn unrecognised_or_missing_extension_is_rejected() {
        for path in ["lena.pcx", "lena.xyz", "lena", "lena."] {
            assert!(codec_for(path).is_err(), "{path} should not be supported");
        }
    }

    #[test]
    fn round_trips_every_supported_format() {
        // Eight distinct colours (24 bytes), so a palette-based GIF keeps the
        // colour type.
        let rgb: Vec<u8> = (0..24u8).map(|i| i * 10).collect();
        let path_of = |ext: &str| {
            std::env::temp_dir()
                .join(format!("fractal_pie_round_trip_{ext}.{ext}"))
                .to_str()
                .unwrap()
                .to_string()
        };

        for extension in ["png", "bmp", "tif", "gif", "webp", "jpg", "jp2", "j2k"] {
            let path = path_of(extension);
            write_rgb(&path, 4, 2, &rgb).unwrap();
            let decoded = read_image(&path).unwrap();
            let _ = std::fs::remove_file(&path);

            let Image::Colour(colour) = decoded else {
                panic!("{extension}: expected a colour image back");
            };
            assert_eq!((colour.width, colour.height), (4, 2), "{extension}");
        }
    }

    #[test]
    fn greyscale_round_trips_through_png() {
        let data = [0u8, 50, 100, 150, 200, 250];
        let path = std::env::temp_dir().join("fractal_pie_grey_round_trip.png");
        let path = path.to_str().unwrap();

        write_greyscale(path, 3, 2, &data).unwrap();
        let grey = read_greyscale(path).unwrap();
        let _ = std::fs::remove_file(path);

        assert_eq!((grey.width, grey.height), (3, 2));
        assert_eq!(grey.data, data);
    }

    #[test]
    fn greyscale_jpeg2000_round_trips_losslessly() {
        let data = [0u8, 50, 100, 150, 200, 250];
        let path = std::env::temp_dir().join("fractal_pie_grey_lossless.jp2");
        let path = path.to_str().unwrap();

        write_greyscale(path, 3, 2, &data).unwrap();
        let decoded = decode_image(path).unwrap();
        let _ = std::fs::remove_file(path);

        assert!(!decoded.color().has_color(), "should stay greyscale");
        assert_eq!(decoded.to_luma8().into_raw(), data);
    }

    #[test]
    fn jpeg2000_round_trips_losslessly() {
        // 4x4 RGB. JPEG 2000 uses the reversible 5-3 kernel, so the samples
        // written must come back byte for byte.
        let rgb: Vec<u8> = (0..48u16).map(|i| (i * 5) as u8).collect();
        for extension in ["jp2", "j2k"] {
            let path = std::env::temp_dir().join(format!("fractal_pie_lossless.{extension}"));
            let path = path.to_str().unwrap();

            write_rgb(path, 4, 4, &rgb).unwrap();
            let decoded = decode_image(path).unwrap();
            let _ = std::fs::remove_file(path);

            assert_eq!(decoded.to_rgb8().into_raw(), rgb, "{extension}");
        }
    }

    #[test]
    fn jpeg2000_writes_the_container_the_extension_asks_for() {
        let rgb = [0u8; 48];
        for (extension, is_container) in [("jp2", true), ("j2k", false)] {
            let path = std::env::temp_dir().join(format!("fractal_pie_container.{extension}"));
            let path = path.to_str().unwrap();

            write_rgb(path, 4, 4, &rgb).unwrap();
            let bytes = std::fs::read(path).unwrap();
            let _ = std::fs::remove_file(path);

            assert_eq!(
                oxideav_jpeg2000::looks_like_jp2(&bytes),
                is_container,
                ".{extension} should {} a JP2 container",
                if is_container { "be" } else { "not be" },
            );
        }
    }

    #[test]
    fn heic_is_decode_only() {
        let path = std::env::temp_dir().join("fractal_pie_never.heic");
        let error = write_rgb(path.to_str().unwrap(), 1, 1, &[0, 0, 0]).unwrap_err();
        assert!(error.to_string().contains("decode-only"), "{error}");
        assert!(!path.exists(), "nothing should have been written");
    }
}
