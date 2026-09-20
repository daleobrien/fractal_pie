//! Fractal image compression by recursively fitting planes to a quadtree.
//!
//! For each square sub-image we fit the plane `pixel = a*i + b*j + c` by
//! least squares. If the mean-squared error of that fit exceeds a threshold
//! the sub-image is split into four quadrants and the process repeats on each.
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
use std::fs::File;
use std::io::BufReader;

use rayon::prelude::*;

/// Maximum mean-squared error tolerated by a fitted plane before the
/// sub-image is subdivided. Matches the original `max_error` of 32. Used for
/// the luma (brightness) plane.
pub const MAX_ERROR: f64 = 32.0;

/// Maximum mean-squared error for the chroma planes. This is deliberately
/// looser than [`MAX_ERROR`]: we do not have to fit chroma as tightly, so the
/// chroma planes subdivide less often and cost far fewer leaves.
pub const CHROMA_MAX_ERROR: f64 = 64.0;

/// Below this edge length the quadtree is walked serially. Larger sub-images
/// are split across threads; the recursion is only 4-way, so this keeps the
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

/// A decoded PNG, tagged with whether it should be compressed as greyscale or
/// as colour.
pub enum Image {
    Grey(Grey),
    Colour(Colour),
}

/// One leaf of the quadtree: a square sub-image described by a single plane.
#[derive(Clone, Copy, Debug)]
pub struct Leaf {
    pub x: usize,
    pub y: usize,
    pub n: usize,
    pub plane: Plane,
}

/// The quadtree produced for one sub-image, flattened into its leaves plus the
/// running lengths of the original `tree` and `parameters` bit/lists used to
/// estimate the compressed size.
struct Subtree {
    leaves: Vec<Leaf>,
    /// Number of entries the original code would have appended to `tree`.
    tree_len: u64,
    /// Number of entries the original code would have appended to `parameters`.
    params_len: u64,
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

/// Fit the plane `pixel = a*i + b*j + c` to the `n` x `n` sub-image whose
/// top-left corner is `(x_offset, y_offset)`, using ordinary least squares.
///
/// The normal-equations matrix only depends on `n`, which the original code
/// exploits to avoid re-summing the pixel coordinates.
pub fn find_plane(image: &[u8], width: usize, x_offset: usize, y_offset: usize, n: usize) -> Fit {
    // A single pixel has a singular normal-equations matrix (the row and
    // column terms are all zero). numpy's lstsq returns the minimum-norm
    // solution, which here is simply the pixel value at the origin.
    if n == 1 {
        let z = image[x_offset * width + y_offset] as i64;
        return Fit {
            plane: Plane { a: 0, b: 0, c: z },
            error: 0.0,
        };
    }

    let ni = n as i64;
    let m = ni - 1;
    let sn_i = ni * ni;

    // Sums of the coordinate terms over an n x n grid. Computed exactly in
    // integers, exactly as the Python version does before it divides.
    let sxx = (m * (2 * ni - 1) * sn_i) as f64 / 6.0;
    let syx = (m * m * sn_i) as f64 / 4.0;
    let sx = (m * sn_i) as f64 / 2.0;
    let sn = sn_i as f64;

    let mut sxz: i64 = 0;
    let mut syz: i64 = 0;
    let mut sz: i64 = 0;
    for i in 0..ni {
        let base = (x_offset + i as usize) * width + y_offset;
        let mut row_sum: i64 = 0;
        let mut row_jz: i64 = 0;
        for j in 0..ni {
            let z = image[base + j as usize] as i64;
            row_sum += z;
            row_jz += j * z;
        }
        sxz += i * row_sum;
        syz += row_jz;
        sz += row_sum;
    }

    let a_mat = [[sxx, syx, sx], [syx, sxx, sx], [sx, sx, sn]];
    let b_vec = [sxz as f64, syz as f64, sz as f64];
    let sol = solve3(&a_mat, &b_vec);

    // `int(v + 0.499999)` in Python truncates toward zero, as does `as i64`.
    let a = (sol[0] + 0.499999) as i64;
    let b = (sol[1] + 0.499999) as i64;
    let c = (sol[2] + 0.499999) as i64;

    let mut err: i64 = 0;
    for i in 0..ni {
        let base = (x_offset + i as usize) * width + y_offset;
        let ai = a * i + c;
        for j in 0..ni {
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

/// Split the square range `(x, y, n)` into four quadrants.
pub fn split_range_into_quad(x: usize, y: usize, n: usize) -> [(usize, usize, usize); 4] {
    let m = (n + 1) / 2;
    [(x, y, m), (x, y + m, m), (x + m, y, m), (x + m, y + m, m)]
}

fn walk(image: &[u8], width: usize, x: usize, y: usize, n: usize, max_error: f64) -> Subtree {
    let fit = find_plane(image, width, x, y, n);

    if fit.error > max_error {
        let quads = split_range_into_quad(x, y, n);

        // The original appends a `1` marker, then the children, then a `0`.
        let mut subtree = Subtree {
            leaves: Vec::new(),
            tree_len: 2,
            params_len: 3,
        };

        let absorb = |child: Subtree, subtree: &mut Subtree| {
            subtree.tree_len += child.tree_len;
            subtree.params_len += child.params_len;
            subtree.leaves.extend(child.leaves);
        };

        if n >= PARALLEL_MIN_EDGE {
            let children: Vec<Subtree> = quads
                .par_iter()
                .map(|&(qx, qy, qn)| walk(image, width, qx, qy, qn, max_error))
                .collect();
            for child in children {
                absorb(child, &mut subtree);
            }
        } else {
            for &(qx, qy, qn) in &quads {
                absorb(walk(image, width, qx, qy, qn, max_error), &mut subtree);
            }
        }

        subtree
    } else {
        Subtree {
            leaves: vec![Leaf {
                x,
                y,
                n,
                plane: fit.plane,
            }],
            tree_len: 1,
            params_len: 3,
        }
    }
}

/// Render the quadtree: every leaf paints its plane over its own sub-image.
/// Leaves tile the whole image, so no parent region is needed.
fn render(width: usize, height: usize, leaves: &[Leaf]) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    for leaf in leaves {
        for i in 0..leaf.n {
            let base = (leaf.x + i) * width + leaf.y;
            let ai = leaf.plane.a * i as i64 + leaf.plane.c;
            for j in 0..leaf.n {
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
fn combine_to_rgb(width: usize, height: usize, y: &[u8], cb: &[u8], cr: &[u8]) -> Vec<u8> {
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

/// A decoded PNG: its colour type, dimensions, and 8-bit samples.
type DecodedPng = (png::ColorType, usize, usize, Vec<u8>);

/// Decode a PNG to plain 8-bit samples, normalising palettes and bit depths.
fn decode_png(path: &str) -> Result<DecodedPng, Box<dyn Error>> {
    let file = BufReader::new(File::open(path)?);
    let mut decoder = png::Decoder::new(file);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);

    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    buf.truncate(info.buffer_size());
    Ok((
        info.color_type,
        info.width as usize,
        info.height as usize,
        buf,
    ))
}

/// Read a PNG, converting it to 8-bit greyscale or 4:2:0 YCbCr depending on
/// whether it carries colour.
pub fn read_image(path: &str) -> Result<Image, Box<dyn Error>> {
    let (color_type, width, height, data) = decode_png(path)?;
    match color_type {
        png::ColorType::Grayscale => Ok(Image::Grey(Grey {
            width,
            height,
            data,
        })),
        png::ColorType::GrayscaleAlpha => Ok(Image::Grey(Grey {
            width,
            height,
            data: data.chunks_exact(2).map(|p| p[0]).collect(),
        })),
        png::ColorType::Rgb => Ok(Image::Colour(Colour::from_rgb(&data, width, height))),
        png::ColorType::Rgba => {
            let rgb: Vec<u8> = data
                .chunks_exact(4)
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect();
            Ok(Image::Colour(Colour::from_rgb(&rgb, width, height)))
        }
        other => Err(format!("unsupported PNG colour type: {other:?}").into()),
    }
}

/// Read a PNG and convert it to 8-bit greyscale.
///
/// RGB(A) pixels are averaged as `(r + g + b) / 3`, matching the original.
pub fn read_greyscale(path: &str) -> Result<Grey, Box<dyn Error>> {
    let (color_type, width, height, data) = decode_png(path)?;
    let data = match color_type {
        png::ColorType::Grayscale => data,
        png::ColorType::GrayscaleAlpha => data.chunks_exact(2).map(|p| p[0]).collect(),
        png::ColorType::Rgb => rgb_to_grey(&data, 3),
        png::ColorType::Rgba => rgb_to_grey(&data, 4),
        other => return Err(format!("unsupported PNG colour type: {other:?}").into()),
    };

    Ok(Grey {
        width,
        height,
        data,
    })
}

fn rgb_to_grey(data: &[u8], stride: usize) -> Vec<u8> {
    data.chunks_exact(stride)
        .map(|p| ((p[0] as u16 + p[1] as u16 + p[2] as u16) / 3) as u8)
        .collect()
}

/// Write an 8-bit greyscale PNG.
pub fn write_greyscale(
    path: &str,
    width: u32,
    height: u32,
    data: &[u8],
) -> Result<(), Box<dyn Error>> {
    let w = std::io::BufWriter::new(File::create(path)?);
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(data)?;
    Ok(())
}

/// Write an 8-bit RGB PNG.
pub fn write_rgb(
    path: &str,
    width: u32,
    height: u32,
    data: &[u8],
) -> Result<(), Box<dyn Error>> {
    let w = std::io::BufWriter::new(File::create(path)?);
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(data)?;
    Ok(())
}

/// Images are compressed as a single square quadtree, so the input must be
/// square.
fn square_edge(width: usize, height: usize) -> Result<usize, Box<dyn Error>> {
    if width != height {
        return Err(format!("image must be square, got {width}x{height}").into());
    }
    Ok(width)
}

/// Print the estimated compressed size and the ratio against the raw size.
fn report(size: u64, raw: u64, kind: &str) {
    println!("maybe need around {size} bytes to store compressed image ({kind})");
    let ratio = raw as f64 / size as f64;
    println!("compression ratio: {ratio:.2}:1 ({raw} bytes raw -> {size} bytes)");
}

/// Tunable error bounds for the quadtree fits.
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

/// Compress `input` into `output` using the default error bounds.
pub fn compress(input: &str, output: &str) -> Result<(), Box<dyn Error>> {
    compress_with(input, output, Options::default())
}

/// Compress `input` into `output` with explicit error bounds, printing
/// progress, the estimated compressed size, and the resulting compression
/// ratio.
///
/// Greyscale PNGs are encoded directly; colour PNGs are converted to 4:2:0
/// YCbCr and each of the three planes is encoded with its own quadtree.
pub fn compress_with(input: &str, output: &str, options: Options) -> Result<(), Box<dyn Error>> {
    println!("processing  {input}");

    match read_image(input)? {
        Image::Grey(image) => {
            let n = square_edge(image.width, image.height)?;
            let root = walk(&image.data, image.width, 0, 0, n, options.max_error);

            let size = root.tree_len / 8 + root.params_len + 2;
            report(size, (image.width * image.height) as u64, "greyscale");

            let out = render(image.width, image.height, &root.leaves);
            write_greyscale(output, image.width as u32, image.height as u32, &out)?;
        }

        Image::Colour(image) => {
            let n = square_edge(image.width, image.height)?;
            let (cw, ch) = image.chroma_dims();

            // Luma gets the tight bound; the quarter-size chroma planes get
            // the looser one.
            let y_tree = walk(&image.y, image.width, 0, 0, n, options.max_error);
            let cb_tree = walk(&image.cb, cw, 0, 0, cw, options.chroma_max_error);
            let cr_tree = walk(&image.cr, cw, 0, 0, cw, options.chroma_max_error);

            let size = (y_tree.tree_len + cb_tree.tree_len + cr_tree.tree_len) / 8
                + (y_tree.params_len + cb_tree.params_len + cr_tree.params_len)
                + 2;
            report(size, (image.width * image.height * 3) as u64, "colour");

            let y = render(image.width, image.height, &y_tree.leaves);
            let cb = render(cw, ch, &cb_tree.leaves);
            let cr = render(cw, ch, &cr_tree.leaves);
            let rgb = combine_to_rgb(image.width, image.height, &y, &cb, &cr);
            write_rgb(output, image.width as u32, image.height as u32, &rgb)?;
        }
    }

    Ok(())
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

        let fit = find_plane(&image, 4, 0, 0, 4);
        assert_eq!(fit.plane, Plane { a: 2, b: 3, c: 5 });
        assert_eq!(fit.error, 0.0);
    }

    #[test]
    fn split_range_into_quad_divides_correctly() {
        assert_eq!(
            split_range_into_quad(8, 8, 4),
            [(8, 8, 2), (8, 10, 2), (10, 8, 2), (10, 10, 2)]
        );
    }

    #[test]
    fn single_pixel_is_a_leaf() {
        let image = [123u8];
        let fit = find_plane(&image, 1, 0, 0, 1);
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
        assert_eq!(
            (image.y.len(), image.cb.len(), image.cr.len()),
            (4, 1, 1)
        );

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
}
