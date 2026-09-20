//! Fractal image compression by recursively fitting planes to a quadtree.
//!
//! For each square sub-image we fit the plane `pixel = a*i + b*j + c` by
//! least squares. If the mean-squared error of that fit exceeds a threshold
//! the sub-image is split into four quadrants and the process repeats on each.
//!
//! This is a Rust port of the original Python prototype (`compress.py`). It
//! deliberately reproduces the original's arithmetic (including its integer
//! rounding and its quadtree bookkeeping) so that the rendered image and the
//! reported compressed-size estimate match.

use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use rayon::prelude::*;

/// Maximum mean-squared error tolerated by a fitted plane before the
/// sub-image is subdivided. Matches the original `max_error` of 32.
pub const MAX_ERROR: f64 = 32.0;

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

fn walk(image: &[u8], width: usize, x: usize, y: usize, n: usize) -> Subtree {
    let fit = find_plane(image, width, x, y, n);

    if fit.error > MAX_ERROR {
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
                .map(|&(qx, qy, qn)| walk(image, width, qx, qy, qn))
                .collect();
            for child in children {
                absorb(child, &mut subtree);
            }
        } else {
            for &(qx, qy, qn) in &quads {
                absorb(walk(image, width, qx, qy, qn), &mut subtree);
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

/// Read a PNG and convert it to 8-bit greyscale.
///
/// RGB(A) pixels are averaged as `(r + g + b) / 3`, matching the original.
pub fn read_greyscale(path: &str) -> Result<Grey, Box<dyn Error>> {
    let file = BufReader::new(File::open(path)?);
    let mut decoder = png::Decoder::new(file);
    // Normalise palettes / bit depths to plain 8-bit samples.
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);

    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    let data = &buf[..info.buffer_size()];

    let grey: Vec<u8> = match info.color_type {
        png::ColorType::Grayscale => data.to_vec(),
        png::ColorType::GrayscaleAlpha => data.chunks_exact(2).map(|p| p[0]).collect(),
        png::ColorType::Rgb => data
            .chunks_exact(3)
            .map(|p| ((p[0] as u16 + p[1] as u16 + p[2] as u16) / 3) as u8)
            .collect(),
        png::ColorType::Rgba => data
            .chunks_exact(4)
            .map(|p| ((p[0] as u16 + p[1] as u16 + p[2] as u16) / 3) as u8)
            .collect(),
        other => return Err(format!("unsupported PNG colour type: {other:?}").into()),
    };

    Ok(Grey {
        width: info.width as usize,
        height: info.height as usize,
        data: grey,
    })
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

/// Compress `input` into `output`, printing progress, the estimated
/// compressed size, and the resulting compression ratio.
pub fn compress(input: &str, output: &str) -> Result<(), Box<dyn Error>> {
    println!("processing  {input}");

    let image = read_greyscale(input)?;
    if image.width != image.height {
        return Err(format!("image must be square, got {}x{}", image.width, image.height).into());
    }

    let n = image.height;
    let root = walk(&image.data, image.width, 0, 0, n);

    let approximate_compressed_size = root.tree_len / 8 + root.params_len + 2;
    println!(
        "maybe need around {} bytes to store compressed image (greyscale)",
        approximate_compressed_size
    );

    // The raw representation is one byte per greyscale pixel.
    let raw_size = (image.width * image.height) as u64;
    let ratio = raw_size as f64 / approximate_compressed_size as f64;
    println!(
        "compression ratio: {ratio:.2}:1 ({raw_size} bytes raw -> {approximate_compressed_size} bytes)"
    );

    let out = render(image.width, image.height, &root.leaves);
    write_greyscale(output, image.width as u32, image.height as u32, &out)?;

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
}
