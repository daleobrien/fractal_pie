use std::env;
use std::process::ExitCode;

use fractal_pie::{compress_with, Options, CHROMA_MAX_ERROR, MAX_ERROR};

const DEFAULT_INPUT: &str = "lena.png";
const DEFAULT_OUTPUT: &str = "output_lena.png";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = env::args().skip(1).collect();

    let mut options = Options::default();
    let mut positional: Vec<String> = Vec::new();

    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].clone();

        // A bare `-` or anything not starting with `-` is a path.
        if arg == "-" || !arg.starts_with('-') {
            positional.push(arg);
            i += 1;
            continue;
        }

        // Options may be written `--name value` or `--name=value`.
        let body = arg.trim_start_matches('-');
        let (name, inline) = match body.split_once('=') {
            Some((n, v)) => (n.to_string(), Some(v.to_string())),
            None => (body.to_string(), None),
        };

        match name.as_str() {
            "max-error" | "chroma-max-error" => {
                let raw = match inline {
                    Some(v) => v,
                    None => {
                        i += 1;
                        argv.get(i)
                            .cloned()
                            .ok_or_else(|| format!("--{name} requires a value"))?
                    }
                };
                let value: f64 = raw
                    .parse()
                    .map_err(|_| format!("--{name}: '{raw}' is not a number"))?;
                if name == "max-error" {
                    options.max_error = value;
                } else {
                    options.chroma_max_error = value;
                }
            }
            "help" | "h" => {
                print_usage();
                return Ok(());
            }
            _ => return Err(format!("unknown option '{arg}'").into()),
        }

        i += 1;
    }

    if positional.len() > 2 {
        return Err(format!("expected at most two paths, got {}", positional.len()).into());
    }
    let input = positional
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_INPUT.to_string());
    let output = positional
        .get(1)
        .cloned()
        .unwrap_or_else(|| DEFAULT_OUTPUT.to_string());

    compress_with(&input, &output, options)
}

fn print_usage() {
    println!("usage: fractal-pie [input.png] [output.png] [--max-error N] [--chroma-max-error N]");
    println!();
    println!("Fits a plane to each region of a quadtree. A larger error bound tolerates a");
    println!("coarser fit, so fewer regions are subdivided and the image compresses harder.");
    println!();
    println!(
        "  --max-error N         luma / greyscale bound (default {})",
        MAX_ERROR
    );
    println!(
        "  --chroma-max-error N  chroma bound for colour input (default {})",
        CHROMA_MAX_ERROR
    );
    println!();
    println!("Greyscale PNGs are encoded directly; colour PNGs use 4:2:0 YCbCr.");
}
