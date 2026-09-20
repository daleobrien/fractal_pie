//! Command-line parsing shared by the two binaries.
//!
//! Options may be written `--name value` or `--name=value`; a bare `-` or
//! anything not starting with `-` is a path.

use crate::{Options, CHROMA_MAX_ERROR, MAX_ERROR};

pub const DEFAULT_INPUT: &str = "lena.png";
pub const DEFAULT_PIE: &str = "lena.pie";
pub const DEFAULT_OUTPUT: &str = "decoded_lena.png";

/// The result of parsing: either run with these arguments, or print help.
pub enum Parse<T> {
    Run(T),
    Help,
}

/// Parsed `pie-encode` arguments.
pub struct EncodeArgs {
    pub input: String,
    pub output: String,
    pub options: Options,
}

/// Parsed `pie-decode` arguments.
pub struct DecodeArgs {
    pub input: String,
    pub output: String,
}

/// Pull the value of an option, accepting `--name value` and `--name=value`.
fn take_value(
    argv: &[String],
    i: &mut usize,
    name: &str,
    inline: Option<String>,
) -> Result<String, String> {
    match inline {
        Some(value) => Ok(value),
        None => {
            *i += 1;
            argv.get(*i)
                .cloned()
                .ok_or_else(|| format!("--{name} requires a value"))
        }
    }
}

/// Split an option token into its name and any inline `=value`.
fn split_option(arg: &str) -> (&str, Option<String>) {
    let body = arg.trim_start_matches('-');
    match body.split_once('=') {
        Some((name, value)) => (name, Some(value.to_string())),
        None => (body, None),
    }
}

pub fn parse_encoder(argv: &[String]) -> Result<Parse<EncodeArgs>, String> {
    let mut options = Options::default();
    let mut positional: Vec<String> = Vec::new();
    let mut i = 0;

    while i < argv.len() {
        let arg = &argv[i];
        if arg == "-" || !arg.starts_with('-') {
            positional.push(arg.clone());
            i += 1;
            continue;
        }

        let (name, inline) = split_option(arg);
        match name {
            "max-error" | "chroma-max-error" => {
                let raw = take_value(argv, &mut i, name, inline)?;
                let value: f64 = raw
                    .parse()
                    .map_err(|_| format!("--{name}: '{raw}' is not a number"))?;
                if name == "max-error" {
                    options.max_error = value;
                } else {
                    options.chroma_max_error = value;
                }
            }
            "help" | "h" => return Ok(Parse::Help),
            _ => return Err(format!("unknown option '{arg}'")),
        }
        i += 1;
    }

    if positional.len() > 2 {
        return Err(format!(
            "expected at most two paths, got {}",
            positional.len()
        ));
    }
    let input = positional
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_INPUT.to_string());
    let output = positional
        .get(1)
        .cloned()
        .unwrap_or_else(|| DEFAULT_PIE.to_string());
    Ok(Parse::Run(EncodeArgs {
        input,
        output,
        options,
    }))
}

pub fn parse_decoder(argv: &[String]) -> Result<Parse<DecodeArgs>, String> {
    let mut positional: Vec<String> = Vec::new();
    let mut i = 0;

    while i < argv.len() {
        let arg = &argv[i];
        if arg == "-" || !arg.starts_with('-') {
            positional.push(arg.clone());
            i += 1;
            continue;
        }

        let (name, _inline) = split_option(arg);
        match name {
            "help" | "h" => return Ok(Parse::Help),
            _ => return Err(format!("unknown option '{arg}'")),
        }
    }

    if positional.len() > 2 {
        return Err(format!(
            "expected at most two paths, got {}",
            positional.len()
        ));
    }
    let input = positional
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_PIE.to_string());
    let output = positional
        .get(1)
        .cloned()
        .unwrap_or_else(|| DEFAULT_OUTPUT.to_string());
    Ok(Parse::Run(DecodeArgs { input, output }))
}

pub fn print_encoder_usage() {
    println!("usage: pie-encode [input.png] [output.pie] [--max-error N] [--chroma-max-error N]");
    println!();
    println!("Fits a plane to each region of a tree, then entropy-codes the tree into a");
    println!(".pie file. A larger error bound tolerates a coarser fit, so fewer regions are");
    println!("subdivided and the file gets smaller.");
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
    println!("Images of any dimensions are accepted.");
}

pub fn print_decoder_usage() {
    println!("usage: pie-decode [input.pie] [output.png]");
    println!();
    println!("Decodes a .pie file written by pie-encode back into a PNG.");
}
