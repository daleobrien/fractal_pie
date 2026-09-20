//! `pie-encode`: compress a PNG into a `.pie` file.

use std::env;
use std::process::ExitCode;

use fractal_pie::cli::{self, Parse};
use fractal_pie::encode_file;

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
    match cli::parse_encoder(&argv)? {
        Parse::Help => {
            cli::print_encoder_usage();
            Ok(())
        }
        Parse::Run(args) => encode_file(&args.input, &args.output, args.options),
    }
}
