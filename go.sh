#!/usr/bin/env bash

cargo build --release


./target/release/pie-encode \
 --max-error 64 \
 --chroma-max-error 128 \
 lena.png lena.pie

./target/release/pie-decode lena.pie output_lena.png
