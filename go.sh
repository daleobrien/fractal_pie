#!/usr/bin/env bash

cargo build --release --quiet

echo ''
./target/release/pie-encode \
 --max-error 32 \
 --chroma-max-error 64 \
 lena.png lena.pie

./target/release/pie-encode \
 --max-error 32 \
 --chroma-max-error 64 \
 input.png input.pie


echo ''

./target/release/pie-decode lena.pie output_lena.png
./target/release/pie-decode input.pie output_input.png

echo ''
