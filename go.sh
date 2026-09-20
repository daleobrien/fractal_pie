#!/usr/bin/env bash

cargo build --release --quiet

echo ''
./target/release/pie-encode \
 --max-error 32 \
 --chroma-max-error 64 \
 lena.png lena.pie

echo ''

./target/release/pie-decode lena.pie output_lena.png

echo ''
