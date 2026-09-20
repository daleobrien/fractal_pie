//! A binary range coder with adaptive bit models.
//!
//! This is the classic LZMA-style range coder: probabilities are kept to 11
//! bits, the range is 32 bits and renormalisation is byte-oriented. Every bit
//! is coded against its *own* adaptive probability, which is what lets the
//! quadtree coder spend well under one bit on the very skewed decisions (most
//! nodes are leaves, most gradients are near zero).
//!
//! Encoder and decoder must drive their models in lockstep: a [`BitModel`]
//! adapts identically on both sides because it only ever sees the bits that
//! were actually coded.

/// Bits of precision in a model probability.
const MODEL_BITS: u32 = 11;
/// Total probability mass, `1 << MODEL_BITS`.
const MODEL_TOTAL: u16 = 1 << MODEL_BITS;
/// Adaptation rate: a probability moves `1/32` of the way to certainty.
const MOVE_BITS: u32 = 5;
/// Renormalise once the range drops below this.
const TOP: u32 = 1 << 24;

/// An adaptive probability for a single binary decision.
///
/// `p` is the probability that the next bit is `0`, scaled by `MODEL_TOTAL`.
/// It starts at 1/2 and drifts towards whichever bit actually occurs.
#[derive(Clone, Copy, Debug)]
pub struct BitModel {
    p: u16,
}

impl BitModel {
    #[inline]
    pub fn new() -> Self {
        BitModel { p: MODEL_TOTAL / 2 }
    }

    #[inline]
    fn update_zero(&mut self) {
        self.p += (MODEL_TOTAL - self.p) >> MOVE_BITS;
    }

    #[inline]
    fn update_one(&mut self) {
        self.p -= self.p >> MOVE_BITS;
    }
}

impl Default for BitModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Range encoder. Feed it bits, then call [`Encoder::finish`] to get the
/// byte stream.
pub struct Encoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    out: Vec<u8>,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    pub fn new() -> Self {
        Encoder {
            low: 0,
            range: 0xFFFF_FFFF,
            cache: 0,
            cache_size: 1,
            out: Vec::new(),
        }
    }

    /// Emit the top byte of `low`, handling any carry into the pending bytes.
    #[inline]
    fn shift_low(&mut self) {
        if self.low < 0xFF00_0000 || self.low > 0xFFFF_FFFF {
            let carry = (self.low >> 32) as u8;
            let mut temp = self.cache;
            while self.cache_size > 0 {
                self.out.push(temp.wrapping_add(carry));
                temp = 0xFF;
                self.cache_size -= 1;
            }
            self.cache = ((self.low >> 24) & 0xFF) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low << 8) & 0xFFFF_FFFF;
    }

    /// Code one bit against `model`, updating the model in place.
    ///
    /// `bit` is 0 or 1; anything non-zero is treated as 1.
    #[inline]
    pub fn encode_bit(&mut self, model: &mut BitModel, bit: u32) {
        let bound = (self.range >> MODEL_BITS) * model.p as u32;
        if bit == 0 {
            self.range = bound;
            model.update_zero();
        } else {
            self.low += bound as u64;
            self.range -= bound;
            model.update_one();
        }
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    /// Flush the coder and return the encoded bytes.
    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

/// Range decoder. Consumes the byte stream produced by [`Encoder`].
pub struct Decoder<'a> {
    code: u32,
    range: u32,
    input: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(input: &'a [u8]) -> Self {
        let mut dec = Decoder {
            code: 0,
            range: 0xFFFF_FFFF,
            input,
            pos: 0,
        };
        // The encoder emits one leading cache byte plus five flush bytes; the
        // first five are primed into `code`.
        for _ in 0..5 {
            dec.code = (dec.code << 8) | dec.next_byte() as u32;
        }
        dec
    }

    /// Bytes read so far. `0` is returned past the end of the input, which
    /// lets a truncated stream decode deterministically instead of panicking.
    #[inline]
    fn next_byte(&mut self) -> u8 {
        let byte = self.input.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        byte
    }

    /// Decode one bit against `model`, updating the model in place.
    #[inline]
    pub fn decode_bit(&mut self, model: &mut BitModel) -> u32 {
        let bound = (self.range >> MODEL_BITS) * model.p as u32;
        let bit;
        if self.code < bound {
            self.range = bound;
            model.update_zero();
            bit = 0;
        } else {
            self.code -= bound;
            self.range -= bound;
            model.update_one();
            bit = 1;
        }
        while self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | self.next_byte() as u32;
        }
        bit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modelled_bits_round_trip() {
        // A heavily skewed stream: 1 in 16 set, but the model should still
        // reconstruct it exactly.
        let bits: Vec<u32> = (0..5000).map(|i| if i % 16 == 0 { 1 } else { 0 }).collect();
        let mut enc = Encoder::new();
        let mut model = BitModel::new();
        for &b in &bits {
            enc.encode_bit(&mut model, b);
        }
        let bytes = enc.finish();

        let mut dec = Decoder::new(&bytes);
        let mut model = BitModel::new();
        for &b in &bits {
            assert_eq!(dec.decode_bit(&mut model), b);
        }
    }

    #[test]
    fn skew_is_compressed() {
        // 5000 bits with one 1 per 16 costs ~0.37 bits each in practice, far
        // below the 1 bit/symbol a fixed-width file would spend.
        let mut enc = Encoder::new();
        let mut model = BitModel::new();
        for i in 0..5000 {
            enc.encode_bit(&mut model, if i % 16 == 0 { 1 } else { 0 });
        }
        let bytes = enc.finish();
        assert!(bytes.len() * 8 < 2500, "coded {} bits", bytes.len() * 8);
    }
}
