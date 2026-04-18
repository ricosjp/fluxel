//! 3D Morton (Z-order) coding: encode and decode `(x, y, z)` into a 96-bit code, and helpers
//! to spread or gather bits along every-third-bit lanes for interleaving.
//!
//! Coordinates are treated as 32-bit logical indices. Interleaving uses one bit from each axis
//! per Morton step, which matches a left-aligned code at the finest representable resolution.

use crate::MAX_LEVEL;

/// Spreads the 32 bits of `v` so that bit `i` moves to bit index `3 * i` in a `u128`.
///
/// Two zero bits are inserted between each original bit, producing a 96-bit pattern that
/// leaves room to interleave `y` (offset `+1`) and `z` (offset `+2`).
///
/// # Example
///
/// A low bits pattern `...0011` becomes `...001_000_001` in the spread word (conceptually).
#[inline(always)]
pub fn expand_bits(v: u32) -> u128 {
    (0..MAX_LEVEL).fold(0u128, |acc, i| acc | (((v as u128 >> i) & 1) << (i * 3)))
}

/// Inverse of [`expand_bits`]: collects bits from indices `0, 3, 6, …` back into a `u32`.
#[inline(always)]
pub fn compact_bits(v: u128) -> u32 {
    (0..MAX_LEVEL).fold(0u32, |acc, i| acc | ((((v >> (i * 3)) & 1) as u32) << i))
}

/// Encodes 3D logical coordinates `(x, y, z)` into a 96-bit Morton code.
///
/// Each coordinate is in `0 ..= u32::MAX`. The code corresponds to the natural left-aligned
/// Morton order at the maximum 32-bit resolution (finer than any fixed octree level would need).
#[inline(always)]
pub fn encode(x: u32, y: u32, z: u32) -> u128 {
    expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2)
}

/// Decodes a 96-bit Morton code produced by [`encode`] back to `(x, y, z)`.
#[inline(always)]
pub fn decode(code: u128) -> (u32, u32, u32) {
    (
        compact_bits(code),
        compact_bits(code >> 1),
        compact_bits(code >> 2),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morton_encode_decode() {
        let coords = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (12345, 67890, 13579),
            (u32::MAX, u32::MAX, u32::MAX), // maximum coordinate range
        ];

        coords.iter().copied().for_each(|(x, y, z)| {
            let code = encode(x, y, z);
            let decoded = decode(code);
            assert_eq!((x, y, z), decoded);
        });
    }
}
