//! Shared helpers for the LoongArch SIMD correctness tests.

use crate::loongarch64::utils::y_bias;
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvRange, YuvSourceChannels};

/// Representative BT.601 limited-range 8-bit transform (PRECISION = 13). The
/// exact values only need to be valid; the tests compare the SIMD kernel
/// against the scalar reference for whatever transform is supplied here.
pub(crate) fn bt601_limited() -> (YuvChromaRange, CbCrForwardTransform<i32>) {
    let range = YuvChromaRange {
        bias_y: 16,
        bias_uv: 128,
        range_y: 219,
        range_uv: 224,
        range: YuvRange::Limited,
    };
    let transform = CbCrForwardTransform::<i32> {
        yr: 2104,
        yg: 4130,
        yb: 802,
        cb_r: -1214,
        cb_g: -2384,
        cb_b: 3598,
        cr_r: 3598,
        cr_g: -3013,
        cr_b: -585,
    };
    (range, transform)
}

/// Scalar luma reference, bit-identical to the SIMD kernel.
pub(crate) fn ref_y<const PRECISION: i32>(
    r: i32,
    g: i32,
    b: i32,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
) -> u8 {
    let bias = y_bias::<PRECISION>(range);
    ((r * transform.yr + g * transform.yg + b * transform.yb + bias) >> PRECISION) as u8
}

/// Build two deterministic pseudo-random pixel rows of `width` pixels each.
pub(crate) fn make_rgba_rows<const ORIGIN_CHANNELS: u8>(width: usize) -> (Vec<u8>, Vec<u8>) {
    let channels = YuvSourceChannels::from(ORIGIN_CHANNELS).get_channels_count();
    let len = width * channels;
    let mut row0 = vec![0u8; len];
    let mut row1 = vec![0u8; len];
    // Small xorshift keeps the test deterministic without pulling in `rand`.
    let mut state: u32 = 0x1234_5678;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state & 0xff) as u8
    };
    for b in row0.iter_mut() {
        *b = next();
    }
    for b in row1.iter_mut() {
        *b = next();
    }
    (row0, row1)
}
