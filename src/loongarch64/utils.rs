use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
use core::arch::loongarch64::*;

/// Pixels processed per LSX block (128-bit registers hold 16 bytes).
pub(crate) const LSX_BLOCK: usize = 16;

/// Fixed-point `bias` used for the luma channel:
/// `bias_y * 2^P + (2^(P-1) - 1)`, matching the scalar reference exactly.
#[inline(always)]
pub(crate) fn y_bias<const PRECISION: i32>(range: &YuvChromaRange) -> i32 {
    range.bias_y as i32 * (1 << PRECISION) + ((1 << (PRECISION - 1)) - 1)
}

/// Fixed-point `bias` used for the chroma channels.
#[inline(always)]
pub(crate) fn uv_bias<const PRECISION: i32>(range: &YuvChromaRange) -> i32 {
    range.bias_uv as i32 * (1 << PRECISION) + ((1 << (PRECISION - 1)) - 1)
}

// ---------------------------------------------------------------------------
// LSX (128-bit) helpers
// ---------------------------------------------------------------------------

/// Load 16 interleaved pixels and deinterleave into planar `(r, g, b)`,
/// one byte per lane.
///
/// 4-channel inputs (`Rgba`/`Bgra`) use a branch-free two-stage
/// `vpickev`/`vpickod` deinterleave. 3-channel inputs (`Rgb`/`Bgr`) — which are
/// never produced by this product's pipeline — fall back to a correct scalar
/// gather; they are functionally valid but not optimized.
#[inline]
#[target_feature(enable = "lsx")]
pub(crate) unsafe fn lsx_load_deinterleave16<const ORIGIN_CHANNELS: u8>(
    ptr: *const u8,
) -> (m128i, m128i, m128i) {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    match source_channels {
        YuvSourceChannels::Rgba | YuvSourceChannels::Bgra => {
            let v0 = lsx_vld::<0>(ptr as *const i8);
            let v1 = lsx_vld::<0>(ptr.add(16) as *const i8);
            let v2 = lsx_vld::<0>(ptr.add(32) as *const i8);
            let v3 = lsx_vld::<0>(ptr.add(48) as *const i8);

            // Stage 1: split even/odd bytes. `vpickev_b(a, b)` = [even(b), even(a)].
            let e01 = lsx_vpickev_b(v1, v0); // ch0,ch2 of px0..7
            let o01 = lsx_vpickod_b(v1, v0); // ch1,ch3 of px0..7
            let e23 = lsx_vpickev_b(v3, v2); // ch0,ch2 of px8..15
            let o23 = lsx_vpickod_b(v3, v2); // ch1,ch3 of px8..15

            // Stage 2: separate the channels.
            let c0 = lsx_vpickev_b(e23, e01); // ch0 (R for Rgba, B for Bgra)
            let c2 = lsx_vpickod_b(e23, e01); // ch2 (B for Rgba, R for Bgra)
            let c1 = lsx_vpickev_b(o23, o01); // ch1 (G)

            if source_channels == YuvSourceChannels::Rgba {
                (c0, c1, c2)
            } else {
                (c2, c1, c0)
            }
        }
        YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
            let channels = source_channels.get_channels_count();
            let r_off = source_channels.get_r_channel_offset();
            let g_off = source_channels.get_g_channel_offset();
            let b_off = source_channels.get_b_channel_offset();
            let mut rr = [0u8; 16];
            let mut gg = [0u8; 16];
            let mut bb = [0u8; 16];
            let mut i = 0usize;
            while i < 16 {
                let px = i * channels;
                rr[i] = *ptr.add(px + r_off);
                gg[i] = *ptr.add(px + g_off);
                bb[i] = *ptr.add(px + b_off);
                i += 1;
            }
            (
                lsx_vld::<0>(rr.as_ptr() as *const i8),
                lsx_vld::<0>(gg.as_ptr() as *const i8),
                lsx_vld::<0>(bb.as_ptr() as *const i8),
            )
        }
    }
}

/// Widen 16 unsigned bytes into four i32x4 vectors (pixels 0-3, 4-7, 8-11, 12-15).
#[inline]
#[target_feature(enable = "lsx")]
unsafe fn lsx_widen_u8_to_i32(x: m128i) -> (m128i, m128i, m128i, m128i) {
    let lo16 = lsx_vsllwil_hu_bu::<0>(x); // px0..7 as u16
    let hi16 = lsx_vexth_hu_bu(x); // px8..15 as u16
    (
        lsx_vsllwil_wu_hu::<0>(lo16), // px0..3
        lsx_vexth_wu_hu(lo16),        // px4..7
        lsx_vsllwil_wu_hu::<0>(hi16), // px8..11
        lsx_vexth_wu_hu(hi16),        // px12..15
    )
}

/// Compute 16 luma samples from planar `(r, g, b)` bytes.
///
/// Bit-identical to the scalar path: each sample is
/// `(r*yr + g*yg + b*yb + bias) >> PRECISION` (arithmetic shift), packed to u8.
#[inline]
#[target_feature(enable = "lsx")]
pub(crate) unsafe fn lsx_encode_y16<const PRECISION: i32>(
    r: m128i,
    g: m128i,
    b: m128i,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
) -> m128i {
    let (r0, r1, r2, r3) = lsx_widen_u8_to_i32(r);
    let (g0, g1, g2, g3) = lsx_widen_u8_to_i32(g);
    let (b0, b1, b2, b3) = lsx_widen_u8_to_i32(b);

    let vyr = lsx_vreplgr2vr_w(transform.yr);
    let vyg = lsx_vreplgr2vr_w(transform.yg);
    let vyb = lsx_vreplgr2vr_w(transform.yb);
    let vbias = lsx_vreplgr2vr_w(y_bias::<PRECISION>(range));

    // vmadd_w(acc, x, c) = acc + x * c
    let a0 = lsx_vmadd_w(lsx_vmadd_w(lsx_vmadd_w(vbias, r0, vyr), g0, vyg), b0, vyb);
    let a1 = lsx_vmadd_w(lsx_vmadd_w(lsx_vmadd_w(vbias, r1, vyr), g1, vyg), b1, vyb);
    let a2 = lsx_vmadd_w(lsx_vmadd_w(lsx_vmadd_w(vbias, r2, vyr), g2, vyg), b2, vyb);
    let a3 = lsx_vmadd_w(lsx_vmadd_w(lsx_vmadd_w(vbias, r3, vyr), g3, vyg), b3, vyb);

    // `>> PRECISION` with saturating narrow i32 -> i16. `vssrani_h_w(a, b)`
    // packs [narrow(b), narrow(a)] so order stays px0..7 then px8..15.
    // PRECISION is guaranteed to be 13 on this path (the dispatch returns early
    // otherwise), so the shift immediate is a literal.
    let h01 = lsx_vssrani_h_w::<13>(a1, a0); // px0..7
    let h23 = lsx_vssrani_h_w::<13>(a3, a2); // px8..15
    lsx_vssrani_bu_h::<0>(h23, h01) // 16 u8 luma, px0..15
}

/// 2x2 box-downsample two rows of planar bytes into 8 averaged i16 samples per
/// channel: `(a + b + c + d + 2) >> 2`, matching the scalar reference.
#[inline]
#[target_feature(enable = "lsx")]
pub(crate) unsafe fn lsx_downsample_channel(row0: m128i, row1: m128i, two: m128i) -> m128i {
    // `vhaddw_hu_bu(x, x)[i] = x[2i] + x[2i+1]` — horizontal adjacent-pair sum.
    let p0 = lsx_vhaddw_hu_bu(row0, row0);
    let p1 = lsx_vhaddw_hu_bu(row1, row1);
    let sum = lsx_vadd_h(p0, p1); // sum of the four samples (max 1020)
    lsx_vsrai_h::<2>(lsx_vadd_h(sum, two)) // (sum + 2) >> 2
}

/// Compute 8 chroma samples (i16x8) from averaged `(r, g, b)` for one
/// coefficient triple. Used twice, once for Cb and once for Cr.
#[inline]
#[target_feature(enable = "lsx")]
unsafe fn lsx_encode_chroma8<const PRECISION: i32>(
    r_avg: m128i,
    g_avg: m128i,
    b_avg: m128i,
    cr: i32,
    cg: i32,
    cb: i32,
    vbias: m128i,
) -> m128i {
    let r_lo = lsx_vsllwil_w_h::<0>(r_avg); // px0..3
    let r_hi = lsx_vexth_w_h(r_avg); // px4..7
    let g_lo = lsx_vsllwil_w_h::<0>(g_avg);
    let g_hi = lsx_vexth_w_h(g_avg);
    let b_lo = lsx_vsllwil_w_h::<0>(b_avg);
    let b_hi = lsx_vexth_w_h(b_avg);

    let vcr = lsx_vreplgr2vr_w(cr);
    let vcg = lsx_vreplgr2vr_w(cg);
    let vcb = lsx_vreplgr2vr_w(cb);

    let lo = lsx_vmadd_w(lsx_vmadd_w(lsx_vmadd_w(vbias, r_lo, vcr), g_lo, vcg), b_lo, vcb);
    let hi = lsx_vmadd_w(lsx_vmadd_w(lsx_vmadd_w(vbias, r_hi, vcr), g_hi, vcg), b_hi, vcb);
    lsx_vssrani_h_w::<13>(hi, lo) // PRECISION == 13; 8 chroma samples, px0..7
}

/// Result of encoding one 16-pixel-wide 2-row block.
pub(crate) struct LsxBlock {
    pub y0: m128i,
    pub y1: m128i,
    /// 8 Cb samples in the low 8 lanes (i16x8).
    pub cb: m128i,
    /// 8 Cr samples in the low 8 lanes (i16x8).
    pub cr: m128i,
}

/// Encode one 16-pixel-wide, 2-row block: 32 luma samples and 8 chroma pairs.
/// Loads and deinterleaves each row once and reuses the bytes for both luma and
/// chroma.
#[inline]
#[target_feature(enable = "lsx")]
pub(crate) unsafe fn lsx_encode_block16<const ORIGIN_CHANNELS: u8, const PRECISION: i32>(
    rgba0: *const u8,
    rgba1: *const u8,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
) -> LsxBlock {
    debug_assert_eq!(PRECISION, 13, "LSX kernel only supports PRECISION == 13");
    let (r0, g0, b0) = lsx_load_deinterleave16::<ORIGIN_CHANNELS>(rgba0);
    let (r1, g1, b1) = lsx_load_deinterleave16::<ORIGIN_CHANNELS>(rgba1);

    let y0 = lsx_encode_y16::<PRECISION>(r0, g0, b0, range, transform);
    let y1 = lsx_encode_y16::<PRECISION>(r1, g1, b1, range, transform);

    let two = lsx_vreplgr2vr_h(2);
    let r_avg = lsx_downsample_channel(r0, r1, two);
    let g_avg = lsx_downsample_channel(g0, g1, two);
    let b_avg = lsx_downsample_channel(b0, b1, two);

    let vbias = lsx_vreplgr2vr_w(uv_bias::<PRECISION>(range));
    let cb = lsx_encode_chroma8::<PRECISION>(
        r_avg,
        g_avg,
        b_avg,
        transform.cb_r,
        transform.cb_g,
        transform.cb_b,
        vbias,
    );
    let cr = lsx_encode_chroma8::<PRECISION>(
        r_avg,
        g_avg,
        b_avg,
        transform.cr_r,
        transform.cr_g,
        transform.cr_b,
        vbias,
    );

    LsxBlock { y0, y1, cb, cr }
}

/// Store 16 luma bytes.
#[inline]
#[target_feature(enable = "lsx")]
pub(crate) unsafe fn lsx_store_y(dst: *mut u8, y: m128i) {
    lsx_vst::<0>(y, dst as *mut i8);
}

/// Saturating-narrow an i16x8 chroma vector to u8, duplicated across both halves.
#[inline]
#[target_feature(enable = "lsx")]
pub(crate) unsafe fn lsx_chroma_to_u8(c: m128i) -> m128i {
    lsx_vssrani_bu_h::<0>(c, c)
}

/// Store the low 8 bytes (one i64 lane) of `v`.
#[inline]
#[target_feature(enable = "lsx")]
pub(crate) unsafe fn lsx_store8(dst: *mut u8, v: m128i) {
    lsx_vstelm_d::<0, 0>(v, dst as *mut i8);
}

/// Store 16 bytes.
#[inline]
#[target_feature(enable = "lsx")]
pub(crate) unsafe fn lsx_store16(dst: *mut u8, v: m128i) {
    lsx_vst::<0>(v, dst as *mut i8);
}

/// Interleave two u8x8 (low halves) into a u8x16. `vilvl_b(a, b)` places `b` in
/// the even lanes, so `even` becomes the even (first) channel of each pair.
#[inline]
#[target_feature(enable = "lsx")]
pub(crate) unsafe fn lsx_interleave_lo(even: m128i, odd: m128i) -> m128i {
    lsx_vilvl_b(odd, even)
}
