//! The whole frame is processed at 8-bit here (SIMD interior + scalar edges), so
//! there is no precision seam between the vectorised columns and the tail.

use crate::images::YuvPlanarImageMut;
use crate::images::YuvBiPlanarImageMut;
use crate::yuv_support::{YuvNVOrder, YuvRange, YuvSourceChannels, YuvStandardMatrix};

// ---------------------------------------------------------------------------
// 8-bit coefficients (transcribed from libyuv `row_common.cc`)
// ---------------------------------------------------------------------------

/// 8-bit fixed-point RGB→YUV coefficients. `Y = (ry*r + gy*g + by*b + ay) >> 8`,
/// `U = (ru*r + gu*g + bu*b + auv) >> 8`, `V = (rv*r + gv*g + bv*b + auv) >> 8`.
///
/// `ay` is `16*256 + 128` for limited range and `128` for full range; `auv` is
/// always `128*256`. The signed coefficients are stored as `i32` for the scalar
/// path and narrowed to `i16` lanes for SIMD (every value fits `i16`).
#[derive(Clone, Copy)]
pub(crate) struct Coeffs8 {
    pub ry: i32,
    pub gy: i32,
    pub by: i32,
    pub ay: i32,
    pub ru: i32,
    pub gu: i32,
    pub bu: i32,
    pub rv: i32,
    pub gv: i32,
    pub bv: i32,
    pub auv: i32,
}

/// Select the libyuv 8-bit coefficient set for a `(matrix, range)` pair, or
/// `None` for combinations libyuv does not tabulate (the caller then falls back
/// to the generic 13-bit path).
pub(crate) fn coeffs8(matrix: YuvStandardMatrix, range: YuvRange) -> Option<Coeffs8> {
    // (ry, gy, by, ru, gu, bu, rv, gv, bv, ay, auv)
    let (ry, gy, by, ru, gu, bu, rv, gv, bv, ay, auv) = match (matrix, range) {
        (YuvStandardMatrix::Bt601, YuvRange::Limited) => {
            (66, 129, 25, -38, -74, 112, 112, -94, -18, 4224, 32768)
        }
        (YuvStandardMatrix::Bt601, YuvRange::Full) => {
            (77, 150, 29, -43, -85, 128, 128, -107, -21, 128, 32768)
        }
        (YuvStandardMatrix::Bt709, YuvRange::Limited) => {
            (47, 157, 16, -26, -86, 112, 112, -102, -10, 4224, 32768)
        }
        (YuvStandardMatrix::Bt709, YuvRange::Full) => {
            (54, 183, 19, -29, -99, 128, 128, -116, -12, 128, 32768)
        }
        (YuvStandardMatrix::Bt2020, YuvRange::Limited) => {
            (59, 148, 13, -31, -81, 112, 112, -103, -9, 4224, 32768)
        }
        (YuvStandardMatrix::Bt2020, YuvRange::Full) => {
            (67, 174, 15, -36, -92, 128, 128, -118, -10, 128, 32768)
        }
        // Smpte240 / Bt470_6: no libyuv 8-bit table — caller falls back to 13-bit.
        _ => return None,
    };
    Some(Coeffs8 {
        ry,
        gy,
        by,
        ay,
        ru,
        gu,
        bu,
        rv,
        gv,
        bv,
        auv,
    })
}

// ---------------------------------------------------------------------------
// Scalar reference (exact libyuv C math) — used for the frame edges and as the
// test oracle.
// ---------------------------------------------------------------------------

#[inline(always)]
pub(crate) fn rgb_to_y(r: i32, g: i32, b: i32, c: &Coeffs8) -> u8 {
    ((c.ry * r + c.gy * g + c.by * b + c.ay) >> 8) as u8
}

#[inline(always)]
pub(crate) fn rgb_to_u(r: i32, g: i32, b: i32, c: &Coeffs8) -> u8 {
    ((c.ru * r + c.gu * g + c.bu * b + c.auv) >> 8) as u8
}

#[inline(always)]
pub(crate) fn rgb_to_v(r: i32, g: i32, b: i32, c: &Coeffs8) -> u8 {
    ((c.rv * r + c.gv * g + c.bv * b + c.auv) >> 8) as u8
}

/// 2x2 box average of one channel over up to two rows / two columns, matching
/// libyuv's edge rules: `(a+b+c+d+2)>>2` for a full quad, `(a+b+1)>>1` for a
/// pair, the sample itself for a single.
#[inline(always)]
fn avg_quad(s00: i32, s01: i32, s10: i32, s11: i32, two_cols: bool, two_rows: bool) -> i32 {
    match (two_cols, two_rows) {
        (true, true) => (s00 + s01 + s10 + s11 + 2) >> 2,
        (true, false) => (s00 + s01 + 1) >> 1,
        (false, true) => (s00 + s10 + 1) >> 1,
        (false, false) => s00,
    }
}

// ---------------------------------------------------------------------------
// SIMD kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "loongarch64")]
mod simd {
    use super::Coeffs8;
    use crate::loongarch64::utils::{lsx_downsample_channel, lsx_load_deinterleave16};
    use crate::yuv_support::YuvSourceChannels;
    use core::arch::loongarch64::*;

    /// Pixels per LSX block.
    pub(super) const LSX_BLOCK: usize = 16;

    /// Broadcast `v` (low 16 bits used) across an `i16x8` register.
    #[inline]
    #[target_feature(enable = "lsx")]
    unsafe fn splat_h(v: i32) -> m128i {
        lsx_vreplgr2vr_h(v)
    }

    /// Byte permutation that interleaves the even-pixel / odd-pixel luma halves
    /// back to linear order after the `maddwev`/`maddwod` split: result byte `i`
    /// picks `[Yev0, Yod0, Yev1, Yod1, …]`.
    #[inline]
    #[target_feature(enable = "lsx")]
    unsafe fn lsx_y_pack_mask() -> m128i {
        let m: [u8; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];
        lsx_vld::<0>(m.as_ptr() as *const i8)
    }

    /// Compute 16 luma bytes from planar `(r, g, b)` u8x16.
    ///
    /// libyuv-style: `vmaddwev/vmaddwod.h.bu` multiply-accumulate `u8×u8 → u16`
    /// **directly on the bytes** (no explicit widen). `maddwev` handles even
    /// pixels, `maddwod` odd pixels; `vssrlni_bu_h::<8>` takes the high byte
    /// (`>>8`) of both halves and one `vshuf_b` re-interleaves them to linear
    /// order. All partial sums stay in `[0, 65536)`, so the `u16` math is exact.
    #[inline]
    #[target_feature(enable = "lsx")]
    unsafe fn lsx_y16(r: m128i, g: m128i, b: m128i, c: &Coeffs8) -> m128i {
        let bias = lsx_vreplgr2vr_h(c.ay);
        let (vry, vgy, vby) = (
            lsx_vreplgr2vr_b(c.ry),
            lsx_vreplgr2vr_b(c.gy),
            lsx_vreplgr2vr_b(c.by),
        );

        let acc_ev = lsx_vmaddwev_h_bu(
            lsx_vmaddwev_h_bu(lsx_vmaddwev_h_bu(bias, r, vry), g, vgy),
            b,
            vby,
        );
        let acc_od = lsx_vmaddwod_h_bu(
            lsx_vmaddwod_h_bu(lsx_vmaddwod_h_bu(bias, r, vry), g, vgy),
            b,
            vby,
        );
        // `vssrlni_bu_h::<8>(a, b)` = [sat_u8(b>>8) : sat_u8(a>>8)] → [Yev | Yod].
        let packed = lsx_vssrlni_bu_h::<8>(acc_od, acc_ev);
        lsx_vshuf_b(packed, packed, lsx_y_pack_mask())
    }

    /// One chroma vector (8 samples) for a coefficient triple. `r_avg/g_avg/b_avg`
    /// are i16x8 box-averaged samples (each in `0..=255`). Accumulation order
    /// (bias, R, G, B) keeps every partial in `[0, 65536)` for both U and V, so
    /// the high byte is the exact `>>8` result.
    #[inline]
    #[target_feature(enable = "lsx")]
    unsafe fn lsx_chroma8(
        r_avg: m128i,
        g_avg: m128i,
        b_avg: m128i,
        cr: i32,
        cg: i32,
        cb: i32,
        bias: m128i,
    ) -> m128i {
        let (vr, vg, vb) = (splat_h(cr), splat_h(cg), splat_h(cb));
        let acc = lsx_vmadd_h(lsx_vmadd_h(lsx_vmadd_h(bias, r_avg, vr), g_avg, vg), b_avg, vb);
        // Pack the 8 high bytes into the low 8 lanes.
        lsx_vpickod_b(acc, acc)
    }

    /// Result of one 16-px-wide, 2-row block: 16+16 luma bytes and 8 U / 8 V
    /// samples (each in the low 8 lanes of its register).
    pub(super) struct LsxBlock {
        pub y0: m128i,
        pub y1: m128i,
        pub u: m128i,
        pub v: m128i,
    }

    /// Encode one 16-px-wide, 2-row block.
    #[inline]
    #[target_feature(enable = "lsx")]
    pub(super) unsafe fn lsx_block16<const ORIGIN_CHANNELS: u8>(
        rgba0: *const u8,
        rgba1: *const u8,
        c: &Coeffs8,
    ) -> LsxBlock {
        let (r0, g0, b0) = lsx_load_deinterleave16::<ORIGIN_CHANNELS>(rgba0);
        let (r1, g1, b1) = lsx_load_deinterleave16::<ORIGIN_CHANNELS>(rgba1);

        let y0 = lsx_y16(r0, g0, b0, c);
        let y1 = lsx_y16(r1, g1, b1, c);

        let two = splat_h(2);
        let r_avg = lsx_downsample_channel(r0, r1, two);
        let g_avg = lsx_downsample_channel(g0, g1, two);
        let b_avg = lsx_downsample_channel(b0, b1, two);

        let bias = splat_h(c.auv);
        let u = lsx_chroma8(r_avg, g_avg, b_avg, c.ru, c.gu, c.bu, bias);
        let v = lsx_chroma8(r_avg, g_avg, b_avg, c.rv, c.gv, c.bv, bias);
        LsxBlock { y0, y1, u, v }
    }

    #[inline]
    #[target_feature(enable = "lsx")]
    pub(super) unsafe fn store16(dst: *mut u8, v: m128i) {
        lsx_vst::<0>(v, dst as *mut i8);
    }

    /// Store the low 8 bytes (one i64 lane).
    #[inline]
    #[target_feature(enable = "lsx")]
    pub(super) unsafe fn store8(dst: *mut u8, v: m128i) {
        lsx_vstelm_d::<0, 0>(v, dst as *mut i8);
    }

    /// Interleave the low 8 bytes of `first`/`second` into 16 bytes (NV12/NV21).
    #[inline]
    #[target_feature(enable = "lsx")]
    pub(super) unsafe fn interleave_lo(first: m128i, second: m128i) -> m128i {
        lsx_vilvl_b(second, first)
    }

    // -----------------------------------------------------------------------
    // LASX (256-bit) — 32 pixels per block
    // -----------------------------------------------------------------------

    /// Pixels per LASX block.
    pub(super) const LASX_BLOCK: usize = 32;

    /// Word-permutation that linearises a channel after the two-stage
    /// `xvpickev`/`xvpickod` deinterleave. LASX pick* operate per 128-bit lane,
    /// so the 8 4-pixel words come out as `[0,4,1,5,2,6,3,7]`; this control
    /// reorders them back to `0..8`. (Identical to libyuv's `shuff`/`control`.)
    #[inline]
    #[target_feature(enable = "lasx")]
    unsafe fn perm_ctrl() -> m256i {
        let c: [i32; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
        lasx_xvld::<0>(c.as_ptr() as *const i8)
    }

    /// Broadcast `v` (low 16 bits) across an i16x16 register.
    #[inline]
    #[target_feature(enable = "lasx")]
    unsafe fn xsplat_h(v: i32) -> m256i {
        lasx_xvreplgr2vr_h(v)
    }

    /// Load 32 interleaved 4-channel pixels and deinterleave into linear planar
    /// `(r, g, b)`, one byte per lane. 4-channel only (the product feeds BGRA);
    /// callers handle 3-channel via the scalar path.
    #[inline]
    #[target_feature(enable = "lasx")]
    unsafe fn lasx_load_deinterleave32<const ORIGIN_CHANNELS: u8>(
        ptr: *const u8,
    ) -> (m256i, m256i, m256i) {
        let v0 = lasx_xvld::<0>(ptr as *const i8);
        let v1 = lasx_xvld::<32>(ptr as *const i8);
        let v2 = lasx_xvld::<64>(ptr as *const i8);
        let v3 = lasx_xvld::<96>(ptr as *const i8);

        // Stage 1: split even/odd bytes (per 128-bit lane).
        let e01 = lasx_xvpickev_b(v1, v0); // ch0,ch2 of px0-3,8-11 | px4-7,12-15
        let o01 = lasx_xvpickod_b(v1, v0); // ch1,ch3
        let e23 = lasx_xvpickev_b(v3, v2);
        let o23 = lasx_xvpickod_b(v3, v2);

        // Stage 2: separate channels (still lane-scrambled in 4-pixel words).
        let c0 = lasx_xvpickev_b(e23, e01); // ch0
        let c2 = lasx_xvpickod_b(e23, e01); // ch2
        let c1 = lasx_xvpickev_b(o23, o01); // ch1

        // Linearise the 4-pixel words.
        let ctrl = perm_ctrl();
        let c0 = lasx_xvperm_w(c0, ctrl);
        let c1 = lasx_xvperm_w(c1, ctrl);
        let c2 = lasx_xvperm_w(c2, ctrl);

        match YuvSourceChannels::from(ORIGIN_CHANNELS) {
            YuvSourceChannels::Rgba => (c0, c1, c2),
            _ => (c2, c1, c0), // Bgra: ch0=B, ch2=R
        }
    }

    /// Per-128-bit-lane byte permutation re-interleaving even/odd-pixel luma
    /// halves to linear order (same `[0,8,1,9,…]` pattern as LSX, duplicated).
    #[inline]
    #[target_feature(enable = "lasx")]
    unsafe fn lasx_y_pack_mask() -> m256i {
        let m: [u8; 32] = [
            0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12,
            5, 13, 6, 14, 7, 15,
        ];
        lasx_xvld::<0>(m.as_ptr() as *const i8)
    }

    /// 32 luma bytes from linear planar `(r, g, b)`, libyuv-style byte-path
    /// multiply-accumulate (see [`lsx_y16`] for the scheme). `xvmaddwev` handles
    /// even pixels, `xvmaddwod` odd; both operate per 128-bit lane, so the
    /// `xvssrlni_bu_h::<8>` + per-lane `xvshuf_b` yields linear `Y0..31`.
    #[inline]
    #[target_feature(enable = "lasx")]
    unsafe fn lasx_y32(r: m256i, g: m256i, b: m256i, c: &Coeffs8) -> m256i {
        let bias = xsplat_h(c.ay);
        let (vry, vgy, vby) = (
            lasx_xvreplgr2vr_b(c.ry),
            lasx_xvreplgr2vr_b(c.gy),
            lasx_xvreplgr2vr_b(c.by),
        );

        let acc_ev = lasx_xvmaddwev_h_bu(
            lasx_xvmaddwev_h_bu(lasx_xvmaddwev_h_bu(bias, r, vry), g, vgy),
            b,
            vby,
        );
        let acc_od = lasx_xvmaddwod_h_bu(
            lasx_xvmaddwod_h_bu(lasx_xvmaddwod_h_bu(bias, r, vry), g, vgy),
            b,
            vby,
        );
        let packed = lasx_xvssrlni_bu_h::<8>(acc_od, acc_ev);
        lasx_xvshuf_b(packed, packed, lasx_y_pack_mask())
    }

    /// 2x2 box-average one channel across two linear rows → 16 i16 samples.
    #[inline]
    #[target_feature(enable = "lasx")]
    unsafe fn lasx_downsample(row0: m256i, row1: m256i) -> m256i {
        let ev = lasx_xvaddwev_h_bu(row0, row1); // c0[2i]+c1[2i]
        let od = lasx_xvaddwod_h_bu(row0, row1); // c0[2i+1]+c1[2i+1]
        lasx_xvsrari_h::<2>(lasx_xvadd_h(ev, od)) // (sum + 2) >> 2
    }

    /// 16 chroma bytes for one coefficient triple, packed as `[s0..7 | dup |
    /// s8..15 | dup]` (low 8 of each 128-bit lane hold the samples).
    #[inline]
    #[target_feature(enable = "lasx")]
    unsafe fn lasx_chroma16(
        r_avg: m256i,
        g_avg: m256i,
        b_avg: m256i,
        cr: i32,
        cg: i32,
        cb: i32,
        bias: m256i,
    ) -> m256i {
        let (vr, vg, vb) = (xsplat_h(cr), xsplat_h(cg), xsplat_h(cb));
        let acc =
            lasx_xvmadd_h(lasx_xvmadd_h(lasx_xvmadd_h(bias, r_avg, vr), g_avg, vg), b_avg, vb);
        lasx_xvpickod_b(acc, acc)
    }

    /// Result of one 32-px-wide, 2-row LASX block.
    pub(super) struct LasxBlock {
        pub y0: m256i,
        pub y1: m256i,
        /// U samples: low 8 bytes of each 128-bit lane (px0-7, px8-15).
        pub u: m256i,
        pub v: m256i,
    }

    #[inline]
    #[target_feature(enable = "lasx")]
    pub(super) unsafe fn lasx_block32<const ORIGIN_CHANNELS: u8>(
        rgba0: *const u8,
        rgba1: *const u8,
        c: &Coeffs8,
    ) -> LasxBlock {
        let (r0, g0, b0) = lasx_load_deinterleave32::<ORIGIN_CHANNELS>(rgba0);
        let (r1, g1, b1) = lasx_load_deinterleave32::<ORIGIN_CHANNELS>(rgba1);

        let y0 = lasx_y32(r0, g0, b0, c);
        let y1 = lasx_y32(r1, g1, b1, c);

        let r_avg = lasx_downsample(r0, r1);
        let g_avg = lasx_downsample(g0, g1);
        let b_avg = lasx_downsample(b0, b1);

        let bias = xsplat_h(c.auv);
        let u = lasx_chroma16(r_avg, g_avg, b_avg, c.ru, c.gu, c.bu, bias);
        let v = lasx_chroma16(r_avg, g_avg, b_avg, c.rv, c.gv, c.bv, bias);
        LasxBlock { y0, y1, u, v }
    }

    /// Store 32 luma bytes.
    #[inline]
    #[target_feature(enable = "lasx")]
    pub(super) unsafe fn xstore32(dst: *mut u8, v: m256i) {
        lasx_xvst::<0>(v, dst as *mut i8);
    }

    /// Store the 16 chroma bytes of a packed register (samples are in the low 8
    /// bytes of each 128-bit lane, i.e. doublewords 0 and 2).
    ///
    /// `xvstelm.d`'s element index is only 1 bit in the Rust bindings (0/1), so
    /// doubleword 2 is unreachable directly. `xvpermi_d` first gathers the two
    /// sample halves into doublewords 0 and 1 (control `[d0, d2, d1, d3]`).
    #[inline]
    #[target_feature(enable = "lasx")]
    pub(super) unsafe fn xstore16_chroma(dst: *mut u8, v: m256i) {
        let p = lasx_xvpermi_d::<0xD8>(v); // d0=s0..7, d1=s8..15
        lasx_xvstelm_d::<0, 0>(p, dst as *mut i8); // s0..7
        lasx_xvstelm_d::<8, 1>(p, dst as *mut i8); // s8..15
    }

    /// Interleave packed U/V (16 each) into 32 linear NV bytes.
    #[inline]
    #[target_feature(enable = "lasx")]
    pub(super) unsafe fn xinterleave(first: m256i, second: m256i) -> m256i {
        lasx_xvilvl_b(second, first)
    }
}

// ---------------------------------------------------------------------------
// Full-frame drivers
// ---------------------------------------------------------------------------

/// SIMD backend selection. `Auto` prefers LASX, then LSX, then scalar (the
/// production choice). `Lsx`/`Lasx` force a single backend so the benchmark can
/// measure each in isolation.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Simd8 {
    Auto,
    #[cfg_attr(not(feature = "loongarch_bench"), allow(dead_code))]
    Lsx,
    #[cfg_attr(not(feature = "loongarch_bench"), allow(dead_code))]
    Lasx,
}

/// Per-row-pair SIMD: process whole 16-px blocks, returning the number of source
/// columns covered (`cx`, a multiple of 16). The caller fills `[cx, width)` with
/// the scalar tail so the whole frame stays 8-bit.
#[cfg(target_arch = "loongarch64")]
#[target_feature(enable = "lsx")]
unsafe fn lsx_i420_rowpair<const ORIGIN_CHANNELS: u8>(
    rgba0: &[u8],
    rgba1: &[u8],
    y0: &mut [u8],
    y1: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
    width: usize,
    c: &Coeffs8,
) -> usize {
    let channels = YuvSourceChannels::from(ORIGIN_CHANNELS).get_channels_count();
    let mut cx = 0usize;
    while cx + simd::LSX_BLOCK <= width {
        let px = cx * channels;
        let blk =
            simd::lsx_block16::<ORIGIN_CHANNELS>(rgba0.as_ptr().add(px), rgba1.as_ptr().add(px), c);
        simd::store16(y0.as_mut_ptr().add(cx), blk.y0);
        simd::store16(y1.as_mut_ptr().add(cx), blk.y1);
        simd::store8(u.as_mut_ptr().add(cx / 2), blk.u);
        simd::store8(v.as_mut_ptr().add(cx / 2), blk.v);
        cx += simd::LSX_BLOCK;
    }
    cx
}

#[cfg(target_arch = "loongarch64")]
#[target_feature(enable = "lsx")]
unsafe fn lsx_nv12_rowpair<const ORIGIN_CHANNELS: u8, const UV_ORDER: u8>(
    rgba0: &[u8],
    rgba1: &[u8],
    y0: &mut [u8],
    y1: &mut [u8],
    uv: &mut [u8],
    width: usize,
    c: &Coeffs8,
) -> usize {
    let channels = YuvSourceChannels::from(ORIGIN_CHANNELS).get_channels_count();
    let u_first = YuvNVOrder::from(UV_ORDER).get_u_position() == 0;
    let mut cx = 0usize;
    while cx + simd::LSX_BLOCK <= width {
        let px = cx * channels;
        let blk =
            simd::lsx_block16::<ORIGIN_CHANNELS>(rgba0.as_ptr().add(px), rgba1.as_ptr().add(px), c);
        simd::store16(y0.as_mut_ptr().add(cx), blk.y0);
        simd::store16(y1.as_mut_ptr().add(cx), blk.y1);
        let packed = if u_first {
            simd::interleave_lo(blk.u, blk.v)
        } else {
            simd::interleave_lo(blk.v, blk.u)
        };
        simd::store16(uv.as_mut_ptr().add(cx), packed);
        cx += simd::LSX_BLOCK;
    }
    cx
}

#[cfg(target_arch = "loongarch64")]
#[target_feature(enable = "lsx,lasx")]
unsafe fn lasx_i420_rowpair<const ORIGIN_CHANNELS: u8>(
    rgba0: &[u8],
    rgba1: &[u8],
    y0: &mut [u8],
    y1: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
    width: usize,
    c: &Coeffs8,
) -> usize {
    let channels = YuvSourceChannels::from(ORIGIN_CHANNELS).get_channels_count();
    let mut cx = 0usize;
    while cx + simd::LASX_BLOCK <= width {
        let px = cx * channels;
        let blk =
            simd::lasx_block32::<ORIGIN_CHANNELS>(rgba0.as_ptr().add(px), rgba1.as_ptr().add(px), c);
        simd::xstore32(y0.as_mut_ptr().add(cx), blk.y0);
        simd::xstore32(y1.as_mut_ptr().add(cx), blk.y1);
        simd::xstore16_chroma(u.as_mut_ptr().add(cx / 2), blk.u);
        simd::xstore16_chroma(v.as_mut_ptr().add(cx / 2), blk.v);
        cx += simd::LASX_BLOCK;
    }
    // Trailing 16-px block (width % 32 in [16, 32)).
    while cx + simd::LSX_BLOCK <= width {
        let px = cx * channels;
        let blk =
            simd::lsx_block16::<ORIGIN_CHANNELS>(rgba0.as_ptr().add(px), rgba1.as_ptr().add(px), c);
        simd::store16(y0.as_mut_ptr().add(cx), blk.y0);
        simd::store16(y1.as_mut_ptr().add(cx), blk.y1);
        simd::store8(u.as_mut_ptr().add(cx / 2), blk.u);
        simd::store8(v.as_mut_ptr().add(cx / 2), blk.v);
        cx += simd::LSX_BLOCK;
    }
    cx
}

#[cfg(target_arch = "loongarch64")]
#[target_feature(enable = "lsx,lasx")]
unsafe fn lasx_nv12_rowpair<const ORIGIN_CHANNELS: u8, const UV_ORDER: u8>(
    rgba0: &[u8],
    rgba1: &[u8],
    y0: &mut [u8],
    y1: &mut [u8],
    uv: &mut [u8],
    width: usize,
    c: &Coeffs8,
) -> usize {
    let channels = YuvSourceChannels::from(ORIGIN_CHANNELS).get_channels_count();
    let u_first = YuvNVOrder::from(UV_ORDER).get_u_position() == 0;
    let mut cx = 0usize;
    while cx + simd::LASX_BLOCK <= width {
        let px = cx * channels;
        let blk =
            simd::lasx_block32::<ORIGIN_CHANNELS>(rgba0.as_ptr().add(px), rgba1.as_ptr().add(px), c);
        simd::xstore32(y0.as_mut_ptr().add(cx), blk.y0);
        simd::xstore32(y1.as_mut_ptr().add(cx), blk.y1);
        let packed = if u_first {
            simd::xinterleave(blk.u, blk.v)
        } else {
            simd::xinterleave(blk.v, blk.u)
        };
        simd::xstore32(uv.as_mut_ptr().add(cx), packed);
        cx += simd::LASX_BLOCK;
    }
    while cx + simd::LSX_BLOCK <= width {
        let px = cx * channels;
        let blk =
            simd::lsx_block16::<ORIGIN_CHANNELS>(rgba0.as_ptr().add(px), rgba1.as_ptr().add(px), c);
        simd::store16(y0.as_mut_ptr().add(cx), blk.y0);
        simd::store16(y1.as_mut_ptr().add(cx), blk.y1);
        let packed = if u_first {
            simd::interleave_lo(blk.u, blk.v)
        } else {
            simd::interleave_lo(blk.v, blk.u)
        };
        simd::store16(uv.as_mut_ptr().add(cx), packed);
        cx += simd::LSX_BLOCK;
    }
    cx
}

/// Scalar luma for source columns `[x0, width)` of one row.
#[inline]
fn scalar_y_tail<const ORIGIN_CHANNELS: u8>(
    rgba: &[u8],
    y: &mut [u8],
    x0: usize,
    width: usize,
    c: &Coeffs8,
) {
    let sc = YuvSourceChannels::from(ORIGIN_CHANNELS);
    let ch = sc.get_channels_count();
    let (ro, go, bo) = (
        sc.get_r_channel_offset(),
        sc.get_g_channel_offset(),
        sc.get_b_channel_offset(),
    );
    for (x, yo) in y[x0..width].iter_mut().enumerate() {
        let p = (x0 + x) * ch;
        *yo = rgb_to_y(
            rgba[p + ro] as i32,
            rgba[p + go] as i32,
            rgba[p + bo] as i32,
            c,
        );
    }
}

/// Average channel `off` of one source pixel quad, handling odd edges.
#[inline]
fn quad_channel(
    rgba0: &[u8],
    rgba1: &[u8],
    p: usize,
    q: usize,
    off: usize,
    ch: usize,
    two_cols: bool,
    two_rows: bool,
) -> i32 {
    let s00 = rgba0[p + off] as i32;
    let s01 = if two_cols { rgba0[q + off] as i32 } else { 0 };
    let s10 = if two_rows { rgba1[p + off] as i32 } else { 0 };
    let s11 = if two_cols && two_rows {
        rgba1[q + off] as i32
    } else {
        0
    };
    let _ = ch;
    avg_quad(s00, s01, s10, s11, two_cols, two_rows)
}

/// Scalar chroma for chroma columns `[cx0/2 .. chroma_w)`; writes U and V via a
/// closure so it serves both planar (I420) and interleaved (NV12) layouts.
#[inline]
fn scalar_chroma_tail<const ORIGIN_CHANNELS: u8>(
    rgba0: &[u8],
    rgba1: &[u8],
    x0: usize, // first source column not covered by SIMD (even)
    width: usize,
    two_rows: bool,
    c: &Coeffs8,
    mut put: impl FnMut(usize, u8, u8),
) {
    let sc = YuvSourceChannels::from(ORIGIN_CHANNELS);
    let ch = sc.get_channels_count();
    let (ro, go, bo) = (
        sc.get_r_channel_offset(),
        sc.get_g_channel_offset(),
        sc.get_b_channel_offset(),
    );
    let mut sx = x0;
    while sx < width {
        let two_cols = sx + 1 < width;
        let p = sx * ch;
        let q = p + ch;
        let r = quad_channel(rgba0, rgba1, p, q, ro, ch, two_cols, two_rows);
        let g = quad_channel(rgba0, rgba1, p, q, go, ch, two_cols, two_rows);
        let b = quad_channel(rgba0, rgba1, p, q, bo, ch, two_cols, two_rows);
        put(sx / 2, rgb_to_u(r, g, b, c), rgb_to_v(r, g, b, c));
        sx += 2;
    }
}

/// BGRA/RGBA → I420 at 8-bit precision. Returns `false` if SIMD is unavailable
/// (the caller falls back); the conversion is still fully performed by scalar in
/// that case only if `force_scalar` is set (used by tests).
#[cfg(target_arch = "loongarch64")]
pub(crate) fn rgba_to_i420_8<const ORIGIN_CHANNELS: u8>(
    image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    c: &Coeffs8,
    backend: Simd8,
) -> bool {
    // The 256-bit deinterleave is 4-channel only; 3-channel uses the LSX path.
    let channels = YuvSourceChannels::from(ORIGIN_CHANNELS).get_channels_count();
    let has_lasx = matches!(backend, Simd8::Auto | Simd8::Lasx)
        && std::arch::is_loongarch_feature_detected!("lasx")
        && channels == 4;
    let has_lsx = matches!(backend, Simd8::Auto | Simd8::Lsx)
        && std::arch::is_loongarch_feature_detected!("lsx");
    let width = image.width as usize;
    let height = image.height as usize;
    let y_stride = image.y_stride as usize;
    let u_stride = image.u_stride as usize;
    let v_stride = image.v_stride as usize;
    let src_stride = rgba_stride as usize;

    let y_plane = image.y_plane.borrow_mut();
    let u_plane = image.u_plane.borrow_mut();
    let v_plane = image.v_plane.borrow_mut();

    let mut row = 0usize;
    while row < height {
        let two_rows = row + 1 < height;
        let r0 = &rgba[row * src_stride..];
        // For an odd final row, reuse row 0 as the (unused for chroma) second row.
        let r1 = if two_rows {
            &rgba[(row + 1) * src_stride..]
        } else {
            r0
        };

        // SIMD covers full blocks for the (typical) two-row case.
        let cx = if (has_lsx || has_lasx) && two_rows {
            // SAFETY: feature checked above; slices are at least `width*ch` long.
            unsafe {
                // Split the two Y output rows without aliasing.
                let (y_top, y_rest) = y_plane.split_at_mut((row + 1) * y_stride);
                let y0 = &mut y_top[row * y_stride..row * y_stride + width];
                let y1 = &mut y_rest[..width];
                let cr = row / 2;
                let u_row = &mut u_plane[cr * u_stride..cr * u_stride + width.div_ceil(2)];
                let v_row = &mut v_plane[cr * v_stride..cr * v_stride + width.div_ceil(2)];
                if has_lasx {
                    lasx_i420_rowpair::<ORIGIN_CHANNELS>(r0, r1, y0, y1, u_row, v_row, width, c)
                } else {
                    lsx_i420_rowpair::<ORIGIN_CHANNELS>(r0, r1, y0, y1, u_row, v_row, width, c)
                }
            }
        } else {
            0
        };

        // Scalar luma tail for both rows.
        let y0 = &mut y_plane[row * y_stride..row * y_stride + width];
        scalar_y_tail::<ORIGIN_CHANNELS>(r0, y0, cx, width, c);
        if two_rows {
            let y1 = &mut y_plane[(row + 1) * y_stride..(row + 1) * y_stride + width];
            scalar_y_tail::<ORIGIN_CHANNELS>(r1, y1, cx, width, c);
        }

        // Scalar chroma tail.
        let cr = row / 2;
        let (u_row, v_row) = (
            &mut u_plane[cr * u_stride..],
            &mut v_plane[cr * v_stride..],
        );
        scalar_chroma_tail::<ORIGIN_CHANNELS>(r0, r1, cx, width, two_rows, c, |j, u, v| {
            u_row[j] = u;
            v_row[j] = v;
        });

        row += 2;
    }
    has_lsx || has_lasx
}

/// BGRA/RGBA → NV12/NV21 at 8-bit precision.
#[cfg(target_arch = "loongarch64")]
pub(crate) fn rgba_to_nv12_8<const ORIGIN_CHANNELS: u8, const UV_ORDER: u8>(
    image: &mut YuvBiPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    c: &Coeffs8,
    backend: Simd8,
) -> bool {
    let channels = YuvSourceChannels::from(ORIGIN_CHANNELS).get_channels_count();
    let has_lasx = matches!(backend, Simd8::Auto | Simd8::Lasx)
        && std::arch::is_loongarch_feature_detected!("lasx")
        && channels == 4;
    let has_lsx = matches!(backend, Simd8::Auto | Simd8::Lsx)
        && std::arch::is_loongarch_feature_detected!("lsx");
    let width = image.width as usize;
    let height = image.height as usize;
    let y_stride = image.y_stride as usize;
    let uv_stride = image.uv_stride as usize;
    let src_stride = rgba_stride as usize;
    let order = YuvNVOrder::from(UV_ORDER);
    let upos = order.get_u_position();
    let vpos = order.get_v_position();

    let y_plane = image.y_plane.borrow_mut();
    let uv_plane = image.uv_plane.borrow_mut();

    let mut row = 0usize;
    while row < height {
        let two_rows = row + 1 < height;
        let r0 = &rgba[row * src_stride..];
        let r1 = if two_rows {
            &rgba[(row + 1) * src_stride..]
        } else {
            r0
        };

        let cx = if (has_lsx || has_lasx) && two_rows {
            unsafe {
                let (y_top, y_rest) = y_plane.split_at_mut((row + 1) * y_stride);
                let y0 = &mut y_top[row * y_stride..row * y_stride + width];
                let y1 = &mut y_rest[..width];
                let cr = row / 2;
                let uv_row = &mut uv_plane[cr * uv_stride..cr * uv_stride + width.div_ceil(2) * 2];
                if has_lasx {
                    lasx_nv12_rowpair::<ORIGIN_CHANNELS, UV_ORDER>(r0, r1, y0, y1, uv_row, width, c)
                } else {
                    lsx_nv12_rowpair::<ORIGIN_CHANNELS, UV_ORDER>(r0, r1, y0, y1, uv_row, width, c)
                }
            }
        } else {
            0
        };

        let y0 = &mut y_plane[row * y_stride..row * y_stride + width];
        scalar_y_tail::<ORIGIN_CHANNELS>(r0, y0, cx, width, c);
        if two_rows {
            let y1 = &mut y_plane[(row + 1) * y_stride..(row + 1) * y_stride + width];
            scalar_y_tail::<ORIGIN_CHANNELS>(r1, y1, cx, width, c);
        }

        let cr = row / 2;
        let uv_row = &mut uv_plane[cr * uv_stride..];
        scalar_chroma_tail::<ORIGIN_CHANNELS>(r0, r1, cx, width, two_rows, c, |j, u, v| {
            uv_row[j * 2 + upos] = u;
            uv_row[j * 2 + vpos] = v;
        });

        row += 2;
    }
    has_lsx || has_lasx
}

#[cfg(all(test, target_arch = "loongarch64"))]
mod tests {
    use super::*;
    use crate::images::{BufferStoreMut, YuvBiPlanarImageMut, YuvPlanarImageMut};
    use crate::yuv_support::{YuvChromaSubsampling, YuvNVOrder};

    /// Deterministic pseudo-random rows (xorshift) without pulling in `rand`.
    fn make_src(width: usize, height: usize, channels: usize) -> Vec<u8> {
        let mut v = vec![0u8; width * height * channels];
        let mut state: u32 = 0x9e37_79b9;
        for b in v.iter_mut() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *b = (state & 0xff) as u8;
        }
        v
    }

    /// Pure-scalar full-frame oracle for I420 (independent of the driver's SIMD).
    fn oracle_i420<const O: u8>(
        src: &[u8],
        width: usize,
        height: usize,
        c: &Coeffs8,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let cw = width.div_ceil(2);
        let chh = height.div_ceil(2);
        let mut y = vec![0u8; width * height];
        let mut u = vec![0u8; cw * chh];
        let mut v = vec![0u8; cw * chh];
        let stride = width * YuvSourceChannels::from(O).get_channels_count();
        for r in 0..height {
            scalar_y_tail::<O>(&src[r * stride..], &mut y[r * width..r * width + width], 0, width, c);
        }
        let mut r = 0usize;
        while r < height {
            let two_rows = r + 1 < height;
            let r0 = &src[r * stride..];
            let r1 = if two_rows { &src[(r + 1) * stride..] } else { r0 };
            let cr = r / 2;
            let (u_row, v_row) = (&mut u[cr * cw..], &mut v[cr * cw..]);
            scalar_chroma_tail::<O>(r0, r1, 0, width, two_rows, c, |j, cu, cv| {
                u_row[j] = cu;
                v_row[j] = cv;
            });
            r += 2;
        }
        (y, u, v)
    }

    /// Backends available on this CPU, so each is validated against scalar.
    fn backends() -> Vec<Simd8> {
        let mut v = Vec::new();
        if std::arch::is_loongarch_feature_detected!("lsx") {
            v.push(Simd8::Lsx);
        }
        if std::arch::is_loongarch_feature_detected!("lasx") {
            v.push(Simd8::Lasx);
        }
        v
    }

    fn run_i420<const O: u8>(width: usize, height: usize, backend: Simd8) {
        let c = coeffs8(YuvStandardMatrix::Bt601, YuvRange::Limited).unwrap();
        let ch = YuvSourceChannels::from(O).get_channels_count();
        let src = make_src(width, height, ch);
        let (ey, eu, ev) = oracle_i420::<O>(&src, width, height, &c);

        let cw = width.div_ceil(2);
        let chh = height.div_ceil(2);
        let mut image = YuvPlanarImageMut {
            y_plane: BufferStoreMut::Owned(vec![0u8; width * height]),
            y_stride: width as u32,
            u_plane: BufferStoreMut::Owned(vec![0u8; cw * chh]),
            u_stride: cw as u32,
            v_plane: BufferStoreMut::Owned(vec![0u8; cw * chh]),
            v_stride: cw as u32,
            width: width as u32,
            height: height as u32,
        };
        rgba_to_i420_8::<O>(&mut image, &src, (width * ch) as u32, &c, backend);
        let b = backend as u8;
        assert_eq!(image.y_plane.borrow(), &ey[..], "Y w={width} h={height} O={O} b={b}");
        assert_eq!(image.u_plane.borrow(), &eu[..], "U w={width} h={height} O={O} b={b}");
        assert_eq!(image.v_plane.borrow(), &ev[..], "V w={width} h={height} O={O} b={b}");
    }

    #[test]
    fn lsx_i420_matches_scalar() {
        const RGBA: u8 = YuvSourceChannels::Rgba as u8;
        const BGRA: u8 = YuvSourceChannels::Bgra as u8;
        for backend in backends() {
            for &(w, h) in &[
                (16usize, 2usize),
                (17, 3),
                (31, 5),
                (32, 4),
                (33, 2),
                (48, 6),
                (63, 3),
                (64, 6),
                (65, 7),
                (256, 8),
                (1920, 4),
            ] {
                run_i420::<RGBA>(w, h, backend);
                run_i420::<BGRA>(w, h, backend);
            }
        }
    }

    fn oracle_nv12<const O: u8, const ORD: u8>(
        src: &[u8],
        width: usize,
        height: usize,
        c: &Coeffs8,
    ) -> (Vec<u8>, Vec<u8>) {
        let cw = width.div_ceil(2);
        let chh = height.div_ceil(2);
        let mut y = vec![0u8; width * height];
        let mut uv = vec![0u8; cw * 2 * chh];
        let order = YuvNVOrder::from(ORD);
        let (upos, vpos) = (order.get_u_position(), order.get_v_position());
        let stride = width * YuvSourceChannels::from(O).get_channels_count();
        for r in 0..height {
            scalar_y_tail::<O>(&src[r * stride..], &mut y[r * width..r * width + width], 0, width, c);
        }
        let mut r = 0usize;
        while r < height {
            let two_rows = r + 1 < height;
            let r0 = &src[r * stride..];
            let r1 = if two_rows { &src[(r + 1) * stride..] } else { r0 };
            let cr = r / 2;
            let uv_row = &mut uv[cr * cw * 2..];
            scalar_chroma_tail::<O>(r0, r1, 0, width, two_rows, c, |j, cu, cv| {
                uv_row[j * 2 + upos] = cu;
                uv_row[j * 2 + vpos] = cv;
            });
            r += 2;
        }
        (y, uv)
    }

    fn run_nv12<const O: u8, const ORD: u8>(width: usize, height: usize, backend: Simd8) {
        let c = coeffs8(YuvStandardMatrix::Bt601, YuvRange::Limited).unwrap();
        let ch = YuvSourceChannels::from(O).get_channels_count();
        let src = make_src(width, height, ch);
        let (ey, euv) = oracle_nv12::<O, ORD>(&src, width, height, &c);

        let cw = width.div_ceil(2);
        let chh = height.div_ceil(2);
        let mut image = YuvBiPlanarImageMut {
            y_plane: BufferStoreMut::Owned(vec![0u8; width * height]),
            y_stride: width as u32,
            uv_plane: BufferStoreMut::Owned(vec![0u8; cw * 2 * chh]),
            uv_stride: (cw * 2) as u32,
            width: width as u32,
            height: height as u32,
        };
        rgba_to_nv12_8::<O, ORD>(&mut image, &src, (width * ch) as u32, &c, backend);
        let b = backend as u8;
        assert_eq!(image.y_plane.borrow(), &ey[..], "Y w={width} h={height} b={b}");
        assert_eq!(image.uv_plane.borrow(), &euv[..], "UV w={width} h={height} b={b}");
    }

    #[test]
    fn lsx_nv12_matches_scalar() {
        const RGBA: u8 = YuvSourceChannels::Rgba as u8;
        const BGRA: u8 = YuvSourceChannels::Bgra as u8;
        const UV: u8 = YuvNVOrder::UV as u8;
        const VU: u8 = YuvNVOrder::VU as u8;
        for backend in backends() {
            for &(w, h) in &[
                (16usize, 2usize),
                (17, 3),
                (31, 5),
                (32, 4),
                (48, 4),
                (65, 7),
                (256, 8),
            ] {
                run_nv12::<RGBA, UV>(w, h, backend);
                run_nv12::<BGRA, UV>(w, h, backend);
                run_nv12::<BGRA, VU>(w, h, backend);
            }
        }
    }

    #[test]
    fn _silence_unused() {
        // `YuvChromaSubsampling` import kept available for future 422/444 work.
        let _ = YuvChromaSubsampling::Yuv420;
    }
}
