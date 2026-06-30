use crate::internals::ProcessedOffset;
use crate::loongarch64::utils::{
    lsx_chroma_to_u8, lsx_encode_block16, lsx_store8, lsx_store_y, LSX_BLOCK,
};
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};

pub(crate) fn loongarch64_rgba_to_yuv420<const ORIGIN_CHANNELS: u8, const PRECISION: i32>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    if PRECISION != 13 {
        return ProcessedOffset {
            cx: start_cx,
            ux: start_ux,
        };
    }

    // LASX implies LSX; until a dedicated 256-bit kernel lands both feature
    // levels use the LSX path.
    if std::arch::is_loongarch_feature_detected!("lasx") {
        unsafe {
            return lasx_rgba_to_yuv420::<ORIGIN_CHANNELS, PRECISION>(
                transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                start_ux, width,
            );
        }
    }

    if std::arch::is_loongarch_feature_detected!("lsx") {
        unsafe {
            return lsx_rgba_to_yuv420::<ORIGIN_CHANNELS, PRECISION>(
                transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                start_ux, width,
            );
        }
    }

    ProcessedOffset {
        cx: start_cx,
        ux: start_ux,
    }
}

#[target_feature(enable = "lsx")]
unsafe fn lsx_rgba_to_yuv420<const ORIGIN_CHANNELS: u8, const PRECISION: i32>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    rgba_to_yuv420_impl::<ORIGIN_CHANNELS, PRECISION>(
        transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx, start_ux,
        width,
    )
}

#[target_feature(enable = "lsx,lasx")]
unsafe fn lasx_rgba_to_yuv420<const ORIGIN_CHANNELS: u8, const PRECISION: i32>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    rgba_to_yuv420_impl::<ORIGIN_CHANNELS, PRECISION>(
        transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx, start_ux,
        width,
    )
}

/// Process all full 16-pixel-wide, 2-row blocks. The trailing columns
/// (`width % 16`) and an odd final row are completed by the scalar remainder
/// loop in the caller, so this only needs to cover whole blocks.
#[inline]
#[target_feature(enable = "lsx")]
unsafe fn rgba_to_yuv420_impl<const ORIGIN_CHANNELS: u8, const PRECISION: i32>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();
    let mut cx = start_cx;
    let mut uv_x = start_ux;

    while cx + LSX_BLOCK <= width {
        let px = cx * channels;
        let blk = lsx_encode_block16::<ORIGIN_CHANNELS, PRECISION>(
            rgba0.as_ptr().add(px),
            rgba1.as_ptr().add(px),
            range,
            transform,
        );
        lsx_store_y(y_plane0.as_mut_ptr().add(cx), blk.y0);
        lsx_store_y(y_plane1.as_mut_ptr().add(cx), blk.y1);
        lsx_store8(u_plane.as_mut_ptr().add(uv_x), lsx_chroma_to_u8(blk.cb));
        lsx_store8(v_plane.as_mut_ptr().add(uv_x), lsx_chroma_to_u8(blk.cr));
        uv_x += LSX_BLOCK / 2;
        cx += LSX_BLOCK;
    }

    ProcessedOffset { cx, ux: uv_x }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loongarch64::tests_common::{bt601_limited, make_rgba_rows, ref_y};

    /// The SIMD kernel must be bit-identical to the scalar reference for every
    /// luma/chroma sample inside the blocks it processed.
    #[test]
    fn lsx_yuv420_matches_scalar() {
        if !std::arch::is_loongarch_feature_detected!("lsx") {
            return;
        }
        const RGBA: u8 = YuvSourceChannels::Rgba as u8;
        const BGRA: u8 = YuvSourceChannels::Bgra as u8;
        let (range, transform) = bt601_limited();

        for &width in &[16usize, 17, 31, 32, 48, 64, 65, 257, 1920] {
            check::<RGBA>(width, &range, &transform);
            check::<BGRA>(width, &range, &transform);
        }
    }

    fn check<const O: u8>(
        width: usize,
        range: &YuvChromaRange,
        transform: &CbCrForwardTransform<i32>,
    ) {
        let channels = (YuvSourceChannels::from(O)).get_channels_count();
        let (rgba0, rgba1) = make_rgba_rows::<O>(width);

        let mut y0 = vec![0u8; width];
        let mut y1 = vec![0u8; width];
        let cw = width.div_ceil(2);
        let mut u = vec![0u8; cw];
        let mut v = vec![0u8; cw];

        let off = unsafe {
            lsx_rgba_to_yuv420::<O, 13>(
                transform, range, &mut y0, &mut y1, &mut u, &mut v, &rgba0, &rgba1, 0, 0, width,
            )
        };

        let sc = YuvSourceChannels::from(O);
        let (ro, go, bo) = (
            sc.get_r_channel_offset(),
            sc.get_g_channel_offset(),
            sc.get_b_channel_offset(),
        );

        for x in 0..off.cx {
            let p = x * channels;
            let e0 = ref_y::<13>(
                rgba0[p + ro] as i32,
                rgba0[p + go] as i32,
                rgba0[p + bo] as i32,
                range,
                transform,
            );
            let e1 = ref_y::<13>(
                rgba1[p + ro] as i32,
                rgba1[p + go] as i32,
                rgba1[p + bo] as i32,
                range,
                transform,
            );
            assert_eq!(y0[x], e0, "Y0 mismatch O={O} w={width} x={x}");
            assert_eq!(y1[x], e1, "Y1 mismatch O={O} w={width} x={x}");
        }

        for ux in 0..off.ux {
            let p = (ux * 2) * channels;
            let q = p + channels;
            let avg = |o: usize| {
                (rgba0[p + o] as i32 + rgba0[q + o] as i32 + rgba1[p + o] as i32
                    + rgba1[q + o] as i32
                    + 2)
                    >> 2
            };
            let bias = crate::loongarch64::utils::uv_bias::<13>(range);
            let (r, g, b) = (avg(ro), avg(go), avg(bo));
            let cb =
                ((r * transform.cb_r + g * transform.cb_g + b * transform.cb_b + bias) >> 13) as u8;
            let cr =
                ((r * transform.cr_r + g * transform.cr_g + b * transform.cr_b + bias) >> 13) as u8;
            assert_eq!(u[ux], cb, "U mismatch O={O} w={width} ux={ux}");
            assert_eq!(v[ux], cr, "V mismatch O={O} w={width} ux={ux}");
        }
    }
}
