use crate::internals::ProcessedOffset;
use crate::loongarch64::utils::{
    lsx_chroma_to_u8, lsx_encode_block16, lsx_interleave_lo, lsx_store16, lsx_store_y, LSX_BLOCK,
};
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvNVOrder, YuvSourceChannels};

pub(crate) fn loongarch64_rgba_to_nv420<
    const ORIGIN_CHANNELS: u8,
    const UV_ORDER: u8,
    const PRECISION: i32,
>(
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    uv_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    if PRECISION != 13 {
        return ProcessedOffset {
            cx: start_cx,
            ux: start_ux,
        };
    }

    if std::arch::is_loongarch_feature_detected!("lasx") {
        unsafe {
            return lasx_rgba_to_nv420::<ORIGIN_CHANNELS, UV_ORDER, PRECISION>(
                y_plane0, y_plane1, uv_plane, rgba0, rgba1, width, range, transform, start_cx,
                start_ux,
            );
        }
    }

    if std::arch::is_loongarch_feature_detected!("lsx") {
        unsafe {
            return lsx_rgba_to_nv420::<ORIGIN_CHANNELS, UV_ORDER, PRECISION>(
                y_plane0, y_plane1, uv_plane, rgba0, rgba1, width, range, transform, start_cx,
                start_ux,
            );
        }
    }

    ProcessedOffset {
        cx: start_cx,
        ux: start_ux,
    }
}

#[target_feature(enable = "lsx")]
unsafe fn lsx_rgba_to_nv420<const ORIGIN_CHANNELS: u8, const UV_ORDER: u8, const PRECISION: i32>(
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    uv_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    rgba_to_nv420_impl::<ORIGIN_CHANNELS, UV_ORDER, PRECISION>(
        y_plane0, y_plane1, uv_plane, rgba0, rgba1, width as usize, range, transform, start_cx,
        start_ux,
    )
}

#[target_feature(enable = "lsx,lasx")]
unsafe fn lasx_rgba_to_nv420<const ORIGIN_CHANNELS: u8, const UV_ORDER: u8, const PRECISION: i32>(
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    uv_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    rgba_to_nv420_impl::<ORIGIN_CHANNELS, UV_ORDER, PRECISION>(
        y_plane0, y_plane1, uv_plane, rgba0, rgba1, width as usize, range, transform, start_cx,
        start_ux,
    )
}

/// Process all full 16-pixel-wide, 2-row blocks. The trailing columns
/// (`width % 16`) and an odd final row are completed by the scalar remainder
/// loop in the caller.
#[inline]
#[target_feature(enable = "lsx")]
unsafe fn rgba_to_nv420_impl<
    const ORIGIN_CHANNELS: u8,
    const UV_ORDER: u8,
    const PRECISION: i32,
>(
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    uv_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    width: usize,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();
    let order: YuvNVOrder = UV_ORDER.into();
    let u_first = order.get_u_position() == 0;
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

        let cb = lsx_chroma_to_u8(blk.cb);
        let cr = lsx_chroma_to_u8(blk.cr);
        // For UV order the first byte of each pair is Cb; for VU it is Cr.
        let uv = if u_first {
            lsx_interleave_lo(cb, cr)
        } else {
            lsx_interleave_lo(cr, cb)
        };
        lsx_store16(uv_plane.as_mut_ptr().add(uv_x), uv);

        uv_x += LSX_BLOCK;
        cx += LSX_BLOCK;
    }

    ProcessedOffset { cx, ux: uv_x }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loongarch64::tests_common::{bt601_limited, make_rgba_rows, ref_y};

    #[test]
    fn lsx_nv420_matches_scalar() {
        if !std::arch::is_loongarch_feature_detected!("lsx") {
            return;
        }
        const RGBA: u8 = YuvSourceChannels::Rgba as u8;
        const BGRA: u8 = YuvSourceChannels::Bgra as u8;
        const UV: u8 = YuvNVOrder::UV as u8;
        const VU: u8 = YuvNVOrder::VU as u8;
        let (range, transform) = bt601_limited();

        for &width in &[16usize, 17, 31, 32, 48, 64, 65, 257, 1920] {
            check::<RGBA, UV>(width, &range, &transform);
            check::<BGRA, UV>(width, &range, &transform);
            check::<BGRA, VU>(width, &range, &transform);
        }
    }

    fn check<const O: u8, const ORD: u8>(
        width: usize,
        range: &YuvChromaRange,
        transform: &CbCrForwardTransform<i32>,
    ) {
        let channels = YuvSourceChannels::from(O).get_channels_count();
        let order: YuvNVOrder = ORD.into();
        let (rgba0, rgba1) = make_rgba_rows::<O>(width);

        let mut y0 = vec![0u8; width];
        let mut y1 = vec![0u8; width];
        let mut uv = vec![0u8; width.div_ceil(2) * 2];

        let off = unsafe {
            lsx_rgba_to_nv420::<O, ORD, 13>(
                &mut y0,
                &mut y1,
                &mut uv,
                &rgba0,
                &rgba1,
                width as u32,
                range,
                transform,
                0,
                0,
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
            assert_eq!(
                y0[x],
                ref_y::<13>(
                    rgba0[p + ro] as i32,
                    rgba0[p + go] as i32,
                    rgba0[p + bo] as i32,
                    range,
                    transform
                )
            );
            assert_eq!(
                y1[x],
                ref_y::<13>(
                    rgba1[p + ro] as i32,
                    rgba1[p + go] as i32,
                    rgba1[p + bo] as i32,
                    range,
                    transform
                )
            );
        }

        let bias = crate::loongarch64::utils::uv_bias::<13>(range);
        // off.ux counts interleaved bytes; there are off.ux / 2 chroma pairs.
        for k in 0..(off.ux / 2) {
            let p = (k * 2) * channels;
            let q = p + channels;
            let avg = |o: usize| {
                (rgba0[p + o] as i32 + rgba0[q + o] as i32 + rgba1[p + o] as i32
                    + rgba1[q + o] as i32
                    + 2)
                    >> 2
            };
            let (r, g, b) = (avg(ro), avg(go), avg(bo));
            let cb =
                ((r * transform.cb_r + g * transform.cb_g + b * transform.cb_b + bias) >> 13) as u8;
            let cr =
                ((r * transform.cr_r + g * transform.cr_g + b * transform.cr_b + bias) >> 13) as u8;
            assert_eq!(uv[k * 2 + order.get_u_position()], cb, "Cb w={width} k={k}");
            assert_eq!(uv[k * 2 + order.get_v_position()], cr, "Cr w={width} k={k}");
        }
    }
}
