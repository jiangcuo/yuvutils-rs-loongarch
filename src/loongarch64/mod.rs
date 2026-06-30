#![deny(unreachable_code, unreachable_pub)]

mod convert8;
mod rgba_to_nv420;
mod rgba_to_yuv420;
#[cfg(test)]
mod tests_common;
mod utils;

pub(crate) use convert8::{coeffs8, rgba_to_i420_8, rgba_to_nv12_8, Simd8};

pub(crate) use rgba_to_nv420::loongarch64_rgba_to_nv420;
pub(crate) use rgba_to_yuv420::loongarch64_rgba_to_yuv420;

#[cfg(feature = "loongarch_bench")]
#[inline]
pub fn has_lsx() -> bool {
    std::arch::is_loongarch_feature_detected!("lsx")
}

#[cfg(feature = "loongarch_bench")]
#[inline]
pub fn has_lasx() -> bool {
    std::arch::is_loongarch_feature_detected!("lasx")
}
