use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use yuv::{YuvBiPlanarImageMut, YuvChromaSubsampling, YuvPlanarImageMut, YuvRange, YuvStandardMatrix};

#[cfg(all(target_arch = "loongarch64", feature = "loongarch_bench"))]
use yuv::loongarch64_bench::{
    bgra_to_yuv420_lasx, bgra_to_yuv420_lsx, bgra_to_yuv420_scalar, bgra_to_yuv_nv12_lasx,
    bgra_to_yuv_nv12_lsx, bgra_to_yuv_nv12_scalar, has_lasx, has_lsx,
};

fn make_bgra(width: u32, height: u32) -> Vec<u8> {
    let mut bgra = vec![0u8; width as usize * height as usize * 4];
    for (i, px) in bgra.chunks_exact_mut(4).enumerate() {
        px[0] = (i.wrapping_mul(17)) as u8;
        px[1] = (i.wrapping_mul(31).wrapping_add(7)) as u8;
        px[2] = (i.wrapping_mul(47).wrapping_add(11)) as u8;
        px[3] = 255;
    }
    bgra
}

#[cfg(all(target_arch = "loongarch64", feature = "loongarch_bench"))]
fn bench_loongarch(c: &mut Criterion) {
    let width = 1920u32;
    let height = 1080u32;
    let bgra = make_bgra(width, height);
    let mut group = c.benchmark_group("LoongArch BGRA -> YUV 4:2:0");
    group.sample_size(100);
    group.measurement_time(std::time::Duration::from_secs(5));

    group.bench_function("I420 scalar", |b| {
        let mut image = YuvPlanarImageMut::<u8>::alloc(width, height, YuvChromaSubsampling::Yuv420);
        b.iter(|| {
            bgra_to_yuv420_scalar(
                black_box(&mut image),
                black_box(&bgra),
                width * 4,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });

    if has_lsx() {
        group.bench_function("I420 LSX", |b| {
            let mut image =
                YuvPlanarImageMut::<u8>::alloc(width, height, YuvChromaSubsampling::Yuv420);
            b.iter(|| {
                bgra_to_yuv420_lsx(
                    black_box(&mut image),
                    black_box(&bgra),
                    width * 4,
                    YuvRange::Limited,
                    YuvStandardMatrix::Bt601,
                )
                .unwrap();
            })
        });
    }

    if has_lasx() {
        group.bench_function("I420 LASX", |b| {
            let mut image =
                YuvPlanarImageMut::<u8>::alloc(width, height, YuvChromaSubsampling::Yuv420);
            b.iter(|| {
                bgra_to_yuv420_lasx(
                    black_box(&mut image),
                    black_box(&bgra),
                    width * 4,
                    YuvRange::Limited,
                    YuvStandardMatrix::Bt601,
                )
                .unwrap();
            })
        });
    }

    group.bench_function("NV12 scalar", |b| {
        let mut image =
            YuvBiPlanarImageMut::<u8>::alloc(width, height, YuvChromaSubsampling::Yuv420);
        b.iter(|| {
            bgra_to_yuv_nv12_scalar(
                black_box(&mut image),
                black_box(&bgra),
                width * 4,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });

    if has_lsx() {
        group.bench_function("NV12 LSX", |b| {
            let mut image =
                YuvBiPlanarImageMut::<u8>::alloc(width, height, YuvChromaSubsampling::Yuv420);
            b.iter(|| {
                bgra_to_yuv_nv12_lsx(
                    black_box(&mut image),
                    black_box(&bgra),
                    width * 4,
                    YuvRange::Limited,
                    YuvStandardMatrix::Bt601,
                )
                .unwrap();
            })
        });
    }

    if has_lasx() {
        group.bench_function("NV12 LASX", |b| {
            let mut image =
                YuvBiPlanarImageMut::<u8>::alloc(width, height, YuvChromaSubsampling::Yuv420);
            b.iter(|| {
                bgra_to_yuv_nv12_lasx(
                    black_box(&mut image),
                    black_box(&bgra),
                    width * 4,
                    YuvRange::Limited,
                    YuvStandardMatrix::Bt601,
                )
                .unwrap();
            })
        });
    }

    group.finish();
}

#[cfg(not(all(target_arch = "loongarch64", feature = "loongarch_bench")))]
fn bench_loongarch(c: &mut Criterion) {
    let mut group = c.benchmark_group("LoongArch BGRA -> YUV 4:2:0");
    group.bench_function("disabled", |b| {
        let bgra = make_bgra(16, 16);
        b.iter(|| black_box(&bgra));
    });
    group.finish();
}

criterion_group!(benches, bench_loongarch);
criterion_main!(benches);
