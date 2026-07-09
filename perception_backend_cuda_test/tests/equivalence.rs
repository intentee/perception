#![cfg(feature = "cuda")]

use rgb::RGB;
use rgb::RGBA;

use perception_backend::Backend;
use perception_backend_cpu::CpuBackend;
use perception_backend_cuda::CudaBackend;
use perception_backend_cuda_test::measure_cross_backend_difference::measure_cross_backend_difference;
use perception_metric_bench::synthetic_pixels::synthetic_pixels;

const SIDE: usize = 256;
const SCALE_COUNT: usize = 5;

const PIPELINE_OPERATION_BOUND: f32 = 1024.0;
const MAP_TOLERANCE: f32 = PIPELINE_OPERATION_BOUND * f32::EPSILON;
const DEVIATION_TOLERANCE: f64 = PIPELINE_OPERATION_BOUND as f64 * f32::EPSILON as f64;

#[test]
fn gray_pipeline_matches_the_cpu_backend() {
    let cpu = CpuBackend::default();
    let cuda = CudaBackend::new().unwrap();
    let reference: Vec<u8> = synthetic_pixels(SIDE, SIDE);
    let distorted: Vec<u8> = reference
        .iter()
        .map(|&value| value.wrapping_add(23))
        .collect();

    let cpu_reference = cpu
        .prepare_gray(&reference, SIDE, SIDE, SCALE_COUNT)
        .unwrap();
    let cpu_distorted = cpu
        .prepare_gray(&distorted, SIDE, SIDE, SCALE_COUNT)
        .unwrap();
    let cuda_reference = cuda
        .prepare_gray(&reference, SIDE, SIDE, SCALE_COUNT)
        .unwrap();
    let cuda_distorted = cuda
        .prepare_gray(&distorted, SIDE, SIDE, SCALE_COUNT)
        .unwrap();

    let difference = measure_cross_backend_difference(
        &cpu,
        &cuda,
        &cpu_reference,
        &cpu_distorted,
        &cuda_reference,
        &cuda_distorted,
    )
    .unwrap();

    assert!(
        difference.max_map_difference < MAP_TOLERANCE,
        "gray map difference {} exceeds {MAP_TOLERANCE}",
        difference.max_map_difference
    );
    assert!(
        difference.max_deviation_difference < DEVIATION_TOLERANCE,
        "gray deviation difference {} exceeds {DEVIATION_TOLERANCE}",
        difference.max_deviation_difference
    );
}

#[test]
fn rgb_pipeline_matches_the_cpu_backend() {
    let cpu = CpuBackend::default();
    let cuda = CudaBackend::new().unwrap();
    let reference: Vec<RGB<u8>> = synthetic_pixels(SIDE, SIDE);
    let distorted: Vec<RGB<u8>> = reference
        .iter()
        .map(|pixel| RGB::new(pixel.r.wrapping_add(23), pixel.g, pixel.b.wrapping_add(9)))
        .collect();

    let cpu_reference = cpu
        .prepare_rgb(&reference, SIDE, SIDE, SCALE_COUNT)
        .unwrap();
    let cpu_distorted = cpu
        .prepare_rgb(&distorted, SIDE, SIDE, SCALE_COUNT)
        .unwrap();
    let cuda_reference = cuda
        .prepare_rgb(&reference, SIDE, SIDE, SCALE_COUNT)
        .unwrap();
    let cuda_distorted = cuda
        .prepare_rgb(&distorted, SIDE, SIDE, SCALE_COUNT)
        .unwrap();

    let difference = measure_cross_backend_difference(
        &cpu,
        &cuda,
        &cpu_reference,
        &cpu_distorted,
        &cuda_reference,
        &cuda_distorted,
    )
    .unwrap();

    assert!(
        difference.max_map_difference < MAP_TOLERANCE,
        "rgb map difference {} exceeds {MAP_TOLERANCE}",
        difference.max_map_difference
    );
    assert!(
        difference.max_deviation_difference < DEVIATION_TOLERANCE,
        "rgb deviation difference {} exceeds {DEVIATION_TOLERANCE}",
        difference.max_deviation_difference
    );
}

#[test]
fn rgba_pipeline_matches_the_cpu_backend() {
    let cpu = CpuBackend::default();
    let cuda = CudaBackend::new().unwrap();
    let reference: Vec<RGBA<u8>> = synthetic_pixels(SIDE, SIDE);
    let distorted: Vec<RGBA<u8>> = reference
        .iter()
        .map(|pixel| {
            RGBA::new(
                pixel.r.wrapping_add(23),
                pixel.g,
                pixel.b.wrapping_add(9),
                pixel.a,
            )
        })
        .collect();

    let cpu_reference = cpu
        .prepare_rgba(&reference, SIDE, SIDE, SCALE_COUNT)
        .unwrap();
    let cpu_distorted = cpu
        .prepare_rgba(&distorted, SIDE, SIDE, SCALE_COUNT)
        .unwrap();
    let cuda_reference = cuda
        .prepare_rgba(&reference, SIDE, SIDE, SCALE_COUNT)
        .unwrap();
    let cuda_distorted = cuda
        .prepare_rgba(&distorted, SIDE, SIDE, SCALE_COUNT)
        .unwrap();

    let difference = measure_cross_backend_difference(
        &cpu,
        &cuda,
        &cpu_reference,
        &cpu_distorted,
        &cuda_reference,
        &cuda_distorted,
    )
    .unwrap();

    assert!(
        difference.max_map_difference < MAP_TOLERANCE,
        "rgba map difference {} exceeds {MAP_TOLERANCE}",
        difference.max_map_difference
    );
    assert!(
        difference.max_deviation_difference < DEVIATION_TOLERANCE,
        "rgba deviation difference {} exceeds {DEVIATION_TOLERANCE}",
        difference.max_deviation_difference
    );
}
