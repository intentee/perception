#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use perception_backend_cuda::CudaBackend;
    use perception_metric::Ssim;
    use perception_metric_bench::benchmark_sizes::BENCHMARK_SIZES;
    use perception_metric_bench_scenarios::create_image::create_image;

    let engine = Ssim::with_backend(CudaBackend::new()?);

    println!("{}", create_image(&engine, &BENCHMARK_SIZES));

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {}
