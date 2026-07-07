#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use ssim_backend_cuda::CudaBackend;
    use ssim_metric::Ssim;
    use ssim_metric_bench::benchmark_sizes::BENCHMARK_SIZES;
    use ssim_metric_bench_scenarios::pipeline::pipeline;

    let engine = Ssim::with_backend(CudaBackend::new()?);

    println!("{}", pipeline(&engine, &BENCHMARK_SIZES));

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {}
