#[cfg(feature = "cpu")]
fn main() {
    use ssim_metric::Ssim;
    use ssim_metric_bench::benchmark_sizes::BENCHMARK_SIZES;
    use ssim_metric_bench_scenarios::create_image::create_image;

    let engine = Ssim::new();

    println!("{}", create_image(&engine, &BENCHMARK_SIZES));
}

#[cfg(not(feature = "cpu"))]
fn main() {}
