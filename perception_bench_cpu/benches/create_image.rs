#[cfg(feature = "cpu")]
fn main() {
    use perception_metric::Ssim;
    use perception_metric_bench::benchmark_sizes::BENCHMARK_SIZES;
    use perception_metric_bench_scenarios::create_image::create_image;

    let engine = Ssim::new();

    println!("{}", create_image(&engine, &BENCHMARK_SIZES));
}

#[cfg(not(feature = "cpu"))]
fn main() {}
