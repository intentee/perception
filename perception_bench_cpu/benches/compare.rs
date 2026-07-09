#[cfg(feature = "cpu")]
fn main() {
    use perception_metric::Ssim;
    use perception_metric_bench::benchmark_sizes::BENCHMARK_SIZES;
    use perception_metric_bench_scenarios::compare::compare;

    let engine = Ssim::new();

    println!("{}", compare(&engine, &BENCHMARK_SIZES));
}

#[cfg(not(feature = "cpu"))]
fn main() {}
