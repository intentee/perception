pub(crate) fn reduce_deviation(map: &[f32], adjusted_mean_exponent: f64) -> f64 {
    let pixel_count = map.len() as f64;
    let mean = map.iter().map(|&value| f64::from(value)).sum::<f64>() / pixel_count;
    let adjusted_mean = mean.max(0.0).powf(adjusted_mean_exponent);

    map.iter()
        .map(|&value| (adjusted_mean - f64::from(value)).abs())
        .sum::<f64>()
        / pixel_count
}
