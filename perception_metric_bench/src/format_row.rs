use crate::measurement::Measurement;

#[must_use]
pub fn format_row(label: &str, side: usize, measurement: &Measurement) -> String {
    format!(
        "{label:<8} {side:>5}x{side:<5} {:>12.3} {:>12.3} {:>12.1}",
        measurement.median_milliseconds(),
        measurement.minimum_milliseconds(),
        measurement.throughput_megapixels_per_second(side * side),
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::measurement::Measurement;

    use super::format_row;

    #[test]
    fn formats_a_row_from_the_measurement_values() {
        let measurement = Measurement::from_samples(vec![Duration::from_secs(1)]);

        let row = format_row("ssim", 1000, &measurement);

        assert_eq!(
            row,
            "ssim      1000x1000      1000.000     1000.000          1.0"
        );
    }
}
