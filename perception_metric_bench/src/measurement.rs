use std::time::Duration;

pub struct Measurement {
    minimum: Duration,
    median: Duration,
}

impl Measurement {
    #[must_use]
    pub fn from_samples(mut samples: Vec<Duration>) -> Self {
        samples.sort_unstable();

        Self {
            minimum: samples[0],
            median: samples[samples.len() / 2],
        }
    }

    #[must_use]
    pub fn median_milliseconds(&self) -> f64 {
        self.median.as_secs_f64() * 1_000.0
    }

    #[must_use]
    pub fn minimum_milliseconds(&self) -> f64 {
        self.minimum.as_secs_f64() * 1_000.0
    }

    #[must_use]
    pub fn throughput_megapixels_per_second(&self, pixel_count: usize) -> f64 {
        pixel_count as f64 / self.minimum.as_secs_f64() / 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::Measurement;

    #[test]
    fn reports_minimum_and_median_from_unsorted_samples() {
        let measurement = Measurement::from_samples(vec![
            Duration::from_millis(5),
            Duration::from_millis(1),
            Duration::from_millis(3),
        ]);

        assert!((measurement.minimum_milliseconds() - 1.0).abs() < 1e-9);
        assert!((measurement.median_milliseconds() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn throughput_scales_with_pixel_count() {
        let measurement = Measurement::from_samples(vec![Duration::from_secs(1)]);

        assert!((measurement.throughput_megapixels_per_second(2_000_000) - 2.0).abs() < 1e-9);
    }
}
