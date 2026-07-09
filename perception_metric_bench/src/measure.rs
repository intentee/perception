use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

use crate::measurement::Measurement;

const WARMUP_ITERATIONS: usize = 3;
const SAMPLE_COUNT: usize = 31;

#[must_use]
pub fn measure<Operation, Output>(mut operation: Operation) -> Measurement
where
    Operation: FnMut() -> Output,
{
    for _ in 0..WARMUP_ITERATIONS {
        black_box(operation());
    }

    let mut samples: Vec<Duration> = Vec::with_capacity(SAMPLE_COUNT);

    for _ in 0..SAMPLE_COUNT {
        let start = Instant::now();
        let output = operation();
        samples.push(start.elapsed());
        black_box(output);
    }

    Measurement::from_samples(samples)
}

#[cfg(test)]
mod tests {
    use super::measure;

    #[test]
    fn reports_ordered_timing_statistics_for_a_workload() {
        let measurement = measure(|| (0..64u64).sum::<u64>());

        assert!(measurement.minimum_milliseconds() >= 0.0);
        assert!(measurement.median_milliseconds() >= measurement.minimum_milliseconds());
    }
}
