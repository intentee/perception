#[must_use]
pub fn max_abs_difference(first: &[f32], second: &[f32]) -> f32 {
    first
        .iter()
        .zip(second)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}
