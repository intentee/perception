use ndarray::Array2;
use ndarray_stats::CorrelationExt;

pub(crate) fn pearson(first: &[f64], second: &[f64]) -> f64 {
    debug_assert_eq!(first.len(), second.len());

    let observations = first.len();
    let mut data = Vec::with_capacity(2 * observations);
    data.extend_from_slice(first);
    data.extend_from_slice(second);
    let matrix = Array2::from_shape_vec((2, observations), data)
        .expect("a two-row matrix always matches two series of equal length");

    matrix
        .pearson_correlation()
        .expect("correlation requires at least one observation")[[0, 1]]
}

#[cfg(test)]
mod tests {
    use super::pearson;

    #[test]
    fn perfect_positive_correlation_is_one() {
        assert!((pearson(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn perfect_negative_correlation_is_minus_one() {
        assert!((pearson(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn matches_reference_value() {
        assert!((pearson(&[1.0, 2.0, 3.0, 4.0], &[2.0, 1.0, 4.0, 3.0]) - 0.6).abs() < 1e-12);
    }
}
