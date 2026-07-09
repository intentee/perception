use crate::average_ranks::average_ranks;
use crate::pearson::pearson;

#[must_use]
pub fn spearman(first: &[f64], second: &[f64]) -> f64 {
    pearson(&average_ranks(first), &average_ranks(second))
}

#[cfg(test)]
mod tests {
    use super::spearman;

    #[test]
    fn monotonic_relationship_is_one() {
        assert!((spearman(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn reversed_relationship_is_minus_one() {
        assert!((spearman(&[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0]) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn matches_reference_value_with_ties() {
        assert!(
            (spearman(&[1.0, 1.0, 2.0, 3.0], &[1.0, 2.0, 3.0, 4.0]) - 0.948_683_298_05).abs()
                < 1e-9
        );
    }
}
