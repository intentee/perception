use std::cmp::Ordering;

#[must_use]
pub fn kendall_tau_b(first: &[f64], second: &[f64]) -> f64 {
    debug_assert_eq!(first.len(), second.len());

    let mut concordant = 0.0f64;
    let mut discordant = 0.0f64;
    let mut tied_first_only = 0.0f64;
    let mut tied_second_only = 0.0f64;

    for left in 0..first.len() {
        for right in (left + 1)..first.len() {
            let first_order = first[left]
                .partial_cmp(&first[right])
                .expect("correlation inputs must not contain NaN");
            let second_order = second[left]
                .partial_cmp(&second[right])
                .expect("correlation inputs must not contain NaN");

            match (first_order, second_order) {
                (Ordering::Equal, Ordering::Equal) => {}
                (Ordering::Equal, _) => tied_first_only += 1.0,
                (_, Ordering::Equal) => tied_second_only += 1.0,
                _ if first_order == second_order => concordant += 1.0,
                _ => discordant += 1.0,
            }
        }
    }

    let ranked_pairs = concordant + discordant;

    (concordant - discordant)
        / ((ranked_pairs + tied_second_only) * (ranked_pairs + tied_first_only)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::kendall_tau_b;

    #[test]
    fn concordant_sequence_is_one() {
        assert!((kendall_tau_b(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn discordant_sequence_is_minus_one() {
        assert!((kendall_tau_b(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn matches_reference_value_with_ties() {
        assert!(
            (kendall_tau_b(&[1.0, 1.0, 2.0], &[1.0, 2.0, 3.0]) - 0.816_496_580_93).abs() < 1e-9
        );
    }

    #[test]
    fn handles_ties_in_both_variables() {
        assert!(
            (kendall_tau_b(&[1.0, 1.0, 2.0, 3.0], &[1.0, 1.0, 2.0, 2.0]) - 0.894_427_191_0).abs()
                < 1e-9
        );
    }
}
