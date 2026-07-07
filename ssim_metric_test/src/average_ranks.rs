use std::cmp::Ordering;

pub(crate) fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();

    order.sort_by(|&left, &right| {
        values[left]
            .partial_cmp(&values[right])
            .expect("correlation inputs must not contain NaN")
    });

    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;

    while start < order.len() {
        let mut end = start;

        while end + 1 < order.len()
            && values[order[end + 1]].partial_cmp(&values[order[start]]) == Some(Ordering::Equal)
        {
            end += 1;
        }

        let average_rank = (start + end) as f64 / 2.0 + 1.0;

        for &index in &order[start..=end] {
            ranks[index] = average_rank;
        }

        start = end + 1;
    }

    ranks
}

#[cfg(test)]
mod tests {
    use super::average_ranks;

    #[test]
    fn distinct_values_get_sequential_ranks() {
        assert_eq!(average_ranks(&[30.0, 10.0, 20.0]), vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn tied_values_share_the_average_rank() {
        assert_eq!(average_ranks(&[10.0, 10.0, 20.0]), vec![1.5, 1.5, 3.0]);
    }
}
