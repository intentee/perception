use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use crate::parallel_pixel_threshold::PARALLEL_PIXEL_THRESHOLD;

pub(crate) fn map_indices<Output, Body>(count: usize, body: Body) -> Vec<Output>
where
    Output: Send,
    Body: Fn(usize) -> Output + Sync + Send,
{
    if count >= PARALLEL_PIXEL_THRESHOLD {
        (0..count)
            .into_par_iter()
            .with_min_len(PARALLEL_PIXEL_THRESHOLD)
            .map(body)
            .collect()
    } else {
        (0..count).map(body).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::map_indices;
    use crate::parallel_pixel_threshold::PARALLEL_PIXEL_THRESHOLD;

    #[test]
    fn small_counts_run_sequentially() {
        assert_eq!(map_indices(4, |index| index * index), vec![0, 1, 4, 9]);
    }

    #[test]
    fn large_counts_match_the_sequential_result() {
        let count = PARALLEL_PIXEL_THRESHOLD;
        let expected: Vec<usize> = (0..count).map(|index| index * 2 + 1).collect();

        assert_eq!(map_indices(count, |index| index * 2 + 1), expected);
    }
}
