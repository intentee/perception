use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;

use crate::parallel_pixel_threshold::PARALLEL_PIXEL_THRESHOLD;

pub(crate) fn map_indices_three<Body>(count: usize, body: Body) -> [Vec<f32>; 3]
where
    Body: Fn(usize) -> [f32; 3] + Sync + Send,
{
    let mut first = vec![0.0f32; count];
    let mut second = vec![0.0f32; count];
    let mut third = vec![0.0f32; count];
    let write =
        |index: usize, first_value: &mut f32, second_value: &mut f32, third_value: &mut f32| {
            let [a, b, c] = body(index);

            *first_value = a;
            *second_value = b;
            *third_value = c;
        };

    if count >= PARALLEL_PIXEL_THRESHOLD {
        first
            .par_iter_mut()
            .zip(second.par_iter_mut())
            .zip(third.par_iter_mut())
            .with_min_len(PARALLEL_PIXEL_THRESHOLD)
            .enumerate()
            .for_each(|(index, ((first_value, second_value), third_value))| {
                write(index, first_value, second_value, third_value);
            });
    } else {
        first
            .iter_mut()
            .zip(second.iter_mut())
            .zip(third.iter_mut())
            .enumerate()
            .for_each(|(index, ((first_value, second_value), third_value))| {
                write(index, first_value, second_value, third_value);
            });
    }

    [first, second, third]
}

#[cfg(test)]
mod tests {
    use super::map_indices_three;
    use crate::parallel_pixel_threshold::PARALLEL_PIXEL_THRESHOLD;

    #[test]
    fn small_counts_fill_all_three_sequentially() {
        let [first, second, third] = map_indices_three(2, |index| {
            [index as f32, index as f32 + 1.0, index as f32 + 2.0]
        });

        assert_eq!(first, vec![0.0, 1.0]);
        assert_eq!(second, vec![1.0, 2.0]);
        assert_eq!(third, vec![2.0, 3.0]);
    }

    #[test]
    fn large_counts_match_the_sequential_result() {
        let count = PARALLEL_PIXEL_THRESHOLD;
        let [first, second, third] = map_indices_three(count, |index| {
            [index as f32, index as f32 * 2.0, index as f32 * 3.0]
        });

        let expected_first: Vec<f32> = (0..count).map(|index| index as f32).collect();
        let expected_second: Vec<f32> = (0..count).map(|index| index as f32 * 2.0).collect();
        let expected_third: Vec<f32> = (0..count).map(|index| index as f32 * 3.0).collect();

        assert_eq!(first, expected_first);
        assert_eq!(second, expected_second);
        assert_eq!(third, expected_third);
    }
}
