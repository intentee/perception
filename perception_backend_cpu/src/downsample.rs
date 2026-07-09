use image::ImageBuffer;
use image::Pixel;
use image::imageops::FilterType;
use image::imageops::resize;

const MINIMUM_DOWNSAMPLEABLE_SIDE: usize = 8;

pub(crate) fn downsample<Channel>(
    components: Vec<f32>,
    width: usize,
    height: usize,
) -> Option<Vec<f32>>
where
    Channel: Pixel<Subpixel = f32> + 'static,
{
    if width < MINIMUM_DOWNSAMPLEABLE_SIDE || height < MINIMUM_DOWNSAMPLEABLE_SIDE {
        return None;
    }

    let source =
        ImageBuffer::<Channel, Vec<f32>>::from_raw(width as u32, height as u32, components)?;
    let resized = resize(
        &source,
        (width / 2) as u32,
        (height / 2) as u32,
        FilterType::Triangle,
    );

    Some(resized.into_raw())
}

#[cfg(test)]
mod tests {
    use image::Luma;

    use super::downsample;

    #[test]
    fn sides_below_the_minimum_are_not_downsampleable() {
        assert_eq!(downsample::<Luma<f32>>(vec![0.0; 7 * 8], 7, 8), None);
        assert_eq!(downsample::<Luma<f32>>(vec![0.0; 8 * 7], 8, 7), None);
    }

    #[test]
    fn an_eight_pixel_side_halves_and_stays_within_the_input_range() {
        let pixels: Vec<f32> = (0..8 * 8).map(|index| index as f32 / 64.0).collect();
        let minimum = pixels.iter().copied().fold(f32::INFINITY, f32::min);
        let maximum = pixels.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let downsampled = downsample::<Luma<f32>>(pixels, 8, 8).expect("an 8x8 plane halves");

        assert_eq!(downsampled.len(), 4 * 4);

        for value in downsampled {
            assert!(
                (minimum..=maximum).contains(&value),
                "triangle resampling stays within the input range: {value}"
            );
        }
    }

    #[test]
    fn a_buffer_that_contradicts_the_dimensions_yields_no_downsample() {
        assert_eq!(downsample::<Luma<f32>>(vec![0.0; 3], 8, 8), None);
    }
}
