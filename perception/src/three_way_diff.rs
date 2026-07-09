use image::DynamicImage;
use image::Rgba;
use image::RgbaImage;

use crate::diff_output_paths::DiffOutputPaths;
use crate::dissimilarity_threshold::DissimilarityThreshold;
use crate::similarity_error::SimilarityError;
use crate::similarity_map::SimilarityMap;

const BASE_FADE_ALPHA: f32 = 0.1;
const LUMA_WEIGHT_BLUE: f32 = 0.11448223;
const LUMA_WEIGHT_GREEN: f32 = 0.586_622_5;
const LUMA_WEIGHT_RED: f32 = 0.298_895_3;
const MAXIMUM_INTENSITY: f32 = 255.0;

fn render_diff_panel(
    base: &RgbaImage,
    values: &[f32],
    threshold: &DissimilarityThreshold,
) -> RgbaImage {
    let (width, height) = base.dimensions();

    RgbaImage::from_fn(width, height, |column, row| {
        let index = row as usize * width as usize + column as usize;
        let dissimilarity = (1.0 - values[index]).clamp(0.0, 1.0);

        if dissimilarity >= threshold.value() {
            Rgba([u8::MAX, 0, 0, u8::MAX])
        } else {
            let Rgba([red, green, blue, alpha]) = *base.get_pixel(column, row);
            let luma = LUMA_WEIGHT_RED * f32::from(red)
                + LUMA_WEIGHT_GREEN * f32::from(green)
                + LUMA_WEIGHT_BLUE * f32::from(blue);
            let fade = BASE_FADE_ALPHA * (f32::from(alpha) / MAXIMUM_INTENSITY);
            let value = MAXIMUM_INTENSITY + (luma - MAXIMUM_INTENSITY) * fade;
            let gray = value.round() as u8;

            Rgba([gray, gray, gray, u8::MAX])
        }
    })
}

pub struct ThreeWayDiff {
    current: DynamicImage,
    expected: DynamicImage,
    map: SimilarityMap,
}

impl ThreeWayDiff {
    pub(crate) fn new(expected: DynamicImage, current: DynamicImage, map: SimilarityMap) -> Self {
        Self {
            current,
            expected,
            map,
        }
    }

    pub fn write(
        self,
        threshold: DissimilarityThreshold,
        output: &DiffOutputPaths<'_>,
    ) -> Result<(), SimilarityError> {
        let Self {
            current,
            expected,
            map,
        } = self;

        expected
            .save(output.expected())
            .map_err(|source| SimilarityError::ExpectedImageWrite {
                path: output.expected().to_path_buf(),
                source,
            })?;
        current
            .save(output.current())
            .map_err(|source| SimilarityError::CurrentImageWrite {
                path: output.current().to_path_buf(),
                source,
            })?;

        let panel = render_diff_panel(&expected.to_rgba8(), map.values(), &threshold);

        panel
            .save(output.diff())
            .map_err(|source| SimilarityError::DiffImageWrite {
                path: output.diff().to_path_buf(),
                source,
            })
    }
}

#[cfg(test)]
mod tests {
    use image::Rgba;
    use image::RgbaImage;

    use perception_test::Scratch;
    use perception_test::write_test_image;

    use super::render_diff_panel;
    use crate::diff_output_paths::DiffOutputPaths;
    use crate::dissimilarity_threshold::DissimilarityThreshold;
    use crate::image_pair::ImagePair;

    #[test]
    fn similar_pixels_fade_to_exact_grayscale() {
        let base = RgbaImage::from_pixel(1, 1, Rgba([0, 0, 100, 255]));
        let threshold = DissimilarityThreshold::new(0.5).unwrap();

        let panel = render_diff_panel(&base, &[1.0], &threshold);

        assert_eq!(*panel.get_pixel(0, 0), Rgba([231, 231, 231, 255]));
    }

    #[test]
    fn source_alpha_weights_the_fade() {
        let base = RgbaImage::from_pixel(1, 1, Rgba([0, 0, 100, 128]));
        let threshold = DissimilarityThreshold::new(0.5).unwrap();

        let panel = render_diff_panel(&base, &[1.0], &threshold);

        assert_eq!(*panel.get_pixel(0, 0), Rgba([243, 243, 243, 255]));
    }

    #[test]
    fn the_inclusive_threshold_paints_red() {
        let base = RgbaImage::from_pixel(1, 1, Rgba([0, 0, 100, 255]));
        let threshold = DissimilarityThreshold::new(0.5).unwrap();

        let panel = render_diff_panel(&base, &[0.5], &threshold);

        assert_eq!(*panel.get_pixel(0, 0), Rgba([255, 0, 0, 255]));
    }

    #[test]
    fn the_expected_output_holds_the_reference_pixels() {
        let scratch = Scratch::new("diff_expected_pixels");
        let original = scratch.path("original.png");
        let distorted = scratch.path("distorted.png");
        let expected = scratch.path("expected.png");
        let current = scratch.path("current.png");
        let diff = scratch.path("diff.png");
        write_test_image(&original, 12, 0);
        write_test_image(&distorted, 12, 40);

        ImagePair::new(&original, &distorted)
            .diff()
            .unwrap()
            .write(
                DissimilarityThreshold::new(0.5).unwrap(),
                &DiffOutputPaths::new(&expected, &current, &diff),
            )
            .unwrap();

        let written = image::open(&expected).unwrap().into_rgba8();

        assert_eq!(*written.get_pixel(0, 0), Rgba([0, 0, 100, 255]));
    }

    #[test]
    fn the_current_output_holds_the_distorted_pixels() {
        let scratch = Scratch::new("diff_current_pixels");
        let original = scratch.path("original.png");
        let distorted = scratch.path("distorted.png");
        let expected = scratch.path("expected.png");
        let current = scratch.path("current.png");
        let diff = scratch.path("diff.png");
        write_test_image(&original, 12, 0);
        write_test_image(&distorted, 12, 40);

        ImagePair::new(&original, &distorted)
            .diff()
            .unwrap()
            .write(
                DissimilarityThreshold::new(0.5).unwrap(),
                &DiffOutputPaths::new(&expected, &current, &diff),
            )
            .unwrap();

        let written = image::open(&current).unwrap().into_rgba8();

        assert_eq!(*written.get_pixel(0, 0), Rgba([40, 0, 100, 255]));
    }

    #[test]
    fn a_failed_expected_write_is_reported() {
        let scratch = Scratch::new("diff_expected_write");
        let original = scratch.path("original.png");
        let distorted = scratch.path("distorted.png");
        write_test_image(&original, 12, 0);
        write_test_image(&distorted, 12, 40);
        let expected = scratch.path("missing").join("expected.png");
        let current = scratch.path("current.png");
        let diff = scratch.path("diff.png");

        let result = ImagePair::new(&original, &distorted).diff().unwrap().write(
            DissimilarityThreshold::new(0.5).unwrap(),
            &DiffOutputPaths::new(&expected, &current, &diff),
        );

        assert!(result.is_err());
        assert!(!expected.exists() && !current.exists() && !diff.exists());
    }

    #[test]
    fn a_failed_current_write_is_reported() {
        let scratch = Scratch::new("diff_current_write");
        let original = scratch.path("original.png");
        let distorted = scratch.path("distorted.png");
        write_test_image(&original, 12, 0);
        write_test_image(&distorted, 12, 40);
        let expected = scratch.path("expected.png");
        let current = scratch.path("missing").join("current.png");
        let diff = scratch.path("diff.png");

        let result = ImagePair::new(&original, &distorted).diff().unwrap().write(
            DissimilarityThreshold::new(0.5).unwrap(),
            &DiffOutputPaths::new(&expected, &current, &diff),
        );

        assert!(result.is_err());
        assert!(expected.exists() && !current.exists() && !diff.exists());
    }

    #[test]
    fn a_failed_diff_write_is_reported() {
        let scratch = Scratch::new("diff_panel_write");
        let original = scratch.path("original.png");
        let distorted = scratch.path("distorted.png");
        write_test_image(&original, 12, 0);
        write_test_image(&distorted, 12, 40);
        let expected = scratch.path("expected.png");
        let current = scratch.path("current.png");
        let diff = scratch.path("missing").join("diff.png");

        let result = ImagePair::new(&original, &distorted).diff().unwrap().write(
            DissimilarityThreshold::new(0.5).unwrap(),
            &DiffOutputPaths::new(&expected, &current, &diff),
        );

        assert!(result.is_err());
        assert!(expected.exists() && current.exists() && !diff.exists());
    }
}
