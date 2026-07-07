use std::path::Path;

use ssim_metric::SsimMap;

use crate::similarity_error::SimilarityError;

const MAXIMUM_INTENSITY: f32 = 255.0;

pub struct SimilarityMap {
    width: usize,
    height: usize,
    values: Vec<f32>,
}

impl SimilarityMap {
    pub(crate) fn from_ssim_map(SsimMap { map, ssim: _ }: SsimMap) -> Self {
        Self {
            width: map.width(),
            height: map.height(),
            values: map.into_buf(),
        }
    }

    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn write(self, output: &Path) -> Result<(), SimilarityError> {
        let Self {
            width,
            height,
            values,
        } = self;
        let pixel_count = values.len();
        let intensities: Vec<u8> = values
            .iter()
            .map(|value| ((1.0 - value).clamp(0.0, 1.0) * MAXIMUM_INTENSITY).round() as u8)
            .collect();
        let buffer = image::GrayImage::from_raw(width as u32, height as u32, intensities).ok_or(
            SimilarityError::InconsistentSimilarityMap {
                width,
                height,
                pixel_count,
            },
        )?;

        buffer
            .save(output)
            .map_err(|source| SimilarityError::MapWrite {
                path: output.to_path_buf(),
                source,
            })
    }
}

#[cfg(test)]
mod tests {
    use ssim_test::Scratch;
    use ssim_test::write_test_image;

    use super::SimilarityMap;
    use crate::image_pair::ImagePair;

    #[test]
    fn writing_a_map_whose_values_contradict_its_dimensions_is_an_error() {
        let scratch = Scratch::new("inconsistent");
        let inconsistent = SimilarityMap {
            width: 2,
            height: 2,
            values: vec![1.0; 3],
        };

        assert!(inconsistent.write(&scratch.path("map.png")).is_err());
    }

    #[test]
    fn exposes_per_pixel_values_sized_to_the_image() {
        let scratch = Scratch::new("values");
        let original = scratch.path("original.png");
        let distorted = scratch.path("distorted.png");
        write_test_image(&original, 12, 0);
        write_test_image(&distorted, 12, 40);

        let map = ImagePair::new(&original, &distorted)
            .compare()
            .unwrap()
            .into_map();

        assert_eq!(map.values().len(), map.width() * map.height());
    }

    #[test]
    fn differing_images_reduce_some_similarity_values() {
        let scratch = Scratch::new("differing");
        let original = scratch.path("original.png");
        let distorted = scratch.path("distorted.png");
        write_test_image(&original, 12, 0);
        write_test_image(&distorted, 12, 40);

        let map = ImagePair::new(&original, &distorted)
            .compare()
            .unwrap()
            .into_map();

        assert!(map.values().iter().any(|&value| value < 1.0));
    }

    #[test]
    fn identical_images_yield_maximal_similarity_values() {
        let scratch = Scratch::new("identical");
        let original = scratch.path("original.png");
        write_test_image(&original, 12, 0);

        let map = ImagePair::new(&original, &original)
            .compare()
            .unwrap()
            .into_map();

        assert!(map.values().iter().all(|&value| (value - 1.0).abs() < 1e-4));
    }

    #[test]
    fn writes_the_map_to_a_file() {
        let scratch = Scratch::new("write");
        let original = scratch.path("original.png");
        let distorted = scratch.path("distorted.png");
        let output = scratch.path("map.png");
        write_test_image(&original, 12, 0);
        write_test_image(&distorted, 12, 40);

        ImagePair::new(&original, &distorted)
            .compare()
            .unwrap()
            .into_map()
            .write(&output)
            .unwrap();

        assert!(output.exists());
    }

    #[test]
    fn writing_to_an_unwritable_path_is_an_error() {
        let scratch = Scratch::new("unwritable");
        let original = scratch.path("original.png");
        let distorted = scratch.path("distorted.png");
        write_test_image(&original, 12, 0);
        write_test_image(&distorted, 12, 40);

        let result = ImagePair::new(&original, &distorted)
            .compare()
            .unwrap()
            .into_map()
            .write(&scratch.path("missing").join("map.png"));

        assert!(result.is_err());
    }
}
