use ssim_metric::ComparisonResult;

use crate::similarity_error::SimilarityError;
use crate::similarity_map::SimilarityMap;

pub struct Comparison {
    similarity: f64,
    map: SimilarityMap,
}

impl Comparison {
    pub(crate) fn from_metric(
        ComparisonResult {
            similarity,
            ssim_maps,
        }: ComparisonResult,
    ) -> Result<Self, SimilarityError> {
        let saved_map = ssim_maps
            .into_iter()
            .next()
            .ok_or(SimilarityError::MissingSimilarityMap)?;

        Ok(Self {
            similarity: similarity.value(),
            map: SimilarityMap::from_ssim_map(saved_map),
        })
    }

    #[must_use]
    pub fn into_map(self) -> SimilarityMap {
        self.map
    }

    #[must_use]
    pub fn similarity(&self) -> f64 {
        self.similarity
    }
}

#[cfg(test)]
mod tests {
    use ssim_metric::Ssim;
    use ssim_test::Scratch;
    use ssim_test::write_test_image;

    use super::Comparison;
    use crate::image_pair::ImagePair;

    #[test]
    fn a_comparison_without_a_saved_map_is_an_error() {
        let engine = Ssim::new();
        let image = engine.create_image_gray(&[0u8; 64], 8, 8).unwrap();
        let result = engine.compare(&image, &image).unwrap();

        assert!(Comparison::from_metric(result).is_err());
    }

    #[test]
    fn a_single_comparison_yields_both_a_score_and_a_matching_map() {
        let scratch = Scratch::new("both");
        let original = scratch.path("original.png");
        let distorted = scratch.path("distorted.png");
        write_test_image(&original, 12, 0);
        write_test_image(&distorted, 12, 40);

        let comparison = ImagePair::new(&original, &distorted).compare().unwrap();
        let similarity = comparison.similarity();
        let map = comparison.into_map();

        assert!(similarity > 0.0 && similarity < 1.0);
        assert_eq!(map.width(), 12);
        assert_eq!(map.height(), 12);
    }
}
