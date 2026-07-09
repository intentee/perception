use std::path::Path;

use perception_metric::Ssim;

use crate::compare_paths::compare_paths;
use crate::comparison::Comparison;
use crate::diff_paths::diff_paths;
use crate::similarity_error::SimilarityError;
use crate::three_way_diff::ThreeWayDiff;

pub struct ImagePair<'paths> {
    original: &'paths Path,
    distorted: &'paths Path,
}

impl<'paths> ImagePair<'paths> {
    #[must_use]
    pub fn new(original: &'paths Path, distorted: &'paths Path) -> Self {
        Self {
            original,
            distorted,
        }
    }

    pub fn compare(self) -> Result<Comparison, SimilarityError> {
        let Self {
            original,
            distorted,
        } = self;
        let engine = Ssim::new().with_saved_map_scales(1);

        compare_paths(&engine, original, distorted)
    }

    pub fn diff(self) -> Result<ThreeWayDiff, SimilarityError> {
        let Self {
            original,
            distorted,
        } = self;
        let engine = Ssim::new().with_saved_map_scales(1);

        diff_paths(&engine, original, distorted)
    }
}

#[cfg(test)]
mod tests {
    use perception_test::Scratch;
    use perception_test::write_test_image;

    use super::ImagePair;

    #[test]
    fn identical_files_have_similarity_of_one() {
        let scratch = Scratch::new("identical");
        let original = scratch.path("original.png");
        write_test_image(&original, 12, 0);

        let similarity = ImagePair::new(&original, &original)
            .compare()
            .unwrap()
            .similarity();

        assert!((similarity - 1.0).abs() < 1e-9);
    }

    #[test]
    fn different_files_have_similarity_below_one() {
        let scratch = Scratch::new("different");
        let original = scratch.path("original.png");
        let distorted = scratch.path("distorted.png");
        write_test_image(&original, 12, 0);
        write_test_image(&distorted, 12, 40);

        let similarity = ImagePair::new(&original, &distorted)
            .compare()
            .unwrap()
            .similarity();

        assert!(similarity < 1.0);
    }

    #[test]
    fn a_missing_original_is_an_error() {
        let scratch = Scratch::new("missing_original");
        let distorted = scratch.path("distorted.png");
        write_test_image(&distorted, 12, 0);

        let result = ImagePair::new(&scratch.path("missing.png"), &distorted).compare();

        assert!(result.is_err());
    }

    #[test]
    fn a_missing_distorted_is_an_error() {
        let scratch = Scratch::new("missing_distorted");
        let original = scratch.path("original.png");
        write_test_image(&original, 12, 0);

        let result = ImagePair::new(&original, &scratch.path("missing.png")).compare();

        assert!(result.is_err());
    }

    #[test]
    fn a_dimension_mismatch_is_an_error() {
        let scratch = Scratch::new("mismatch");
        let original = scratch.path("original.png");
        let distorted = scratch.path("distorted.png");
        write_test_image(&original, 12, 0);
        write_test_image(&distorted, 8, 0);

        let result = ImagePair::new(&original, &distorted).compare();

        assert!(result.is_err());
    }

    #[test]
    fn a_missing_original_makes_diff_an_error() {
        let scratch = Scratch::new("diff_missing_original");
        let distorted = scratch.path("distorted.png");
        write_test_image(&distorted, 12, 0);

        let result = ImagePair::new(&scratch.path("missing.png"), &distorted).diff();

        assert!(result.is_err());
    }

    #[test]
    fn a_missing_distorted_makes_diff_an_error() {
        let scratch = Scratch::new("diff_missing_distorted");
        let original = scratch.path("original.png");
        write_test_image(&original, 12, 0);

        let result = ImagePair::new(&original, &scratch.path("missing.png")).diff();

        assert!(result.is_err());
    }

    #[test]
    fn a_dimension_mismatch_makes_diff_an_error() {
        let scratch = Scratch::new("diff_mismatch");
        let original = scratch.path("original.png");
        let distorted = scratch.path("distorted.png");
        write_test_image(&original, 12, 0);
        write_test_image(&distorted, 8, 0);

        let result = ImagePair::new(&original, &distorted).diff();

        assert!(result.is_err());
    }
}
