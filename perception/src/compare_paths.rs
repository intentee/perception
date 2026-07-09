use std::path::Path;

use rayon::join;

use perception_backend::Backend;
use perception_metric::Ssim;

use crate::comparison::Comparison;
use crate::decoded_image::DecodedImage;
use crate::similarity_error::SimilarityError;

fn compare_decoded<Strategy>(
    engine: &Ssim<Strategy>,
    original: DecodedImage,
    distorted: DecodedImage,
) -> Result<Comparison, SimilarityError>
where
    Strategy: Backend,
{
    let original_image = original.to_ssim_image(engine)?;
    let distorted_image = distorted.to_ssim_image(engine)?;

    Comparison::from_metric(engine.compare(&original_image, &distorted_image)?)
}

pub(crate) fn compare_paths<Strategy>(
    engine: &Ssim<Strategy>,
    original: &Path,
    distorted: &Path,
) -> Result<Comparison, SimilarityError>
where
    Strategy: Backend,
{
    let (original_decoded, distorted_decoded) = join(
        || DecodedImage::decode(original),
        || DecodedImage::decode(distorted),
    );

    compare_decoded(engine, original_decoded?, distorted_decoded?)
}

#[cfg(test)]
mod tests {
    use perception_metric::Ssim;

    use super::compare_decoded;
    use crate::decoded_image::DecodedImage;

    fn consistent_image() -> DecodedImage {
        DecodedImage::Rgb {
            width: 8,
            height: 8,
            raw: vec![0u8; 8 * 8 * 3],
        }
    }

    fn inconsistent_image() -> DecodedImage {
        DecodedImage::Rgb {
            width: 8,
            height: 8,
            raw: vec![0u8; 3],
        }
    }

    #[test]
    fn an_inconsistent_original_image_is_an_error() {
        let engine = Ssim::new().with_saved_map_scales(1);

        assert!(compare_decoded(&engine, inconsistent_image(), consistent_image()).is_err());
    }

    #[test]
    fn an_inconsistent_distorted_image_is_an_error() {
        let engine = Ssim::new().with_saved_map_scales(1);

        assert!(compare_decoded(&engine, consistent_image(), inconsistent_image()).is_err());
    }
}
