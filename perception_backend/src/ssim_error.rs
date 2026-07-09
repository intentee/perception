use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SsimError {
    #[error(
        "image dimensions differ: reference is {reference_width}x{reference_height}, distorted is {distorted_width}x{distorted_height}"
    )]
    DimensionMismatch {
        reference_width: usize,
        reference_height: usize,
        distorted_width: usize,
        distorted_height: usize,
    },
    #[error("image has a zero dimension: {width}x{height}")]
    EmptyImage { width: usize, height: usize },
    #[error("preparing a {width}x{height} image on the backend failed: {reason}")]
    ImagePreparation {
        width: usize,
        height: usize,
        reason: String,
    },
    #[error("pixel buffer holds {actual} pixels but {width}x{height} requires {expected}")]
    PixelCountMismatch {
        width: usize,
        height: usize,
        expected: usize,
        actual: usize,
    },
    #[error("comparing a {width}x{height} scale on the backend failed: {reason}")]
    ScaleComparison {
        width: usize,
        height: usize,
        reason: String,
    },
    #[error("at least one scale weight is required, but none were provided")]
    ScaleWeightsEmpty,
}

#[cfg(test)]
mod tests {
    use super::SsimError;

    #[test]
    fn each_variant_renders_its_diagnostic_message() {
        assert_eq!(
            SsimError::DimensionMismatch {
                reference_width: 4,
                reference_height: 5,
                distorted_width: 6,
                distorted_height: 7,
            }
            .to_string(),
            "image dimensions differ: reference is 4x5, distorted is 6x7"
        );
        assert_eq!(
            SsimError::EmptyImage {
                width: 0,
                height: 4,
            }
            .to_string(),
            "image has a zero dimension: 0x4"
        );
        assert_eq!(
            SsimError::ImagePreparation {
                width: 8,
                height: 9,
                reason: "device unavailable".to_string(),
            }
            .to_string(),
            "preparing a 8x9 image on the backend failed: device unavailable"
        );
        assert_eq!(
            SsimError::PixelCountMismatch {
                width: 4,
                height: 4,
                expected: 16,
                actual: 3,
            }
            .to_string(),
            "pixel buffer holds 3 pixels but 4x4 requires 16"
        );
        assert_eq!(
            SsimError::ScaleComparison {
                width: 2,
                height: 3,
                reason: "kernel launch failed".to_string(),
            }
            .to_string(),
            "comparing a 2x3 scale on the backend failed: kernel launch failed"
        );
        assert_eq!(
            SsimError::ScaleWeightsEmpty.to_string(),
            "at least one scale weight is required, but none were provided"
        );
    }
}
