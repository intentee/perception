use std::path::PathBuf;

use thiserror::Error;

use perception_metric::SsimError;

#[derive(Debug, Error)]
pub enum SimilarityError {
    #[error("failed to decode image {}", .path.display())]
    Decode {
        path: PathBuf,
        source: image::ImageError,
    },
    #[error(
        "a similarity map of {width}x{height} does not match its {pixel_count}-pixel intensity buffer"
    )]
    InconsistentSimilarityMap {
        width: usize,
        height: usize,
        pixel_count: usize,
    },
    #[error("failed to write similarity map {}", .path.display())]
    MapWrite {
        path: PathBuf,
        source: image::ImageError,
    },
    #[error("the similarity metric failed to compare the images")]
    Metric(#[from] SsimError),
    #[error("the comparison produced no saved similarity map")]
    MissingSimilarityMap,
}
