use thiserror::Error;

#[derive(Debug, Error)]
pub(crate) enum PlaneError {
    #[error("a gaussian blur over a {width}x{height} plane failed: {reason}")]
    GaussianBlur {
        width: usize,
        height: usize,
        reason: String,
    },
}
