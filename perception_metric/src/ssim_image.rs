use perception_backend::Backend;
use perception_backend::ScaleDimensions;

pub struct SsimImage<BackendStrategy>
where
    BackendStrategy: Backend,
{
    width: usize,
    height: usize,
    scales: Vec<ScaleDimensions>,
    prepared: BackendStrategy::Prepared,
}

impl<BackendStrategy> SsimImage<BackendStrategy>
where
    BackendStrategy: Backend,
{
    pub(crate) fn new(
        width: usize,
        height: usize,
        scales: Vec<ScaleDimensions>,
        prepared: BackendStrategy::Prepared,
    ) -> Self {
        Self {
            width,
            height,
            scales,
            prepared,
        }
    }

    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn prepared(&self) -> &BackendStrategy::Prepared {
        &self.prepared
    }

    pub(crate) fn scales(&self) -> &[ScaleDimensions] {
        &self.scales
    }
}
