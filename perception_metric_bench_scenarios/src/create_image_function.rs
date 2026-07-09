use perception_metric::Ssim;
use perception_metric::SsimError;
use perception_metric::SsimImage;

pub(crate) type CreateImageFunction<BackendStrategy, Pixel> =
    fn(
        &Ssim<BackendStrategy>,
        &[Pixel],
        usize,
        usize,
    ) -> Result<SsimImage<BackendStrategy>, SsimError>;
