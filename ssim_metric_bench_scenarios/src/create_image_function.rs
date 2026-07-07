use ssim_metric::Ssim;
use ssim_metric::SsimError;
use ssim_metric::SsimImage;

pub(crate) type CreateImageFunction<BackendStrategy, Pixel> =
    fn(
        &Ssim<BackendStrategy>,
        &[Pixel],
        usize,
        usize,
    ) -> Result<SsimImage<BackendStrategy>, SsimError>;
