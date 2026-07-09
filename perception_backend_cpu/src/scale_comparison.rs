use crate::gaussian_blur::GaussianBlur;
use crate::map_indices::map_indices;
use crate::plane_error::PlaneError;
use crate::prepared_scale::PreparedScale;

const STABILIZER_LUMINANCE: f32 = 0.01 * 0.01;
const STABILIZER_CONTRAST: f32 = 0.03 * 0.03;

pub(crate) fn compare_scale(
    original: &PreparedScale,
    distorted: &PreparedScale,
    blur: &GaussianBlur,
) -> Result<Vec<f32>, PlaneError> {
    let channel_count = original.channels().len();
    let pixel_count = original.width() * original.height();

    let mut cross_blur: Vec<Vec<f32>> = Vec::with_capacity(channel_count);

    for (original_channel, distorted_channel) in
        original.channels().iter().zip(distorted.channels())
    {
        cross_blur
            .push(blur.blur_of_product(original_channel.values(), distorted_channel.values())?);
    }

    let inverse_channels = 1.0 / channel_count as f32;

    Ok(map_indices(pixel_count, |pixel| {
        let mut mu1_squared = 0.0f32;
        let mut mu2_squared = 0.0f32;
        let mut mu1_mu2 = 0.0f32;
        let mut sigma1_squared = 0.0f32;
        let mut sigma2_squared = 0.0f32;
        let mut sigma12 = 0.0f32;

        for ((original_channel, distorted_channel), cross) in original
            .channels()
            .iter()
            .zip(distorted.channels())
            .zip(&cross_blur)
        {
            let mu1 = original_channel.mu()[pixel];
            let mu2 = distorted_channel.mu()[pixel];
            let mu1_mu1 = mu1 * mu1;
            let mu2_mu2 = mu2 * mu2;
            let mu1_times_mu2 = mu1 * mu2;

            mu1_squared += mu1_mu1;
            mu2_squared += mu2_mu2;
            mu1_mu2 += mu1_times_mu2;
            sigma1_squared += original_channel.squared_blur()[pixel] - mu1_mu1;
            sigma2_squared += distorted_channel.squared_blur()[pixel] - mu2_mu2;
            sigma12 += cross[pixel] - mu1_times_mu2;
        }

        mu1_squared *= inverse_channels;
        mu2_squared *= inverse_channels;
        mu1_mu2 *= inverse_channels;
        sigma1_squared *= inverse_channels;
        sigma2_squared *= inverse_channels;
        sigma12 *= inverse_channels;

        2.0f32.mul_add(mu1_mu2, STABILIZER_LUMINANCE) * 2.0f32.mul_add(sigma12, STABILIZER_CONTRAST)
            / ((mu1_squared + mu2_squared + STABILIZER_LUMINANCE)
                * (sigma1_squared + sigma2_squared + STABILIZER_CONTRAST))
    }))
}
