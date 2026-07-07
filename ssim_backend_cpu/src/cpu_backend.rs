use palette::stimulus::FromStimulus;
use rgb::RGB;
use rgb::RGBA;

use ssim_backend::Backend;
use ssim_backend::MapPlane;
use ssim_backend::ScaleDimensions;
use ssim_backend::ScaleScore;
use ssim_backend::ScaleScoreWithMap;
use ssim_backend::SsimError;

use crate::cpu_prepared::CpuPrepared;
use crate::gaussian_blur::GaussianBlur;
use crate::linear_gray_image::LinearGrayImage;
use crate::linear_pyramid_level::LinearPyramidLevel;
use crate::linear_rgb_image::LinearRgbImage;
use crate::linear_rgba_image::LinearRgbaImage;
use crate::plane_error::PlaneError;
use crate::prepared_scale::PreparedScale;
use crate::reduce_deviation::reduce_deviation;
use crate::scale_comparison::compare_scale;

fn build<Level>(
    full_resolution: Level,
    scale_count: usize,
    blur: &GaussianBlur,
) -> Result<CpuPrepared, PlaneError>
where
    Level: LinearPyramidLevel,
{
    let mut scales = Vec::with_capacity(scale_count);
    let mut level = full_resolution;

    loop {
        scales.push(PreparedScale::prepare(level.to_lab_planes(), blur)?);

        if scales.len() >= scale_count {
            break;
        }

        match level.downsampled() {
            Some(smaller) => level = smaller,
            None => break,
        }
    }

    Ok(CpuPrepared::new(scales))
}

pub struct CpuBackend {
    blur: GaussianBlur,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self {
            blur: GaussianBlur::new(),
        }
    }
}

impl Backend for CpuBackend {
    type Prepared = CpuPrepared;

    fn compare_scale(
        &self,
        reference: &Self::Prepared,
        distorted: &Self::Prepared,
        scale_index: usize,
        adjusted_mean_exponent: f64,
    ) -> Result<ScaleScore, SsimError> {
        let reference_scale = &reference.scales()[scale_index];
        let width = reference_scale.width();
        let height = reference_scale.height();
        let map = compare_scale(
            reference_scale,
            &distorted.scales()[scale_index],
            &self.blur,
        )
        .map_err(|error| SsimError::ScaleComparison {
            width,
            height,
            reason: error.to_string(),
        })?;

        Ok(ScaleScore {
            deviation: reduce_deviation(&map, adjusted_mean_exponent),
        })
    }

    fn compare_scale_with_map(
        &self,
        reference: &Self::Prepared,
        distorted: &Self::Prepared,
        scale_index: usize,
        adjusted_mean_exponent: f64,
    ) -> Result<ScaleScoreWithMap, SsimError> {
        let reference_scale = &reference.scales()[scale_index];
        let width = reference_scale.width();
        let height = reference_scale.height();
        let map = compare_scale(
            reference_scale,
            &distorted.scales()[scale_index],
            &self.blur,
        )
        .map_err(|error| SsimError::ScaleComparison {
            width,
            height,
            reason: error.to_string(),
        })?;
        let deviation = reduce_deviation(&map, adjusted_mean_exponent);

        Ok(ScaleScoreWithMap {
            deviation,
            map: MapPlane {
                width,
                height,
                pixels: map,
            },
        })
    }

    fn prepare_gray<Component>(
        &self,
        srgb: &[Component],
        width: usize,
        height: usize,
        scale_count: usize,
    ) -> Result<Self::Prepared, SsimError>
    where
        Component: Copy + Send + Sync,
        f32: FromStimulus<Component>,
    {
        build(
            LinearGrayImage::from_srgb(srgb, width, height),
            scale_count,
            &self.blur,
        )
        .map_err(|error| SsimError::ImagePreparation {
            width,
            height,
            reason: error.to_string(),
        })
    }

    fn prepare_rgb<Component>(
        &self,
        srgb: &[RGB<Component>],
        width: usize,
        height: usize,
        scale_count: usize,
    ) -> Result<Self::Prepared, SsimError>
    where
        Component: Copy + Send + Sync,
        f32: FromStimulus<Component>,
    {
        build(
            LinearRgbImage::from_srgb(srgb, width, height),
            scale_count,
            &self.blur,
        )
        .map_err(|error| SsimError::ImagePreparation {
            width,
            height,
            reason: error.to_string(),
        })
    }

    fn prepare_rgba<Component>(
        &self,
        srgb: &[RGBA<Component>],
        width: usize,
        height: usize,
        scale_count: usize,
    ) -> Result<Self::Prepared, SsimError>
    where
        Component: Copy + Send + Sync,
        f32: FromStimulus<Component>,
    {
        build(
            LinearRgbaImage::from_srgb(srgb, width, height),
            scale_count,
            &self.blur,
        )
        .map_err(|error| SsimError::ImagePreparation {
            width,
            height,
            reason: error.to_string(),
        })
    }

    fn scale_dimensions(&self, prepared: &Self::Prepared) -> Vec<ScaleDimensions> {
        prepared
            .scales()
            .iter()
            .map(|scale| ScaleDimensions {
                width: scale.width(),
                height: scale.height(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rgb::RGB;
    use rgb::RGBA;

    use ssim_backend::Backend;

    use super::CpuBackend;

    const SCALE_COUNT: usize = 5;
    const INCONSISTENT_PIXELS: usize = 3;
    const SIDE: usize = 8;

    #[test]
    fn preparing_gray_whose_buffer_contradicts_its_dimensions_is_an_error() {
        let backend = CpuBackend::default();

        assert!(
            backend
                .prepare_gray(&[0u8; INCONSISTENT_PIXELS], SIDE, SIDE, SCALE_COUNT)
                .is_err()
        );
    }

    #[test]
    fn preparing_rgb_whose_buffer_contradicts_its_dimensions_is_an_error() {
        let backend = CpuBackend::default();

        assert!(
            backend
                .prepare_rgb(
                    &[RGB::new(0u8, 0, 0); INCONSISTENT_PIXELS],
                    SIDE,
                    SIDE,
                    SCALE_COUNT
                )
                .is_err()
        );
    }

    #[test]
    fn preparing_rgba_whose_buffer_contradicts_its_dimensions_is_an_error() {
        let backend = CpuBackend::default();

        assert!(
            backend
                .prepare_rgba(
                    &[RGBA::new(0u8, 0, 0, 255); INCONSISTENT_PIXELS],
                    SIDE,
                    SIDE,
                    SCALE_COUNT
                )
                .is_err()
        );
    }

    #[test]
    fn comparing_scales_whose_dimensions_disagree_is_an_error() {
        let backend = CpuBackend::default();
        let larger = backend.prepare_gray(&[0u8; 16 * 16], 16, 16, 1).unwrap();
        let smaller = backend.prepare_gray(&[0u8; 8 * 8], 8, 8, 1).unwrap();

        assert!(backend.compare_scale(&larger, &smaller, 0, 1.0).is_err());
        assert!(
            backend
                .compare_scale_with_map(&larger, &smaller, 0, 1.0)
                .is_err()
        );
    }
}
