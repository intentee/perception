use imgref::ImgVec;
use palette::stimulus::FromStimulus;
use rgb::RGB;
use rgb::RGBA;

use ssim_backend::Backend;
use ssim_backend::ScaleScoreWithMap;
use ssim_backend::SsimError;
#[cfg(feature = "cpu")]
use ssim_backend_cpu::CpuBackend;

use crate::comparison_result::ComparisonResult;
use crate::similarity::Similarity;
use crate::ssim_image::SsimImage;
use crate::ssim_map::SsimMap;
use crate::weights::DEFAULT_WEIGHTS;

fn validate_dimensions(pixel_count: usize, width: usize, height: usize) -> Result<(), SsimError> {
    if width == 0 || height == 0 {
        return Err(SsimError::EmptyImage { width, height });
    }

    let expected = width * height;

    if pixel_count != expected {
        return Err(SsimError::PixelCountMismatch {
            width,
            height,
            expected,
            actual: pixel_count,
        });
    }

    Ok(())
}

pub struct Ssim<BackendStrategy>
where
    BackendStrategy: Backend,
{
    backend: BackendStrategy,
    scale_weights: Vec<f64>,
    saved_map_scales: u8,
}

#[cfg(feature = "cpu")]
impl Ssim<CpuBackend> {
    #[must_use]
    pub fn new() -> Self {
        Self::with_backend(CpuBackend::default())
    }
}

impl<BackendStrategy> Ssim<BackendStrategy>
where
    BackendStrategy: Backend,
{
    #[must_use]
    pub fn with_backend(backend: BackendStrategy) -> Self {
        Self {
            backend,
            scale_weights: DEFAULT_WEIGHTS.to_vec(),
            saved_map_scales: 0,
        }
    }

    #[must_use]
    pub fn with_saved_map_scales(mut self, count: u8) -> Self {
        self.saved_map_scales = count;

        self
    }

    pub fn with_scale_weights(mut self, weights: &[f64]) -> Result<Self, SsimError> {
        if weights.is_empty() {
            return Err(SsimError::ScaleWeightsEmpty);
        }

        self.scale_weights = weights.to_vec();

        Ok(self)
    }

    pub fn compare(
        &self,
        original: &SsimImage<BackendStrategy>,
        distorted: &SsimImage<BackendStrategy>,
    ) -> Result<ComparisonResult, SsimError> {
        if original.width() != distorted.width() || original.height() != distorted.height() {
            return Err(SsimError::DimensionMismatch {
                reference_width: original.width(),
                reference_height: original.height(),
                distorted_width: distorted.width(),
                distorted_height: distorted.height(),
            });
        }

        let mut weighted_score_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut ssim_maps = Vec::new();

        for (scale_index, weight) in self.scale_weights.iter().copied().enumerate() {
            if scale_index >= original.scales().len() {
                break;
            }

            let adjusted_mean_exponent = 0.5f64.powf(scale_index as f64);
            let deviation = if usize::from(self.saved_map_scales) > scale_index {
                let ScaleScoreWithMap { deviation, map } = self.backend.compare_scale_with_map(
                    original.prepared(),
                    distorted.prepared(),
                    scale_index,
                    adjusted_mean_exponent,
                )?;

                ssim_maps.push(SsimMap {
                    map: ImgVec::new(map.pixels, map.width, map.height),
                    ssim: 1.0 - deviation,
                });

                deviation
            } else {
                self.backend
                    .compare_scale(
                        original.prepared(),
                        distorted.prepared(),
                        scale_index,
                        adjusted_mean_exponent,
                    )?
                    .deviation
            };
            let score = 1.0 - deviation;

            weighted_score_sum += score * weight;
            weight_sum += weight;
        }

        Ok(ComparisonResult {
            similarity: Similarity::new(weighted_score_sum / weight_sum),
            ssim_maps,
        })
    }

    pub fn create_image_gray<Component>(
        &self,
        bitmap: &[Component],
        width: usize,
        height: usize,
    ) -> Result<SsimImage<BackendStrategy>, SsimError>
    where
        Component: Copy + Send + Sync,
        f32: FromStimulus<Component>,
    {
        validate_dimensions(bitmap.len(), width, height)?;

        let prepared =
            self.backend
                .prepare_gray(bitmap, width, height, self.scale_weights.len())?;
        let scales = self.backend.scale_dimensions(&prepared);

        Ok(SsimImage::new(width, height, scales, prepared))
    }

    pub fn create_image_rgb<Component>(
        &self,
        bitmap: &[RGB<Component>],
        width: usize,
        height: usize,
    ) -> Result<SsimImage<BackendStrategy>, SsimError>
    where
        Component: Copy + Send + Sync,
        f32: FromStimulus<Component>,
    {
        validate_dimensions(bitmap.len(), width, height)?;

        let prepared = self
            .backend
            .prepare_rgb(bitmap, width, height, self.scale_weights.len())?;
        let scales = self.backend.scale_dimensions(&prepared);

        Ok(SsimImage::new(width, height, scales, prepared))
    }

    pub fn create_image_rgba<Component>(
        &self,
        bitmap: &[RGBA<Component>],
        width: usize,
        height: usize,
    ) -> Result<SsimImage<BackendStrategy>, SsimError>
    where
        Component: Copy + Send + Sync,
        f32: FromStimulus<Component>,
    {
        validate_dimensions(bitmap.len(), width, height)?;

        let prepared =
            self.backend
                .prepare_rgba(bitmap, width, height, self.scale_weights.len())?;
        let scales = self.backend.scale_dimensions(&prepared);

        Ok(SsimImage::new(width, height, scales, prepared))
    }
}

#[cfg(feature = "cpu")]
impl Default for Ssim<CpuBackend> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use rgb::RGB;
    use rgb::RGBA;

    use super::Ssim;
    use ssim_backend::SsimError;

    const SIDE: usize = 24;

    fn gradient_rgba(shift: u8) -> Vec<RGBA<u8>> {
        (0..SIDE * SIDE)
            .map(|index| {
                let column = (index % SIDE) as u8;
                let row = (index / SIDE) as u8;

                RGBA::new(
                    column.wrapping_mul(10).wrapping_add(shift),
                    row.wrapping_mul(10),
                    128,
                    255,
                )
            })
            .collect()
    }

    #[test]
    fn identical_rgba_images_have_similarity_of_one() {
        let context = Ssim::new();
        let pixels = gradient_rgba(0);
        let reference = context.create_image_rgba(&pixels, SIDE, SIDE).unwrap();
        let distorted = context.create_image_rgba(&pixels, SIDE, SIDE).unwrap();

        let result = context.compare(&reference, &distorted).unwrap();

        assert!((result.similarity.value() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn different_rgba_images_have_similarity_below_one() {
        let context = Ssim::new();
        let reference = context
            .create_image_rgba(&gradient_rgba(0), SIDE, SIDE)
            .unwrap();
        let distorted = context
            .create_image_rgba(&gradient_rgba(90), SIDE, SIDE)
            .unwrap();

        let result = context.compare(&reference, &distorted).unwrap();

        assert!(result.similarity.value() < 1.0);
    }

    #[test]
    fn similarity_is_symmetric() {
        let context = Ssim::new();
        let reference = context
            .create_image_rgba(&gradient_rgba(0), SIDE, SIDE)
            .unwrap();
        let distorted = context
            .create_image_rgba(&gradient_rgba(60), SIDE, SIDE)
            .unwrap();

        let forward = context
            .compare(&reference, &distorted)
            .unwrap()
            .similarity
            .value();
        let backward = context
            .compare(&distorted, &reference)
            .unwrap()
            .similarity
            .value();

        assert_eq!(forward, backward);
    }

    #[test]
    fn larger_distortion_scores_lower() {
        let context = Ssim::new();
        let reference = context
            .create_image_rgba(&gradient_rgba(0), SIDE, SIDE)
            .unwrap();
        let small = context
            .create_image_rgba(&gradient_rgba(20), SIDE, SIDE)
            .unwrap();
        let large = context
            .create_image_rgba(&gradient_rgba(120), SIDE, SIDE)
            .unwrap();

        let small_difference = context.compare(&reference, &small).unwrap().similarity;
        let large_difference = context.compare(&reference, &large).unwrap().similarity;

        assert!(large_difference < small_difference);
    }

    #[test]
    fn rgb_and_gray_inputs_are_supported() {
        let context = Ssim::new();
        let rgb_pixels = vec![RGB::new(40u8, 90, 160); SIDE * SIDE];
        let gray_pixels = vec![100u8; SIDE * SIDE];

        let rgb_image = context.create_image_rgb(&rgb_pixels, SIDE, SIDE).unwrap();
        let gray_image = context.create_image_gray(&gray_pixels, SIDE, SIDE).unwrap();

        assert_eq!(rgb_image.width(), SIDE);
        assert_eq!(gray_image.height(), SIDE);
    }

    #[test]
    fn empty_dimension_is_rejected() {
        let context = Ssim::new();

        assert_eq!(
            context.create_image_rgba::<u8>(&[], 0, 4).err(),
            Some(SsimError::EmptyImage {
                width: 0,
                height: 4
            })
        );
    }

    #[test]
    fn pixel_count_mismatch_is_rejected() {
        let context = Ssim::new();
        let pixels = vec![RGBA::new(0u8, 0, 0, 255); 3];

        assert_eq!(
            context.create_image_rgba(&pixels, 4, 4).err(),
            Some(SsimError::PixelCountMismatch {
                width: 4,
                height: 4,
                expected: 16,
                actual: 3,
            })
        );
    }

    #[test]
    fn gray_pixel_count_mismatch_is_rejected() {
        let context = Ssim::new();

        assert_eq!(
            context.create_image_gray(&[0u8; 3], 4, 4).err(),
            Some(SsimError::PixelCountMismatch {
                width: 4,
                height: 4,
                expected: 16,
                actual: 3,
            })
        );
    }

    #[test]
    fn rgb_empty_dimension_is_rejected() {
        let context = Ssim::new();

        assert_eq!(
            context.create_image_rgb::<u8>(&[], 4, 0).err(),
            Some(SsimError::EmptyImage {
                width: 4,
                height: 0
            })
        );
    }

    #[test]
    fn rgb16_pixel_count_mismatch_is_rejected() {
        let context = Ssim::new();
        let pixels = vec![RGB::new(0u16, 0, 0); 3];

        assert_eq!(
            context.create_image_rgb(&pixels, 4, 4).err(),
            Some(SsimError::PixelCountMismatch {
                width: 4,
                height: 4,
                expected: 16,
                actual: 3,
            })
        );
    }

    #[test]
    fn rgba16_empty_dimension_is_rejected() {
        let context = Ssim::new();

        assert_eq!(
            context.create_image_rgba::<u16>(&[], 0, 4).err(),
            Some(SsimError::EmptyImage {
                width: 0,
                height: 4
            })
        );
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let context = Ssim::new();
        let reference = context
            .create_image_rgba(&gradient_rgba(0), SIDE, SIDE)
            .unwrap();
        let smaller = vec![RGBA::new(0u8, 0, 0, 255); 8 * 8];
        let distorted = context.create_image_rgba(&smaller, 8, 8).unwrap();

        assert_eq!(
            context.compare(&reference, &distorted).err(),
            Some(SsimError::DimensionMismatch {
                reference_width: SIDE,
                reference_height: SIDE,
                distorted_width: 8,
                distorted_height: 8,
            })
        );
    }

    #[test]
    fn empty_scale_weights_are_rejected() {
        assert_eq!(
            Ssim::new().with_scale_weights(&[]).err(),
            Some(SsimError::ScaleWeightsEmpty)
        );
    }

    #[test]
    fn saved_ssim_maps_are_returned_per_scale() {
        let context = Ssim::new()
            .with_scale_weights(&[0.5, 0.5])
            .unwrap()
            .with_saved_map_scales(1);
        let reference = context
            .create_image_rgba(&gradient_rgba(0), SIDE, SIDE)
            .unwrap();
        let distorted = context
            .create_image_rgba(&gradient_rgba(40), SIDE, SIDE)
            .unwrap();

        let result = context.compare(&reference, &distorted).unwrap();

        assert_eq!(result.ssim_maps.len(), 1);
        assert_eq!(result.ssim_maps[0].map.width(), SIDE);
    }

    #[test]
    fn default_matches_new() {
        let default_context = Ssim::default();
        let reference = default_context
            .create_image_rgba(&gradient_rgba(0), SIDE, SIDE)
            .unwrap();
        let distorted = default_context
            .create_image_rgba(&gradient_rgba(0), SIDE, SIDE)
            .unwrap();

        assert!(
            (default_context
                .compare(&reference, &distorted)
                .unwrap()
                .similarity
                .value()
                - 1.0)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn translucent_rgba_images_are_supported() {
        let context = Ssim::new();
        let pixels: Vec<RGBA<u8>> = (0..SIDE * SIDE)
            .map(|index| RGBA::new((index % SIDE) as u8, (index / SIDE) as u8, 100, 128))
            .collect();
        let reference = context.create_image_rgba(&pixels, SIDE, SIDE).unwrap();
        let distorted = context.create_image_rgba(&pixels, SIDE, SIDE).unwrap();

        let result = context.compare(&reference, &distorted).unwrap();

        assert!((result.similarity.value() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn matching_width_but_different_height_is_rejected() {
        let context = Ssim::new();
        let reference = context
            .create_image_rgba(&gradient_rgba(0), SIDE, SIDE)
            .unwrap();
        let shorter = vec![RGBA::new(0u8, 0, 0, 255); SIDE * 8];
        let distorted = context.create_image_rgba(&shorter, SIDE, 8).unwrap();

        assert_eq!(
            context.compare(&reference, &distorted).err(),
            Some(SsimError::DimensionMismatch {
                reference_width: SIDE,
                reference_height: SIDE,
                distorted_width: SIDE,
                distorted_height: 8,
            })
        );
    }

    #[test]
    fn opaque_rgba_matches_the_rgb_path() {
        let context = Ssim::new();
        let gradient_rgb = |shift: u8| -> Vec<RGB<u8>> {
            (0..SIDE * SIDE)
                .map(|index| {
                    let column = (index % SIDE) as u8;
                    let row = (index / SIDE) as u8;

                    RGB::new(
                        column.wrapping_mul(10).wrapping_add(shift),
                        row.wrapping_mul(10),
                        128,
                    )
                })
                .collect()
        };

        let rgb_reference = context
            .create_image_rgb(&gradient_rgb(0), SIDE, SIDE)
            .unwrap();
        let rgb_distorted = context
            .create_image_rgb(&gradient_rgb(60), SIDE, SIDE)
            .unwrap();
        let rgba_reference = context
            .create_image_rgba(&gradient_rgba(0), SIDE, SIDE)
            .unwrap();
        let rgba_distorted = context
            .create_image_rgba(&gradient_rgba(60), SIDE, SIDE)
            .unwrap();

        let rgb_similarity = context
            .compare(&rgb_reference, &rgb_distorted)
            .unwrap()
            .similarity
            .value();
        let rgba_similarity = context
            .compare(&rgba_reference, &rgba_distorted)
            .unwrap()
            .similarity
            .value();

        assert_eq!(rgb_similarity, rgba_similarity);
    }

    #[test]
    fn sixteen_bit_detail_below_the_eighth_bit_is_not_truncated() {
        let context = Ssim::new();
        let side = 12;
        let base = 117u16 * 256;
        let reference: Vec<RGB<u16>> = vec![RGB::new(base, base, base); side * side];
        let distorted: Vec<RGB<u16>> = (0..side * side)
            .map(|index| {
                let value = if index % side < side / 2 {
                    base
                } else {
                    base + 200
                };

                RGB::new(value, value, value)
            })
            .collect();
        let truncate = |pixels: &[RGB<u16>]| -> Vec<RGB<u8>> {
            pixels
                .iter()
                .map(|pixel| {
                    RGB::new(
                        (pixel.r >> 8) as u8,
                        (pixel.g >> 8) as u8,
                        (pixel.b >> 8) as u8,
                    )
                })
                .collect()
        };

        let reference16 = context.create_image_rgb(&reference, side, side).unwrap();
        let distorted16 = context.create_image_rgb(&distorted, side, side).unwrap();
        let reference8 = context
            .create_image_rgb(&truncate(&reference), side, side)
            .unwrap();
        let distorted8 = context
            .create_image_rgb(&truncate(&distorted), side, side)
            .unwrap();

        let sixteen_bit = context
            .compare(&reference16, &distorted16)
            .unwrap()
            .similarity
            .value();
        let eight_bit = context
            .compare(&reference8, &distorted8)
            .unwrap()
            .similarity
            .value();

        assert!(
            (eight_bit - 1.0).abs() < 1e-9,
            "eight-bit truncation erases the sub-byte edge: {eight_bit}"
        );
        assert!(
            sixteen_bit < 1.0 - 1e-6,
            "sixteen-bit path resolves detail the eight-bit path cannot: {sixteen_bit}"
        );
    }

    #[test]
    fn a_small_image_yields_a_single_pyramid_scale() {
        let context = Ssim::new();
        let pixels = vec![100u8; 7 * 7];

        let image = context.create_image_gray(&pixels, 7, 7).unwrap();

        assert_eq!(image.scales().len(), 1);
    }

    #[test]
    fn a_large_image_builds_the_full_five_scale_pyramid() {
        let context = Ssim::new();
        let pixels: Vec<u8> = (0..64 * 64).map(|index| (index % 256) as u8).collect();

        let image = context.create_image_gray(&pixels, 64, 64).unwrap();

        assert_eq!(image.scales().len(), 5);
    }

    #[test]
    fn similarity_magnitude_is_stable_for_a_fixed_pair() {
        let context = Ssim::new();
        let reference = context
            .create_image_rgba(&gradient_rgba(0), SIDE, SIDE)
            .unwrap();
        let distorted = context
            .create_image_rgba(&gradient_rgba(60), SIDE, SIDE)
            .unwrap();

        let similarity = context
            .compare(&reference, &distorted)
            .unwrap()
            .similarity
            .value();

        assert!(
            (similarity - 0.812_181_189_694_920_3).abs() < 1e-9,
            "pinned similarity drifted: {similarity}"
        );
    }

    fn preparation_result(
        fail_preparation: bool,
        width: usize,
        height: usize,
    ) -> Result<usize, SsimError> {
        if fail_preparation {
            Err(SsimError::ImagePreparation {
                width,
                height,
                reason: "stub backend preparation failure".to_string(),
            })
        } else {
            Ok(1)
        }
    }

    struct StubBackend {
        fail_preparation: bool,
    }

    impl ssim_backend::Backend for StubBackend {
        type Prepared = usize;

        fn prepare_gray<Component>(
            &self,
            _srgb: &[Component],
            width: usize,
            height: usize,
            _scale_count: usize,
        ) -> Result<Self::Prepared, SsimError>
        where
            Component: Copy + Send + Sync,
            f32: palette::stimulus::FromStimulus<Component>,
        {
            preparation_result(self.fail_preparation, width, height)
        }

        fn prepare_rgb<Component>(
            &self,
            _srgb: &[RGB<Component>],
            width: usize,
            height: usize,
            _scale_count: usize,
        ) -> Result<Self::Prepared, SsimError>
        where
            Component: Copy + Send + Sync,
            f32: palette::stimulus::FromStimulus<Component>,
        {
            preparation_result(self.fail_preparation, width, height)
        }

        fn prepare_rgba<Component>(
            &self,
            _srgb: &[RGBA<Component>],
            width: usize,
            height: usize,
            _scale_count: usize,
        ) -> Result<Self::Prepared, SsimError>
        where
            Component: Copy + Send + Sync,
            f32: palette::stimulus::FromStimulus<Component>,
        {
            preparation_result(self.fail_preparation, width, height)
        }

        fn scale_dimensions(
            &self,
            prepared: &Self::Prepared,
        ) -> Vec<ssim_backend::ScaleDimensions> {
            (0..*prepared)
                .map(|_| ssim_backend::ScaleDimensions {
                    width: 1,
                    height: 1,
                })
                .collect()
        }

        fn compare_scale(
            &self,
            _reference: &Self::Prepared,
            _distorted: &Self::Prepared,
            _scale_index: usize,
            _adjusted_mean_exponent: f64,
        ) -> Result<ssim_backend::ScaleScore, SsimError> {
            Err(SsimError::ScaleComparison {
                width: 1,
                height: 1,
                reason: "stub backend comparison failure".to_string(),
            })
        }

        fn compare_scale_with_map(
            &self,
            _reference: &Self::Prepared,
            _distorted: &Self::Prepared,
            _scale_index: usize,
            _adjusted_mean_exponent: f64,
        ) -> Result<ssim_backend::ScaleScoreWithMap, SsimError> {
            Err(SsimError::ScaleComparison {
                width: 1,
                height: 1,
                reason: "stub backend comparison failure".to_string(),
            })
        }
    }

    #[test]
    fn a_backend_preparation_failure_propagates_for_every_pixel_kind() {
        let context = Ssim::with_backend(StubBackend {
            fail_preparation: true,
        });

        assert!(context.create_image_gray(&[0u8; 4], 2, 2).is_err());
        assert!(
            context
                .create_image_rgb(&[RGB::new(0u8, 0, 0); 4], 2, 2)
                .is_err()
        );
        assert!(
            context
                .create_image_rgba(&[RGBA::new(0u8, 0, 0, 255); 4], 2, 2)
                .is_err()
        );
    }

    #[test]
    fn a_backend_comparison_failure_propagates() {
        let context = Ssim::with_backend(StubBackend {
            fail_preparation: false,
        });
        let reference = context.create_image_gray(&[0u8; 4], 2, 2).unwrap();
        let distorted = context.create_image_gray(&[0u8; 4], 2, 2).unwrap();

        assert!(context.compare(&reference, &distorted).is_err());
    }

    #[test]
    fn a_backend_comparison_failure_propagates_when_a_map_is_saved() {
        let context = Ssim::with_backend(StubBackend {
            fail_preparation: false,
        })
        .with_saved_map_scales(1);
        let reference = context.create_image_gray(&[0u8; 4], 2, 2).unwrap();
        let distorted = context.create_image_gray(&[0u8; 4], 2, 2).unwrap();

        assert!(context.compare(&reference, &distorted).is_err());
    }
}
