use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::CudaModule;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use palette::stimulus::FromStimulus;
use rgb::RGB;
use rgb::RGBA;

use ssim_backend::Backend;
use ssim_backend::MapPlane;
use ssim_backend::ScaleDimensions;
use ssim_backend::ScaleScore;
use ssim_backend::ScaleScoreWithMap;
use ssim_backend::SsimError;

use crate::compile_ptx::compile_ptx;
use crate::create_context::create_context;
use crate::cuda_error::CudaError;
use crate::cuda_prepared::CudaPrepared;
use crate::cuda_prepared_channel::CudaPreparedChannel;
use crate::cuda_scale::CudaScale;
use crate::device_capability::DeviceCapability;
use crate::download_plane::download_plane;
use crate::elementwise_product::ElementwiseProduct;
use crate::gaussian_blur::GaussianBlur;
use crate::kernel_source::KERNEL_SOURCE;
use crate::lab_conversion::LabConversion;
use crate::load_module::load_module;
use crate::plane_reduction::PlaneReduction;
use crate::srgb_linearization::SrgbLinearization;
use crate::ssim_map::SsimMap;
use crate::triangle_downsample::TriangleDownsample;
use crate::upload_plane::upload_plane;
use crate::virtual_arch::virtual_arch;

const MINIMUM_DOWNSAMPLEABLE_SIDE: usize = 8;

pub struct CudaBackend {
    stream: Arc<CudaStream>,
    linearization: SrgbLinearization,
    lab: LabConversion,
    blur: GaussianBlur,
    downsample: TriangleDownsample,
    product: ElementwiseProduct,
    reduction: PlaneReduction,
    ssim_map: SsimMap,
}

impl CudaBackend {
    pub fn new() -> Result<Self, CudaError> {
        create_context(0).and_then(Self::from_context)
    }

    fn from_arch(context: Arc<CudaContext>, arch: &'static str) -> Result<Self, CudaError> {
        compile_ptx(KERNEL_SOURCE, arch)
            .and_then(|ptx| load_module(&context, ptx))
            .and_then(|module| Self::from_module(context, module))
    }

    fn from_capability(
        context: Arc<CudaContext>,
        capability: DeviceCapability,
    ) -> Result<Self, CudaError> {
        let DeviceCapability { major, minor } = capability;

        virtual_arch(major, minor).and_then(|arch| Self::from_arch(context, arch))
    }

    fn from_context(context: Arc<CudaContext>) -> Result<Self, CudaError> {
        DeviceCapability::query(&context)
            .and_then(|capability| Self::from_capability(context, capability))
    }

    fn from_module(context: Arc<CudaContext>, module: Arc<CudaModule>) -> Result<Self, CudaError> {
        let stream = context.default_stream();

        SrgbLinearization::new(&module, stream.clone()).and_then(|linearization| {
            LabConversion::new(&module, stream.clone()).and_then(|lab| {
                GaussianBlur::new(&module, stream.clone()).and_then(|blur| {
                    TriangleDownsample::new(&module, stream.clone()).and_then(|downsample| {
                        ElementwiseProduct::new(&module, stream.clone()).and_then(|product| {
                            PlaneReduction::new(&module, stream.clone()).and_then(|reduction| {
                                SsimMap::new(&module, stream.clone()).map(|ssim_map| Self {
                                    linearization,
                                    lab,
                                    blur,
                                    downsample,
                                    product,
                                    reduction,
                                    ssim_map,
                                    stream,
                                })
                            })
                        })
                    })
                })
            })
        })
    }

    fn build_pyramid(
        &self,
        full_resolution: Vec<CudaSlice<f32>>,
        width: usize,
        height: usize,
        scale_count: usize,
        to_lab: impl Fn(&[CudaSlice<f32>], usize, usize) -> Result<Vec<CudaSlice<f32>>, CudaError>,
    ) -> Result<CudaPrepared, CudaError> {
        let mut dimensions = Vec::with_capacity(scale_count);
        let mut current_width = width;
        let mut current_height = height;

        loop {
            dimensions.push(ScaleDimensions {
                width: current_width,
                height: current_height,
            });

            let room_for_another_level = dimensions.len() < scale_count;
            let downsampleable = current_width >= MINIMUM_DOWNSAMPLEABLE_SIDE
                && current_height >= MINIMUM_DOWNSAMPLEABLE_SIDE;

            if room_for_another_level && downsampleable {
                current_width /= 2;
                current_height /= 2;
            } else {
                break;
            }
        }

        let built_levels = dimensions.iter().take(dimensions.len() - 1).try_fold(
            vec![full_resolution],
            |mut levels,
             ScaleDimensions {
                 width: level_width,
                 height: level_height,
             }| {
                levels[levels.len() - 1]
                    .iter()
                    .map(|plane| {
                        self.downsample
                            .downsample(plane, *level_width, *level_height)
                    })
                    .collect::<Result<Vec<CudaSlice<f32>>, CudaError>>()
                    .map(|smaller| {
                        levels.push(smaller);
                        levels
                    })
            },
        );

        built_levels.and_then(|levels| {
            levels
                .into_iter()
                .zip(dimensions)
                .map(|(planes, ScaleDimensions { width, height })| {
                    to_lab(&planes, width, height)
                        .and_then(|lab| self.prepare_scale(lab, width, height))
                })
                .collect::<Result<Vec<_>, CudaError>>()
                .map(CudaPrepared::new)
        })
    }

    fn prepare_scale(
        &self,
        lab_planes: Vec<CudaSlice<f32>>,
        width: usize,
        height: usize,
    ) -> Result<CudaScale, CudaError> {
        lab_planes
            .into_iter()
            .enumerate()
            .map(|(index, plane)| {
                CudaPreparedChannel::prepare(
                    &self.blur,
                    &self.product,
                    plane,
                    index > 0,
                    width,
                    height,
                )
            })
            .collect::<Result<Vec<CudaPreparedChannel>, CudaError>>()
            .map(|channels| {
                let mut values = Vec::with_capacity(channels.len());
                let mut mu = Vec::with_capacity(channels.len());
                let mut squared_blur = Vec::with_capacity(channels.len());

                for CudaPreparedChannel {
                    mu: channel_mu,
                    squared_blur: channel_squared_blur,
                    value,
                } in channels
                {
                    values.push(value);
                    mu.push(channel_mu);
                    squared_blur.push(channel_squared_blur);
                }

                CudaScale::new(width, height, values, mu, squared_blur)
            })
    }

    fn prepared_gray<Component>(
        &self,
        srgb: &[Component],
        width: usize,
        height: usize,
        scale_count: usize,
    ) -> Result<CudaPrepared, CudaError>
    where
        Component: Copy + Send + Sync,
        f32: FromStimulus<Component>,
    {
        let normalized: Vec<f32> = srgb
            .iter()
            .map(|&value| f32::from_stimulus(value))
            .collect();
        let uploaded = upload_plane(&self.stream, &normalized)?;
        let linear = self.linearization.gray(&uploaded, width * height)?;

        self.build_pyramid(
            linear,
            width,
            height,
            scale_count,
            |planes, plane_width, plane_height| self.lab.gray(planes, plane_width * plane_height),
        )
    }

    fn prepared_rgb<Component>(
        &self,
        srgb: &[RGB<Component>],
        width: usize,
        height: usize,
        scale_count: usize,
    ) -> Result<CudaPrepared, CudaError>
    where
        Component: Copy + Send + Sync,
        f32: FromStimulus<Component>,
    {
        let mut interleaved = Vec::with_capacity(srgb.len() * 3);

        for pixel in srgb {
            interleaved.push(f32::from_stimulus(pixel.r));
            interleaved.push(f32::from_stimulus(pixel.g));
            interleaved.push(f32::from_stimulus(pixel.b));
        }

        let uploaded = upload_plane(&self.stream, &interleaved)?;
        let linear = self.linearization.rgb(&uploaded, width * height)?;

        self.build_pyramid(
            linear,
            width,
            height,
            scale_count,
            |planes, plane_width, plane_height| self.lab.rgb(planes, plane_width * plane_height),
        )
    }

    fn prepared_rgba<Component>(
        &self,
        srgb: &[RGBA<Component>],
        width: usize,
        height: usize,
        scale_count: usize,
    ) -> Result<CudaPrepared, CudaError>
    where
        Component: Copy + Send + Sync,
        f32: FromStimulus<Component>,
    {
        let mut interleaved = Vec::with_capacity(srgb.len() * 4);

        for pixel in srgb {
            interleaved.push(f32::from_stimulus(pixel.r));
            interleaved.push(f32::from_stimulus(pixel.g));
            interleaved.push(f32::from_stimulus(pixel.b));
            interleaved.push(f32::from_stimulus(pixel.a));
        }

        let uploaded = upload_plane(&self.stream, &interleaved)?;
        let linear = self.linearization.rgba(&uploaded, width * height)?;

        self.build_pyramid(
            linear,
            width,
            height,
            scale_count,
            |planes, plane_width, plane_height| self.lab.rgba(planes, plane_width, plane_height),
        )
    }

    fn scale_deviation(
        &self,
        reference: &CudaScale,
        distorted: &CudaScale,
        adjusted_mean_exponent: f64,
    ) -> Result<f64, CudaError> {
        let len = reference.width() * reference.height();
        let map = self
            .ssim_map
            .compute(&self.blur, &self.product, reference, distorted)?;

        self.score(&map, len, adjusted_mean_exponent)
    }

    fn scale_deviation_with_map(
        &self,
        reference: &CudaScale,
        distorted: &CudaScale,
        adjusted_mean_exponent: f64,
    ) -> Result<ScaleScoreWithMap, CudaError> {
        let width = reference.width();
        let height = reference.height();

        self.ssim_map
            .compute(&self.blur, &self.product, reference, distorted)
            .and_then(|map| {
                self.score(&map, width * height, adjusted_mean_exponent)
                    .and_then(|deviation| {
                        download_plane(&self.stream, &map).map(|pixels| ScaleScoreWithMap {
                            deviation,
                            map: MapPlane {
                                width,
                                height,
                                pixels,
                            },
                        })
                    })
            })
    }

    fn score(
        &self,
        map: &CudaSlice<f32>,
        len: usize,
        adjusted_mean_exponent: f64,
    ) -> Result<f64, CudaError> {
        self.reduction.sum(map, len).and_then(|total| {
            let adjusted_mean = (total / len as f64).max(0.0).powf(adjusted_mean_exponent);

            self.reduction
                .absolute_deviation_sum(map, adjusted_mean, len)
                .map(|deviation| deviation / len as f64)
        })
    }
}

impl Backend for CudaBackend {
    type Prepared = CudaPrepared;

    fn compare_scale(
        &self,
        reference: &Self::Prepared,
        distorted: &Self::Prepared,
        scale_index: usize,
        adjusted_mean_exponent: f64,
    ) -> Result<ScaleScore, SsimError> {
        let reference_scale = &reference.scales()[scale_index];
        let distorted_scale = &distorted.scales()[scale_index];
        let width = reference_scale.width();
        let height = reference_scale.height();
        let deviation = self
            .scale_deviation(reference_scale, distorted_scale, adjusted_mean_exponent)
            .map_err(|error| SsimError::ScaleComparison {
                width,
                height,
                reason: error.to_string(),
            })?;

        Ok(ScaleScore { deviation })
    }

    fn compare_scale_with_map(
        &self,
        reference: &Self::Prepared,
        distorted: &Self::Prepared,
        scale_index: usize,
        adjusted_mean_exponent: f64,
    ) -> Result<ScaleScoreWithMap, SsimError> {
        let reference_scale = &reference.scales()[scale_index];
        let distorted_scale = &distorted.scales()[scale_index];
        let width = reference_scale.width();
        let height = reference_scale.height();

        self.scale_deviation_with_map(reference_scale, distorted_scale, adjusted_mean_exponent)
            .map_err(|error| SsimError::ScaleComparison {
                width,
                height,
                reason: error.to_string(),
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
        self.prepared_gray(srgb, width, height, scale_count)
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
        self.prepared_rgb(srgb, width, height, scale_count)
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
        self.prepared_rgba(srgb, width, height, scale_count)
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

    use super::CudaBackend;
    use crate::cuda_prepared::CudaPrepared;
    use crate::cuda_scale::CudaScale;

    const IMPOSSIBLE_SIDE: usize = usize::MAX / 8;

    fn impossibly_large_prepared() -> CudaPrepared {
        CudaPrepared::new(vec![CudaScale::new(
            IMPOSSIBLE_SIDE,
            1,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )])
    }

    #[test]
    fn scale_dimensions_reports_every_prepared_scale() {
        let backend = CudaBackend::new().unwrap();
        let prepared = backend.prepare_gray(&[10u8, 20, 30, 40], 2, 2, 1).unwrap();

        let dimensions = backend.scale_dimensions(&prepared);

        assert_eq!(dimensions.len(), 1);
        assert_eq!(dimensions[0].width, 2);
        assert_eq!(dimensions[0].height, 2);
    }

    #[test]
    fn preparing_an_impossibly_large_gray_image_is_an_image_preparation_error() {
        let backend = CudaBackend::new().unwrap();

        assert!(
            backend
                .prepare_gray(&[10u8], IMPOSSIBLE_SIDE, 1, 1)
                .is_err()
        );
    }

    #[test]
    fn preparing_an_impossibly_large_rgb_image_is_an_image_preparation_error() {
        let backend = CudaBackend::new().unwrap();

        assert!(
            backend
                .prepare_rgb(&[RGB::<u8>::new(10, 20, 30)], IMPOSSIBLE_SIDE, 1, 1)
                .is_err()
        );
    }

    #[test]
    fn preparing_an_impossibly_large_rgba_image_is_an_image_preparation_error() {
        let backend = CudaBackend::new().unwrap();

        assert!(
            backend
                .prepare_rgba(&[RGBA::<u8>::new(10, 20, 30, 40)], IMPOSSIBLE_SIDE, 1, 1)
                .is_err()
        );
    }

    #[test]
    fn comparing_impossibly_large_scales_is_a_scale_comparison_error() {
        let backend = CudaBackend::new().unwrap();
        let reference = impossibly_large_prepared();
        let distorted = impossibly_large_prepared();

        assert!(
            backend
                .compare_scale(&reference, &distorted, 0, 1.0)
                .is_err()
        );
    }

    #[test]
    fn comparing_impossibly_large_scales_with_map_is_a_scale_comparison_error() {
        let backend = CudaBackend::new().unwrap();
        let reference = impossibly_large_prepared();
        let distorted = impossibly_large_prepared();

        assert!(
            backend
                .compare_scale_with_map(&reference, &distorted, 0, 1.0)
                .is_err()
        );
    }

    fn poison(backend: &CudaBackend) {
        let recorded = cudarc::driver::CudaContext::new(usize::MAX).unwrap_err();

        backend.stream.context().record_err::<()>(Err(recorded));
    }

    #[test]
    fn a_backend_without_a_visible_device_is_an_error() {
        unsafe {
            std::env::set_var("CUDA_VISIBLE_DEVICES", "");
        }

        assert!(CudaBackend::new().is_err());
    }

    #[test]
    fn preparing_gray_after_a_recorded_error_is_an_error() {
        let backend = CudaBackend::new().unwrap();
        poison(&backend);

        assert!(backend.prepare_gray(&[10u8, 20, 30, 40], 2, 2, 1).is_err());
    }

    #[test]
    fn preparing_rgb_after_a_recorded_error_is_an_error() {
        let backend = CudaBackend::new().unwrap();
        poison(&backend);

        assert!(
            backend
                .prepare_rgb(&[RGB::<u8>::new(10, 20, 30); 4], 2, 2, 1)
                .is_err()
        );
    }

    #[test]
    fn preparing_rgba_after_a_recorded_error_is_an_error() {
        let backend = CudaBackend::new().unwrap();
        poison(&backend);

        assert!(
            backend
                .prepare_rgba(&[RGBA::<u8>::new(10, 20, 30, 40); 4], 2, 2, 1)
                .is_err()
        );
    }

    #[test]
    fn comparing_after_a_recorded_error_is_an_error() {
        let backend = CudaBackend::new().unwrap();
        let reference = backend.prepare_gray(&[10u8, 20, 30, 40], 2, 2, 1).unwrap();
        let distorted = backend.prepare_gray(&[40u8, 30, 20, 10], 2, 2, 1).unwrap();
        poison(&backend);

        assert!(
            backend
                .compare_scale(&reference, &distorted, 0, 1.0)
                .is_err()
        );
    }

    #[test]
    fn comparing_with_map_after_a_recorded_error_is_an_error() {
        let backend = CudaBackend::new().unwrap();
        let reference = backend.prepare_gray(&[10u8, 20, 30, 40], 2, 2, 1).unwrap();
        let distorted = backend.prepare_gray(&[40u8, 30, 20, 10], 2, 2, 1).unwrap();
        poison(&backend);

        assert!(
            backend
                .compare_scale_with_map(&reference, &distorted, 0, 1.0)
                .is_err()
        );
    }
}
