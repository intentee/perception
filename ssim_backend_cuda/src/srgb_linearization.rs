use std::sync::Arc;

use cudarc::driver::CudaFunction;
use cudarc::driver::CudaModule;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;

use crate::allocate_plane::allocate_plane;
use crate::allocate_planes::allocate_planes;
use crate::cuda_error::CudaError;
use crate::load_kernel::load_kernel;

pub(crate) struct SrgbLinearization {
    stream: Arc<CudaStream>,
    gray: CudaFunction,
    rgb: CudaFunction,
    rgba: CudaFunction,
}

impl SrgbLinearization {
    pub(crate) fn new(
        module: &Arc<CudaModule>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaError> {
        Ok(Self {
            stream,
            gray: load_kernel(module, "srgb_gray_to_linear")?,
            rgb: load_kernel(module, "srgb_rgb_to_linear")?,
            rgba: load_kernel(module, "srgba_to_linear_premult")?,
        })
    }

    pub(crate) fn gray(
        &self,
        srgb: &CudaSlice<f32>,
        pixel_count: usize,
    ) -> Result<Vec<CudaSlice<f32>>, CudaError> {
        let mut linear = allocate_plane(&self.stream, pixel_count)?;
        let count = pixel_count as i32;

        unsafe {
            self.stream
                .launch_builder(&self.gray)
                .arg(srgb)
                .arg(&mut linear)
                .arg(&count)
                .launch(LaunchConfig::for_num_elems(pixel_count as u32))
        }
        .ok();

        Ok(vec![linear])
    }

    pub(crate) fn rgb(
        &self,
        interleaved: &CudaSlice<f32>,
        pixel_count: usize,
    ) -> Result<Vec<CudaSlice<f32>>, CudaError> {
        let mut planes = allocate_planes::<f32>(&self.stream, &[pixel_count; 3])?;
        let count = pixel_count as i32;

        {
            let (red, rest) = planes.split_at_mut(1);
            let (green, blue) = rest.split_at_mut(1);

            unsafe {
                self.stream
                    .launch_builder(&self.rgb)
                    .arg(interleaved)
                    .arg(&mut red[0])
                    .arg(&mut green[0])
                    .arg(&mut blue[0])
                    .arg(&count)
                    .launch(LaunchConfig::for_num_elems(pixel_count as u32))
            }
            .ok();
        }

        Ok(planes)
    }

    pub(crate) fn rgba(
        &self,
        interleaved: &CudaSlice<f32>,
        pixel_count: usize,
    ) -> Result<Vec<CudaSlice<f32>>, CudaError> {
        let mut planes = allocate_planes::<f32>(&self.stream, &[pixel_count; 4])?;
        let count = pixel_count as i32;

        {
            let (red, rest) = planes.split_at_mut(1);
            let (green, rest) = rest.split_at_mut(1);
            let (blue, alpha) = rest.split_at_mut(1);

            unsafe {
                self.stream
                    .launch_builder(&self.rgba)
                    .arg(interleaved)
                    .arg(&mut red[0])
                    .arg(&mut green[0])
                    .arg(&mut blue[0])
                    .arg(&mut alpha[0])
                    .arg(&count)
                    .launch(LaunchConfig::for_num_elems(pixel_count as u32))
            }
            .ok();
        }

        Ok(planes)
    }
}

#[cfg(test)]
mod tests {
    use super::SrgbLinearization;
    use crate::allocate_plane::allocate_plane;
    use crate::compile_ptx::compile_ptx;
    use crate::create_context::create_context;
    use crate::kernel_source::KERNEL_SOURCE;
    use crate::load_module::load_module;

    const IMPOSSIBLE_PIXELS: usize = usize::MAX / 8;
    const WITHOUT_GRAY: &str = "extern \"C\" __global__ void srgb_rgb_to_linear() {}\nextern \"C\" __global__ void srgba_to_linear_premult() {}";
    const WITHOUT_RGB: &str = "extern \"C\" __global__ void srgb_gray_to_linear() {}\nextern \"C\" __global__ void srgba_to_linear_premult() {}";
    const WITHOUT_RGBA: &str = "extern \"C\" __global__ void srgb_gray_to_linear() {}\nextern \"C\" __global__ void srgb_rgb_to_linear() {}";

    fn linearization() -> SrgbLinearization {
        let context = create_context(0).unwrap();
        let module =
            load_module(&context, compile_ptx(KERNEL_SOURCE, "compute_75").unwrap()).unwrap();

        SrgbLinearization::new(&module, context.default_stream()).unwrap()
    }

    #[test]
    fn a_module_without_the_gray_kernel_is_an_error() {
        let context = create_context(0).unwrap();
        let module =
            load_module(&context, compile_ptx(WITHOUT_GRAY, "compute_75").unwrap()).unwrap();

        assert!(SrgbLinearization::new(&module, context.default_stream()).is_err());
    }

    #[test]
    fn a_module_without_the_rgb_kernel_is_an_error() {
        let context = create_context(0).unwrap();
        let module =
            load_module(&context, compile_ptx(WITHOUT_RGB, "compute_75").unwrap()).unwrap();

        assert!(SrgbLinearization::new(&module, context.default_stream()).is_err());
    }

    #[test]
    fn a_module_without_the_rgba_kernel_is_an_error() {
        let context = create_context(0).unwrap();
        let module =
            load_module(&context, compile_ptx(WITHOUT_RGBA, "compute_75").unwrap()).unwrap();

        assert!(SrgbLinearization::new(&module, context.default_stream()).is_err());
    }

    #[test]
    fn linearizing_an_impossibly_large_gray_plane_is_an_error() {
        let linearization = linearization();
        let srgb = allocate_plane::<f32>(&linearization.stream, 4).unwrap();

        assert!(linearization.gray(&srgb, IMPOSSIBLE_PIXELS).is_err());
    }

    #[test]
    fn linearizing_an_impossibly_large_rgb_plane_is_an_error() {
        let linearization = linearization();
        let srgb = allocate_plane::<f32>(&linearization.stream, 4).unwrap();

        assert!(linearization.rgb(&srgb, IMPOSSIBLE_PIXELS).is_err());
    }

    #[test]
    fn linearizing_an_impossibly_large_rgba_plane_is_an_error() {
        let linearization = linearization();
        let srgb = allocate_plane::<f32>(&linearization.stream, 4).unwrap();

        assert!(linearization.rgba(&srgb, IMPOSSIBLE_PIXELS).is_err());
    }
}
