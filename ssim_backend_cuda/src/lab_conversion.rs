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

pub(crate) struct LabConversion {
    stream: Arc<CudaStream>,
    gray: CudaFunction,
    rgb: CudaFunction,
    rgba: CudaFunction,
}

impl LabConversion {
    pub(crate) fn new(
        module: &Arc<CudaModule>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaError> {
        Ok(Self {
            stream,
            gray: load_kernel(module, "linear_gray_to_lab")?,
            rgb: load_kernel(module, "linear_rgb_to_lab")?,
            rgba: load_kernel(module, "linear_rgba_to_lab")?,
        })
    }

    pub(crate) fn gray(
        &self,
        linear: &[CudaSlice<f32>],
        pixel_count: usize,
    ) -> Result<Vec<CudaSlice<f32>>, CudaError> {
        let mut lightness = allocate_plane(&self.stream, pixel_count)?;
        let count = pixel_count as i32;

        unsafe {
            self.stream
                .launch_builder(&self.gray)
                .arg(&linear[0])
                .arg(&mut lightness)
                .arg(&count)
                .launch(LaunchConfig::for_num_elems(pixel_count as u32))
        }
        .ok();

        Ok(vec![lightness])
    }

    pub(crate) fn rgb(
        &self,
        linear: &[CudaSlice<f32>],
        pixel_count: usize,
    ) -> Result<Vec<CudaSlice<f32>>, CudaError> {
        let mut planes = allocate_planes::<f32>(&self.stream, &[pixel_count; 3])?;
        let count = pixel_count as i32;

        {
            let (lightness, rest) = planes.split_at_mut(1);
            let (green_red, blue_yellow) = rest.split_at_mut(1);

            unsafe {
                self.stream
                    .launch_builder(&self.rgb)
                    .arg(&linear[0])
                    .arg(&linear[1])
                    .arg(&linear[2])
                    .arg(&mut lightness[0])
                    .arg(&mut green_red[0])
                    .arg(&mut blue_yellow[0])
                    .arg(&count)
                    .launch(LaunchConfig::for_num_elems(pixel_count as u32))
            }
            .ok();
        }

        Ok(planes)
    }

    pub(crate) fn rgba(
        &self,
        linear: &[CudaSlice<f32>],
        width: usize,
        height: usize,
    ) -> Result<Vec<CudaSlice<f32>>, CudaError> {
        let pixel_count = width * height;
        let mut planes = allocate_planes::<f32>(&self.stream, &[pixel_count; 3])?;
        let width_arg = width as i32;
        let height_arg = height as i32;

        {
            let (lightness, rest) = planes.split_at_mut(1);
            let (green_red, blue_yellow) = rest.split_at_mut(1);

            unsafe {
                self.stream
                    .launch_builder(&self.rgba)
                    .arg(&linear[0])
                    .arg(&linear[1])
                    .arg(&linear[2])
                    .arg(&linear[3])
                    .arg(&mut lightness[0])
                    .arg(&mut green_red[0])
                    .arg(&mut blue_yellow[0])
                    .arg(&width_arg)
                    .arg(&height_arg)
                    .launch(LaunchConfig::for_num_elems(pixel_count as u32))
            }
            .ok();
        }

        Ok(planes)
    }
}

#[cfg(test)]
mod tests {
    use super::LabConversion;
    use crate::allocate_plane::allocate_plane;
    use crate::compile_ptx::compile_ptx;
    use crate::create_context::create_context;
    use crate::kernel_source::KERNEL_SOURCE;
    use crate::load_module::load_module;

    const IMPOSSIBLE_PIXELS: usize = usize::MAX / 8;
    const WITHOUT_GRAY: &str = "extern \"C\" __global__ void linear_rgb_to_lab() {}\nextern \"C\" __global__ void linear_rgba_to_lab() {}";
    const WITHOUT_RGB: &str = "extern \"C\" __global__ void linear_gray_to_lab() {}\nextern \"C\" __global__ void linear_rgba_to_lab() {}";
    const WITHOUT_RGBA: &str = "extern \"C\" __global__ void linear_gray_to_lab() {}\nextern \"C\" __global__ void linear_rgb_to_lab() {}";

    fn conversion() -> LabConversion {
        let context = create_context(0).unwrap();
        let module =
            load_module(&context, compile_ptx(KERNEL_SOURCE, "compute_75").unwrap()).unwrap();

        LabConversion::new(&module, context.default_stream()).unwrap()
    }

    fn linear_planes(
        conversion: &LabConversion,
        count: usize,
    ) -> Vec<cudarc::driver::CudaSlice<f32>> {
        (0..count)
            .map(|_| allocate_plane::<f32>(&conversion.stream, 4).unwrap())
            .collect()
    }

    #[test]
    fn a_module_without_the_gray_kernel_is_an_error() {
        let context = create_context(0).unwrap();
        let module =
            load_module(&context, compile_ptx(WITHOUT_GRAY, "compute_75").unwrap()).unwrap();

        assert!(LabConversion::new(&module, context.default_stream()).is_err());
    }

    #[test]
    fn a_module_without_the_rgb_kernel_is_an_error() {
        let context = create_context(0).unwrap();
        let module =
            load_module(&context, compile_ptx(WITHOUT_RGB, "compute_75").unwrap()).unwrap();

        assert!(LabConversion::new(&module, context.default_stream()).is_err());
    }

    #[test]
    fn a_module_without_the_rgba_kernel_is_an_error() {
        let context = create_context(0).unwrap();
        let module =
            load_module(&context, compile_ptx(WITHOUT_RGBA, "compute_75").unwrap()).unwrap();

        assert!(LabConversion::new(&module, context.default_stream()).is_err());
    }

    #[test]
    fn converting_an_impossibly_large_gray_plane_is_an_error() {
        let conversion = conversion();
        let linear = linear_planes(&conversion, 1);

        assert!(conversion.gray(&linear, IMPOSSIBLE_PIXELS).is_err());
    }

    #[test]
    fn converting_an_impossibly_large_rgb_plane_is_an_error() {
        let conversion = conversion();
        let linear = linear_planes(&conversion, 3);

        assert!(conversion.rgb(&linear, IMPOSSIBLE_PIXELS).is_err());
    }

    #[test]
    fn converting_an_impossibly_large_rgba_plane_is_an_error() {
        let conversion = conversion();
        let linear = linear_planes(&conversion, 4);

        assert!(conversion.rgba(&linear, IMPOSSIBLE_PIXELS, 1).is_err());
    }
}
