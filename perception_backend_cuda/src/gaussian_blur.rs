use std::sync::Arc;

use cudarc::driver::CudaFunction;
use cudarc::driver::CudaModule;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use libblur::gaussian_kernel_1d;

use crate::allocate_planes::allocate_planes;
use crate::cuda_error::CudaError;
use crate::load_kernel::load_kernel;
use crate::upload_plane::upload_plane;

const BLUR_SIGMA: f32 = 1.5;
const BLUR_KERNEL_SIZE: u32 = 7;

pub(crate) struct GaussianBlur {
    stream: Arc<CudaStream>,
    horizontal: CudaFunction,
    vertical: CudaFunction,
    taps: CudaSlice<f32>,
}

impl GaussianBlur {
    pub(crate) fn new(
        module: &Arc<CudaModule>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaError> {
        let horizontal = load_kernel(module, "blur_horizontal")?;
        let vertical = load_kernel(module, "blur_vertical")?;
        let taps = upload_plane(&stream, &gaussian_kernel_1d(BLUR_KERNEL_SIZE, BLUR_SIGMA))?;

        Ok(Self {
            stream,
            horizontal,
            vertical,
            taps,
        })
    }

    pub(crate) fn blur(
        &self,
        source: &CudaSlice<f32>,
        width: usize,
        height: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let pixel_count = width * height;
        let mut planes = allocate_planes::<f32>(&self.stream, &[pixel_count, pixel_count])?;
        let width_arg = width as i32;
        let height_arg = height as i32;
        let config = LaunchConfig::for_num_elems(pixel_count as u32);

        unsafe {
            self.stream
                .launch_builder(&self.horizontal)
                .arg(source)
                .arg(&mut planes[0])
                .arg(&self.taps)
                .arg(&width_arg)
                .arg(&height_arg)
                .launch(config)
        }
        .ok();

        {
            let (horizontal, blurred) = planes.split_at_mut(1);

            unsafe {
                self.stream
                    .launch_builder(&self.vertical)
                    .arg(&horizontal[0])
                    .arg(&mut blurred[0])
                    .arg(&self.taps)
                    .arg(&width_arg)
                    .arg(&height_arg)
                    .launch(config)
            }
            .ok();
        }

        Ok(planes.swap_remove(1))
    }
}

#[cfg(test)]
mod tests {
    use super::GaussianBlur;
    use crate::allocate_plane::allocate_plane;
    use crate::compile_ptx::compile_ptx;
    use crate::create_context::create_context;
    use crate::kernel_source::KERNEL_SOURCE;
    use crate::load_module::load_module;

    const IMPOSSIBLE_SIDE: usize = usize::MAX / 8;
    const WITHOUT_HORIZONTAL: &str = "extern \"C\" __global__ void blur_vertical() {}";
    const WITHOUT_VERTICAL: &str = "extern \"C\" __global__ void blur_horizontal() {}";

    #[test]
    fn a_module_without_the_horizontal_kernel_is_an_error() {
        let context = create_context(0).unwrap();
        let module = load_module(
            &context,
            compile_ptx(WITHOUT_HORIZONTAL, "compute_75").unwrap(),
        )
        .unwrap();

        assert!(GaussianBlur::new(&module, context.default_stream()).is_err());
    }

    #[test]
    fn a_module_without_the_vertical_kernel_is_an_error() {
        let context = create_context(0).unwrap();
        let module = load_module(
            &context,
            compile_ptx(WITHOUT_VERTICAL, "compute_75").unwrap(),
        )
        .unwrap();

        assert!(GaussianBlur::new(&module, context.default_stream()).is_err());
    }

    #[test]
    fn uploading_taps_after_a_recorded_error_is_an_error() {
        let context = create_context(0).unwrap();
        let module =
            load_module(&context, compile_ptx(KERNEL_SOURCE, "compute_75").unwrap()).unwrap();
        let stream = context.default_stream();
        let recorded = cudarc::driver::CudaContext::new(usize::MAX).unwrap_err();

        context.record_err::<()>(Err(recorded));

        assert!(GaussianBlur::new(&module, stream).is_err());
    }

    #[test]
    fn blurring_an_impossibly_large_plane_is_an_error() {
        let context = create_context(0).unwrap();
        let module =
            load_module(&context, compile_ptx(KERNEL_SOURCE, "compute_75").unwrap()).unwrap();
        let stream = context.default_stream();
        let blur = GaussianBlur::new(&module, stream.clone()).unwrap();
        let source = allocate_plane::<f32>(&stream, 4).unwrap();

        assert!(blur.blur(&source, IMPOSSIBLE_SIDE, 1).is_err());
    }
}
