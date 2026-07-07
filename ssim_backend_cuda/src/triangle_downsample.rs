use std::sync::Arc;

use cudarc::driver::CudaFunction;
use cudarc::driver::CudaModule;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;

use crate::allocate_planes::allocate_planes;
use crate::cuda_error::CudaError;
use crate::load_kernel::load_kernel;

pub(crate) struct TriangleDownsample {
    stream: Arc<CudaStream>,
    vertical: CudaFunction,
    horizontal: CudaFunction,
}

impl TriangleDownsample {
    pub(crate) fn new(
        module: &Arc<CudaModule>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaError> {
        let vertical = load_kernel(module, "downsample_vertical")?;
        let horizontal = load_kernel(module, "downsample_horizontal")?;

        Ok(Self {
            stream,
            vertical,
            horizontal,
        })
    }

    pub(crate) fn downsample(
        &self,
        source: &CudaSlice<f32>,
        width: usize,
        height: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let target_width = width / 2;
        let target_height = height / 2;
        let mut planes = allocate_planes::<f32>(
            &self.stream,
            &[width * target_height, target_width * target_height],
        )?;
        let width_arg = width as i32;
        let source_height = height as i32;
        let target_height_arg = target_height as i32;

        unsafe {
            self.stream
                .launch_builder(&self.vertical)
                .arg(source)
                .arg(&mut planes[0])
                .arg(&width_arg)
                .arg(&source_height)
                .arg(&target_height_arg)
                .launch(LaunchConfig::for_num_elems((width * target_height) as u32))
        }
        .ok();

        let source_width = width as i32;
        let target_width_arg = target_width as i32;

        {
            let (vertical, horizontal) = planes.split_at_mut(1);

            unsafe {
                self.stream
                    .launch_builder(&self.horizontal)
                    .arg(&vertical[0])
                    .arg(&mut horizontal[0])
                    .arg(&source_width)
                    .arg(&target_width_arg)
                    .arg(&target_height_arg)
                    .launch(LaunchConfig::for_num_elems(
                        (target_width * target_height) as u32,
                    ))
            }
            .ok();
        }

        Ok(planes.swap_remove(1))
    }
}

#[cfg(test)]
mod tests {
    use super::TriangleDownsample;
    use crate::allocate_plane::allocate_plane;
    use crate::compile_ptx::compile_ptx;
    use crate::create_context::create_context;
    use crate::kernel_source::KERNEL_SOURCE;
    use crate::load_module::load_module;

    const IMPOSSIBLE_SIDE: usize = usize::MAX / 8;
    const WITHOUT_VERTICAL: &str = "extern \"C\" __global__ void downsample_horizontal() {}";
    const WITHOUT_HORIZONTAL: &str = "extern \"C\" __global__ void downsample_vertical() {}";

    #[test]
    fn a_module_without_the_vertical_kernel_is_an_error() {
        let context = create_context(0).unwrap();
        let module = load_module(
            &context,
            compile_ptx(WITHOUT_VERTICAL, "compute_75").unwrap(),
        )
        .unwrap();

        assert!(TriangleDownsample::new(&module, context.default_stream()).is_err());
    }

    #[test]
    fn a_module_without_the_horizontal_kernel_is_an_error() {
        let context = create_context(0).unwrap();
        let module = load_module(
            &context,
            compile_ptx(WITHOUT_HORIZONTAL, "compute_75").unwrap(),
        )
        .unwrap();

        assert!(TriangleDownsample::new(&module, context.default_stream()).is_err());
    }

    #[test]
    fn downsampling_an_impossibly_large_plane_is_an_error() {
        let context = create_context(0).unwrap();
        let module =
            load_module(&context, compile_ptx(KERNEL_SOURCE, "compute_75").unwrap()).unwrap();
        let stream = context.default_stream();
        let downsample = TriangleDownsample::new(&module, stream.clone()).unwrap();
        let source = allocate_plane::<f32>(&stream, 4).unwrap();

        assert!(downsample.downsample(&source, IMPOSSIBLE_SIDE, 2).is_err());
    }
}
