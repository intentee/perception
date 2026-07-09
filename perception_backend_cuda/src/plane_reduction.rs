use std::sync::Arc;

use cudarc::driver::CudaFunction;
use cudarc::driver::CudaModule;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;

use crate::allocate_plane::allocate_plane;
use crate::cuda_error::CudaError;
use crate::download_plane::download_plane;
use crate::load_kernel::load_kernel;

const REDUCTION_BLOCK: usize = 256;

pub(crate) struct PlaneReduction {
    stream: Arc<CudaStream>,
    sum: CudaFunction,
    absolute_deviation: CudaFunction,
}

impl PlaneReduction {
    pub(crate) fn new(
        module: &Arc<CudaModule>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaError> {
        Ok(Self {
            stream,
            sum: load_kernel(module, "reduce_sum")?,
            absolute_deviation: load_kernel(module, "reduce_abs_deviation")?,
        })
    }

    pub(crate) fn absolute_deviation_sum(
        &self,
        data: &CudaSlice<f32>,
        adjusted_mean: f64,
        len: usize,
    ) -> Result<f64, CudaError> {
        let blocks = len.div_ceil(REDUCTION_BLOCK);
        let mut partials = allocate_plane::<f64>(&self.stream, blocks)?;
        let count = len as i32;

        unsafe {
            self.stream
                .launch_builder(&self.absolute_deviation)
                .arg(data)
                .arg(&adjusted_mean)
                .arg(&mut partials)
                .arg(&count)
                .launch(Self::config(blocks))
        }
        .ok();

        self.combine_partials(&partials)
    }

    pub(crate) fn sum(&self, data: &CudaSlice<f32>, len: usize) -> Result<f64, CudaError> {
        let blocks = len.div_ceil(REDUCTION_BLOCK);
        let mut partials = allocate_plane::<f64>(&self.stream, blocks)?;
        let count = len as i32;

        unsafe {
            self.stream
                .launch_builder(&self.sum)
                .arg(data)
                .arg(&mut partials)
                .arg(&count)
                .launch(Self::config(blocks))
        }
        .ok();

        self.combine_partials(&partials)
    }

    fn combine_partials(&self, partials: &CudaSlice<f64>) -> Result<f64, CudaError> {
        Ok(download_plane(&self.stream, partials)?.iter().sum())
    }

    fn config(blocks: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (REDUCTION_BLOCK as u32, 1, 1),
            shared_mem_bytes: (REDUCTION_BLOCK * size_of::<f64>()) as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use cudarc::driver::CudaContext;

    use super::PlaneReduction;
    use crate::allocate_plane::allocate_plane;
    use crate::compile_ptx::compile_ptx;
    use crate::create_context::create_context;
    use crate::kernel_source::KERNEL_SOURCE;
    use crate::load_module::load_module;

    const IMPOSSIBLE_LEN: usize = usize::MAX / 8;
    const WITHOUT_SUM: &str = "extern \"C\" __global__ void reduce_abs_deviation() {}";
    const WITHOUT_ABS_DEVIATION: &str = "extern \"C\" __global__ void reduce_sum() {}";

    #[test]
    fn a_module_without_the_sum_kernel_is_an_error() {
        let context = create_context(0).unwrap();
        let module =
            load_module(&context, compile_ptx(WITHOUT_SUM, "compute_75").unwrap()).unwrap();

        assert!(PlaneReduction::new(&module, context.default_stream()).is_err());
    }

    #[test]
    fn a_module_without_the_absolute_deviation_kernel_is_an_error() {
        let context = create_context(0).unwrap();
        let module = load_module(
            &context,
            compile_ptx(WITHOUT_ABS_DEVIATION, "compute_75").unwrap(),
        )
        .unwrap();

        assert!(PlaneReduction::new(&module, context.default_stream()).is_err());
    }

    #[test]
    fn summing_an_impossibly_large_plane_is_an_error() {
        let context = create_context(0).unwrap();
        let module =
            load_module(&context, compile_ptx(KERNEL_SOURCE, "compute_75").unwrap()).unwrap();
        let stream = context.default_stream();
        let reduction = PlaneReduction::new(&module, stream.clone()).unwrap();
        let data = allocate_plane::<f32>(&stream, 4).unwrap();

        assert!(reduction.sum(&data, IMPOSSIBLE_LEN).is_err());
    }

    #[test]
    fn absolute_deviation_of_an_impossibly_large_plane_is_an_error() {
        let context = create_context(0).unwrap();
        let module =
            load_module(&context, compile_ptx(KERNEL_SOURCE, "compute_75").unwrap()).unwrap();
        let stream = context.default_stream();
        let reduction = PlaneReduction::new(&module, stream.clone()).unwrap();
        let data = allocate_plane::<f32>(&stream, 4).unwrap();

        assert!(
            reduction
                .absolute_deviation_sum(&data, 0.5, IMPOSSIBLE_LEN)
                .is_err()
        );
    }

    #[test]
    fn combining_partials_after_a_recorded_error_is_an_error() {
        let context = create_context(0).unwrap();
        let module =
            load_module(&context, compile_ptx(KERNEL_SOURCE, "compute_75").unwrap()).unwrap();
        let stream = context.default_stream();
        let reduction = PlaneReduction::new(&module, stream.clone()).unwrap();
        let partials = allocate_plane::<f64>(&stream, 4).unwrap();
        let recorded = CudaContext::new(usize::MAX).unwrap_err();

        context.record_err::<()>(Err(recorded));

        assert!(reduction.combine_partials(&partials).is_err());
    }
}
