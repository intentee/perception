use std::sync::Arc;

use cudarc::driver::CudaFunction;
use cudarc::driver::CudaModule;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;

use crate::allocate_plane::allocate_plane;
use crate::cuda_error::CudaError;
use crate::load_kernel::load_kernel;

pub(crate) struct ElementwiseProduct {
    stream: Arc<CudaStream>,
    multiply: CudaFunction,
}

impl ElementwiseProduct {
    pub(crate) fn new(
        module: &Arc<CudaModule>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaError> {
        Ok(Self {
            stream,
            multiply: load_kernel(module, "multiply")?,
        })
    }

    pub(crate) fn product(
        &self,
        first: &CudaSlice<f32>,
        second: &CudaSlice<f32>,
        len: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let mut product = allocate_plane(&self.stream, len)?;
        let count = len as i32;

        unsafe {
            self.stream
                .launch_builder(&self.multiply)
                .arg(first)
                .arg(second)
                .arg(&mut product)
                .arg(&count)
                .launch(LaunchConfig::for_num_elems(len as u32))
        }
        .ok();

        Ok(product)
    }
}

#[cfg(test)]
mod tests {
    use super::ElementwiseProduct;
    use crate::allocate_plane::allocate_plane;
    use crate::compile_ptx::compile_ptx;
    use crate::create_context::create_context;
    use crate::load_module::load_module;

    const IMPOSSIBLE_LEN: usize = usize::MAX / 8;
    const UNRELATED_KERNEL: &str = "extern \"C\" __global__ void unrelated() {}";

    #[test]
    fn a_module_without_the_multiply_kernel_is_an_error() {
        let context = create_context(0).unwrap();
        let ptx = compile_ptx(UNRELATED_KERNEL, "compute_75").unwrap();
        let module = load_module(&context, ptx).unwrap();

        assert!(ElementwiseProduct::new(&module, context.default_stream()).is_err());
    }

    #[test]
    fn an_impossibly_large_product_is_an_error() {
        let context = create_context(0).unwrap();
        let ptx = compile_ptx(crate::kernel_source::KERNEL_SOURCE, "compute_75").unwrap();
        let module = load_module(&context, ptx).unwrap();
        let stream = context.default_stream();
        let product = ElementwiseProduct::new(&module, stream.clone()).unwrap();
        let first = allocate_plane::<f32>(&stream, 4).unwrap();
        let second = allocate_plane::<f32>(&stream, 4).unwrap();

        assert!(product.product(&first, &second, IMPOSSIBLE_LEN).is_err());
    }
}
