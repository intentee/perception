use std::sync::Arc;

use cudarc::driver::CudaFunction;
use cudarc::driver::CudaModule;

use crate::cuda_error::CudaError;

pub(crate) fn load_kernel(module: &Arc<CudaModule>, name: &str) -> Result<CudaFunction, CudaError> {
    module
        .load_function(name)
        .map_err(|source| CudaError::KernelNotFound {
            name: name.to_string(),
            source,
        })
}

#[cfg(test)]
mod tests {
    use super::load_kernel;
    use crate::compile_ptx::compile_ptx;
    use crate::create_context::create_context;
    use crate::load_module::load_module;

    const TINY_KERNEL: &str = "extern \"C\" __global__ void noop() {}";

    #[test]
    fn an_unknown_kernel_name_is_a_kernel_not_found_error() {
        let context = create_context(0).unwrap();
        let ptx = compile_ptx(TINY_KERNEL, "compute_75").unwrap();
        let module = load_module(&context, ptx).unwrap();

        assert!(load_kernel(&module, "does_not_exist").is_err());
    }
}
