use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::CudaModule;
use cudarc::nvrtc::Ptx;

use crate::cuda_error::CudaError;

pub(crate) fn load_module(
    context: &Arc<CudaContext>,
    ptx: Ptx,
) -> Result<Arc<CudaModule>, CudaError> {
    context
        .load_module(ptx)
        .map_err(|source| CudaError::ModuleLoad { source })
}

#[cfg(test)]
mod tests {
    use cudarc::nvrtc::Ptx;

    use super::load_module;
    use crate::create_context::create_context;

    #[test]
    fn invalid_ptx_is_a_module_load_error() {
        let context = create_context(0).unwrap();

        assert!(load_module(&context, Ptx::from_src("this is not valid ptx")).is_err());
    }
}
