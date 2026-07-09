use std::sync::Arc;

use cudarc::driver::CudaContext;

use crate::cuda_error::CudaError;

pub(crate) fn create_context(ordinal: usize) -> Result<Arc<CudaContext>, CudaError> {
    CudaContext::new(ordinal).map_err(|source| CudaError::DeviceCreation { source })
}

#[cfg(test)]
mod tests {
    use super::create_context;

    #[test]
    fn a_nonexistent_device_ordinal_is_a_device_creation_error() {
        assert!(create_context(usize::MAX).is_err());
    }
}
