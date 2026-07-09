use std::sync::Arc;

use cudarc::driver::CudaContext;

use crate::cuda_error::CudaError;

pub(crate) struct DeviceCapability {
    pub(crate) major: i32,
    pub(crate) minor: i32,
}

impl DeviceCapability {
    pub(crate) fn query(context: &Arc<CudaContext>) -> Result<Self, CudaError> {
        let (major, minor) = context
            .compute_capability()
            .map_err(|source| CudaError::ComputeCapabilityQuery { source })?;

        Ok(Self { major, minor })
    }
}

#[cfg(test)]
mod tests {
    use cudarc::driver::CudaContext;

    use super::DeviceCapability;
    use crate::create_context::create_context;

    #[test]
    fn a_context_with_a_recorded_error_is_a_compute_capability_query_error() {
        let context = create_context(0).unwrap();
        let recorded = CudaContext::new(usize::MAX).unwrap_err();

        context.record_err::<()>(Err(recorded));

        assert!(DeviceCapability::query(&context).is_err());
    }
}
