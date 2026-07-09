use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;

use crate::cuda_error::CudaError;

pub(crate) fn upload_plane(
    stream: &Arc<CudaStream>,
    host: &[f32],
) -> Result<CudaSlice<f32>, CudaError> {
    stream
        .clone_htod(host)
        .map_err(|source| CudaError::HostToDeviceCopy {
            len: host.len(),
            source,
        })
}

#[cfg(test)]
mod tests {
    use cudarc::driver::CudaContext;

    use super::upload_plane;
    use crate::create_context::create_context;

    #[test]
    fn copying_to_a_context_with_a_recorded_error_is_a_host_to_device_copy_error() {
        let context = create_context(0).unwrap();
        let stream = context.default_stream();
        let recorded = CudaContext::new(usize::MAX).unwrap_err();

        context.record_err::<()>(Err(recorded));

        assert!(upload_plane(&stream, &[0.0f32; 4]).is_err());
    }
}
