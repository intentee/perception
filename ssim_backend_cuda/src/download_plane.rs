use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DeviceRepr;

use crate::cuda_error::CudaError;

pub(crate) fn download_plane<Element>(
    stream: &Arc<CudaStream>,
    plane: &CudaSlice<Element>,
) -> Result<Vec<Element>, CudaError>
where
    Element: DeviceRepr,
{
    stream
        .clone_dtoh(plane)
        .map_err(|source| CudaError::DeviceToHostCopy {
            len: plane.len(),
            source,
        })
}

#[cfg(test)]
mod tests {
    use cudarc::driver::CudaContext;

    use super::download_plane;
    use crate::allocate_plane::allocate_plane;
    use crate::create_context::create_context;

    #[test]
    fn copying_from_a_context_with_a_recorded_error_is_a_device_to_host_copy_error() {
        let context = create_context(0).unwrap();
        let stream = context.default_stream();
        let plane = allocate_plane::<f32>(&stream, 4).unwrap();
        let recorded = CudaContext::new(usize::MAX).unwrap_err();

        context.record_err::<()>(Err(recorded));

        assert!(download_plane(&stream, &plane).is_err());
    }
}
