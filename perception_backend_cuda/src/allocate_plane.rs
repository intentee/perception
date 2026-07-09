use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DeviceRepr;
use cudarc::driver::ValidAsZeroBits;

use crate::cuda_error::CudaError;

pub(crate) fn allocate_plane<Element>(
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<CudaSlice<Element>, CudaError>
where
    Element: DeviceRepr + ValidAsZeroBits,
{
    stream
        .alloc_zeros::<Element>(len)
        .map_err(|source| CudaError::DeviceAllocation { len, source })
}

#[cfg(test)]
mod tests {
    use super::allocate_plane;
    use crate::create_context::create_context;

    #[test]
    fn an_impossibly_large_plane_is_a_device_allocation_error() {
        let context = create_context(0).unwrap();
        let stream = context.default_stream();

        assert!(allocate_plane::<f32>(&stream, usize::MAX / 8).is_err());
    }
}
