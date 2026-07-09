use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DeviceRepr;
use cudarc::driver::ValidAsZeroBits;

use crate::allocate_plane::allocate_plane;
use crate::cuda_error::CudaError;

pub(crate) fn allocate_planes<Element>(
    stream: &Arc<CudaStream>,
    lengths: &[usize],
) -> Result<Vec<CudaSlice<Element>>, CudaError>
where
    Element: DeviceRepr + ValidAsZeroBits,
{
    let mut planes = Vec::with_capacity(lengths.len());

    for &len in lengths {
        planes.push(allocate_plane(stream, len)?);
    }

    Ok(planes)
}

#[cfg(test)]
mod tests {
    use super::allocate_planes;
    use crate::create_context::create_context;

    const IMPOSSIBLE_LEN: usize = usize::MAX / 8;

    #[test]
    fn an_impossibly_large_plane_group_is_an_error() {
        let context = create_context(0).unwrap();
        let stream = context.default_stream();

        assert!(allocate_planes::<f32>(&stream, &[IMPOSSIBLE_LEN]).is_err());
    }
}
