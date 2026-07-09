use crate::cuda_error::CudaError;
use crate::select_virtual_arch::select_virtual_arch;

pub(crate) fn virtual_arch(major: i32, minor: i32) -> Result<&'static str, CudaError> {
    select_virtual_arch(major, minor)
        .ok_or(CudaError::UnsupportedComputeCapability { major, minor })
}

#[cfg(test)]
mod tests {
    use super::virtual_arch;

    #[test]
    fn a_supported_capability_resolves_to_a_virtual_arch() {
        assert_eq!(virtual_arch(7, 5).unwrap(), "compute_75");
    }

    #[test]
    fn an_unsupported_capability_is_an_error() {
        assert!(virtual_arch(5, 0).is_err());
    }
}
