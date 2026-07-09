use cudarc::nvrtc::CompileOptions;
use cudarc::nvrtc::Ptx;
use cudarc::nvrtc::compile_ptx_with_opts;

use crate::cuda_error::CudaError;

pub(crate) fn compile_ptx(kernel_source: &str, arch: &'static str) -> Result<Ptx, CudaError> {
    compile_ptx_with_opts(
        kernel_source,
        CompileOptions {
            arch: Some(arch),
            ..Default::default()
        },
    )
    .map_err(|source| CudaError::KernelCompilation { source })
}

#[cfg(test)]
mod tests {
    use super::compile_ptx;

    #[test]
    fn invalid_kernel_source_is_a_kernel_compilation_error() {
        assert!(compile_ptx("this is not valid cuda", "compute_75").is_err());
    }
}
