use cudarc::driver::DriverError;
use cudarc::nvrtc::CompileError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("querying the CUDA device compute capability failed")]
    ComputeCapabilityQuery {
        #[source]
        source: DriverError,
    },
    #[error("allocating a {len}-element device plane failed")]
    DeviceAllocation {
        len: usize,
        #[source]
        source: DriverError,
    },
    #[error("creating the CUDA device context failed")]
    DeviceCreation {
        #[source]
        source: DriverError,
    },
    #[error("copying a {len}-element device plane back to the host failed")]
    DeviceToHostCopy {
        len: usize,
        #[source]
        source: DriverError,
    },
    #[error("copying a {len}-element host plane to the device failed")]
    HostToDeviceCopy {
        len: usize,
        #[source]
        source: DriverError,
    },
    #[error("compiling the ssim CUDA kernels failed")]
    KernelCompilation {
        #[source]
        source: CompileError,
    },
    #[error("failed to load CUDA kernel {name:?}")]
    KernelNotFound {
        name: String,
        #[source]
        source: DriverError,
    },
    #[error("loading the compiled ssim CUDA module failed")]
    ModuleLoad {
        #[source]
        source: DriverError,
    },
    #[error("the CUDA device compute capability {major}.{minor} is not supported")]
    UnsupportedComputeCapability { major: i32, minor: i32 },
}
