use crate::cuda_scale::CudaScale;

pub struct CudaPrepared {
    scales: Vec<CudaScale>,
}

impl CudaPrepared {
    pub(crate) fn new(scales: Vec<CudaScale>) -> Self {
        Self { scales }
    }

    pub(crate) fn scales(&self) -> &[CudaScale] {
        &self.scales
    }
}
