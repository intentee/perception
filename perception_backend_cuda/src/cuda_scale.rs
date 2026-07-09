use cudarc::driver::CudaSlice;

pub(crate) struct CudaScale {
    width: usize,
    height: usize,
    values: Vec<CudaSlice<f32>>,
    mu: Vec<CudaSlice<f32>>,
    squared_blur: Vec<CudaSlice<f32>>,
}

impl CudaScale {
    pub(crate) fn new(
        width: usize,
        height: usize,
        values: Vec<CudaSlice<f32>>,
        mu: Vec<CudaSlice<f32>>,
        squared_blur: Vec<CudaSlice<f32>>,
    ) -> Self {
        Self {
            width,
            height,
            values,
            mu,
            squared_blur,
        }
    }

    pub(crate) fn channel_count(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn height(&self) -> usize {
        self.height
    }

    pub(crate) fn mu(&self) -> &[CudaSlice<f32>] {
        &self.mu
    }

    pub(crate) fn squared_blur(&self) -> &[CudaSlice<f32>] {
        &self.squared_blur
    }

    pub(crate) fn values(&self) -> &[CudaSlice<f32>] {
        &self.values
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }
}
