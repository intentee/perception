use crate::prepared_scale::PreparedScale;

pub struct CpuPrepared {
    scales: Vec<PreparedScale>,
}

impl CpuPrepared {
    pub(crate) fn new(scales: Vec<PreparedScale>) -> Self {
        Self { scales }
    }

    pub(crate) fn scales(&self) -> &[PreparedScale] {
        &self.scales
    }
}
