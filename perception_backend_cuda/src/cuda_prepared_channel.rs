use cudarc::driver::CudaSlice;

use crate::cuda_error::CudaError;
use crate::elementwise_product::ElementwiseProduct;
use crate::gaussian_blur::GaussianBlur;

pub(crate) struct CudaPreparedChannel {
    pub(crate) mu: CudaSlice<f32>,
    pub(crate) squared_blur: CudaSlice<f32>,
    pub(crate) value: CudaSlice<f32>,
}

impl CudaPreparedChannel {
    pub(crate) fn prepare(
        blur: &GaussianBlur,
        product: &ElementwiseProduct,
        plane: CudaSlice<f32>,
        is_chroma: bool,
        width: usize,
        height: usize,
    ) -> Result<Self, CudaError> {
        let pixel_count = width * height;
        let value_source = if is_chroma {
            blur.blur(&plane, width, height)
        } else {
            Ok(plane)
        };

        value_source.and_then(|value| {
            blur.blur(&value, width, height).and_then(|mu| {
                product
                    .product(&value, &value, pixel_count)
                    .and_then(|cross| {
                        blur.blur(&cross, width, height).map(|squared_blur| Self {
                            mu,
                            squared_blur,
                            value,
                        })
                    })
            })
        })
    }
}
