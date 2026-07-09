use std::sync::Arc;

use cudarc::driver::CudaFunction;
use cudarc::driver::CudaModule;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;

use crate::allocate_planes::allocate_planes;
use crate::cuda_error::CudaError;
use crate::cuda_scale::CudaScale;
use crate::elementwise_product::ElementwiseProduct;
use crate::gaussian_blur::GaussianBlur;
use crate::load_kernel::load_kernel;

pub(crate) struct SsimMap {
    stream: Arc<CudaStream>,
    accumulate: CudaFunction,
    finalize: CudaFunction,
}

impl SsimMap {
    pub(crate) fn new(
        module: &Arc<CudaModule>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaError> {
        Ok(Self {
            stream,
            accumulate: load_kernel(module, "ssim_accumulate")?,
            finalize: load_kernel(module, "ssim_finalize")?,
        })
    }

    pub(crate) fn compute(
        &self,
        blur: &GaussianBlur,
        product: &ElementwiseProduct,
        reference: &CudaScale,
        distorted: &CudaScale,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let width = reference.width();
        let height = reference.height();
        let pixel_count = width * height;
        let count = pixel_count as i32;
        let channel_count = reference.channel_count() as i32;
        let config = LaunchConfig::for_num_elems(pixel_count as u32);
        let mut buffers = allocate_planes::<f32>(&self.stream, &[pixel_count; 7])?;

        let accumulated = (0..reference.channel_count()).try_for_each(|channel| {
            product
                .product(
                    &reference.values()[channel],
                    &distorted.values()[channel],
                    pixel_count,
                )
                .and_then(|cross| blur.blur(&cross, width, height))
                .map(|cross_blur| {
                    let (mu1_sq, rest) = buffers.split_at_mut(1);
                    let (mu2_sq, rest) = rest.split_at_mut(1);
                    let (mu1_mu2, rest) = rest.split_at_mut(1);
                    let (sigma1, rest) = rest.split_at_mut(1);
                    let (sigma2, rest) = rest.split_at_mut(1);
                    let (sigma12, _map) = rest.split_at_mut(1);

                    unsafe {
                        self.stream
                            .launch_builder(&self.accumulate)
                            .arg(&reference.mu()[channel])
                            .arg(&distorted.mu()[channel])
                            .arg(&reference.squared_blur()[channel])
                            .arg(&distorted.squared_blur()[channel])
                            .arg(&cross_blur)
                            .arg(&mut mu1_sq[0])
                            .arg(&mut mu2_sq[0])
                            .arg(&mut mu1_mu2[0])
                            .arg(&mut sigma1[0])
                            .arg(&mut sigma2[0])
                            .arg(&mut sigma12[0])
                            .arg(&count)
                            .launch(config)
                    }
                    .ok();
                })
        });

        accumulated.map(|()| {
            {
                let (accumulators, map) = buffers.split_at_mut(6);

                unsafe {
                    self.stream
                        .launch_builder(&self.finalize)
                        .arg(&accumulators[0])
                        .arg(&accumulators[1])
                        .arg(&accumulators[2])
                        .arg(&accumulators[3])
                        .arg(&accumulators[4])
                        .arg(&accumulators[5])
                        .arg(&channel_count)
                        .arg(&count)
                        .arg(&mut map[0])
                        .launch(config)
                }
                .ok();
            }

            buffers.swap_remove(6)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::SsimMap;
    use crate::compile_ptx::compile_ptx;
    use crate::create_context::create_context;
    use crate::load_module::load_module;

    const WITHOUT_ACCUMULATE: &str = "extern \"C\" __global__ void ssim_finalize() {}";
    const WITHOUT_FINALIZE: &str = "extern \"C\" __global__ void ssim_accumulate() {}";

    #[test]
    fn a_module_without_the_accumulate_kernel_is_an_error() {
        let context = create_context(0).unwrap();
        let module = load_module(
            &context,
            compile_ptx(WITHOUT_ACCUMULATE, "compute_75").unwrap(),
        )
        .unwrap();

        assert!(SsimMap::new(&module, context.default_stream()).is_err());
    }

    #[test]
    fn a_module_without_the_finalize_kernel_is_an_error() {
        let context = create_context(0).unwrap();
        let module = load_module(
            &context,
            compile_ptx(WITHOUT_FINALIZE, "compute_75").unwrap(),
        )
        .unwrap();

        assert!(SsimMap::new(&module, context.default_stream()).is_err());
    }
}
