use std::path::Path;

use perception_metric::Ssim;

use crate::compare_paths::compare_paths;
use crate::comparison::Comparison;
use crate::similarity_error::SimilarityError;

const SAVED_MAP_SCALES: u8 = 1;

pub enum Engine {
    #[cfg(feature = "cpu")]
    Cpu(Box<Ssim<perception_backend_cpu::CpuBackend>>),
    #[cfg(feature = "cuda")]
    Cuda(Box<Ssim<perception_backend_cuda::CudaBackend>>),
}

impl Engine {
    #[cfg(feature = "cpu")]
    #[must_use]
    pub fn cpu() -> Self {
        Self::Cpu(Box::new(
            Ssim::new().with_saved_map_scales(SAVED_MAP_SCALES),
        ))
    }

    #[cfg(feature = "cuda")]
    pub fn cuda() -> Result<Self, perception_backend_cuda::CudaError> {
        let backend = perception_backend_cuda::CudaBackend::new()?;

        Ok(Self::Cuda(Box::new(
            Ssim::with_backend(backend).with_saved_map_scales(SAVED_MAP_SCALES),
        )))
    }

    pub fn compare(
        &self,
        original: &Path,
        distorted: &Path,
    ) -> Result<Comparison, SimilarityError> {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => compare_paths(engine, original, distorted),
            #[cfg(feature = "cuda")]
            Self::Cuda(engine) => compare_paths(engine, original, distorted),
        }
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use perception_test::Scratch;
    use perception_test::write_test_image;

    use super::Engine;

    #[test]
    fn the_cpu_engine_scores_identical_images_as_fully_similar() {
        let scratch = Scratch::new("engine_cpu_identical");
        let original = scratch.path("original.png");
        write_test_image(&original, 12, 0);

        let similarity = Engine::cpu()
            .compare(&original, &original)
            .unwrap()
            .similarity();

        assert!((similarity - 1.0).abs() < 1e-9);
    }

    #[test]
    fn the_cpu_engine_scores_different_images_below_one() {
        let scratch = Scratch::new("engine_cpu_different");
        let original = scratch.path("original.png");
        let distorted = scratch.path("distorted.png");
        write_test_image(&original, 12, 0);
        write_test_image(&distorted, 12, 40);

        let similarity = Engine::cpu()
            .compare(&original, &distorted)
            .unwrap()
            .similarity();

        assert!(similarity < 1.0);
    }

    #[test]
    fn the_cpu_engine_reports_a_missing_file_as_an_error() {
        let scratch = Scratch::new("engine_cpu_missing");
        let distorted = scratch.path("distorted.png");
        write_test_image(&distorted, 12, 0);

        let result = Engine::cpu().compare(&scratch.path("missing.png"), &distorted);

        assert!(result.is_err());
    }
}
