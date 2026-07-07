use libblur::BlurImage;
use libblur::BlurImageMut;
use libblur::EdgeMode;
use libblur::EdgeMode2D;
use libblur::FastBlurChannels;
use libblur::GaussianBlurParams;
use libblur::IeeeBinaryConvolutionMode;
use libblur::ThreadingPolicy;
use libblur::gaussian_blur_f32;

use crate::plane::Plane;
use crate::plane_error::PlaneError;

const BLUR_SIGMA: f64 = 1.5;

pub(crate) struct GaussianBlur {
    params: GaussianBlurParams,
}

impl GaussianBlur {
    pub(crate) fn new() -> Self {
        Self {
            params: GaussianBlurParams::new_from_sigma(BLUR_SIGMA),
        }
    }

    pub(crate) fn blur(&self, plane: &Plane) -> Result<Plane, PlaneError> {
        let pixels = self.convolve(plane.pixels(), plane.width(), plane.height())?;

        Ok(Plane::new(plane.width(), plane.height(), pixels))
    }

    pub(crate) fn blur_of_product(
        &self,
        first: &Plane,
        second: &Plane,
    ) -> Result<Vec<f32>, PlaneError> {
        let product: Vec<f32> = first
            .pixels()
            .iter()
            .zip(second.pixels())
            .map(|(left, right)| left * right)
            .collect();

        self.convolve(&product, first.width(), first.height())
    }

    fn convolve(
        &self,
        pixels: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, PlaneError> {
        let source =
            BlurImage::borrow(pixels, width as u32, height as u32, FastBlurChannels::Plane);
        let mut destination = vec![0.0f32; width * height];
        let mut target = BlurImageMut::borrow(
            &mut destination,
            width as u32,
            height as u32,
            FastBlurChannels::Plane,
        );

        gaussian_blur_f32(
            &source,
            &mut target,
            self.params,
            EdgeMode2D::new(EdgeMode::Clamp),
            ThreadingPolicy::Single,
            IeeeBinaryConvolutionMode::Normal,
        )
        .map_err(|error| PlaneError::GaussianBlur {
            width,
            height,
            reason: error.to_string(),
        })?;

        Ok(destination)
    }

    pub(crate) fn blur_consistent(&self, plane: &Plane) -> Plane {
        let pixels = self.convolve_consistent(plane.pixels(), plane.width(), plane.height());

        Plane::new(plane.width(), plane.height(), pixels)
    }

    pub(crate) fn blur_of_product_consistent(&self, first: &Plane, second: &Plane) -> Vec<f32> {
        let product: Vec<f32> = first
            .pixels()
            .iter()
            .zip(second.pixels())
            .map(|(left, right)| left * right)
            .collect();

        self.convolve_consistent(&product, first.width(), first.height())
    }

    fn convolve_consistent(&self, pixels: &[f32], width: usize, height: usize) -> Vec<f32> {
        let source =
            BlurImage::borrow(pixels, width as u32, height as u32, FastBlurChannels::Plane);
        let mut destination = vec![0.0f32; width * height];
        let mut target = BlurImageMut::borrow(
            &mut destination,
            width as u32,
            height as u32,
            FastBlurChannels::Plane,
        );

        gaussian_blur_f32(
            &source,
            &mut target,
            self.params,
            EdgeMode2D::new(EdgeMode::Clamp),
            ThreadingPolicy::Single,
            IeeeBinaryConvolutionMode::Normal,
        )
        .ok();

        destination
    }
}

#[cfg(test)]
mod tests {
    use super::GaussianBlur;
    use crate::plane::Plane;

    fn uniform_plane(width: usize, height: usize, value: f32) -> Plane {
        Plane::new(width, height, vec![value; width * height])
    }

    #[test]
    fn preserves_uniform_plane_at_all_sizes() {
        let blur = GaussianBlur::new();

        for size in 1..=6 {
            let blurred = blur.blur(&uniform_plane(size, size, 0.4)).unwrap();

            for &value in blurred.pixels() {
                assert!((value - 0.4).abs() < 1e-5, "size {size}: got {value}");
            }
        }
    }

    #[test]
    fn blur_of_product_preserves_uniform_at_all_sizes() {
        let blur = GaussianBlur::new();

        for size in 1..=6 {
            let plane = uniform_plane(size, size, 0.5);
            let product = blur.blur_of_product(&plane, &plane).unwrap();

            for &value in &product {
                assert!((value - 0.25).abs() < 1e-5, "size {size}: got {value}");
            }
        }
    }

    #[test]
    fn a_step_edge_spreads_while_flat_regions_are_preserved() {
        let blur = GaussianBlur::new();
        let width = 21;
        let height = 5;
        let pixels: Vec<f32> = (0..width * height)
            .map(|index| if index % width < width / 2 { 0.0 } else { 1.0 })
            .collect();

        let blurred = blur.blur(&Plane::new(width, height, pixels)).unwrap();
        let at = |row: usize, column: usize| blurred.pixels()[row * width + column];
        let far_dark = at(2, 0);
        let far_bright = at(2, width - 1);
        let boundary = at(2, width / 2);

        assert!(far_dark.abs() < 1e-4, "far dark side leaked: {far_dark}");
        assert!(
            (far_bright - 1.0).abs() < 1e-4,
            "far bright side leaked: {far_bright}"
        );
        assert!(
            boundary > 0.0 && boundary < 1.0,
            "the boundary must blend the two sides: {boundary}"
        );
    }

    #[test]
    fn a_plane_whose_buffer_contradicts_its_dimensions_is_an_error() {
        let blur = GaussianBlur::new();
        let inconsistent = Plane::new(8, 8, vec![0.0f32; 3]);

        assert!(blur.blur(&inconsistent).is_err());
    }
}
