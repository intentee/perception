use crate::gaussian_blur::GaussianBlur;
use crate::plane::Plane;
use crate::plane_error::PlaneError;

pub(crate) struct PreparedChannel {
    values: Plane,
    mu: Vec<f32>,
    squared_blur: Vec<f32>,
}

impl PreparedChannel {
    pub(crate) fn prepare(
        plane: Plane,
        is_chroma: bool,
        blur: &GaussianBlur,
    ) -> Result<Self, PlaneError> {
        let (values, mu) = if is_chroma {
            let values = blur.blur(&plane)?;
            let mu = blur.blur_consistent(&values).into_pixels();

            (values, mu)
        } else {
            let mu = blur.blur(&plane)?.into_pixels();

            (plane, mu)
        };
        let squared_blur = blur.blur_of_product_consistent(&values, &values);

        Ok(Self {
            values,
            mu,
            squared_blur,
        })
    }

    pub(crate) fn mu(&self) -> &[f32] {
        &self.mu
    }

    pub(crate) fn squared_blur(&self) -> &[f32] {
        &self.squared_blur
    }

    pub(crate) fn values(&self) -> &Plane {
        &self.values
    }
}

#[cfg(test)]
mod tests {
    use super::PreparedChannel;
    use crate::gaussian_blur::GaussianBlur;
    use crate::plane::Plane;

    const SIDE: usize = 8;
    const INCONSISTENT_PIXELS: usize = 3;

    #[test]
    fn a_chroma_channel_whose_plane_contradicts_its_dimensions_is_an_error() {
        let blur = GaussianBlur::new();
        let inconsistent = Plane::new(SIDE, SIDE, vec![0.0f32; INCONSISTENT_PIXELS]);

        assert!(PreparedChannel::prepare(inconsistent, true, &blur).is_err());
    }

    #[test]
    fn a_luminance_channel_whose_plane_contradicts_its_dimensions_is_an_error() {
        let blur = GaussianBlur::new();
        let inconsistent = Plane::new(SIDE, SIDE, vec![0.0f32; INCONSISTENT_PIXELS]);

        assert!(PreparedChannel::prepare(inconsistent, false, &blur).is_err());
    }
}
