use palette::IntoColor;
use palette::Lab;
use palette::LinSrgb;

const LIGHTNESS_RANGE: f32 = 100.0;

pub(crate) fn linear_to_lab_gray(luminance: f32) -> f32 {
    let lab: Lab = LinSrgb::new(luminance, luminance, luminance).into_color();

    lab.l / LIGHTNESS_RANGE
}

#[cfg(test)]
mod tests {
    use super::linear_to_lab_gray;

    #[test]
    fn full_luminance_maps_to_one() {
        assert!((linear_to_lab_gray(1.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn black_maps_to_zero() {
        assert!(linear_to_lab_gray(0.0).abs() < 1e-6);
    }
}
