use palette::IntoColor;
use palette::Lab;
use palette::LinSrgb;
use rgb::RGB;

const LIGHTNESS_RANGE: f32 = 100.0;
const CHROMA_RANGE: f32 = 256.0;
const CHROMA_OFFSET: f32 = 0.5;

pub(crate) fn linear_to_lab_color(rgb: RGB<f32>) -> [f32; 3] {
    let lab: Lab = LinSrgb::new(rgb.r, rgb.g, rgb.b).into_color();

    [
        lab.l / LIGHTNESS_RANGE,
        lab.a / CHROMA_RANGE + CHROMA_OFFSET,
        lab.b / CHROMA_RANGE + CHROMA_OFFSET,
    ]
}

#[cfg(test)]
mod tests {
    use rgb::RGB;

    use super::linear_to_lab_color;

    #[test]
    fn white_maps_to_full_neutral_lightness() {
        let [lightness, green_red, blue_yellow] = linear_to_lab_color(RGB::new(1.0, 1.0, 1.0));

        assert!((lightness - 1.0).abs() < 1e-4, "lightness {lightness}");
        assert!((green_red - 0.5).abs() < 1e-4, "green_red {green_red}");
        assert!(
            (blue_yellow - 0.5).abs() < 1e-4,
            "blue_yellow {blue_yellow}"
        );
    }

    #[test]
    fn a_red_pixel_shifts_the_green_red_axis_up() {
        let neutral = linear_to_lab_color(RGB::new(0.5, 0.5, 0.5));
        let reddish = linear_to_lab_color(RGB::new(0.9, 0.2, 0.2));

        assert!(reddish[1] > neutral[1]);
    }
}
