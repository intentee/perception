use rgb::RGB;

use ssim_backend::Backend;
use ssim_metric::Ssim;
use ssim_metric_bench::format_header::format_header;
use ssim_metric_bench::format_row::format_row;
use ssim_metric_bench::measure::measure;
use ssim_metric_bench::synthetic_pixels::synthetic_pixels;

#[must_use]
pub fn compare<BackendStrategy>(engine: &Ssim<BackendStrategy>, sizes: &[usize]) -> String
where
    BackendStrategy: Backend,
{
    let mut lines = vec!["compare".to_string(), format_header()];

    for &side in sizes {
        let reference_pixels = synthetic_pixels::<RGB<u8>>(side, side);
        let mut distorted_pixels = reference_pixels.clone();
        distorted_pixels.rotate_left(side);

        let reference = engine
            .create_image_rgb(&reference_pixels, side, side)
            .unwrap();
        let distorted = engine
            .create_image_rgb(&distorted_pixels, side, side)
            .unwrap();

        lines.push(format_row(
            "ssim",
            side,
            &measure(|| engine.compare(&reference, &distorted).unwrap()),
        ));
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use ssim_metric::Ssim;

    use super::compare;

    #[test]
    fn reports_a_titled_header_and_one_row_per_size() {
        let engine = Ssim::new();

        let report = compare(&engine, &[1, 2]);
        let lines: Vec<&str> = report.lines().collect();

        assert_eq!(lines[0], "compare");
        assert_eq!(lines.len(), 4);
        assert!(lines[2].starts_with("ssim"));
    }
}
