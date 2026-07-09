use perception_backend::Backend;
use perception_metric::Ssim;
use perception_metric_bench::format_header::format_header;
use perception_metric_bench::format_row::format_row;
use perception_metric_bench::from_coordinate_hash::FromCoordinateHash;
use perception_metric_bench::measure::measure;
use perception_metric_bench::synthetic_pixels::synthetic_pixels;

use crate::create_image_function::CreateImageFunction;

fn create_image_row<BackendStrategy, Pixel>(
    engine: &Ssim<BackendStrategy>,
    side: usize,
    label: &str,
    create: CreateImageFunction<BackendStrategy, Pixel>,
) -> String
where
    BackendStrategy: Backend,
    Pixel: FromCoordinateHash,
{
    let pixels = synthetic_pixels::<Pixel>(side, side);

    format_row(
        label,
        side,
        &measure(|| create(engine, &pixels, side, side).unwrap()),
    )
}

#[must_use]
pub fn create_image<BackendStrategy>(engine: &Ssim<BackendStrategy>, sizes: &[usize]) -> String
where
    BackendStrategy: Backend,
{
    let mut lines = vec!["create_image".to_string(), format_header()];

    for &side in sizes {
        lines.push(create_image_row(
            engine,
            side,
            "gray",
            Ssim::<BackendStrategy>::create_image_gray::<u8>,
        ));
        lines.push(create_image_row(
            engine,
            side,
            "rgb8",
            Ssim::<BackendStrategy>::create_image_rgb::<u8>,
        ));
        lines.push(create_image_row(
            engine,
            side,
            "rgb16",
            Ssim::<BackendStrategy>::create_image_rgb::<u16>,
        ));
        lines.push(create_image_row(
            engine,
            side,
            "rgba8",
            Ssim::<BackendStrategy>::create_image_rgba::<u8>,
        ));
        lines.push(create_image_row(
            engine,
            side,
            "rgba16",
            Ssim::<BackendStrategy>::create_image_rgba::<u16>,
        ));
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use perception_metric::Ssim;

    use super::create_image;

    #[test]
    fn reports_a_titled_header_and_one_row_per_pixel_kind() {
        let engine = Ssim::new();

        let report = create_image(&engine, &[1]);
        let lines: Vec<&str> = report.lines().collect();

        assert_eq!(lines[0], "create_image");
        assert_eq!(lines.len(), 7);
        assert!(lines.iter().any(|line| line.starts_with("rgba16")));
    }
}
