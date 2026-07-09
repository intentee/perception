use std::path::Path;

pub fn write_test_image(path: &Path, side: u32, shift: u8) {
    let mut buffer = image::RgbImage::new(side, side);

    for (column, row, pixel) in buffer.enumerate_pixels_mut() {
        *pixel = image::Rgb([
            (column as u8).wrapping_mul(7).wrapping_add(shift),
            (row as u8).wrapping_mul(7),
            100,
        ]);
    }

    buffer.save(path).expect("failed to write a test image");
}
