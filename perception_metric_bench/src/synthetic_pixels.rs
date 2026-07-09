use crate::from_coordinate_hash::FromCoordinateHash;

fn hash_coordinate(x: usize, y: usize) -> u64 {
    let mut state = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(
        (y as u64)
            .rotate_left(32)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9),
    );
    state = (state ^ (state >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

    state ^ (state >> 31)
}

#[must_use]
pub fn synthetic_pixels<Pixel>(width: usize, height: usize) -> Vec<Pixel>
where
    Pixel: FromCoordinateHash,
{
    (0..width * height)
        .map(|index| Pixel::from_coordinate_hash(hash_coordinate(index % width, index / width)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::synthetic_pixels;

    #[test]
    fn produces_the_requested_pixel_count() {
        let pixels: Vec<u8> = synthetic_pixels(4, 3);

        assert_eq!(pixels.len(), 12);
    }

    #[test]
    fn generation_is_deterministic_for_the_same_dimensions() {
        let first: Vec<u8> = synthetic_pixels(5, 5);
        let second: Vec<u8> = synthetic_pixels(5, 5);

        assert_eq!(first, second);
    }

    #[test]
    fn distinct_coordinates_do_not_collapse_to_one_value() {
        let pixels: Vec<u8> = synthetic_pixels(4, 4);

        assert!(pixels.iter().any(|&pixel| pixel != pixels[0]));
    }
}
