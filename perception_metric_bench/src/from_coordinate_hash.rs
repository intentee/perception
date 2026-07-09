use rgb::RGB;
use rgb::RGBA;

pub trait FromCoordinateHash {
    fn from_coordinate_hash(hash: u64) -> Self;
}

impl FromCoordinateHash for u8 {
    fn from_coordinate_hash(hash: u64) -> Self {
        hash as Self
    }
}

impl FromCoordinateHash for RGB<u8> {
    fn from_coordinate_hash(hash: u64) -> Self {
        Self::new(hash as u8, (hash >> 8) as u8, (hash >> 16) as u8)
    }
}

impl FromCoordinateHash for RGBA<u8> {
    fn from_coordinate_hash(hash: u64) -> Self {
        Self::new(
            hash as u8,
            (hash >> 8) as u8,
            (hash >> 16) as u8,
            (hash >> 24) as u8,
        )
    }
}

impl FromCoordinateHash for RGB<u16> {
    fn from_coordinate_hash(hash: u64) -> Self {
        Self::new(hash as u16, (hash >> 16) as u16, (hash >> 32) as u16)
    }
}

impl FromCoordinateHash for RGBA<u16> {
    fn from_coordinate_hash(hash: u64) -> Self {
        Self::new(
            hash as u16,
            (hash >> 16) as u16,
            (hash >> 32) as u16,
            (hash >> 48) as u16,
        )
    }
}

#[cfg(test)]
mod tests {
    use rgb::RGB;
    use rgb::RGBA;

    use super::FromCoordinateHash;

    const HASH: u64 = 0x1122_3344_5566_7788;

    #[test]
    fn u8_takes_the_low_byte() {
        assert_eq!(u8::from_coordinate_hash(HASH), 0x88);
    }

    #[test]
    fn rgb8_unpacks_three_low_bytes() {
        assert_eq!(
            RGB::<u8>::from_coordinate_hash(HASH),
            RGB::new(0x88, 0x77, 0x66)
        );
    }

    #[test]
    fn rgba8_unpacks_four_low_bytes() {
        assert_eq!(
            RGBA::<u8>::from_coordinate_hash(HASH),
            RGBA::new(0x88, 0x77, 0x66, 0x55)
        );
    }

    #[test]
    fn rgb16_unpacks_three_low_words() {
        assert_eq!(
            RGB::<u16>::from_coordinate_hash(HASH),
            RGB::new(0x7788, 0x5566, 0x3344)
        );
    }

    #[test]
    fn rgba16_unpacks_four_low_words() {
        assert_eq!(
            RGBA::<u16>::from_coordinate_hash(HASH),
            RGBA::new(0x7788, 0x5566, 0x3344, 0x1122)
        );
    }
}
