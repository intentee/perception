use std::path::Path;

use image::ColorType;
use rgb::FromSlice;

use ssim_backend::Backend;
use ssim_metric::Ssim;
use ssim_metric::SsimImage;

use crate::similarity_error::SimilarityError;

fn is_sixteen_bit(color: ColorType) -> bool {
    matches!(
        color,
        ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16
    )
}

pub(crate) enum DecodedImage {
    Rgb {
        width: usize,
        height: usize,
        raw: Vec<u8>,
    },
    Rgb16 {
        width: usize,
        height: usize,
        raw: Vec<u16>,
    },
    Rgba {
        width: usize,
        height: usize,
        raw: Vec<u8>,
    },
    Rgba16 {
        width: usize,
        height: usize,
        raw: Vec<u16>,
    },
}

impl DecodedImage {
    pub(crate) fn decode(path: &Path) -> Result<Self, SimilarityError> {
        let decoded = image::open(path).map_err(|source| SimilarityError::Decode {
            path: path.to_path_buf(),
            source,
        })?;
        let width = decoded.width() as usize;
        let height = decoded.height() as usize;
        let color = decoded.color();

        Ok(if is_sixteen_bit(color) {
            if color.has_alpha() {
                Self::Rgba16 {
                    width,
                    height,
                    raw: decoded.into_rgba16().into_raw(),
                }
            } else {
                Self::Rgb16 {
                    width,
                    height,
                    raw: decoded.into_rgb16().into_raw(),
                }
            }
        } else if color.has_alpha() {
            Self::Rgba {
                width,
                height,
                raw: decoded.into_rgba8().into_raw(),
            }
        } else {
            Self::Rgb {
                width,
                height,
                raw: decoded.into_rgb8().into_raw(),
            }
        })
    }

    pub(crate) fn to_ssim_image<Strategy>(
        &self,
        context: &Ssim<Strategy>,
    ) -> Result<SsimImage<Strategy>, SimilarityError>
    where
        Strategy: Backend,
    {
        Ok(match self {
            Self::Rgb { width, height, raw } => {
                context.create_image_rgb(raw.as_rgb(), *width, *height)?
            }
            Self::Rgb16 { width, height, raw } => {
                context.create_image_rgb(raw.as_rgb(), *width, *height)?
            }
            Self::Rgba { width, height, raw } => {
                context.create_image_rgba(raw.as_rgba(), *width, *height)?
            }
            Self::Rgba16 { width, height, raw } => {
                context.create_image_rgba(raw.as_rgba(), *width, *height)?
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use image::GrayImage;
    use image::ImageBuffer;
    use image::Luma;
    use image::Rgb;
    use image::RgbImage;
    use image::Rgba;
    use image::RgbaImage;

    use ssim_metric::Ssim;
    use ssim_test::Scratch;

    use super::DecodedImage;

    fn assert_decodes_to_eight_by_eight(path: &Path) {
        let image = DecodedImage::decode(path)
            .unwrap()
            .to_ssim_image(&Ssim::new())
            .unwrap();

        assert_eq!(image.width(), 8);
        assert_eq!(image.height(), 8);
    }

    #[test]
    fn opaque_color_image_decodes_as_rgb() {
        let scratch = Scratch::new("rgb");
        let path = scratch.path("rgb.png");
        RgbImage::from_pixel(8, 8, Rgb([10, 120, 240]))
            .save(&path)
            .unwrap();

        assert_decodes_to_eight_by_eight(&path);
    }

    #[test]
    fn grayscale_image_decodes_through_the_rgb_path() {
        let scratch = Scratch::new("gray");
        let path = scratch.path("gray.png");
        GrayImage::from_pixel(8, 8, Luma([100]))
            .save(&path)
            .unwrap();

        assert_decodes_to_eight_by_eight(&path);
    }

    #[test]
    fn image_with_alpha_decodes_as_rgba() {
        let scratch = Scratch::new("rgba");
        let path = scratch.path("rgba.png");
        RgbaImage::from_pixel(8, 8, Rgba([10, 120, 240, 128]))
            .save(&path)
            .unwrap();

        assert_decodes_to_eight_by_eight(&path);
    }

    #[test]
    fn sixteen_bit_color_image_decodes_as_rgb16() {
        let scratch = Scratch::new("rgb16");
        let path = scratch.path("rgb16.png");
        ImageBuffer::<Rgb<u16>, Vec<u16>>::from_pixel(8, 8, Rgb([1000, 30000, 60000]))
            .save(&path)
            .unwrap();

        assert_decodes_to_eight_by_eight(&path);
    }

    #[test]
    fn sixteen_bit_image_with_alpha_decodes_as_rgba16() {
        let scratch = Scratch::new("rgba16");
        let path = scratch.path("rgba16.png");
        ImageBuffer::<Rgba<u16>, Vec<u16>>::from_pixel(8, 8, Rgba([1000, 30000, 60000, 20000]))
            .save(&path)
            .unwrap();

        assert_decodes_to_eight_by_eight(&path);
    }

    #[test]
    fn missing_file_is_an_error() {
        let scratch = Scratch::new("missing");
        assert!(DecodedImage::decode(&scratch.path("does_not_exist.png")).is_err());
    }

    #[test]
    fn an_rgb_buffer_that_contradicts_its_dimensions_is_an_error() {
        let inconsistent = DecodedImage::Rgb {
            width: 2,
            height: 2,
            raw: vec![0u8; 3],
        };

        assert!(inconsistent.to_ssim_image(&Ssim::new()).is_err());
    }

    #[test]
    fn an_rgb16_buffer_that_contradicts_its_dimensions_is_an_error() {
        let inconsistent = DecodedImage::Rgb16 {
            width: 2,
            height: 2,
            raw: vec![0u16; 3],
        };

        assert!(inconsistent.to_ssim_image(&Ssim::new()).is_err());
    }

    #[test]
    fn an_rgba_buffer_that_contradicts_its_dimensions_is_an_error() {
        let inconsistent = DecodedImage::Rgba {
            width: 2,
            height: 2,
            raw: vec![0u8; 4],
        };

        assert!(inconsistent.to_ssim_image(&Ssim::new()).is_err());
    }

    #[test]
    fn an_rgba16_buffer_that_contradicts_its_dimensions_is_an_error() {
        let inconsistent = DecodedImage::Rgba16 {
            width: 2,
            height: 2,
            raw: vec![0u16; 4],
        };

        assert!(inconsistent.to_ssim_image(&Ssim::new()).is_err());
    }
}
