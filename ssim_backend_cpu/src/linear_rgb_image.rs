use image::Rgb;
use palette::Srgb;
use palette::stimulus::FromStimulus;
use rgb::RGB;

use crate::downsample::downsample;
use crate::linear_pyramid_level::LinearPyramidLevel;
use crate::linear_to_lab_color::linear_to_lab_color;
use crate::map_indices::map_indices;
use crate::map_indices_three::map_indices_three;
use crate::plane::Plane;

pub(crate) struct LinearRgbImage {
    width: usize,
    height: usize,
    pixels: Vec<RGB<f32>>,
}

impl LinearRgbImage {
    pub(crate) fn from_srgb<Component>(
        bitmap: &[RGB<Component>],
        width: usize,
        height: usize,
    ) -> Self
    where
        Component: Copy + Send + Sync,
        f32: FromStimulus<Component>,
    {
        let pixels = map_indices(bitmap.len(), |index| {
            let pixel = bitmap[index];
            let linear = Srgb::new(pixel.r, pixel.g, pixel.b)
                .into_format::<f32>()
                .into_linear();

            RGB {
                r: linear.red,
                g: linear.green,
                b: linear.blue,
            }
        });

        Self {
            width,
            height,
            pixels,
        }
    }
}

impl LinearPyramidLevel for LinearRgbImage {
    fn downsampled(&self) -> Option<Self> {
        let components = self
            .pixels
            .iter()
            .flat_map(|pixel| [pixel.r, pixel.g, pixel.b])
            .collect();

        downsample::<Rgb<f32>>(components, self.width, self.height).map(|resized| Self {
            width: self.width / 2,
            height: self.height / 2,
            pixels: resized
                .chunks_exact(3)
                .map(|channels| RGB::new(channels[0], channels[1], channels[2]))
                .collect(),
        })
    }

    fn to_lab_planes(&self) -> Vec<Plane> {
        let [lightness, green_red, blue_yellow] = map_indices_three(self.pixels.len(), |index| {
            linear_to_lab_color(self.pixels[index])
        });

        vec![
            Plane::new(self.width, self.height, lightness),
            Plane::new(self.width, self.height, green_red),
            Plane::new(self.width, self.height, blue_yellow),
        ]
    }
}
