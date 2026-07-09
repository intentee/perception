use image::Rgba;
use palette::Srgba;
use palette::stimulus::FromStimulus;
use rgb::RGB;
use rgb::RGBA;

use crate::downsample::downsample;
use crate::linear_pyramid_level::LinearPyramidLevel;
use crate::linear_to_lab_color::linear_to_lab_color;
use crate::map_indices::map_indices;
use crate::map_indices_three::map_indices_three;
use crate::plane::Plane;

const DITHER_OFFSET: usize = 11;
const DITHER_RED_BIT: usize = 16;
const DITHER_GREEN_BIT: usize = 8;
const DITHER_BLUE_BIT: usize = 32;

fn composite_over_dither(pixel: RGBA<f32>, dither: usize) -> RGB<f32> {
    let mut red = pixel.r;
    let mut green = pixel.g;
    let mut blue = pixel.b;

    if pixel.a < 1.0 {
        let uncovered = 1.0 - pixel.a;

        if dither & DITHER_RED_BIT != 0 {
            red += uncovered;
        }
        if dither & DITHER_GREEN_BIT != 0 {
            green += uncovered;
        }
        if dither & DITHER_BLUE_BIT != 0 {
            blue += uncovered;
        }
    }

    RGB::new(red, green, blue)
}

pub(crate) struct LinearRgbaImage {
    width: usize,
    height: usize,
    pixels: Vec<RGBA<f32>>,
}

impl LinearRgbaImage {
    pub(crate) fn from_srgb<Component>(
        bitmap: &[RGBA<Component>],
        width: usize,
        height: usize,
    ) -> Self
    where
        Component: Copy + Send + Sync,
        f32: FromStimulus<Component>,
    {
        let pixels = map_indices(bitmap.len(), |index| {
            let pixel = bitmap[index];
            let srgba = Srgba::new(pixel.r, pixel.g, pixel.b, pixel.a).into_format::<f32, f32>();
            let linear = srgba.color.into_linear();
            let alpha = srgba.alpha;

            RGBA {
                r: linear.red * alpha,
                g: linear.green * alpha,
                b: linear.blue * alpha,
                a: alpha,
            }
        });

        Self {
            width,
            height,
            pixels,
        }
    }
}

impl LinearPyramidLevel for LinearRgbaImage {
    fn downsampled(&self) -> Option<Self> {
        let components = self
            .pixels
            .iter()
            .flat_map(|pixel| [pixel.r, pixel.g, pixel.b, pixel.a])
            .collect();

        downsample::<Rgba<f32>>(components, self.width, self.height).map(|resized| Self {
            width: self.width / 2,
            height: self.height / 2,
            pixels: resized
                .chunks_exact(4)
                .map(|channels| RGBA::new(channels[0], channels[1], channels[2], channels[3]))
                .collect(),
        })
    }

    fn to_lab_planes(&self) -> Vec<Plane> {
        let width = self.width;
        let [lightness, green_red, blue_yellow] = map_indices_three(self.pixels.len(), |index| {
            let row = index / width;
            let column = index % width;
            let dither = (column + DITHER_OFFSET) ^ (row + DITHER_OFFSET);

            linear_to_lab_color(composite_over_dither(self.pixels[index], dither))
        });

        vec![
            Plane::new(self.width, self.height, lightness),
            Plane::new(self.width, self.height, green_red),
            Plane::new(self.width, self.height, blue_yellow),
        ]
    }
}
