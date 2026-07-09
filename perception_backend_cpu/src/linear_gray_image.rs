use image::Luma;
use palette::SrgbLuma;
use palette::stimulus::FromStimulus;

use crate::downsample::downsample;
use crate::linear_pyramid_level::LinearPyramidLevel;
use crate::linear_to_lab_gray::linear_to_lab_gray;
use crate::map_indices::map_indices;
use crate::plane::Plane;

pub(crate) struct LinearGrayImage {
    width: usize,
    height: usize,
    pixels: Vec<f32>,
}

impl LinearGrayImage {
    pub(crate) fn from_srgb<Component>(bitmap: &[Component], width: usize, height: usize) -> Self
    where
        Component: Copy + Send + Sync,
        f32: FromStimulus<Component>,
    {
        let pixels = map_indices(bitmap.len(), |index| {
            SrgbLuma::new(bitmap[index])
                .into_format::<f32>()
                .into_linear()
                .luma
        });

        Self {
            width,
            height,
            pixels,
        }
    }
}

impl LinearPyramidLevel for LinearGrayImage {
    fn downsampled(&self) -> Option<Self> {
        downsample::<Luma<f32>>(self.pixels.clone(), self.width, self.height).map(|pixels| Self {
            width: self.width / 2,
            height: self.height / 2,
            pixels,
        })
    }

    fn to_lab_planes(&self) -> Vec<Plane> {
        let values = map_indices(self.pixels.len(), |index| {
            linear_to_lab_gray(self.pixels[index])
        });

        vec![Plane::new(self.width, self.height, values)]
    }
}
