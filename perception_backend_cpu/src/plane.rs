pub(crate) struct Plane {
    width: usize,
    height: usize,
    pixels: Vec<f32>,
}

impl Plane {
    pub(crate) fn new(width: usize, height: usize, pixels: Vec<f32>) -> Self {
        Self {
            width,
            height,
            pixels,
        }
    }

    pub(crate) fn height(&self) -> usize {
        self.height
    }

    pub(crate) fn into_pixels(self) -> Vec<f32> {
        self.pixels
    }

    pub(crate) fn pixels(&self) -> &[f32] {
        &self.pixels
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }
}
