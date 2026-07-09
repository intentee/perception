use crate::gaussian_blur::GaussianBlur;
use crate::plane::Plane;
use crate::plane_error::PlaneError;
use crate::prepared_channel::PreparedChannel;

pub(crate) struct PreparedScale {
    channels: Vec<PreparedChannel>,
}

impl PreparedScale {
    pub(crate) fn prepare(planes: Vec<Plane>, blur: &GaussianBlur) -> Result<Self, PlaneError> {
        let mut channels = Vec::with_capacity(planes.len());

        for (index, plane) in planes.into_iter().enumerate() {
            channels.push(PreparedChannel::prepare(plane, index > 0, blur)?);
        }

        Ok(Self { channels })
    }

    pub(crate) fn channels(&self) -> &[PreparedChannel] {
        &self.channels
    }

    pub(crate) fn height(&self) -> usize {
        self.channels[0].values().height()
    }

    pub(crate) fn width(&self) -> usize {
        self.channels[0].values().width()
    }
}
