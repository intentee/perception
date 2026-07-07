use palette::stimulus::FromStimulus;
use rgb::RGB;
use rgb::RGBA;

use crate::scale_dimensions::ScaleDimensions;
use crate::scale_score::ScaleScore;
use crate::scale_score_with_map::ScaleScoreWithMap;
use crate::ssim_error::SsimError;

pub trait Backend {
    type Prepared;

    fn prepare_gray<Component>(
        &self,
        srgb: &[Component],
        width: usize,
        height: usize,
        scale_count: usize,
    ) -> Result<Self::Prepared, SsimError>
    where
        Component: Copy + Send + Sync,
        f32: FromStimulus<Component>;

    fn prepare_rgb<Component>(
        &self,
        srgb: &[RGB<Component>],
        width: usize,
        height: usize,
        scale_count: usize,
    ) -> Result<Self::Prepared, SsimError>
    where
        Component: Copy + Send + Sync,
        f32: FromStimulus<Component>;

    fn prepare_rgba<Component>(
        &self,
        srgb: &[RGBA<Component>],
        width: usize,
        height: usize,
        scale_count: usize,
    ) -> Result<Self::Prepared, SsimError>
    where
        Component: Copy + Send + Sync,
        f32: FromStimulus<Component>;

    fn scale_dimensions(&self, prepared: &Self::Prepared) -> Vec<ScaleDimensions>;

    fn compare_scale(
        &self,
        reference: &Self::Prepared,
        distorted: &Self::Prepared,
        scale_index: usize,
        adjusted_mean_exponent: f64,
    ) -> Result<ScaleScore, SsimError>;

    fn compare_scale_with_map(
        &self,
        reference: &Self::Prepared,
        distorted: &Self::Prepared,
        scale_index: usize,
        adjusted_mean_exponent: f64,
    ) -> Result<ScaleScoreWithMap, SsimError>;
}
