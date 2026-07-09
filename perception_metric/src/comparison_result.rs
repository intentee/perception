use crate::similarity::Similarity;
use crate::ssim_map::SsimMap;

pub struct ComparisonResult {
    pub similarity: Similarity,
    pub ssim_maps: Vec<SsimMap>,
}
