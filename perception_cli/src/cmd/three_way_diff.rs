use std::path::PathBuf;

use anyhow::Context as _;
use anyhow::Result;
use clap::Parser;

use perception::DiffOutputPaths;
use perception::DissimilarityThreshold;

use crate::cmd::backend::Backend;

#[derive(Parser)]
pub struct ThreeWayDiff {
    #[arg(long, default_value_t = Backend::Cpu, help = "Backend that runs the comparison")]
    pub backend: Backend,
    #[arg(long, help = "Path to the original (reference) image")]
    pub original: PathBuf,
    #[arg(long, help = "Path to the distorted (current) image")]
    pub distorted: PathBuf,
    #[arg(
        long,
        help = "Dissimilarity, from 0 to 1, at or above which a pixel is painted red"
    )]
    pub threshold: f32,
    #[arg(long, help = "Where to write the expected (original) image panel")]
    pub expected: PathBuf,
    #[arg(long, help = "Where to write the current (distorted) image panel")]
    pub current: PathBuf,
    #[arg(long, help = "Where to write the red-highlighted diff panel")]
    pub diff: PathBuf,
}

impl ThreeWayDiff {
    pub(crate) fn handle(self) -> Result<()> {
        let ThreeWayDiff {
            backend,
            original,
            distorted,
            threshold,
            expected,
            current,
            diff,
        } = self;
        let threshold =
            DissimilarityThreshold::new(threshold).context("invalid dissimilarity threshold")?;
        let output = DiffOutputPaths::new(&expected, &current, &diff);

        backend.into_engine().and_then(|engine| {
            engine
                .diff(&original, &distorted)
                .context("failed to compute the three-way diff")?
                .write(threshold, &output)
                .context("failed to write the three-way diff images")
        })
    }
}
