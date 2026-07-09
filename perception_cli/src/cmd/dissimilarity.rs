use std::path::PathBuf;

use anyhow::Context as _;
use anyhow::Result;
use clap::Parser;

use crate::cmd::backend::Backend;

#[derive(Parser)]
pub struct Dissimilarity {
    #[arg(long, default_value_t = Backend::Cpu, help = "Backend that runs the comparison")]
    pub backend: Backend,
    #[arg(long, help = "Path to the original (reference) image")]
    pub original: PathBuf,
    #[arg(long, help = "Path to the distorted (current) image")]
    pub distorted: PathBuf,
    #[arg(long, help = "Where to write the grayscale dissimilarity heatmap")]
    pub output: PathBuf,
}

impl Dissimilarity {
    pub(crate) fn handle(self) -> Result<()> {
        let Dissimilarity {
            backend,
            original,
            distorted,
            output,
        } = self;

        backend.into_engine().and_then(|engine| {
            engine
                .compare(&original, &distorted)
                .context("failed to compare the images")?
                .into_map()
                .write(&output)
                .context("failed to write the dissimilarity image")
        })
    }
}
