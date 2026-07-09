use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;

#[cfg(feature = "cuda")]
use anyhow::Context as _;
use anyhow::Result;
use clap::ValueEnum;

use perception::Engine;

#[derive(Clone, ValueEnum)]
pub enum Backend {
    #[cfg(feature = "cpu")]
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

impl Backend {
    pub(crate) fn into_engine(self) -> Result<Engine> {
        Ok(match self {
            #[cfg(feature = "cpu")]
            Backend::Cpu => Engine::cpu(),
            #[cfg(feature = "cuda")]
            Backend::Cuda => Engine::cuda().context("failed to initialize the CUDA backend")?,
        })
    }
}

impl Display for Backend {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        self.to_possible_value()
            .expect("backend variants are never skipped")
            .get_name()
            .fmt(formatter)
    }
}

#[cfg(test)]
mod tests {
    use super::Backend;

    #[test]
    fn the_cpu_backend_builds_an_engine() {
        Backend::Cpu
            .into_engine()
            .expect("the cpu backend always builds an engine");
    }
}
