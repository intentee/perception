use anyhow::Result;

use crate::cmd::Commands;

pub fn dispatch(command: Commands) -> Result<()> {
    match command {
        Commands::ThreeWayDiff(command) => command.handle(),
        Commands::Dissimilarity(command) => command.handle(),
    }
}
