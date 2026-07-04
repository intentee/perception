mod cmd;

use anyhow::Result;
use clap::Parser;
use command_handler::handler::Handler as _;
use command_handler::shutdown_signal::register_shutdown_signals;
use tokio_util::sync::CancellationToken;

use crate::cmd::Commands;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cancellation_token: CancellationToken = register_shutdown_signals()
        .expect("failed to register shutdown signal handlers")
        .into();

    match Cli::parse().command {
        Commands::Hello(handler) => handler.handle(cancellation_token).await,
    }
}
