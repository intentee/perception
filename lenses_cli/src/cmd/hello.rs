use anyhow::Result;
use async_trait::async_trait;
use clap::Parser;
use command_handler::handler::Handler;
use tokio_util::sync::CancellationToken;

#[derive(Parser)]
pub struct Hello;

#[async_trait(?Send)]
impl Handler for Hello {
    async fn handle(self, _cancellation_token: CancellationToken) -> Result<()> {
        println!("hello, world");

        Ok(())
    }
}
