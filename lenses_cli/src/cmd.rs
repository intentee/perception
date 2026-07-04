pub mod hello;

use clap::Subcommand;

use self::hello::Hello;

#[derive(Subcommand)]
pub enum Commands {
    #[command(about = "Prints a greeting")]
    Hello(Hello),
}
