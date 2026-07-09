pub mod backend;
pub mod dissimilarity;
pub mod three_way_diff;

use clap::Subcommand;

use self::dissimilarity::Dissimilarity;
use self::three_way_diff::ThreeWayDiff;

#[derive(Subcommand)]
pub enum Commands {
    #[command(about = "Write expected, current, and red-highlighted diff panels for two images")]
    ThreeWayDiff(ThreeWayDiff),

    #[command(about = "Write a grayscale heatmap of where two images differ")]
    Dissimilarity(Dissimilarity),
}
