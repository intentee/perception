use std::ffi::OsString;

use anyhow::Result;
use clap::Parser as _;

use crate::cli::Cli;
use crate::dispatch::dispatch;

pub fn run(arguments: Vec<OsString>) -> Result<()> {
    let cli = Cli::try_parse_from(arguments)?;

    dispatch(cli.command)
}
