use std::process::Command;

use anyhow::Result;

#[test]
fn errors_without_a_subcommand() -> Result<()> {
    let output = Command::new(env!("CARGO_BIN_EXE_lenses")).output()?;

    assert!(!output.status.success());

    Ok(())
}
