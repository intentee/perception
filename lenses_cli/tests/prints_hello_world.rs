use std::process::Command;

use anyhow::Result;

#[test]
fn prints_hello_world() -> Result<()> {
    let output = Command::new(env!("CARGO_BIN_EXE_lenses"))
        .arg("hello")
        .output()?;

    assert!(output.status.success());
    assert_eq!(output.stdout, b"hello, world\n");

    Ok(())
}
