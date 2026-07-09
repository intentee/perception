use std::path::PathBuf;

use clap::Parser as _;

use perception_cli::cli::Cli;
use perception_cli::cmd::Commands;

#[test]
fn parses_the_three_way_diff_subcommand_with_all_arguments() {
    let cli = Cli::try_parse_from([
        "perception",
        "three-way-diff",
        "--backend",
        "cpu",
        "--original",
        "a.png",
        "--distorted",
        "b.png",
        "--threshold",
        "0.8",
        "--expected",
        "e.png",
        "--current",
        "c.png",
        "--diff",
        "d.png",
    ])
    .expect("the three-way-diff subcommand parses");

    let Commands::ThreeWayDiff(three_way_diff) = cli.command else {
        panic!("expected the three-way-diff subcommand");
    };

    assert_eq!(three_way_diff.original, PathBuf::from("a.png"));
    assert_eq!(three_way_diff.distorted, PathBuf::from("b.png"));
    assert_eq!(three_way_diff.threshold, 0.8);
    assert_eq!(three_way_diff.expected, PathBuf::from("e.png"));
    assert_eq!(three_way_diff.current, PathBuf::from("c.png"));
    assert_eq!(three_way_diff.diff, PathBuf::from("d.png"));
}

#[test]
fn parses_the_dissimilarity_subcommand_with_all_arguments() {
    let cli = Cli::try_parse_from([
        "perception",
        "dissimilarity",
        "--original",
        "a.png",
        "--distorted",
        "b.png",
        "--output",
        "m.png",
    ])
    .expect("the dissimilarity subcommand parses");

    let Commands::Dissimilarity(dissimilarity) = cli.command else {
        panic!("expected the dissimilarity subcommand");
    };

    assert_eq!(dissimilarity.original, PathBuf::from("a.png"));
    assert_eq!(dissimilarity.distorted, PathBuf::from("b.png"));
    assert_eq!(dissimilarity.output, PathBuf::from("m.png"));
}

#[test]
fn requires_a_subcommand() {
    assert!(Cli::try_parse_from(["perception"]).is_err());
}

#[test]
fn rejects_an_unknown_backend() {
    assert!(
        Cli::try_parse_from([
            "perception",
            "dissimilarity",
            "--backend",
            "quantum",
            "--original",
            "a.png",
            "--distorted",
            "b.png",
            "--output",
            "m.png",
        ])
        .is_err()
    );
}
