use image::Rgba;

use perception_cli::run::run;
use perception_test::Scratch;
use perception_test::write_test_image;

#[test]
fn runs_the_three_way_diff_subcommand() {
    let scratch = Scratch::new("cli_run_three_way_diff");
    write_test_image(&scratch.path("original.png"), 8, 0);
    write_test_image(&scratch.path("distorted.png"), 8, 40);

    run(vec![
        "perception".into(),
        "three-way-diff".into(),
        "--backend".into(),
        "cpu".into(),
        "--original".into(),
        scratch.path("original.png").into_os_string(),
        "--distorted".into(),
        scratch.path("distorted.png").into_os_string(),
        "--threshold".into(),
        "0.8".into(),
        "--expected".into(),
        scratch.path("expected.png").into_os_string(),
        "--current".into(),
        scratch.path("current.png").into_os_string(),
        "--diff".into(),
        scratch.path("diff.png").into_os_string(),
    ])
    .expect("the three-way-diff subcommand runs");

    let expected = image::open(scratch.path("expected.png"))
        .unwrap()
        .into_rgba8();
    let current = image::open(scratch.path("current.png"))
        .unwrap()
        .into_rgba8();

    assert_eq!(*expected.get_pixel(0, 0), Rgba([0, 0, 100, 255]));
    assert_eq!(*current.get_pixel(0, 0), Rgba([40, 0, 100, 255]));
    assert!(scratch.path("diff.png").is_file());
}

#[test]
fn runs_the_dissimilarity_subcommand() {
    let scratch = Scratch::new("cli_run_dissimilarity");
    write_test_image(&scratch.path("original.png"), 8, 0);
    write_test_image(&scratch.path("distorted.png"), 8, 40);

    run(vec![
        "perception".into(),
        "dissimilarity".into(),
        "--original".into(),
        scratch.path("original.png").into_os_string(),
        "--distorted".into(),
        scratch.path("distorted.png").into_os_string(),
        "--output".into(),
        scratch.path("dissimilarity.png").into_os_string(),
    ])
    .expect("the dissimilarity subcommand runs");

    assert!(scratch.path("dissimilarity.png").is_file());
}

#[test]
fn rejects_an_out_of_range_threshold() {
    let scratch = Scratch::new("cli_run_threshold");
    write_test_image(&scratch.path("original.png"), 8, 0);
    write_test_image(&scratch.path("distorted.png"), 8, 40);

    run(vec![
        "perception".into(),
        "three-way-diff".into(),
        "--original".into(),
        scratch.path("original.png").into_os_string(),
        "--distorted".into(),
        scratch.path("distorted.png").into_os_string(),
        "--threshold".into(),
        "5".into(),
        "--expected".into(),
        scratch.path("expected.png").into_os_string(),
        "--current".into(),
        scratch.path("current.png").into_os_string(),
        "--diff".into(),
        scratch.path("diff.png").into_os_string(),
    ])
    .expect_err("an out-of-range threshold fails");
}

#[test]
fn reports_a_missing_image_for_the_three_way_diff() {
    let scratch = Scratch::new("cli_run_three_way_diff_missing");
    write_test_image(&scratch.path("distorted.png"), 8, 0);

    run(vec![
        "perception".into(),
        "three-way-diff".into(),
        "--original".into(),
        scratch.path("missing.png").into_os_string(),
        "--distorted".into(),
        scratch.path("distorted.png").into_os_string(),
        "--threshold".into(),
        "0.8".into(),
        "--expected".into(),
        scratch.path("expected.png").into_os_string(),
        "--current".into(),
        scratch.path("current.png").into_os_string(),
        "--diff".into(),
        scratch.path("diff.png").into_os_string(),
    ])
    .expect_err("a missing original image fails");
}

#[test]
fn reports_a_missing_image_for_the_dissimilarity() {
    let scratch = Scratch::new("cli_run_dissimilarity_missing");
    write_test_image(&scratch.path("distorted.png"), 8, 0);

    run(vec![
        "perception".into(),
        "dissimilarity".into(),
        "--original".into(),
        scratch.path("missing.png").into_os_string(),
        "--distorted".into(),
        scratch.path("distorted.png").into_os_string(),
        "--output".into(),
        scratch.path("dissimilarity.png").into_os_string(),
    ])
    .expect_err("a missing original image fails");
}

#[test]
fn reports_an_error_for_an_unknown_subcommand() {
    run(vec!["perception".into(), "sharpen".into()]).expect_err("an unknown subcommand fails");
}
