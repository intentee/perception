use image::DynamicImage;
use image::imageops::FilterType;
use rgb::FromSlice;

use perception_metric::CpuBackend;
use perception_metric::Ssim;
use perception_metric::SsimImage;
use perception_metric_test::kendall_tau_b::kendall_tau_b;
use perception_metric_test::spearman::spearman;

const CAT_TABBY: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fixtures/cat_tabby.png"
));
const CAT_TUXEDO: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fixtures/cat_tuxedo.png"
));

const LADDER_SIDE: u32 = 24;
const BLUR_STEPS: usize = 4;
const BLUR_SIGMA_STEP: f32 = 0.5;

fn resized(bytes: &[u8]) -> DynamicImage {
    image::load_from_memory(bytes)
        .expect("fixture is a valid image")
        .resize_exact(LADDER_SIDE, LADDER_SIDE, FilterType::Triangle)
}

fn build_rgb8(context: &Ssim<CpuBackend>, image: &DynamicImage) -> SsimImage<CpuBackend> {
    let buffer = image.to_rgb8();

    context
        .create_image_rgb(
            buffer.as_raw().as_rgb(),
            buffer.width() as usize,
            buffer.height() as usize,
        )
        .expect("resized fixture has a consistent buffer")
}

fn build_rgba8(context: &Ssim<CpuBackend>, image: &DynamicImage) -> SsimImage<CpuBackend> {
    let buffer = image.to_rgba8();

    context
        .create_image_rgba(
            buffer.as_raw().as_rgba(),
            buffer.width() as usize,
            buffer.height() as usize,
        )
        .expect("resized fixture has a consistent buffer")
}

fn build_gray(context: &Ssim<CpuBackend>, image: &DynamicImage) -> SsimImage<CpuBackend> {
    let buffer = image.to_luma8();

    context
        .create_image_gray(
            buffer.as_raw(),
            buffer.width() as usize,
            buffer.height() as usize,
        )
        .expect("resized fixture has a consistent buffer")
}

fn blur_ladder_similarities(
    base: &DynamicImage,
    build: fn(&Ssim<CpuBackend>, &DynamicImage) -> SsimImage<CpuBackend>,
) -> Vec<f64> {
    let context = Ssim::new();
    let reference = build(&context, base);
    let mut similarities = Vec::with_capacity(BLUR_STEPS + 1);

    similarities.push(
        context
            .compare(&reference, &reference)
            .expect("matching dimensions")
            .similarity
            .value(),
    );

    for step in 1..=BLUR_STEPS {
        let distorted = build(&context, &base.blur(step as f32 * BLUR_SIGMA_STEP));

        similarities.push(
            context
                .compare(&reference, &distorted)
                .expect("matching dimensions")
                .similarity
                .value(),
        );
    }

    similarities
}

fn assert_similarity_ranks_blur_severity(
    base: &DynamicImage,
    build: fn(&Ssim<CpuBackend>, &DynamicImage) -> SsimImage<CpuBackend>,
) {
    let similarities = blur_ladder_similarities(base, build);

    assert!(
        (similarities[0] - 1.0).abs() < 1e-9,
        "identical inputs must score one: {}",
        similarities[0]
    );

    for window in similarities.windows(2) {
        assert!(
            window[1] < window[0],
            "similarity must shrink with blur: {similarities:?}"
        );
    }

    let ladder: Vec<f64> = (0..similarities.len()).map(|index| index as f64).collect();

    assert!(
        (spearman(&ladder, &similarities) + 1.0).abs() < 1e-9,
        "the metric must rank the blur ladder perfectly and inversely (Spearman)"
    );
    assert!(
        (kendall_tau_b(&ladder, &similarities) + 1.0).abs() < 1e-9,
        "the metric must rank the blur ladder perfectly and inversely (Kendall)"
    );
}

#[test]
fn tabby_rgb8_similarity_ranks_blur_severity() {
    assert_similarity_ranks_blur_severity(&resized(CAT_TABBY), build_rgb8);
}

#[test]
fn tabby_rgba8_similarity_ranks_blur_severity() {
    assert_similarity_ranks_blur_severity(&resized(CAT_TABBY), build_rgba8);
}

#[test]
fn tabby_gray_similarity_ranks_blur_severity() {
    assert_similarity_ranks_blur_severity(&resized(CAT_TABBY), build_gray);
}

#[test]
fn tuxedo_rgb8_similarity_ranks_blur_severity() {
    assert_similarity_ranks_blur_severity(&resized(CAT_TUXEDO), build_rgb8);
}

#[test]
fn tuxedo_rgba8_similarity_ranks_blur_severity() {
    assert_similarity_ranks_blur_severity(&resized(CAT_TUXEDO), build_rgba8);
}

#[test]
fn tuxedo_gray_similarity_ranks_blur_severity() {
    assert_similarity_ranks_blur_severity(&resized(CAT_TUXEDO), build_gray);
}
