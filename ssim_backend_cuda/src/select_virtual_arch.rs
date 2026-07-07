const KNOWN_VIRTUAL_ARCHS: [(i32, &str); 10] = [
    (52, "compute_52"),
    (60, "compute_60"),
    (61, "compute_61"),
    (70, "compute_70"),
    (75, "compute_75"),
    (80, "compute_80"),
    (86, "compute_86"),
    (89, "compute_89"),
    (90, "compute_90"),
    (120, "compute_120"),
];

pub(crate) fn select_virtual_arch(major: i32, minor: i32) -> Option<&'static str> {
    let capability = major * 10 + minor;

    KNOWN_VIRTUAL_ARCHS
        .iter()
        .rev()
        .find(|(arch_capability, _)| *arch_capability <= capability)
        .map(|(_, arch)| *arch)
}

#[cfg(test)]
mod tests {
    use super::select_virtual_arch;

    #[test]
    fn exact_match_returns_that_arch() {
        assert_eq!(select_virtual_arch(7, 5), Some("compute_75"));
    }

    #[test]
    fn newer_than_every_known_arch_falls_back_to_the_highest() {
        assert_eq!(select_virtual_arch(13, 0), Some("compute_120"));
    }

    #[test]
    fn between_known_archs_picks_the_largest_below_or_equal() {
        assert_eq!(select_virtual_arch(8, 7), Some("compute_86"));
    }

    #[test]
    fn older_than_every_known_arch_is_unsupported() {
        assert_eq!(select_virtual_arch(5, 0), None);
    }
}
