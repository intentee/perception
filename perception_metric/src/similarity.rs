use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Similarity {
    value: f64,
}

impl Similarity {
    pub(crate) fn new(value: f64) -> Self {
        Self { value }
    }

    #[must_use]
    pub fn value(self) -> f64 {
        self.value
    }
}

impl Display for Similarity {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        Display::fmt(&self.value, formatter)
    }
}

impl From<Similarity> for f64 {
    fn from(similarity: Similarity) -> Self {
        similarity.value
    }
}

#[cfg(test)]
mod tests {
    use super::Similarity;

    #[test]
    fn perfect_ssim_maps_to_one() {
        assert_eq!(Similarity::new(1.0).value(), 1.0);
    }

    #[test]
    fn displays_the_underlying_value() {
        assert_eq!(Similarity::new(1.0).to_string(), "1");
    }

    #[test]
    fn converts_into_f64() {
        let value: f64 = Similarity::new(0.5).into();

        assert!((value - 0.5).abs() < 1e-12);
    }

    #[test]
    fn is_comparable_copyable_and_debuggable() {
        let identical = Similarity::new(1.0);
        let different = Similarity::new(0.5);

        assert_ne!(identical, different);
        assert!(different < identical);

        let copied = identical;

        assert_eq!(copied, identical);
        assert!(!format!("{identical:?}").is_empty());
    }
}
