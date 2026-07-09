use crate::similarity_error::SimilarityError;

pub struct DissimilarityThreshold {
    value: f32,
}

impl DissimilarityThreshold {
    pub fn new(value: f32) -> Result<Self, SimilarityError> {
        if (0.0..=1.0).contains(&value) {
            Ok(Self { value })
        } else {
            Err(SimilarityError::ThresholdOutOfRange { value })
        }
    }

    pub(crate) fn value(&self) -> f32 {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::DissimilarityThreshold;

    #[test]
    fn rejects_a_value_outside_the_unit_interval() {
        assert!(DissimilarityThreshold::new(1.5).is_err());
    }
}
