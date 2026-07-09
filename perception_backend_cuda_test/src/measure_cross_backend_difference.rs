use perception_backend::Backend;
use perception_backend::SsimError;

use crate::cross_backend_difference::CrossBackendDifference;
use crate::max_abs_difference::max_abs_difference;

pub fn measure_cross_backend_difference<ReferenceBackend, CandidateBackend>(
    reference_backend: &ReferenceBackend,
    candidate_backend: &CandidateBackend,
    reference_original: &ReferenceBackend::Prepared,
    reference_distorted: &ReferenceBackend::Prepared,
    candidate_original: &CandidateBackend::Prepared,
    candidate_distorted: &CandidateBackend::Prepared,
) -> Result<CrossBackendDifference, SsimError>
where
    ReferenceBackend: Backend,
    CandidateBackend: Backend,
{
    let scale_count = reference_backend.scale_dimensions(reference_original).len();

    (0..scale_count).try_fold(
        CrossBackendDifference {
            max_map_difference: 0.0,
            max_deviation_difference: 0.0,
        },
        |CrossBackendDifference {
             max_map_difference,
             max_deviation_difference,
         },
         scale_index| {
            let exponent = 0.5f64.powf(scale_index as f64);

            if scale_index == 0 {
                reference_backend
                    .compare_scale_with_map(
                        reference_original,
                        reference_distorted,
                        scale_index,
                        exponent,
                    )
                    .and_then(|reference| {
                        candidate_backend
                            .compare_scale_with_map(
                                candidate_original,
                                candidate_distorted,
                                scale_index,
                                exponent,
                            )
                            .map(|candidate| CrossBackendDifference {
                                max_map_difference: max_map_difference.max(max_abs_difference(
                                    &reference.map.pixels,
                                    &candidate.map.pixels,
                                )),
                                max_deviation_difference: max_deviation_difference
                                    .max((reference.deviation - candidate.deviation).abs()),
                            })
                    })
            } else {
                reference_backend
                    .compare_scale(
                        reference_original,
                        reference_distorted,
                        scale_index,
                        exponent,
                    )
                    .and_then(|reference| {
                        candidate_backend
                            .compare_scale(
                                candidate_original,
                                candidate_distorted,
                                scale_index,
                                exponent,
                            )
                            .map(|candidate| CrossBackendDifference {
                                max_map_difference,
                                max_deviation_difference: max_deviation_difference
                                    .max((reference.deviation - candidate.deviation).abs()),
                            })
                    })
            }
        },
    )
}
