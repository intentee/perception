use crate::plane::Plane;

pub(crate) trait LinearPyramidLevel: Sized {
    fn downsampled(&self) -> Option<Self>;

    fn to_lab_planes(&self) -> Vec<Plane>;
}
