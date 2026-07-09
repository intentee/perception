use std::path::Path;

pub struct DiffOutputPaths<'paths> {
    current: &'paths Path,
    diff: &'paths Path,
    expected: &'paths Path,
}

impl<'paths> DiffOutputPaths<'paths> {
    #[must_use]
    pub fn new(expected: &'paths Path, current: &'paths Path, diff: &'paths Path) -> Self {
        Self {
            current,
            diff,
            expected,
        }
    }

    #[must_use]
    pub fn current(&self) -> &Path {
        self.current
    }

    #[must_use]
    pub fn diff(&self) -> &Path {
        self.diff
    }

    #[must_use]
    pub fn expected(&self) -> &Path {
        self.expected
    }
}
