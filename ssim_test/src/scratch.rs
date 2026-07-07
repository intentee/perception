use std::path::PathBuf;

pub struct Scratch {
    root: PathBuf,
}

impl Scratch {
    #[must_use]
    pub fn new(label: &str) -> Self {
        let root = std::env::temp_dir().join(format!("ssim_test_{}_{label}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("failed to create scratch directory");

        Self { root }
    }

    #[must_use]
    pub fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
