mod agents {
    pub trait Policy {
        fn action(&self, state: usize) -> usize;
    }
}