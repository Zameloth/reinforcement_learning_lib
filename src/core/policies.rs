pub trait Policy {
    fn action(&self, action: usize) -> usize;
}
