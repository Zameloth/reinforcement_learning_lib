mod environments {
    pub trait Environment {
        fn reset(&mut self) -> usize; // renvoie l’état initial
        fn step(&mut self, action: usize) -> (usize, f64, bool); // (next_state, reward, done)
        fn num_states(&self) -> usize;
        fn num_actions(&self) -> usize;
        
        //fn render(&self);
    }
}
