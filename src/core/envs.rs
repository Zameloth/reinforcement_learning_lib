pub trait Environment {
    fn reset(&mut self); // renvoie l’état initial
    fn step(&mut self, action: usize); // (next_state, reward, done)
    fn score(&self) -> f64;
    fn num_states(&self) -> usize;
    fn num_actions(&self) -> usize;
    fn is_game_over(&self) -> bool;
    //fn render(&self);
}

pub trait StatefulEnvironment: Environment {
    fn set_state(&self, state: usize);
    fn get_state(&self) -> usize;
}
