use crate::core::envs::Environment;

struct LineWorld {
    agent_pos: usize,
}

impl Environment for LineWorld {
    fn reset(&mut self) {
        self.agent_pos = 2;
    }

    fn step(&mut self, action: usize) {
        assert!(action > 0 && action < self.num_actions());
        assert!(!self.is_game_over());

        match action {
            0 => self.agent_pos -= 1,
            1 => self.agent_pos += 1,
            _ => unreachable!(),
        }
    }

    fn score(&self) -> f64 {
        match self.agent_pos {
            0 => -1.0,
            4 => 1.0,
            1..=3 => 0.0,
            _ => unreachable!(),
        }
    }

    fn num_states(&self) -> usize {
        5
    }

    fn num_actions(&self) -> usize {
        2
    }

    fn is_game_over(&self) -> bool {
        self.agent_pos == 0 || self.agent_pos == 4
    }
}
