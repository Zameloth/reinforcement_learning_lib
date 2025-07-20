pub mod dynamic_programming {
    use crate::core::envs::{DPEnvironment, DynamicProgramingEnvironment};

    pub fn grid_world() -> DPEnvironment {
        let num_states = 25;
        let num_actions = 4; // Haut bas gauche droite
        let num_rewards = 3;
        let rewards = vec![-3.0, 0.0, 1.0];
        let terminal_states = vec![4, 24]; // 4 = en haut à droite, 24 en bas à droite

        let mut env = DPEnvironment::new(
            num_states,
            num_actions,
            num_rewards,
            rewards.clone(),
            terminal_states.clone(),
        );

        fn reward_index(a: usize) -> usize {
            match a {
                4 => 0,
                24 => 2,
                _ => 1,
            }
        }

        for row in 0..5 {
            for col in 0..5 {
                let s = row * 5 + col;

                if terminal_states.contains(&s) {
                    continue;
                }

                // Haut
                let next_state = if row > 0 { (row - 1) * 5 + col } else { s };
                env.set_transition_prob(s, 0, next_state, reward_index(next_state), 1.0);

                // Bas
                let next_state = if row < 4 { (row + 1) * 5 + col } else { s };
                env.set_transition_prob(s, 1, next_state, reward_index(next_state), 1.0);

                // Gauche
                let next_state = if col > 0 { row * 5 + col - 1 } else { s };
                env.set_transition_prob(s, 2, next_state, reward_index(next_state), 1.0);

                // Droite
                let next_state = if col < 4 { row * 5 + col + 1 } else { s };
                env.set_transition_prob(s, 3, next_state, reward_index(next_state), 1.0);
            }
        }

        env
    }
}

use crate::core::envs::{Environment, MonteCarloEnvironment};
use rand::random_range;

pub struct GridWorld {
    agent_pos: usize,
}

impl GridWorld {
    pub fn new() -> Self {
        let mut env = Self { agent_pos: 0 };
        env.reset();
        env
    }
}

impl Environment for GridWorld {
    fn num_states(&self) -> usize {
        25
    }

    fn num_actions(&self) -> usize {
        4
    }

    fn num_rewards(&self) -> usize {
        3
    }
}

impl MonteCarloEnvironment for GridWorld {
    fn reset(&mut self) {
        self.agent_pos = 0;
    }

    fn step(&mut self, action: usize) -> (usize, f64) {
        let row = self.agent_pos / 5;
        let col = self.agent_pos % 5;

        let result = match action {
            0 => {
                if row > 0 {
                    (row - 1) * 5 + col
                } else {
                    self.agent_pos
                }
            }
            1 => {
                if row < 4 {
                    (row + 1) * 5 + col
                } else {
                    self.agent_pos
                }
            }
            2 => {
                if col > 0 {
                    row * 5 + col - 1
                } else {
                    self.agent_pos
                }
            }
            3 => {
                if col < 4 {
                    row * 5 + col + 1
                } else {
                    self.agent_pos
                }
            }
            _ => unreachable!(),
        };

        self.agent_pos = result;
        (self.agent_pos, self.score())
    }

    fn score(&self) -> f64 {
        match self.agent_pos {
            4 => -3.0,
            24 => 1.0,
            _ => 0.0,
        }
    }

    fn is_game_over(&self) -> bool {
        self.agent_pos == 4 || self.agent_pos == 24
    }

    fn display(&self) {
        let size = 5;
        for row in 0..size {
            let line: String = (0..size)
                .map(|col| {
                    let idx = row * size + col;
                    if idx == self.agent_pos {
                        'A'
                    } else {
                        '.'
                    }
                })
                .collect();
            println!("{}", line);
        }
    }

    fn start_from_random_state(&mut self) {
        self.agent_pos = random_range(0..self.num_states())
    }

    fn state_id(&self) -> usize {
        self.agent_pos
    }

    fn is_forbidden(&self, action: usize) -> bool {
        action >= self.num_actions()
    }

    fn action_name(&self, action: usize) -> String {
        match action {
            0 => "Haut".to_string(),
            1 => "Bas".to_string(),
            2 => "Gauche".to_string(),
            3 => "Droite".to_string(),
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::dynamic_programming::grid_world;
    use crate::core::envs::DynamicProgramingEnvironment;

    #[test]
    fn test_grid_world_transitions() {
        let env = grid_world();

        // (0,0) = état 0
        // Action droite (3) → (0,1) = état 1
        assert_eq!(env.get_transition_prob(0, 3, 1, 1), 1.0); // reward = 0 → index 1
                                                              // Action gauche (2) → reste en place
        assert_eq!(env.get_transition_prob(0, 2, 0, 1), 1.0);

        // (3,4) = état 19 → action bas (1) → (4,4) = état 24 (terminal)
        assert_eq!(env.get_transition_prob(19, 1, 24, 2), 1.0); // reward +1 → index 2

        // (0,3) = état 3 → action droite (3) → (0,4) = état 4 (terminal)
        assert_eq!(env.get_transition_prob(3, 3, 4, 0), 1.0); // reward -3 → index 0
    }

    use super::*;
    use crate::core::envs::MonteCarloEnvironment;
    #[test]
    fn test_initial_state() {
        let mut env = GridWorld { agent_pos: 0 };
        env.reset();
        assert_eq!(env.agent_pos, 0); // En haut à gauche (0,0)
    }

    #[test]
    fn test_valid_transition() {
        let mut env = GridWorld { agent_pos: 0 };
        env.reset();
        env.step(3); // droite depuis (0,0)
        assert_eq!(env.agent_pos, 1); // (0,1)
        assert_eq!(env.score(), 0.0);
        assert_eq!(env.is_game_over(), false);
    }

    #[test]
    fn test_wall_collision() {
        let mut env = GridWorld { agent_pos: 0 };
        env.reset();
        env.step(2); // gauche depuis (0,0)
        assert_eq!(env.agent_pos, 0); // reste sur place
        assert_eq!(env.score(), 0.0);
        assert_eq!(env.is_game_over(), false);
    }

    #[test]
    fn test_terminal_state_negative_reward() {
        let mut env = GridWorld { agent_pos: 0 };
        env.reset();
        env.step(3); // (0,0) → (0,1)
        env.step(3); // → (0,2)
        env.step(3); // → (0,3)
        env.step(3); // → (0,4), terminal
        assert_eq!(env.agent_pos, 4);
        assert_eq!(env.score(), -3.0);
        assert!(env.is_game_over());
    }

    #[test]
    fn test_terminal_state_positive_reward() {
        let mut env = GridWorld { agent_pos: 0 };
        env.reset();
        // Move down 4 times to reach (4,0)
        for _ in 0..4 {
            env.step(1);
        }
        env.step(3); // → (4,1)
        env.step(3); // → (4,2)
        env.step(3); // → (4,3)
        env.step(3); // → (4,4), terminal
        assert_eq!(env.agent_pos, 24);
        assert_eq!(env.score(), 1.0);
        assert!(env.is_game_over());
    }

    #[test]
    fn test_display() {
        let mut env = GridWorld { agent_pos: 0 };
        env.reset();
        env.display();
    }
}
