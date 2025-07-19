use crate::core::envs;
use crate::core::envs::{DPEnvironment, DynamicProgramingEnvironment, Environment, MonteCarloEnvironment};
use rand::random_range;

#[derive(Debug)]
pub struct LineWorld {
    agent_pos: usize,
}

impl LineWorld {
    pub fn new() -> LineWorld {
        let mut env = LineWorld { agent_pos: 0 };
        env.reset();
        env
    }
}

impl Environment for LineWorld {
    fn num_states(&self) -> usize {
        5
    }

    fn num_actions(&self) -> usize {
        2
    }

    fn num_rewards(&self) -> usize {
        3
    }
}

impl envs::MonteCarloEnvironment for LineWorld {
    fn reset(&mut self) {
        self.agent_pos = 2;
    }

    fn step(&mut self, action: usize) -> (usize, f64) {
        assert!(action < self.num_actions());
        assert!(!self.is_game_over());

        match action {
            0 => self.agent_pos -= 1,
            1 => self.agent_pos += 1,
            _ => unreachable!(),
        }
        
        (self.state_id(), self.score())
    }

    fn score(&self) -> f64 {
        match self.agent_pos {
            0 => -1.0,
            4 => 1.0,
            1..=3 => 0.0,
            _ => unreachable!(),
        }
    }

    fn is_game_over(&self) -> bool {
        self.agent_pos == 0 || self.agent_pos == 4
    }

    fn display(&self) {
        let line: String = (0..self.num_states())
            .map(|i| if i == self.agent_pos { 'A' } else { '-' })
            .collect();
        println!("{}", line);
    }

    fn start_from_random_state(&mut self) {
        self.reset();
        self.agent_pos = random_range(0..5);
    }

    fn state_id(&self) -> usize {
        self.agent_pos
    }

    fn is_forbidden(&self, action: usize) -> bool {
        action >= self.num_actions()
    }
}

pub fn line_world_dp() -> DPEnvironment {
    let mut env = DPEnvironment::new(5, 2, 3, vec![-1.0, 0.0, 1.0], vec![0, 4]);

    env.set_transition_prob(3, 0, 2, 1, 1.0);
    env.set_transition_prob(2, 0, 1, 1, 1.0);
    env.set_transition_prob(1, 0, 0, 0, 1.0);

    env.set_transition_prob(3, 1, 4, 2, 1.0);
    env.set_transition_prob(2, 1, 3, 1, 1.0);
    env.set_transition_prob(1, 1, 2, 1, 1.0);

    env
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::envs::MonteCarloEnvironment;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[test]
    fn test_line_world_monte_carlo_initial_state() {
        let mut env = LineWorld { agent_pos: 0 };
        env.reset();
        // Position initiale
        assert_eq!(env.agent_pos, 2);
        // Vérification des dimensions
        assert_eq!(env.num_states(), 5);
        assert_eq!(env.num_actions(), 2);
        // Pas de fin de partie au départ
        assert!(!env.is_game_over());
        // Score initial (position 2) : ni gain ni perte
        assert_eq!(env.score(), 0.0);
    }

    #[test]
    fn test_line_world_monte_carlo_step_and_terminal() {
        let mut env = LineWorld { agent_pos: 0 };
        env.reset(); // agent_pos = 2

        // Un pas vers la gauche
        env.step(0);
        assert_eq!(env.agent_pos, 1);
        assert_eq!(env.score(), 0.0);
        assert!(!env.is_game_over());

        // Un deuxième pas vers la gauche → état terminal 0
        env.step(0);
        assert_eq!(env.agent_pos, 0);
        assert_eq!(env.score(), -1.0);
        assert!(env.is_game_over());

        env.reset(); // agent_pos = 2
        env.step(1);
        assert_eq!(env.agent_pos, 3);
        assert_eq!(env.score(), 0.0);
        assert!(!env.is_game_over());
        // Deuxième pas à droite => état terminal 4
        env.step(1);
        assert_eq!(env.agent_pos, 4);
        assert_eq!(env.score(), 1.0);
        assert!(env.is_game_over());

        // Test d'un pas après fin de partie panique - on utilise AssertUnwindSafe
        let result = catch_unwind(AssertUnwindSafe(|| env.step(1)));
        assert!(result.is_err());
    }

    #[test]
    fn test_display_line_world() {
        let mut env = LineWorld { agent_pos: 0 };
        env.reset();
        env.display();
    }

    #[test]
    fn test_line_world_dp_structure_and_transitions() {
        let env = line_world_dp();
        // Dimensions et récompenses
        assert_eq!(env.num_states, 5);
        assert_eq!(env.num_actions, 2);
        assert_eq!(env.num_rewards, 3);
        assert_eq!(env.rewards, vec![-1.0, 0.0, 1.0]);

        // États terminaux
        println!("{:?}", env.terminal_states);
        assert!(env.terminal_states.contains(&0));
        assert!(env.terminal_states.contains(&4));

        // Quelques probabilités de transition
        // Action 0 (gauche) depuis l’état 3 → vers 2 avec reward_index 1 (0.0)
        assert_eq!(env.get_transition_prob(3, 0, 2, 1), 1.0);
        // Action 0 depuis l’état 1 → vers 0 avec reward_index 0 (-1.0)
        assert_eq!(env.get_transition_prob(1, 0, 0, 0), 1.0);
        // Action 1 (droite) depuis l’état 3 → vers 4 avec reward_index 2 (+1.0)
        assert_eq!(env.get_transition_prob(3, 1, 4, 2), 1.0);
    }
}
