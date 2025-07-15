use rand::{random_range};
use crate::core::envs::{DPEnvironment, DynamicProgramingEnvironment, Environment, MonteCarloEnvironment};

/// Actions : 
///     Pierre = 0
///     Feuille = 1
///     Ciseau = 2

fn pierre_feuille_ciseaux_dp() -> DPEnvironment {
    let num_states = 13; // état initial + 3 états possibles + 3*3 états finaux
    let num_actions = 3; // pierre, feuille, ciseaux
    let num_rewards = 3;
    let rewards = vec![-1.0, 0.0, 1.0];
    let terminal_states: Vec<usize> = vec![4, 5, 6, 7, 8, 9, 10, 11, 12];

    let mut env = DPEnvironment::new(
        num_states,
        num_actions,
        num_rewards,
        rewards,
        terminal_states,
    );

    // Round 1 : l'adversaire est random
    for a in 0..num_actions {
        for r in 0..num_rewards {
            env.set_transition_prob(0, a, 1 + a, r, 1.0 / num_rewards as f64);
        }
    }


    // Round 2 : L'adversaire refait l'action de l'agent :
    for s in 1..4 {
        let a_adv = s - 1;
        for a in 0..num_actions {
            let s_prime = 4;
            let reward_ind = match (a, a_adv) {
                // égalité
                (0, 0) => 1,
                (1, 1) => 1,
                (2, 2) => 1,

                // Perdu
                (0, 1) => 0,
                (1, 2) => 0,
                (2, 0) => 0,

                // Gagné
                (0, 2) => 2,
                (1, 0) => 2,
                (2, 1) => 2,

                _ => panic!("Invalid action")
            };

            env.set_transition_prob(s, a, s_prime, reward_ind, 1.0);
        }
    }
    env
}

struct PierreFeuilleCiseaux {
    round_number: usize,
    last_action: usize,
    adv_action: usize,
}

impl Environment for PierreFeuilleCiseaux {
    fn num_states(&self) -> usize {
        13
    }

    fn num_actions(&self) -> usize {
        3
    }
    fn num_rewards(&self) -> usize {
        3
    }
}

impl MonteCarloEnvironment for PierreFeuilleCiseaux {
    fn reset(&mut self) {
        self.round_number=0;
        self.last_action=0;
        self.adv_action=0;
    }

    fn step(&mut self, action: usize) {
        if ! self.is_game_over(){
            match self.round_number {
                0 => {
                    self.adv_action = random_range(0..3)
                }
                1=> self.adv_action = self.last_action,
                _ => unreachable!()
            }
            self.round_number += 1;
            self.last_action = action;
        }
        else {
            unreachable!()
        }
    }

    fn score(&self) -> f64 {
        match (self.last_action, self.adv_action) {
            // égalité
            (0, 0) => 0.0,
            (1, 1) => 0.0,
            (2, 2) => 0.0,

            // Perdu
            (1, 2) => -1.0,
            (2, 0) => -1.0,
            (0, 1) => -1.0,

            // Gagné
            (0, 2) => 1.0,
            (1, 0) => 1.0,
            (2, 1) => 1.0,

            _ => unreachable!()
        }
    }

    fn is_game_over(&self) -> bool {
        self.round_number >= 2
    }



    fn display(&self) {
        todo!()
    }

    fn start_from_random_state(&mut self) {
        todo!()
    }

    fn state_id(&self) -> usize {
        todo!()
    }

    fn is_forbidden(&self, action: usize) -> bool {
        todo!()
    }
}