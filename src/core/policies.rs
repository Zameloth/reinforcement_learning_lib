use crate::core::envs::{DPEnvironment, Environment, MonteCarloEnvironment};
use rand;
use std::collections::HashMap;

/// Interface générale pour les policies
pub trait Policy {
    fn set_proba(&mut self, state: usize, action: usize, proba: f64);

    fn get_proba(&self, state: usize, action: usize) -> f64;
}

pub struct ProbabilisticPolicy {
    pub policy_table: Vec<f64>,
    num_states: usize,
    num_actions: usize,
}

impl ProbabilisticPolicy {
    pub fn new<E: Environment>(env: &E) -> Self {
        Self {
            policy_table: vec![
                1.0 / env.num_actions() as f64;
                env.num_states() * env.num_actions()
            ],
            num_states: env.num_states(),
            num_actions: env.num_actions(),
        }
    }

    // pub fn new(env: MonteCarloEnvironment) -> Self {
    //
    // }
}

impl Policy for ProbabilisticPolicy {
    fn set_proba(&mut self, state: usize, action: usize, proba: f64) {
        self.policy_table[state * self.num_actions + action] = proba;
    }

    fn get_proba(&self, state: usize, action: usize) -> f64 {
        self.policy_table[state * self.num_actions + action]
    }
}
