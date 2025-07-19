use crate::core::envs::{DynamicProgramingEnvironment, Environment};
use rand;
use std::fmt::{Display, Formatter};
use rand::{random, Rng};

/// Interface générale pour les policies
pub trait Policy {
    fn get_action(&self, state: &usize) -> usize;
}

pub struct ProbabilisticPolicy {
    pub policy_table: Vec<f64>,
    num_states: usize,
    num_actions: usize,
}

pub struct DeterministicPolicy {
    pub policy_table: Vec<usize>,
    num_states: usize,
    num_actions: usize,
}

impl Display for DeterministicPolicy {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeterministicPolicy : {:?}", self.policy_table)
    }
}

impl DeterministicPolicy {
    pub fn new_det_pol(env: &dyn DynamicProgramingEnvironment) -> Self {
        Self {
            policy_table: vec![rand::random_range(0..env.num_actions()); env.num_states()],
            num_states: env.num_states(),
            num_actions: env.num_actions(),
        }
    }

    pub fn set_action(&mut self, state: &usize, action: usize) {
        self.policy_table[*state] = action;
    }
    
    pub fn get_action(&self, state: &usize) -> usize {
        self.policy_table[*state]
    }
}

impl Policy for DeterministicPolicy {
    fn get_action(&self, state: &usize) -> usize {
        self.policy_table[*state]
    }
}

impl ProbabilisticPolicy {
    pub fn new_pb_pol<E: Environment>(env: &E) -> Self {
        Self {
            policy_table: vec![
                1.0 / env.num_actions() as f64;
                env.num_states() * env.num_actions()
            ],
            num_states: env.num_states(),
            num_actions: env.num_actions(),
        }
    }

    fn set_proba(&mut self, state: usize, action: usize, proba: f64) {
        self.policy_table[state * self.num_actions + action] = proba;
    }

    fn get_proba(&self, state: usize, action: usize) -> f64 {
        self.policy_table[state * self.num_actions + action]
    }
    
}

impl Policy for ProbabilisticPolicy {
    fn get_action(&self, state: &usize) -> usize {
        let mut probs: Vec<f64> = vec![0.0; self.num_actions];
        for action in 0..self.num_actions {
            probs.insert(action, self.get_proba(*state, action));
        }
        
        assert_eq!(probs.iter().sum::<f64>(), 1.0);
        
        let mut rand_num:f64 = random();
        for (i, prob) in probs.iter().enumerate() {
            rand_num -= prob;
            if rand_num <= 0.0 {
                return i;
            }
        }
        
        panic!("Error in get_action");
    }
}
