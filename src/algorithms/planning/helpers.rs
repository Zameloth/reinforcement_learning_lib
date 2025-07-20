use rand::prelude::IndexedRandom;
use rand::Rng;
use std::collections::HashMap;

use crate::core::envs::MonteCarloEnvironment;
use crate::core::policies::DeterministicPolicy;

type State = usize;
type Action = usize;

/// Choisit une action selon une stratégie ε‑greedy
pub fn choose_action(
    q: &HashMap<(State, Action), f64>,
    s: State,
    actions: &[Action],
    epsilon: f64,
) -> Action {
    let mut rng = rand::rng();

    if rng.random::<f64>() < epsilon {
        // Exploration : choisir une action au hasard
        *actions.choose(&mut rng).unwrap()
    } else {
        // Exploitation : choisir la meilleure action connue
        actions
            .iter()
            .copied()
            .max_by(|&a1, &a2| {
                q.get(&(s, a1))
                    .unwrap_or(&0.0)
                    .partial_cmp(q.get(&(s, a2)).unwrap_or(&0.0))
                    .unwrap()
            })
            .unwrap_or(actions[0])
    }
}

/// Construit une politique déterministe à partir d’une Q‑table
pub fn build_policy(
    q: &HashMap<(State, Action), f64>,
    states: &[State],
    actions: &[Action],
    env: &dyn MonteCarloEnvironment,
) -> DeterministicPolicy {
    let policy_table = states
        .iter()
        .map(|&s| {
            actions
                .iter()
                .copied()
                .max_by(|&a1, &a2| {
                    q.get(&(s, a1))
                        .unwrap_or(&0.0)
                        .partial_cmp(q.get(&(s, a2)).unwrap_or(&0.0))
                        .unwrap()
                })
                .unwrap_or(actions[0])
        })
        .collect();

    DeterministicPolicy::from_vec(env, policy_table)
}
