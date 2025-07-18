use rand::Rng;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use rand::prelude::IndexedRandom;
use super::line_world::LineWorld;
use crate::core::envs::MonteCarloEnvironment; // pour step(), score(), etc.

/// Retourne la position actuelle de l'agent
pub fn current_state(env: &LineWorld) -> usize {
    env.agent_pos
}

/// Effectue l'action dans l'environnement et retourne (reward, next_state)
pub fn environment_step(env: &mut LineWorld, action: usize) -> (f64, usize) {
    env.step(action);
    let reward = env.score();
    let next_state = env.agent_pos;
    (reward, next_state)
}

/// Choisit une action selon une stratégie ε‑greedy
pub fn choose_action(
    q: &HashMap<(usize, usize), f64>,
    s: usize,
    actions: &[usize],
    epsilon: f64
) -> usize {
    let mut rng = rand::thread_rng();

    if rng.gen::<f64>() < epsilon {
        // exploration
        *actions.choose(&mut rng).unwrap()
    } else {
        // exploitation
        let mut best_action = actions[0];
        let mut best_value = f64::MIN;
        for &a in actions {
            let val = *q.get(&(s, a)).unwrap_or(&0.0);
            if val > best_value {
                best_value = val;
                best_action = a;
            }
        }
        best_action
    }
}
