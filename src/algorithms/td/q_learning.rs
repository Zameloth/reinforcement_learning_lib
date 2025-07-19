use crate::algorithms::planning::helpers::{choose_action, build_policy};
use crate::core::envs::MonteCarloEnvironment;
use crate::core::policies::DeterministicPolicy;

use std::collections::HashMap;

type State = usize;
type Action = usize;

/// Q‑Learning (off‑policy TD control)
///
/// Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
///
/// où max_a' est la meilleure action possible dans s′ (greedy)
pub fn q_learning(
    env: &mut dyn MonteCarloEnvironment,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    episodes: usize,
) -> DeterministicPolicy {
    let states = (0..env.num_states()).collect::<Vec<_>>();
    let actions = env.available_actions();

    let mut q: HashMap<(State, Action), f64> = HashMap::new();

    for &s in &states {
        for &a in &actions {
            q.insert((s, a), 0.0);
        }
    }

    for episode in 1..=episodes {
        println!("=== Épisode {} ===", episode);
        env.reset();

        while !env.is_game_over() {
            let s = env.state_id();
            let actions = env.available_actions();
            let a = choose_action(&q, s, &actions, epsilon);

            env.step(a);
            let s_prime = env.state_id();
            let r = env.score();

            println!("State: {}, Action: {}, Reward: {}, Next State: {}", s, a, r, s_prime);

            let q_sa = *q.get(&(s, a)).unwrap();

            let actions_prime = env.available_actions();
            let max_q_sprime = actions_prime
                .iter()
                .map(|&ap| *q.get(&(s_prime, ap)).unwrap_or(&0.0))
                .fold(f64::MIN, f64::max);

            q.insert(
                (s, a),
                q_sa + alpha * (r + gamma * max_q_sprime - q_sa),
            );
        }
    }

    build_policy(&q, &states, &actions, env)
}
