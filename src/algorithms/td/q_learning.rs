use std::collections::HashMap;
use rand::seq::IteratorRandom;
use rand::thread_rng;

use crate::core::envs::MonteCarloEnvironment;
use crate::environments::helpers::{current_state, environment_step, choose_action};
use crate::environments::line_world::LineWorld;

type State = usize;
type Action = usize;

/// Q‑Learning (off‑policy TD control)
///
/// Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
///
/// où max_a' est la meilleure action possible dans s′ (greedy)
pub fn q_learning(
    states: &[State],
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    episodes: usize,
) {
    let mut q: HashMap<(State, Action), f64> = HashMap::new();

    for &s in states {
        q.insert((s, 0), 0.0);
        q.insert((s, 1), 0.0);
    }

    for episode in 1..=episodes {
        println!("=== Épisode {} ===", episode);

        let mut env = LineWorld { agent_pos: 2 };
        env.reset();

        while !env.is_game_over() {
            let s = current_state(&env);
            let actions = env.available_actions();
            let a = choose_action(&q, s, &actions, epsilon);

            let (r, s_prime) = environment_step(&mut env, a);

            println!(
                "State: {}, Action: {}, Reward: {}, Next State: {}",
                s, a, r, s_prime
            );

            let q_sa = *q.get(&(s, a)).unwrap();

            let actions_prime = env.available_actions();
            let max_q_sprime = actions_prime
                .iter()
                .map(|&ap| *q.get(&(s_prime, ap)).unwrap_or(&0.0))
                .fold(f64::MIN, f64::max);

            q.insert(
                (s, a),
                q_sa + alpha * (r + gamma * max_q_sprime - q_sa)
            );
        }

        println!("Q-table à la fin de l’épisode {} :", episode);
        for ((state, action), val) in &q {
            println!("Q[({}, {})] = {:.3}", state, action, val);
        }
    }
}
