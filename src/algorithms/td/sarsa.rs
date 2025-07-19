use std::collections::HashMap;
use rand::seq::IteratorRandom;
use rand::thread_rng;

use crate::core::envs::MonteCarloEnvironment;
use crate::environments::helpers::{current_state, environment_step, choose_action};
use crate::environments::line_world::LineWorld;

type State = usize;
type Action = usize;

/// SARSA (on‑policy TD control)
///
/// Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') − Q(s,a) ]
///
/// où a′ est l’action effectivement choisie dans s′ (ε‑greedy)
pub fn sarsa(
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

        let mut s = current_state(&env);
        let mut actions = env.available_actions();
        let mut a = choose_action(&q, s, &actions, epsilon);

        while !env.is_game_over() {
            let (r, s_prime) = environment_step(&mut env, a);

            println!(
                "State: {}, Action: {}, Reward: {}, Next State: {}",
                s, a, r, s_prime
            );

            let q_sa = *q.get(&(s, a)).unwrap();

            if env.is_game_over() {
                q.insert((s, a), q_sa + alpha * (r - q_sa));
                break;
            }

            actions = env.available_actions();
            let a_prime = choose_action(&q, s_prime, &actions, epsilon);
            let q_saprime = *q.get(&(s_prime, a_prime)).unwrap_or(&0.0);

            q.insert(
                (s, a),
                q_sa + alpha * (r + gamma * q_saprime - q_sa)
            );

            s = s_prime;
            a = a_prime;
        }

        println!("Q-table à la fin de l’épisode {} :", episode);
        for ((state, action), val) in &q {
            println!("Q[({}, {})] = {:.3}", state, action, val);
        }
    }
}
