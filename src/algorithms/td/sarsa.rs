use crate::algorithms::planning::helpers::{choose_action, build_policy};
use crate::core::envs::MonteCarloEnvironment;
use crate::core::policies::DeterministicPolicy;

use std::collections::HashMap;

type State = usize;
type Action = usize;

/// SARSA (on‑policy TD control)
///
/// Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') − Q(s,a) ]
///
/// où a′ est l’action effectivement choisie dans s′ (ε‑greedy)
pub fn sarsa(
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

        let mut s = env.state_id();
        let mut actions = env.available_actions();
        let mut a = choose_action(&q, s, &actions, epsilon);

        while !env.is_game_over() {
            env.step(a);
            let r = env.score();
            let s_prime = env.state_id();

            println!(
                "State: {}, Action: {}, Reward: {}, Next State: {}",
                s, a, r, s_prime
            );

            let q_sa = *q.get(&(s, a)).unwrap();

            if env.is_game_over() {
                q.insert((s, a), q_sa + alpha * (r - q_sa));
                break;
            }

            let actions_prime = env.available_actions();
            let a_prime = choose_action(&q, s_prime, &actions_prime, epsilon);
            let q_saprime = *q.get(&(s_prime, a_prime)).unwrap_or(&0.0);

            q.insert(
                (s, a),
                q_sa + alpha * (r + gamma * q_saprime - q_sa)
            );

            s = s_prime;
            a = a_prime;
        }
    }

    build_policy(&q, &states, &actions, env)
}
