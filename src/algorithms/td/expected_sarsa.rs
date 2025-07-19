use std::collections::HashMap;
use rand::seq::IteratorRandom;
use rand::thread_rng;

use crate::core::envs::MonteCarloEnvironment;
use crate::environments::helpers::{current_state, environment_step, choose_action};
use crate::environments::line_world::LineWorld;

type State = usize;
type Action = usize;

/// Expected SARSA
/// TD Control - On-policy - Expected update
///
/// Q(s,a) ← Q(s,a) + α [ r + γ Σ_a' π(a'|s') Q(s',a') - Q(s,a) ]
///
/// où π(a'|s') est la policy ε-greedy
pub fn expected_sarsa(
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

            // calcul de l'espérance sur Q(s',a') sous policy ε-greedy
            let actions_prime = env.available_actions();
            let mut expected_q = 0.0;
            let greedy_action = actions_prime
                .iter()
                .max_by(|&&a1, &&a2| {
                    q.get(&(s_prime, a1))
                        .unwrap_or(&0.0)
                        .partial_cmp(q.get(&(s_prime, a2)).unwrap_or(&0.0))
                        .unwrap()
                })
                .cloned()
                .unwrap();

            for &ap in &actions_prime {
                let prob = if ap == greedy_action {
                    1.0 - epsilon + (epsilon / actions_prime.len() as f64)
                } else {
                    epsilon / actions_prime.len() as f64
                };
                expected_q += prob * q.get(&(s_prime, ap)).unwrap_or(&0.0);
            }

            let q_sa = *q.get(&(s, a)).unwrap();
            q.insert(
                (s, a),
                q_sa + alpha * (r + gamma * expected_q - q_sa)
            );
        }

        println!("Q-table à la fin de l’épisode {} :", episode);
        for ((state, action), val) in &q {
            println!("Q[({}, {})] = {:.3}", state, action, val);
        }
    }
}
