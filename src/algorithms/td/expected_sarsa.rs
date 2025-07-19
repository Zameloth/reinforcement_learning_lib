use crate::algorithms::planning::helpers::{choose_action, build_policy};
use crate::core::envs::MonteCarloEnvironment;
use crate::core::policies::DeterministicPolicy;

use std::collections::HashMap;

type State = usize;
type Action = usize;

/// Expected SARSA
/// TD Control - On-policy - Expected update
///
/// Q(s,a) ← Q(s,a) + α [ r + γ Σ_a' π(a'|s') Q(s',a') - Q(s,a) ]
pub fn expected_sarsa(
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

            let actions_prime = env.available_actions();

            // calcul de l'espérance sous ε-greedy dans s'
            let greedy_action = actions_prime
                .iter()
                .copied()
                .max_by(|&a1, &a2| {
                    q.get(&(s_prime, a1))
                        .unwrap_or(&0.0)
                        .partial_cmp(q.get(&(s_prime, a2)).unwrap_or(&0.0))
                        .unwrap()
                })
                .unwrap_or(actions_prime[0]);

            let mut expected_q = 0.0;

            for &ap in &actions_prime {
                let prob = if ap == greedy_action {
                    1.0 - epsilon + (epsilon / actions_prime.len() as f64)
                } else {
                    epsilon / actions_prime.len() as f64
                };
                expected_q += prob * q.get(&(s_prime, ap)).unwrap_or(&0.0);
            }

            let q_sa = *q.get(&(s, a)).unwrap();
            q.insert((s, a), q_sa + alpha * (r + gamma * expected_q - q_sa));
        }
    }

    build_policy(&q, &states, &actions, env)
}
