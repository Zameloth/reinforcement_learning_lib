use crate::algorithms::planning::helpers::{choose_action, build_policy};
use crate::core::envs::MonteCarloEnvironment;
use crate::core::policies::DeterministicPolicy;

use std::collections::HashMap;

type State = usize;
type Action = usize;
type QTable = HashMap<(State, Action), f64>;

pub fn expected_sarsa(
    env: &mut dyn MonteCarloEnvironment,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    episodes: usize,
) -> DeterministicPolicy {
    // États et actions globaux pour la construction de la policy
    let all_states  = (0..env.num_states()).collect::<Vec<_>>();
    env.reset();
    let all_actions = env.available_actions();

    let mut q: QTable = HashMap::new();

    for ep in 1..=episodes {
        println!("=== Épisode {} ===", ep);
        run_episode(env, &mut q, alpha, gamma, epsilon);
    }

    build_policy(&q, &all_states, &all_actions, env)
}

fn run_episode(
    env: &mut dyn MonteCarloEnvironment,
    q: &mut QTable,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
) {
    env.reset();
    while !env.is_game_over() {
        let s       = env.state_id();
        let actions = env.available_actions();
        let a       = choose_action(q, s, &actions, epsilon);

        env.step(a);
        let s_next  = env.state_id();
        let reward  = env.score();
        println!("S={} A={} R={} S'={}", s, a, reward, s_next);

        let next_actions = env.available_actions();
        let expected_q   = compute_expected_q(q, s_next, &next_actions, epsilon);

        update_q(q, s, a, reward, expected_q, gamma, alpha);
    }
}

/// Calcule E[Q(s',·)] sous la politique ε-greedy
fn compute_expected_q(
    q: &QTable,
    s_next: State,
    actions: &[Action],
    epsilon: f64,
) -> f64 {
    // choisir l'action greedy (max Q)
    let greedy = actions.iter().copied()
        .max_by(|&a1, &a2| {
            q.get(&(s_next,a1)).unwrap_or(&0.0)
                .partial_cmp(q.get(&(s_next,a2)).unwrap_or(&0.0))
                .unwrap()
        })
        .unwrap_or(actions[0]);

    let n = actions.len() as f64;
    actions.iter().copied().fold(0.0, |sum, ap| {
        let p = if ap == greedy {
            1.0 - epsilon + (epsilon / n)
        } else {
            epsilon / n
        };
        sum + p * q.get(&(s_next,ap)).unwrap_or(&0.0)
    })
}

/// Mise à jour TD : Q(s,a) += α [r + γ E[Q] − Q(s,a)]
fn update_q(
    q: &mut QTable,
    s: State,
    a: Action,
    reward: f64,
    expected_q: f64,
    gamma: f64,
    alpha: f64,
) {
    let entry = q.entry((s,a)).or_insert(0.0);
    *entry += alpha * (reward + gamma * expected_q - *entry);
}
