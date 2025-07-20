use crate::algorithms::planning::helpers::{build_policy, choose_action};
use crate::core::envs::MonteCarloEnvironment;
use crate::core::policies::DeterministicPolicy;

use std::collections::HashMap;

type State = usize;
type Action = usize;
type QTable = HashMap<(State, Action), f64>;

/// Expected SARSA with tracking of total reward per episode
pub fn expected_sarsa(
    env: &mut dyn MonteCarloEnvironment,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    episodes: usize,
) -> (DeterministicPolicy, Vec<f64>) {
    // États et actions globaux pour la construction de la policy
    let all_states = (0..env.num_states()).collect::<Vec<_>>();
    env.reset();
    let all_actions = env.available_actions();

    let mut q: QTable = HashMap::new();
    let mut rewards_per_episode = Vec::with_capacity(episodes);

    for ep in 1..=episodes {
        println!("=== Épisode {} ===", ep);
        let total_reward = run_episode(env, &mut q, alpha, gamma, epsilon);
        rewards_per_episode.push(total_reward);
    }

    let policy = build_policy(&q, &all_states, &all_actions, env);
    (policy, rewards_per_episode)
}

fn run_episode(
    env: &mut dyn MonteCarloEnvironment,
    q: &mut QTable,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
) -> f64 {
    env.reset();
    let mut total_reward = 0.0;

    while !env.is_game_over() {
        let s = env.state_id();
        let actions = env.available_actions();
        let a = choose_action(q, s, &actions, epsilon);

        env.step(a);
        let s_next = env.state_id();
        let reward = env.score();
        total_reward += reward;
        println!("S={} A={} R={} S'={}", s, a, reward, s_next);

        let next_actions = env.available_actions();
        let expected_q = compute_expected_q(q, s_next, &next_actions, epsilon);

        update_q(q, s, a, reward, expected_q, gamma, alpha);
    }

    total_reward
}

/// Calcule E[Q(s', ·)] sous la politique ε-greedy
fn compute_expected_q(q: &QTable, s_next: State, actions: &[Action], epsilon: f64) -> f64 {
    // Choisir l'action greedy (max Q)
    let greedy = actions
        .iter()
        .copied()
        .max_by(|&a1, &a2| {
            q.get(&(s_next, a1))
                .unwrap_or(&0.0)
                .partial_cmp(q.get(&(s_next, a2)).unwrap_or(&0.0))
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
        sum + p * q.get(&(s_next, ap)).unwrap_or(&0.0)
    })
}

/// Mise à jour TD : Q(s, a) += α [r + γ E[Q] − Q(s, a)]
fn update_q(
    q: &mut QTable,
    s: State,
    a: Action,
    reward: f64,
    expected_q: f64,
    gamma: f64,
    alpha: f64,
) {
    let entry = q.entry((s, a)).or_insert(0.0);
    *entry += alpha * (reward + gamma * expected_q - *entry);
}
