use crate::algorithms::planning::helpers::{choose_action, build_policy};
use crate::core::envs::MonteCarloEnvironment;
use crate::core::policies::DeterministicPolicy;

use std::collections::HashMap;

type State = usize;
type Action = usize;
type QTable = HashMap<(State, Action), f64>;

/// SARSA (on-policy TD control) with tracking of total reward per episode
/// Q(s,a) ← Q + α [r + γ Q(s',a') − Q]
pub fn sarsa(
    env: &mut dyn MonteCarloEnvironment,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    episodes: usize,
) -> (DeterministicPolicy, Vec<f64>) {
    // États et actions globaux pour la construction de la policy
    let all_states  = (0..env.num_states()).collect::<Vec<_>>();
    env.reset();
    let all_actions = env.available_actions();

    let mut q: QTable = HashMap::new();
    let mut rewards_per_episode = Vec::with_capacity(episodes);

    for ep in 1..=episodes {
        if ep % 100 == 0 {
            println!("=== Épisode {} ===", ep);
        }        let total_reward = run_episode(env, &mut q, alpha, gamma, epsilon);
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

    // initial state and action
    let mut s = env.state_id();
    let actions = env.available_actions();
    let mut a = choose_action(q, s, &actions, epsilon);

    while !env.is_game_over() {
        env.step(a);
        let r = env.score();
        total_reward += r;
        let s_next = env.state_id();

        if env.is_game_over() {
            update_terminal(q, s, a, r, alpha);
            break;
        }

        let next_actions = env.available_actions();
        let a_next = choose_action(q, s_next, &next_actions, epsilon);
        update_q(q, s, a, r, s_next, a_next, gamma, alpha);

        s = s_next;
        a = a_next;
    }

    total_reward
}

/// Q(s,a) += α [r + γ Q(s',a') − Q(s,a)]
fn update_q(
    q: &mut QTable,
    s: State,
    a: Action,
    reward: f64,
    s_next: State,
    a_next: Action,
    gamma: f64,
    alpha: f64,
) {
    let q_sa   = *q.get(&(s, a)).unwrap_or(&0.0);
    let q_next = *q.get(&(s_next, a_next)).unwrap_or(&0.0);
    let new_q = q_sa + alpha * (reward + gamma * q_next - q_sa);
    q.insert((s, a), new_q);
}

/// Si s' est terminal, Q(s,a) += α [r − Q(s,a)]
fn update_terminal(
    q: &mut QTable,
    s: State,
    a: Action,
    reward: f64,
    alpha: f64,
) {
    let q_sa = *q.get(&(s, a)).unwrap_or(&0.0);
    let new_q = q_sa + alpha * (reward - q_sa);
    q.insert((s, a), new_q);
}
