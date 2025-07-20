use crate::algorithms::planning::helpers::{choose_action, build_policy};
use crate::core::envs::MonteCarloEnvironment;
use crate::core::policies::DeterministicPolicy;

use std::collections::HashMap;

type State = usize;
type Action = usize;
type QTable = HashMap<(State, Action), f64>;

pub fn q_learning(
    env: &mut dyn MonteCarloEnvironment,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    episodes: usize,
) -> DeterministicPolicy {
    // pour build_policy à la fin
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
        let s        = env.state_id();
        let actions  = env.available_actions();
        let a        = choose_action(q, s, &actions, epsilon);

        env.step(a);
        let s_next   = env.state_id();
        let reward   = env.score();
        println!("S={} A={} R={} S'={}", s, a, reward, s_next);

        let next_actions = env.available_actions();
        let max_q_next   = compute_max_q(q, s_next, &next_actions);

        apply_q_update(q, s, a, reward, max_q_next, gamma, alpha);
    }
}

/// max_{a'} Q(s', a')
fn compute_max_q(q: &QTable, s_next: State, actions: &[Action]) -> f64 {
    actions
        .iter()
        .copied()
        .map(|ap| *q.get(&(s_next, ap)).unwrap_or(&0.0))
        .fold(0.0, f64::max)
}

/// Q(s,a) += α [r + γ max_q_next − Q(s,a)]
fn apply_q_update(
    q: &mut QTable,
    s: State,
    a: Action,
    reward: f64,
    max_q_next: f64,
    gamma: f64,
    alpha: f64,
) {
    let entry = q.entry((s, a)).or_insert(0.0);
    *entry += alpha * (reward + gamma * max_q_next - *entry);
}
