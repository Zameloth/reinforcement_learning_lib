use crate::algorithms::planning::helpers::{choose_action, build_policy};
use crate::core::envs::MonteCarloEnvironment;
use crate::core::policies::DeterministicPolicy;

use std::collections::HashMap;

type State = usize;
type Action = usize;
type QTable = HashMap<(State, Action), f64>;

/// SARSA (on-policy TD control)
/// Q(s,a) ← Q + α [r + γ Q(s',a') − Q]
pub fn sarsa(
    env: &mut dyn MonteCarloEnvironment,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    episodes: usize,
) -> DeterministicPolicy {
    // Pour build_policy à la fin
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
    // état et action initiaux
    let mut s = env.state_id();
    let actions = env.available_actions();
    let mut a = choose_action(q, s, &actions, epsilon);

    while !env.is_game_over() {
        env.step(a);
        let r = env.score();
        let s_next = env.state_id();
        println!("S={} A={} R={} S'={}", s, a, r, s_next);

        // si terminal, on fait un dernier update sur r seul
        if env.is_game_over() {
            update_terminal(q, s, a, r, alpha);
            break;
        }

        // sinon on choisit a' et on met à jour Q(s,a)
        let next_actions = env.available_actions();
        let a_next = choose_action(q, s_next, &next_actions, epsilon);
        update_q(q, s, a, r, s_next, a_next, gamma, alpha);

        s = s_next;
        a = a_next;
    }
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
    // Lire d'abord les valeurs actuelles hors du mutable borrow
    let q_sa   = *q.get(&(s, a)).unwrap_or(&0.0);
    let q_next = *q.get(&(s_next, a_next)).unwrap_or(&0.0);

    // Calcul TD
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

