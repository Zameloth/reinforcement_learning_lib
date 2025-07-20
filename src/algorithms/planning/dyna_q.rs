use crate::algorithms::planning::helpers::{build_policy, choose_action};
use crate::core::envs::MonteCarloEnvironment;
use crate::core::policies::DeterministicPolicy;
use rand::prelude::StdRng;
use rand::seq::IteratorRandom;
use rand::SeedableRng;
use std::collections::HashMap;

type State = usize;
type Action = usize;
type QTable = HashMap<(State, Action), f64>;
type Model = HashMap<(State, Action), (f64, State)>;

/// Dyna-Q with tracking of total reward per episode
pub fn dyna_q(
    env: &mut dyn MonteCarloEnvironment,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    planning_steps: usize,
    episodes: usize,
) -> (DeterministicPolicy, Vec<f64>) {
    let mut q = QTable::new();
    let mut model = Model::new();
    let mut rng = <StdRng as SeedableRng>::seed_from_u64(0);
    let mut rewards_per_episode = Vec::with_capacity(episodes);

    for ep in 1..=episodes {
        println!("=== Épisode {} ===", ep);
        // Exécute l'épisode et récupère la récompense totale
        let total_reward = run_episode(
            env,
            &mut q,
            &mut model,
            alpha,
            gamma,
            epsilon,
            planning_steps,
            &mut rng,
        );
        rewards_per_episode.push(total_reward);
    }

    // Construction de la policy basée sur QTable
    let states: Vec<usize> = q.keys().map(|&(s, _)| s).collect();
    let actions: Vec<usize> = q.keys().map(|&(_, a)| a).collect();
    let policy = build_policy(&q, &states, &actions, env);
    (policy, rewards_per_episode)
}

fn run_episode<R: rand::Rng>(
    env: &mut dyn MonteCarloEnvironment,
    q: &mut QTable,
    model: &mut Model,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    planning_steps: usize,
    rng: &mut R,
) -> f64 {
    env.reset();
    let mut total_reward = 0.0;

    while !env.is_game_over() {
        let s = env.state_id();
        let a = choose_action(q, s, &env.available_actions(), epsilon);

        env.step(a);
        let s_next = env.state_id();
        let reward = env.score();
        total_reward += reward;
        println!("S={} A={} R={} S'={}", s, a, reward, s_next);

        update_q(q, s, a, reward, s_next, gamma, alpha);
        model.insert((s, a), (reward, s_next));

        for _ in 0..planning_steps {
            planning_step(q, model, gamma, alpha, rng);
        }
    }

    total_reward
}

/// Q-update selon TD : Q(s,a) ← Q + α [r + γ max Q(s',·) – Q]
fn update_q(
    q: &mut QTable,
    s: State,
    a: Action,
    reward: f64,
    s_next: State,
    gamma: f64,
    alpha: f64,
) {
    let q_sa = *q.get(&(s, a)).unwrap_or(&0.0);
    let max_q_next = q
        .iter()
        .filter(|&(&(s2, _), _)| s2 == s_next)
        .map(|(_, &v)| v)
        .fold(0.0, f64::max);

    let td = reward + gamma * max_q_next - q_sa;
    q.insert((s, a), q_sa + alpha * td);
}

/// Simule une transition passée tirée au hasard du modèle
fn planning_step<R: rand::Rng>(q: &mut QTable, model: &Model, gamma: f64, alpha: f64, rng: &mut R) {
    let (&(s, a), &(r, s_next)) = model.iter().choose(rng).unwrap();
    update_q(q, s, a, r, s_next, gamma, alpha);
}
