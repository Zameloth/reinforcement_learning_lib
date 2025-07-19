use crate::algorithms::planning::helpers::{choose_action, build_policy};
use crate::core::envs::MonteCarloEnvironment;
use crate::core::policies::DeterministicPolicy;

use rand::seq::IteratorRandom;
use rand::thread_rng;
use std::collections::{HashMap, HashSet};

type State = usize;
type Action = usize;
type QTable = HashMap<(State, Action), f64>;
type Model  = HashMap<(State, Action), (f64, State)>;
type Tau    = HashMap<(State, Action), usize>;

pub fn dyna_q_plus(
    env: &mut dyn MonteCarloEnvironment,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    kappa: f64,
    planning_steps: usize,
    episodes: usize,
) -> DeterministicPolicy {
    // Récupérer la liste complète des états et des actions (supposées identiques partout)
    let all_states  = (0..env.num_states()).collect::<Vec<_>>();
    env.reset();
    let all_actions = env.available_actions();

    let mut q     = QTable::new();
    let mut model = Model::new();
    let mut tau   = Tau::new();
    let mut rng   = thread_rng();

    for ep in 1..=episodes {
        println!("=== Épisode {} ===", ep);
        env.reset();

        while !env.is_game_over() {
            let s     = env.state_id();
            let a     = choose_action(&q, s, &env.available_actions(), epsilon);
            env.step(a);
            let s_n   = env.state_id();
            let r     = env.score();
            println!("S={} A={} R={} S'={}", s, a, r, s_n);

            // lazy init
            q.entry((s,a)).or_insert(0.0);
            model.entry((s,a)).or_insert((0.0, s));
            tau.entry((s,a)).or_insert(0);

            // mise à jour réelle
            update_q(&mut q, s, a, r, s_n, gamma, alpha);

            // maj modèle et τ
            model.insert((s,a), (r, s_n));
            tau.insert((s,a), 0);
            for (&sa, t) in tau.iter_mut() {
                if sa != (s,a) { *t += 1 }
            }

            // planning Dyna-Q+ avec bonus
            for _ in 0..planning_steps {
                planning_step_plus(&mut q, &model, &tau, gamma, alpha, kappa, &mut rng);
            }
        }
    }

    // Construire la policy sur tous les états/actions
    build_policy(&q, &all_states, &all_actions, env)
}

fn update_q(
    q: &mut QTable,
    s: State,
    a: Action,
    reward: f64,
    s_next: State,
    gamma: f64,
    alpha: f64,
) {
    let q_sa = *q.get(&(s,a)).unwrap_or(&0.0);
    let max_q = q.iter()
        .filter(|&(&(s2,_), _)| s2 == s_next)
        .map(|(_, &v)| v)
        .fold(0.0, f64::max);
    let td   = reward + gamma * max_q - q_sa;
    q.insert((s,a), q_sa + alpha * td);
}

fn planning_step_plus<R: rand::Rng>(
    q: &mut QTable,
    model: &Model,
    tau: &Tau,
    gamma: f64,
    alpha: f64,
    kappa: f64,
    rng: &mut R,
) {
    let (&(s,a), &(r, s_next)) = model.iter().choose(rng).unwrap();
    let bonus = kappa * (tau[&(s,a)] as f64).sqrt();
    update_q(q, s, a, r + bonus, s_next, gamma, alpha);
}
