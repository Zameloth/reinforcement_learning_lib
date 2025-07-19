use crate::algorithms::planning::helpers::{choose_action, build_policy};
use crate::core::envs::MonteCarloEnvironment;
use crate::core::policies::DeterministicPolicy;

use rand::seq::IteratorRandom;
use std::collections::HashMap;

type State = usize;
type Action = usize;

pub fn dyna_q_plus(
    env: &mut dyn MonteCarloEnvironment,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    kappa: f64,
    n: usize,
    episodes: usize,
) -> DeterministicPolicy {
    let states = (0..env.num_states()).collect::<Vec<_>>();
    let actions = env.available_actions();

    let mut q = HashMap::new();
    let mut model = HashMap::new();
    let mut tau = HashMap::new();
    let rng = &mut rand::rng();

    for &s in &states {
        for &a in &actions {
            q.insert((s, a), 0.0);
            model.insert((s, a), (0.0, s));
            tau.insert((s, a), 0);
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

            let q_sa = *q.get(&(s, a)).unwrap();
            let max_q_sprime = actions.iter()
                .map(|&ap| *q.get(&(s_prime, ap)).unwrap_or(&0.0))
                .fold(f64::MIN, f64::max);

            q.insert((s, a), q_sa + alpha * (r + gamma * max_q_sprime - q_sa));
            model.insert((s, a), (r, s_prime));
            tau.insert((s, a), 0);

            for ((state, action), t) in tau.iter_mut() {
                if (*state, *action) != (s, a) {
                    *t += 1;
                }
            }

            for _ in 0..n {
                let &(sp, ap) = model.keys().choose(rng).unwrap();
                let (rp, s_primep) = model[&(sp, ap)];

                let tau_val = tau[&(sp, ap)] as f64;
                let bonus = kappa * tau_val.sqrt();

                let max_q_sprimep = actions.iter()
                    .map(|&ap2| *q.get(&(s_primep, ap2)).unwrap_or(&0.0))
                    .fold(f64::MIN, f64::max);

                let q_sap = *q.get(&(sp, ap)).unwrap();
                q.insert(
                    (sp, ap),
                    q_sap + alpha * ((rp + bonus) + gamma * max_q_sprimep - q_sap),
                );
            }
        }
    }

    build_policy(&q, &states, &actions, env)
}
