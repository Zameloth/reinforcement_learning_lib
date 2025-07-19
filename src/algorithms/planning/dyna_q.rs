use crate::algorithms::planning::helpers::choose_action;
use crate::core::envs::MonteCarloEnvironment;
use rand::seq::IteratorRandom;
use std::collections::HashMap;

type State = usize;
type Action = usize;

pub fn dyna_q(
    env: &mut dyn MonteCarloEnvironment,
    states: &[State],
    actions: &[Action],
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    n: usize,
    episodes: usize, // 👈 paramètre ajouté
) {
    let mut q: HashMap<(State, Action), f64> = HashMap::new();
    let mut model: HashMap<(State, Action), (f64, State)> = HashMap::new();
    let rng = &mut rand::rng();

    for &s in states {
        for &a in actions {
            q.insert((s, a), 0.0);
            model.insert((s, a), (0.0, s));
        }
    }

    for episode in 1..=episodes {
        println!("=== Épisode {} ===", episode);

        env.reset();

        while !env.is_game_over() {
            let actions = &*env.available_actions();
            let s = env.state_id();
            let a = choose_action(&q, s, actions, epsilon);
            let (s_prime, r) = env.step(a);

            println!(
                "State: {}, Action: {}, Reward: {}, Next State: {}",
                s, a, r, s_prime
            );

            let q_sa = *q.get(&(s, a)).unwrap();
            let max_q_sprime = actions
                .iter()
                .map(|&ap| *q.get(&(s_prime, ap)).unwrap_or(&0.0))
                .fold(f64::MIN, f64::max);

            q.insert((s, a), q_sa + alpha * (r + gamma * max_q_sprime - q_sa));

            model.insert((s, a), (r, s_prime));

            for _ in 0..n {
                let &(sp, ap) = model.keys().choose(rng).unwrap();
                let (rp, s_primep) = model[&(sp, ap)];

                let q_sap = *q.get(&(sp, ap)).unwrap();
                let max_q_sprimep = actions
                    .iter()
                    .map(|&ap2| *q.get(&(s_primep, ap2)).unwrap_or(&0.0))
                    .fold(f64::MIN, f64::max);

                q.insert(
                    (sp, ap),
                    q_sap + alpha * (rp + gamma * max_q_sprimep - q_sap),
                );
            }
        }

        println!("Q-table à la fin de l’épisode {} :", episode);
        for ((state, action), val) in &q {
            println!("Q[({}, {})] = {:.3}", state, action, val);
        }
    }
}
