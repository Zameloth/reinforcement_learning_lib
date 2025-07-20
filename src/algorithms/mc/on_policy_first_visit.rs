use crate::core::envs::MonteCarloEnvironment;
use crate::core::policies::{DeterministicPolicy, Policy};
use rand::prelude::IndexedRandom;
use rand::{rng, Rng};

pub fn on_policy_first_visit_mc_control(
    env: &mut dyn MonteCarloEnvironment,
    episodes: usize,
    gamma: f64,
    epsilon: f64,
) -> (DeterministicPolicy, Vec<Vec<f64>>) {
    let num_states = env.num_states();
    let num_actions = env.num_actions();
    let mut q = vec![vec![0.0; num_actions]; num_states];
    let mut returns_count = vec![vec![0; num_actions]; num_states];
    let mut policy = DeterministicPolicy::new_det_pol_mc(env);

    let mut rng = rng();

    for _ in 0..episodes {
        env.reset();
        let mut episode = Vec::new();
        while !env.is_game_over() {
            let s = env.state_id();
            let a = if rng.random::<f64>() < epsilon {
                *env.available_actions().choose(&mut rng).unwrap()
            } else {
                policy.get_action(&s)
            };
            env.step(a);
            let r = env.score();
            episode.push((s, a, r));
        }

        let mut g = 0.0;
        for i in (0..episode.len()).rev() {
            let (s, a, r) = episode[i];
            g = gamma * g + r;
            if !episode[..i].iter().any(|(s2, a2, _)| *s2 == s && *a2 == a) {
                returns_count[s][a] += 1;
                let alpha = 1.0 / returns_count[s][a] as f64;
                q[s][a] += alpha * (g - q[s][a]);
                let best_action = (0..num_actions)
                    .max_by(|&a1, &a2| q[s][a1].partial_cmp(&q[s][a2]).unwrap())
                    .unwrap();
                policy.set_action(&s, best_action);
            }
        }
    }

    (policy, q)
}
