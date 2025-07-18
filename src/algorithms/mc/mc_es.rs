use crate::core::envs::MonteCarloEnvironment;
use rand::{thread_rng, Rng};

pub fn monte_carlo_es(
    env: &mut dyn MonteCarloEnvironment,
    episodes: usize,
    gamma: f64
) -> (Vec<usize>, Vec<Vec<f64>>) {
    let num_states = env.num_states();
    let num_actions = env.num_actions();
    let mut q = vec![vec![0.0; num_actions]; num_states];
    let mut returns_count = vec![vec![0; num_actions]; num_states];
    let mut policy = vec![0; num_states];

    let mut rng = thread_rng();

    for _ in 0..episodes {
        let s0 = rng.random_range(0..num_states);
        let a0 = rng.random_range(0..num_actions);
        env.force_state_and_action(s0, a0);

        let mut episode = Vec::new();
        while !env.is_game_over() {
            let s = env.state_id();
            let a = rng.gen_range(0..num_actions);
            env.step(a);
            let r = env.score();
            episode.push((s, a, r));
        }

        let mut g = 0.0;
        for &(s, a, r) in episode.iter().rev() {
            g = gamma * g + r;
            if !episode.iter().rev().skip(1).any(|(s2, a2, _)| *s2 == s && *a2 == a) {
                returns_count[s][a] += 1;
                let alpha = 1.0 / returns_count[s][a] as f64;
                q[s][a] += alpha * (g - q[s][a]);
                policy[s] = (0..num_actions)
                    .max_by(|&a1, &a2| q[s][a1].partial_cmp(&q[s][a2]).unwrap())
                    .unwrap();
            }
        }
    }

    (policy, q)
}