use crate::core::envs::MonteCarloEnvironment;
use rand::{thread_rng, Rng};

pub fn off_policy_mc_control(
    env: &mut dyn MonteCarloEnvironment,
    episodes: usize,
    gamma: f64,
    epsilon_behavior: f64,
) -> (Vec<usize>, Vec<Vec<f64>>) {
    let num_states = env.num_states();
    let num_actions = env.num_actions();
    let mut q = vec![vec![0.0; num_actions]; num_states];
    let mut c = vec![vec![0.0; num_actions]; num_states];
    let mut policy = vec![0; num_states];

    let mut rng = thread_rng();

    for _ in 0..episodes {
        env.reset();
        let mut episode = Vec::new();

        while !env.is_game_over() {
            let s = env.state_id();
            let a = if rng.gen::<f64>() < epsilon_behavior {
                rng.gen_range(0..num_actions)
            } else {
                policy[s]
            };
            env.step(a);
            let r = env.score();
            episode.push((s, a, r));
        }

        let mut g = 0.0;
        let mut w = 1.0;
        for &(s, a, r) in episode.iter().rev() {
            g = gamma * g + r;
            c[s][a] += w;
            q[s][a] += (w / c[s][a]) * (g - q[s][a]);
            policy[s] = (0..num_actions)
                .max_by(|&a1, &a2| q[s][a1].partial_cmp(&q[s][a2]).unwrap())
                .unwrap();
            if a != policy[s] {
                break;
            }
            w /= 1.0 / num_actions as f64;
        }
    }

    (policy, q)
}
