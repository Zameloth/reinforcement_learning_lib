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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::envs::{Environment, MonteCarloEnvironment};
    use crate::environments::line_world::LineWorld;

    struct LineWorldTest {
        env: LineWorld,
    }

    impl Environment for LineWorldTest {
        fn num_states(&self) -> usize {
            self.env.num_states()
        }

        fn num_actions(&self) -> usize {
            self.env.num_actions()
        }

        fn num_rewards(&self) -> usize {
            self.env.num_rewards()
        }
    }

    impl MonteCarloEnvironment for LineWorldTest {
        fn reset(&mut self) {
            self.env.reset();
        }

        fn step(&mut self, action: usize) {
            self.env.step(action);
        }

        fn score(&self) -> f64 {
            self.env.score()
        }

        fn is_game_over(&self) -> bool {
            self.env.is_game_over()
        }

        fn start_from_random_state(&mut self) {
            self.env.start_from_random_state();
        }

        fn state_id(&self) -> usize {
            self.env.agent_pos
        }

        fn display(&self) {
            self.env.display();
        }

        fn is_forbidden(&self, _action: usize) -> bool {
            false
        }
    }

    #[test]
    fn test_off_policy_mc_control_learns() {
        let mut env = LineWorldTest {
            env: LineWorld { agent_pos: 0 },
        };

        let (policy, q) = off_policy_mc_control(&mut env, 10_000, 0.9, 0.1);

        // Export CSV
        use std::fs::File;
        use std::io::{Write, BufWriter};

        let mut file = BufWriter::new(File::create("output_off_policy.csv").unwrap());

        for (s, actions) in q.iter().enumerate() {
            for (a, &q_val) in actions.iter().enumerate() {
                writeln!(file, "{},{},{}", s, a, q_val).unwrap();
            }
        }

        assert_eq!(policy.len(), env.num_states());

        for &a in &policy {
            assert!(a < env.num_actions(), "Action hors borne: {}", a);
        }

        for actions in q.iter() {
            for &value in actions {
                assert!(value.is_finite());
            }
        }

        let s = 3;
        let best_action = policy[s];
        assert!(
            q[s][best_action] > 0.5,
            "Q-value trop faible sur état positif : {}",
            q[s][best_action]
        );
    }
}
