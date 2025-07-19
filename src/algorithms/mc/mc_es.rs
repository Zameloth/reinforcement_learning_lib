use crate::core::envs::MonteCarloEnvironment;
use rand::rng;
use rand::seq::IndexedRandom;

pub fn monte_carlo_es(
    env: &mut dyn MonteCarloEnvironment,
    episodes: usize,
    gamma: f64,
) -> (Vec<usize>, Vec<Vec<f64>>) {
    let num_states = env.num_states();
    let num_actions = env.num_actions();
    let mut q = vec![vec![0.0; num_actions]; num_states];
    let mut returns_count = vec![vec![0; num_actions]; num_states];
    let mut policy = vec![0; num_states];

    let mut rng = rng();

    for _ in 0..episodes {
        env.start_from_random_state();

        // Déroulement épisode
        let mut episode = Vec::new();
        while !env.is_game_over() {
            let s = env.state_id();
            let a = *env.available_actions().choose(&mut rng).unwrap();
            env.step(a);
            let r = env.score();
            episode.push((s, a, r));
        }

        let mut g = 0.0;
        let mut visited = vec![];

        //boucle de mise à jour
        for &(s, a, r) in episode.iter().rev() {
            g = gamma * g + r;
            if !visited.contains(&(s, a)) {
                visited.push((s, a));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::envs::{Environment};
    use crate::environments::line_world::LineWorld;

    #[test]
    fn test_monte_carlo_es_runs_and_learns() {
        let mut env = LineWorld::new();

        let (policy, q) = monte_carlo_es(&mut env, 10_000, 0.9);

        // Export CSV pour visualisation
        use std::fs::File;
        use std::io::{BufWriter, Write};
        let mut file = BufWriter::new(File::create("output_mc_es.csv").unwrap());

        for (s, actions) in q.iter().enumerate() {
            for (a, &q_val) in actions.iter().enumerate() {
                writeln!(file, "{},{},{}", s, a, q_val).unwrap();
            }
        }

        //  Vérifications
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
