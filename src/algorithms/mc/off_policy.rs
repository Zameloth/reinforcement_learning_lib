use crate::core::envs::MonteCarloEnvironment;
use crate::core::policies::DeterministicPolicy;
use rand::prelude::{IndexedRandom, StdRng};
use rand::{Rng, SeedableRng};

/// Off-policy Monte Carlo Control with Importance Sampling and tracking total reward per episode
pub fn off_policy_mc_control(
    env: &mut dyn MonteCarloEnvironment,
    episodes: usize,
    gamma: f64,
    epsilon_behavior: f64,
) -> (DeterministicPolicy, Vec<Vec<f64>>, Vec<f64>) {
    let num_states = env.num_states();
    let num_actions = env.num_actions();
    let mut q = vec![vec![0.0; num_actions]; num_states];
    let mut c = vec![vec![0.0; num_actions]; num_states];
    let mut policy = DeterministicPolicy::new_det_pol(env);
    let mut rewards_per_episode = Vec::with_capacity(episodes);
    let mut rng = <StdRng as SeedableRng>::seed_from_u64(0);

    for ep in 0..episodes {
        if ep % 100 == 0 {
            println!("=== Épisode {} ===", ep);
        }
        env.reset();
        let mut episode = Vec::new();
        let mut total_reward = 0.0;

        // Génération de l'épisode suivant la politique de comportement eps-greedy
        while !env.is_game_over() {
            let s = env.state_id();
            let a = if rng.random::<f64>() < epsilon_behavior {
                *env.available_actions().choose(&mut rng).unwrap()
            } else {
                policy.get_action(&s)
            };
            env.step(a);
            let r = env.score();
            episode.push((s, a, r));
            total_reward += r;
        }

        // Sauvegarde de la récompense totale de l'épisode
        rewards_per_episode.push(total_reward);

        // Mise à jour off-policy par importance sampling
        let mut g = 0.0;
        let mut w = 1.0;
        for &(s, a, r) in episode.iter().rev() {
            g = gamma * g + r;
            c[s][a] += w;
            let alpha = w / c[s][a];
            q[s][a] += alpha * (g - q[s][a]);

            // Mise à jour de la politique cible
            let best_action = (0..num_actions)
                .max_by(|&a1, &a2| q[s][a1].partial_cmp(&q[s][a2]).unwrap())
                .unwrap();
            policy.set_action(&s, best_action);

            if a != best_action {
                break;
            }
            w /= epsilon_behavior / ((1.0 - epsilon_behavior) / (num_actions as f64 - 1.0));
        }
    }

    (policy, q, rewards_per_episode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::envs::Environment;
    use crate::environments::line_world::LineWorld;

    #[test]
    fn test_off_policy_mc_control_learns() {
        let mut env = LineWorld::new();

        let (policy, q, rewards) = off_policy_mc_control(&mut env, 10_000, 0.9, 0.1);
        assert_eq!(rewards.len(), 10_000);

        // Export CSV des Q-values
        use std::fs::File;
        use std::io::{BufWriter, Write};
        let mut file = BufWriter::new(File::create("q_values_off_policy.csv").unwrap());
        writeln!(file, "state,action,q_value").unwrap();
        for (s, actions) in q.iter().enumerate() {
            for (a, &q_val) in actions.iter().enumerate() {
                writeln!(file, "{},{},{}", s, a, q_val).unwrap();
            }
        }

        // Vérification politique
        assert_eq!(policy.policy_table.len(), env.num_states());
        for s in 0..env.num_states() {
            let a = policy.get_action(&s);
            assert!(a < env.num_actions());
        }

        // Q-values finies
        for actions in &q {
            for &value in actions {
                assert!(value.is_finite());
            }
        }

        // Q-value élevée à un état positif
        let s = 3;
        let best = policy.get_action(&s);
        assert!(q[s][best] > 0.5, "Q-value trop faible: {}", q[s][best]);
    }
}
