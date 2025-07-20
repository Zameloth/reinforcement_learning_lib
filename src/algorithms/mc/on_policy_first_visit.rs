use crate::core::envs::MonteCarloEnvironment;
use crate::core::policies::DeterministicPolicy;
use rand::prelude::{IndexedRandom, StdRng};
use rand::{Rng, SeedableRng};

/// On-policy First-Visit Monte Carlo Control with tracking of total reward per episode
pub fn on_policy_first_visit_mc_control(
    env: &mut dyn MonteCarloEnvironment,
    episodes: usize,
    gamma: f64,
    epsilon: f64,
) -> (DeterministicPolicy, Vec<Vec<f64>>, Vec<f64>) {
    let num_states = env.num_states();
    let num_actions = env.num_actions();
    let mut q = vec![vec![0.0; num_actions]; num_states];
    let mut returns_count = vec![vec![0; num_actions]; num_states];
    let mut policy = DeterministicPolicy::new_det_pol(env);
    let mut rewards_per_episode = Vec::with_capacity(episodes);
    let mut rng = <StdRng as SeedableRng>::seed_from_u64(0);

    for ep in 0..episodes {
        if ep % 1000 == 0 {
            println!("=== Épisode {} ===", ep);
        }
        env.reset();
        let mut episode = Vec::new();
        let mut total_reward = 0.0;

        // Génération de l'épisode suivant la politique ε-greedy
        while !env.is_game_over() {
            let s = env.state_id();
            let a = if rng.random::<f64>() < epsilon {
                *env.available_actions().choose(&mut rng).unwrap()
            } else {
                // Si l'action de la policy est forbidden, choisi une action aléatoire
                let a = policy.get_action(&s);
                if env.is_forbidden(a) {
                    *env.available_actions().choose(&mut rng).unwrap()
                } else {
                    a
                }
            };
            env.step(a);
            let r = env.score();
            episode.push((s, a, r));
            total_reward += r;
        }

        // Stockage de la récompense totale de l'épisode
        rewards_per_episode.push(total_reward);

        // First-visit updates
        let mut g = 0.0;
        for i in (0..episode.len()).rev() {
            let (s, a, r) = episode[i];
            g = gamma * g + r;
            // Vérifier première visite
            if !episode[..i].iter().any(|&(s2, a2, _)| s2 == s && a2 == a) {
                returns_count[s][a] += 1;
                let alpha = 1.0 / returns_count[s][a] as f64;
                q[s][a] += alpha * (g - q[s][a]);

                // Mise à jour de la politique greedy
                let best_action = (0..num_actions)
                    .max_by(|&a1, &a2| q[s][a1].partial_cmp(&q[s][a2]).unwrap())
                    .unwrap();
                policy.set_action(&s, best_action);
            }
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
    fn test_on_policy_first_visit_mc_control_learns() {
        let mut env = LineWorld::new();

        let (policy, q, rewards) = on_policy_first_visit_mc_control(&mut env, 10_000, 0.9, 0.1);
        assert_eq!(rewards.len(), 10_000);

        // Export CSV des récompenses
        use std::fs::File;
        use std::io::{BufWriter, Write};
        let mut file = BufWriter::new(File::create("rewards_on_policy.csv").unwrap());
        writeln!(file, "episode,total_reward").unwrap();
        for (i, &r) in rewards.iter().enumerate() {
            writeln!(file, "{},{}", i, r).unwrap();
        }

        // Vérification politique et Q-values finies
        assert_eq!(policy.policy_table.len(), env.num_states());
        for s in 0..env.num_states() {
            let a = policy.get_action(&s);
            assert!(a < env.num_actions());
        }
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
