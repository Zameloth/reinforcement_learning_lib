use crate::core::envs::MonteCarloEnvironment;
use crate::core::policies::DeterministicPolicy;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::SeedableRng;

/// Monte Carlo Exploring Starts control with tracking of total reward per episode
pub fn monte_carlo_es(
    env: &mut dyn MonteCarloEnvironment,
    episodes: usize,
    gamma: f64,
) -> (DeterministicPolicy, Vec<Vec<f64>>, Vec<f64>) {
    let num_states = env.num_states();
    let num_actions = env.num_actions();
    let mut q = vec![vec![0.0; num_actions]; num_states];
    let mut returns_count = vec![vec![0; num_actions]; num_states];
    let mut policy = DeterministicPolicy::new_det_pol(env);
    // Vector to store total (undiscounted) reward per episode
    let mut rewards_per_episode = Vec::with_capacity(episodes);
    let mut rng = <StdRng as SeedableRng>::seed_from_u64(0);

    for ep in 0..episodes {
        if ep % 100 == 0 {
            println!("=== Épisode {} ===", ep);
        }
        // Start from a random state for ES
        env.start_from_random_state();

        let mut episode = Vec::new();
        let mut total_reward = 0.0;

        // Génération de l'épisode
        while !env.is_game_over() {
            let s = env.state_id();
            let a = *env.available_actions().choose(&mut rng).unwrap();
            env.step(a);
            let r = env.score();
            episode.push((s, a, r));
            total_reward += r; // accumulation de la récompense
        }

        // Sauvegarde de la récompense totale de l'épisode
        rewards_per_episode.push(total_reward);

        // Mise à jour des valeurs Q et de la politique
        let mut g = 0.0;
        let mut visited = Vec::new();
        for &(s, a, r) in episode.iter().rev() {
            g = gamma * g + r;
            if !visited.contains(&(s, a)) {
                visited.push((s, a));
                returns_count[s][a] += 1;
                let alpha = 1.0 / returns_count[s][a] as f64;
                q[s][a] += alpha * (g - q[s][a]);

                // Mise à jour de la politique
                let best_action = (0..num_actions)
                    .filter(|&a| !env.is_forbidden(a))
                    .max_by(|&a1, &a2| q[s][a1].partial_cmp(&q[s][a2]).unwrap())
                    .unwrap();
                policy.set_action(&s, best_action);
            }
        }
    }

    (policy, q, rewards_per_episode)
}
