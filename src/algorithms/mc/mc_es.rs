use crate::core::envs::MonteCarloEnvironment;
use rand::rng;
use rand::seq::IndexedRandom;
use crate::core::policies::DeterministicPolicy;
use crate::core::envs::Environment;
use crate::environments::line_world::LineWorld;

pub fn monte_carlo_es(
    env: &mut dyn MonteCarloEnvironment,
    episodes: usize,
    gamma: f64,
) -> (DeterministicPolicy, Vec<Vec<f64>>) {
    let num_states = env.num_states();
    let num_actions = env.num_actions();
    let mut q = vec![vec![0.0; num_actions]; num_states];
    let mut returns_count = vec![vec![0; num_actions]; num_states];
    let mut policy = DeterministicPolicy::new_det_pol_mc(env);
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

                //  Mise à jour avec set_action
                let best_action = (0..num_actions)
                    .max_by(|&a1, &a2| q[s][a1].partial_cmp(&q[s][a2]).unwrap())
                    .unwrap();
                policy.set_action(&s, best_action);
            }
        }
    }

    (policy, q)
}
#[test]
fn test_monte_carlo_es_runs_and_learns() {
    let mut env = LineWorld::new();


    let (policy, q) = monte_carlo_es(&mut env, 10_000, 0.9);


    println!("\n=== Q-values ===");
    for (s, actions) in q.iter().enumerate() {
        println!("État {}: {:?}", s, actions);
    }

    // EXPORT CSV
    use std::fs::File;
    use std::io::{Write, BufWriter};

    let file = File::create("q_values_mc_es.csv").expect("Erreur création fichier");
    let mut writer = BufWriter::new(file);

    writeln!(writer, "state,action,q_value").unwrap();

    for (s, actions) in q.iter().enumerate() {
        for (a, &q_val) in actions.iter().enumerate() {
            writeln!(writer, "{},{},{}", s, a, q_val).unwrap();
        }
    }
    println!("✅ Fichier exporté : q_values_mc_es.csv");

    //  VÉRIFICATIONS
    assert_eq!(policy.policy_table.len(), env.num_states());


    for s in 0..env.num_states() {
        let action = policy.get_action(&s);
        assert!(
            action < env.num_actions(),
            "Action hors borne pour état {} : {}", s, action
        );
    }

    for actions in q.iter() {
        for &value in actions {
            assert!(value.is_finite(), "Q-value non finie détectée !");
        }
    }

    //
    let s = 3;
    let best_action = policy.get_action(&s);
    assert!(
        q[s][best_action] > 0.5,
        "Q-value trop faible à l'état 3 : {}", q[s][best_action]
    );
}
