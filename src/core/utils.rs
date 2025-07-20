use std::io::{stderr, Write};
use rand::{prelude::StdRng, seq::IndexedRandom, SeedableRng};

use crate::{
    algorithms::{
        dp::{
            policy_iteration::policy_iteration,
            value_iteration::value_iteration,
        },
        mc::{
            mc_es::monte_carlo_es,
            off_policy::off_policy_mc_control,
            on_policy_first_visit::on_policy_first_visit_mc_control,
        },
        planning::{dyna_q::dyna_q, dyna_q_plus::dyna_q_plus},
        td::{
            expected_sarsa::expected_sarsa,
            q_learning::q_learning,
            sarsa::sarsa,
        },
    },
    core::{
        envs::{DynamicProgramingEnvironment, MonteCarloEnvironment},
        policies::{save_to_file, Policy},
    },
    environments::{
        grid_world::{dynamic_programming::grid_world, GridWorld},
        line_world::{line_world_dp, LineWorld},
        pierre_feuille_ciseaux::{pierre_feuille_ciseaux_dp, PierreFeuilleCiseaux},
        secret_envs::SecretEnv,
    },
};

/// Configuration de l'entraînement
pub struct Config {
    pub env_name: String,
    pub algorithm: String,
    pub alpha: f64,
    pub epsilon: f64,
    pub gamma: f64,
    pub theta: f64,
    pub kappa: f64,
    pub max_iter: usize,
    pub planning_steps: usize,
    pub output_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            env_name: "".into(),
            algorithm: "".into(),
            alpha: 0.1,
            epsilon: 0.1,
            gamma: 0.99,
            theta: 1e-4,
            kappa: 0.001,
            max_iter: 1000,
            planning_steps: 10,
            output_dir: "output/default".into(),
        }
    }
}

/// Résultat d'une expérience, pour un traitement uniforme
/// P : Policy, SV : state values, QV : Q-values, RV : rewards vector
enum ExperimentResult<P, SV, QV, RV> {
    PolicyValues { policy: P, values: SV },
    PolicyQValues { policy: P, q_values: QV, rewards: RV },
    PolicyOnly { policy: P, rewards: RV },
}

/// Évalue une policy sur un environnement MC pour calculer la récompense totale moyenne
fn evaluate_policy(
    env: &mut dyn MonteCarloEnvironment,
    policy: &dyn Policy,
    num_episodes: usize,
    gamma: f64,
) -> f64 {
    let mut total_return = 0.0;
    let mut rng = <StdRng as SeedableRng>::seed_from_u64(0);
    for _ in 0..num_episodes {
        env.reset();
        let mut g = 0.0;
        let mut t = 0;
        loop {
            let s = env.state_id();
            let a = policy.get_action(&s);
            let a = if !env.is_forbidden(a) {
                a
            } else {
                println!("Forbidden action: {}, choosing random_action", a);
                *env.available_actions().choose(&mut rng).unwrap()
            };
            let (_s_next, r) = env.step(a);
            g += r * gamma.powi(t);
            t += 1;
            if env.is_game_over() {
                break;
            }
        }
        total_return += g;
    }
    total_return / num_episodes as f64
}

/// Lance l'entraînement selon la configuration et sauvegarde les résultats
pub fn run_experiment(cfg: &Config) {
    // Init environnements
    let mut env_dp: Box<dyn DynamicProgramingEnvironment> = match cfg.env_name.as_str() {
        "line_world" => Box::new(line_world_dp()),
        "grid_world" => Box::new(grid_world()),
        "pfs" => Box::new(pierre_feuille_ciseaux_dp()),
        "secret_0" => Box::new(SecretEnv::new(0).unwrap()),
        "secret_1" => Box::new(SecretEnv::new(1).unwrap()),
        "secret_2" => Box::new(SecretEnv::new(2).unwrap()),
        _ => panic!("Environnement DP inconnu: {}", cfg.env_name),
    };
    let mut env_mc: Box<dyn MonteCarloEnvironment> = match cfg.env_name.as_str() {
        "line_world" => Box::new(LineWorld::new()),
        "grid_world" => Box::new(GridWorld::new()),
        "pfs" => Box::new(PierreFeuilleCiseaux::new()),
        "secret_0" => Box::new(SecretEnv::new(0).unwrap()),
        "secret_1" => Box::new(SecretEnv::new(1).unwrap()),
        "secret_2" => Box::new(SecretEnv::new(2).unwrap()),
        _ => panic!("Environnement MC inconnu: {}", cfg.env_name),
    };

    let start = std::time::Instant::now();
    // Exécution
    let result = match cfg.algorithm.as_str() {
        "policy_iteration" => {
            let (policy, values) = policy_iteration(&*env_dp, cfg.theta, cfg.gamma, cfg.max_iter);
            ExperimentResult::PolicyValues { policy, values }
        }
        "value_iteration" => {
            let (policy, values) = value_iteration(&*env_dp, cfg.theta, cfg.gamma, cfg.max_iter);
            ExperimentResult::PolicyValues { policy, values }
        }
        "mc_es" => {
            let (policy, q_values, rewards) = monte_carlo_es(&mut *env_mc, cfg.max_iter, cfg.gamma);
            ExperimentResult::PolicyQValues { policy, q_values, rewards }
        }
        "on_policy_mc" => {
            let (policy, q_values, rewards) = on_policy_first_visit_mc_control(
                &mut *env_mc,
                cfg.max_iter,
                cfg.gamma,
                cfg.epsilon,
            );
            ExperimentResult::PolicyQValues { policy, q_values, rewards }
        }
        "off_policy_mc" => {
            let (policy, q_values, rewards) = off_policy_mc_control(
                &mut *env_mc,
                cfg.max_iter,
                cfg.gamma,
                cfg.epsilon,
            );
            ExperimentResult::PolicyQValues { policy, q_values, rewards }
        }
        "sarsa" => {
            let (policy, rewards) = sarsa(&mut *env_mc, cfg.alpha, cfg.gamma, cfg.epsilon, cfg.max_iter);
            ExperimentResult::PolicyOnly { policy, rewards }
        }
        "expected_sarsa" => {
            let (policy, rewards) = expected_sarsa(&mut *env_mc, cfg.alpha, cfg.gamma, cfg.epsilon, cfg.max_iter);
            ExperimentResult::PolicyOnly { policy, rewards }
        }
        "q_learning" => {
            let (policy, rewards) = q_learning(&mut *env_mc, cfg.alpha, cfg.gamma, cfg.epsilon, cfg.max_iter);
            ExperimentResult::PolicyOnly { policy, rewards }
        }
        "dyna_q" => {
            let (policy, rewards) = dyna_q(&mut *env_mc, cfg.alpha, cfg.gamma, cfg.epsilon, cfg.planning_steps, cfg.max_iter);
            ExperimentResult::PolicyOnly { policy, rewards }
        }
        "dyna_q_plus" => {
            let (policy, rewards) = dyna_q_plus(&mut *env_mc, cfg.alpha, cfg.gamma, cfg.epsilon, cfg.kappa, cfg.planning_steps, cfg.max_iter);
            ExperimentResult::PolicyOnly { policy, rewards }
        }
        _ => panic!("Algorithme inconnu: {}", cfg.algorithm),
    };

    let duration = start.elapsed();
    println!("Durée de l'entraînement : {:?}", duration);

    // Création dossier sortie
    std::fs::create_dir_all(&cfg.output_dir).expect("Impossible de créer le répertoire");

    // Sauvegarde
    let eval_episodes = 1000;
    match result {
        ExperimentResult::PolicyValues { policy, values } => {
            save_to_file(&policy, &format!("{}/policy.json", cfg.output_dir)).unwrap();
            save_to_file(&values, &format!("{}/values.csv", cfg.output_dir)).unwrap();
            let avg = evaluate_policy(&mut *env_mc, &policy, eval_episodes, cfg.gamma);
            save_to_file(&avg, &format!("{}/avg_reward.txt", cfg.output_dir)).unwrap();
            println!("Avg reward ({} eps): {:.4}", eval_episodes, avg);
        }
        ExperimentResult::PolicyQValues { policy, q_values, rewards } => {
            save_to_file(&policy, &format!("{}/policy.json", cfg.output_dir)).unwrap();
            save_to_file(&q_values, &format!("{}/q_values.csv", cfg.output_dir)).unwrap();
            save_to_file(&rewards, &format!("{}/rewards.csv", cfg.output_dir)).unwrap();
            let avg = evaluate_policy(&mut *env_mc, &policy, eval_episodes, cfg.gamma);
            save_to_file(&avg, &format!("{}/avg_reward.txt", cfg.output_dir)).unwrap();
            println!("Avg reward ({} eps): {:.4}", eval_episodes, avg);
        }
        ExperimentResult::PolicyOnly { policy, rewards } => {
            save_to_file(&policy, &format!("{}/policy.json", cfg.output_dir)).unwrap();
            save_to_file(&rewards, &format!("{}/rewards.csv", cfg.output_dir)).unwrap();
            let avg = evaluate_policy(&mut *env_mc, &policy, eval_episodes, cfg.gamma);
            save_to_file(&avg, &format!("{}/avg_reward.txt", cfg.output_dir)).unwrap();
            println!("Avg reward ({} eps): {:.4}", eval_episodes, avg);
        }
    }

    println!("Expérience terminée: {} - {} -> {}", cfg.env_name, cfg.algorithm, cfg.output_dir);
}
