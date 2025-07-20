use crate::algorithms::dp::policy_iteration::policy_iteration;
use crate::algorithms::dp::value_iteration::value_iteration;
use crate::algorithms::mc::{
    mc_es::monte_carlo_es, off_policy::off_policy_mc_control,
    on_policy_first_visit::on_policy_first_visit_mc_control,
};
use crate::algorithms::planning::{dyna_q::dyna_q, dyna_q_plus::dyna_q_plus};
use crate::algorithms::td::{expected_sarsa::expected_sarsa, q_learning::q_learning, sarsa::sarsa};
use crate::core::envs::{DynamicProgramingEnvironment, MonteCarloEnvironment};
use crate::core::policies::{save_to_file, Policy};
use crate::environments::grid_world::dynamic_programming::grid_world;
use crate::environments::grid_world::GridWorld;
use crate::environments::line_world::{line_world_dp, LineWorld};
use crate::environments::pierre_feuille_ciseaux::{
    pierre_feuille_ciseaux_dp, PierreFeuilleCiseaux,
};
use crate::environments::secret_envs::SecretEnv;

/// Configuration de l'entraînement
///
/// # Champs
///
/// * `env_name` - Nom de l'environnement (e.g., "line_world", "grid_world", "pfs", "secret_0").
/// * `algorithm` - Algorithme à exécuter (e.g., "policy_iteration", "q_learning", "on_policy_mc").
/// * `alpha` - Taux d'apprentissage pour les algorithmes basés sur le gradient (TD, Planning).
/// * `epsilon` - Paramètre d'exploration ε pour les algorithmes à politique ε-greedy.
/// * `gamma` - Facteur d'actualisation (discount) pour la valeur future des récompenses.
/// * `theta` - Seuil de convergence pour les algorithmes de programmation dynamique.
/// * `kappa` - Bonus temporel pour l'exploration dans Dyna-Q+.
/// * `max_iter` - Nombre maximal d'itérations/épisodes à réaliser.
/// * `planning_steps` - Nombre d'étapes de planification pour Dyna-Q et Dyna-Q+.
/// * `output_dir` - Chemin du répertoire où seront sauvegardés les résultats.
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
enum ExperimentResult<P, SV, QV> {
    PolicyValues { policy: P, values: SV }, // DP: Vec<f64> state values
    PolicyOnly { policy: P },               // TD & Planning: policy seule
    PolicyQValues { policy: P, q_values: QV }, // MC: Vec<Vec<f64>> Q-values
}

/// Évalue une policy sur un environnement MC pour calculer la récompense totale moyenne
fn evaluate_policy(
    env: &mut dyn MonteCarloEnvironment,
    policy: &dyn Policy,
    num_episodes: usize,
    gamma: f64,
) -> f64 {
    let mut total_return = 0.0;
    for _ in 0..num_episodes {
        env.reset();
        let mut G = 0.0;
        let mut t = 0;
        loop {
            let state = env.state_id();
            let action = policy.get_action(&state);
            let (next_state, reward) = env.step(action);
            G += reward * gamma.powi(t);
            t += 1;
            if env.is_game_over() {
                break;
            }
        }
        total_return += G;
    }
    total_return / (num_episodes as f64)
}

/// Lance l'entraînement selon la configuration et sauvegarde les résultats
pub fn run_experiment(cfg: &Config) {
    // Initialiser les environnements DP et MC
    let mut env_dp: Box<dyn DynamicProgramingEnvironment> = match cfg.env_name.as_str() {
        "line_world" => Box::new(line_world_dp()),
        "grid_world" => Box::new(grid_world()),
        "pfs" => Box::new(pierre_feuille_ciseaux_dp()),
        "secret_0" => Box::new(SecretEnv::new(0).unwrap()),
        _ => panic!("Environnement DP inconnu: {}", cfg.env_name),
    };
    let mut env_mc: Box<dyn MonteCarloEnvironment> = match cfg.env_name.as_str() {
        "line_world" => Box::new(LineWorld::new()),
        "grid_world" => Box::new(GridWorld::new()),
        "pfs" => Box::new(PierreFeuilleCiseaux::new()),
        "secret_0" => Box::new(SecretEnv::new(0).unwrap()),
        _ => panic!("Environnement MC inconnu: {}", cfg.env_name),
    };

    // Exécuter l'algorithme choisi
    let result: ExperimentResult<_, Vec<f64>, Vec<Vec<f64>>> = match cfg.algorithm.as_str() {
        // DP algorithms return (policy, Vec<f64>)
        "policy_iteration" => {
            let (policy, values) =
                policy_iteration(&mut *env_dp, cfg.gamma, cfg.theta, cfg.max_iter);
            ExperimentResult::PolicyValues { policy, values }
        }
        "value_iteration" => {
            let (policy, values) =
                value_iteration(&mut *env_dp, cfg.gamma, cfg.theta, cfg.max_iter);
            ExperimentResult::PolicyValues { policy, values }
        }
        // MC algorithms return (policy, Vec<Vec<f64>>)
        "mc_es" => {
            let (policy, q_values) = monte_carlo_es(&mut *env_mc, cfg.max_iter, cfg.gamma);
            ExperimentResult::PolicyQValues { policy, q_values }
        }
        "on_policy_mc" => {
            let (policy, q_values) = on_policy_first_visit_mc_control(
                &mut *env_mc,
                cfg.max_iter,
                cfg.gamma,
                cfg.epsilon,
            );
            ExperimentResult::PolicyQValues { policy, q_values }
        }
        "off_policy_mc" => {
            let (policy, q_values) =
                off_policy_mc_control(&mut *env_mc, cfg.max_iter, cfg.gamma, cfg.epsilon);
            ExperimentResult::PolicyQValues { policy, q_values }
        }
        // TD & Planning return only policy
        "sarsa" => {
            let policy = sarsa(
                &mut *env_mc,
                cfg.alpha,
                cfg.gamma,
                cfg.epsilon,
                cfg.max_iter,
            );
            ExperimentResult::PolicyOnly { policy }
        }
        "expected_sarsa" => {
            let policy = expected_sarsa(
                &mut *env_mc,
                cfg.alpha,
                cfg.gamma,
                cfg.epsilon,
                cfg.max_iter,
            );
            ExperimentResult::PolicyOnly { policy }
        }
        "q_learning" => {
            let policy = q_learning(
                &mut *env_mc,
                cfg.alpha,
                cfg.gamma,
                cfg.epsilon,
                cfg.max_iter,
            );
            ExperimentResult::PolicyOnly { policy }
        }
        "dyna_q" => {
            let policy = dyna_q(
                &mut *env_mc,
                cfg.alpha,
                cfg.gamma,
                cfg.epsilon,
                cfg.planning_steps,
                cfg.max_iter,
            );
            ExperimentResult::PolicyOnly { policy }
        }
        "dyna_q_plus" => {
            let policy = dyna_q_plus(
                &mut *env_mc,
                cfg.alpha,
                cfg.gamma,
                cfg.epsilon,
                cfg.kappa,
                cfg.planning_steps,
                cfg.max_iter,
            );
            ExperimentResult::PolicyOnly { policy }
        }
        _ => panic!("Algorithme inconnu: {}", cfg.algorithm),
    };

    // Créer répertoire de sortie
    std::fs::create_dir_all(&cfg.output_dir).expect("Impossible de créer le répertoire de sortie");

    // Sauvegarde des résultats et évaluation de la policy
    // Nombre d'épisodes pour l'évaluation
    let eval_episodes = 1000;
    match result {
        ExperimentResult::PolicyValues { policy, values } => {
            save_to_file(&policy, &format!("{}/policy.json", cfg.output_dir)).unwrap();
            save_to_file(&values, &format!("{}/values.csv", cfg.output_dir)).unwrap();

            let avg_reward = evaluate_policy(&mut *env_mc, &policy, eval_episodes, cfg.gamma);
            save_to_file(&avg_reward, &format!("{}/avg_reward.txt", cfg.output_dir)).unwrap();
            println!(
                "Average total reward ({} eps): {:.4}",
                eval_episodes, avg_reward
            );
        }
        ExperimentResult::PolicyQValues { policy, q_values } => {
            save_to_file(&policy, &format!("{}/policy.json", cfg.output_dir)).unwrap();
            save_to_file(&q_values, &format!("{}/q_values.csv", cfg.output_dir)).unwrap();

            let avg_reward = evaluate_policy(&mut *env_mc, &policy, eval_episodes, cfg.gamma);
            save_to_file(&avg_reward, &format!("{}/avg_reward.txt", cfg.output_dir)).unwrap();
            println!(
                "Average total reward ({} eps): {:.4}",
                eval_episodes, avg_reward
            );
        }
        ExperimentResult::PolicyOnly { policy } => {
            save_to_file(&policy, &format!("{}/policy.json", cfg.output_dir)).unwrap();

            let avg_reward = evaluate_policy(&mut *env_mc, &policy, eval_episodes, cfg.gamma);
            save_to_file(&avg_reward, &format!("{}/avg_reward.txt", cfg.output_dir)).unwrap();
            println!(
                "Average total reward ({} eps): {:.4}",
                eval_episodes, avg_reward
            );
        }
    }

    println!(
        "Expérience terminée: env={}, alg={} -> {}",
        cfg.env_name, cfg.algorithm, cfg.output_dir
    );
}
