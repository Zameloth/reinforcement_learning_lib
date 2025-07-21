use reinforcement_learning_lib::core::utils::{run_experiment, Config};

fn main() {
    let config = Config {
        env_name: "pierre_feuille_ciseaux".into(),
        algorithm: "policy_iteration".into(),
        alpha: 0.1,
        epsilon: 0.1,
        gamma: 0.99,
        theta: 1e-4,
        kappa: 0.001,
        max_iter: 1000000,
        planning_steps: 10,
        output_dir: "output/pierre_feuille_ciseaux/policy_iteration".into(),
    };

    run_experiment(&config);
}
