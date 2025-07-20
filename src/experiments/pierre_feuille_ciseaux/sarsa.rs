use reinforcement_learning_lib::core::utils::{run_experiment, Config};

fn main() {
    let config = Config {
        env_name: "pierre_feuille_ciseaux".into(),
        algorithm: "sarsa".into(),
        alpha: 0.1,
        epsilon: 0.1,
        gamma: 0.999,
        theta: 1e-4,
        kappa: 0.001,
        max_iter: 1_000_000,
        planning_steps: 10,
        output_dir: "output/pierre_feuille_ciseaux/sarsa/".into(),
    };

    run_experiment(&config);
}
