use reinforcement_learning_lib::core::utils::{run_experiment, Config};

fn main() {
    let config = Config {
        env_name: "secret_0".into(),
        algorithm: "sarsa".into(),
        alpha: 0.1,
        epsilon: 0.3,
        gamma: 0.999,
        theta: 1e-4,
        kappa: 0.001,
        max_iter: 1_000_00,
        planning_steps: 10,
        output_dir: "output/secret_env_0/sarsa/".into(),
    };

    run_experiment(&config);
}
