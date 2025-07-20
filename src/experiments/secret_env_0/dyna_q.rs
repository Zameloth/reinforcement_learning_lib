use reinforcement_learning_lib::core::utils::{run_experiment, Config};

fn main() {
    let config = Config {
        env_name: "secret_0".into(),
        algorithm: "dyna_q".into(),
        alpha: 0.01,
        epsilon: 0.1,
        gamma: 0.999,
        theta: 1e-4,
        kappa: 0.001,
        max_iter: 100_000,
        planning_steps: 30,
        output_dir: "output/secret_env_0/dyna_q/".into(),
    };

    run_experiment(&config);
}
