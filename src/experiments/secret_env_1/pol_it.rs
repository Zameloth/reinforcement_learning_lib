use reinforcement_learning_lib::core::utils::{run_experiment, Config};

fn main() {
    let config = Config {
        env_name: "secret_1".into(),
        algorithm: "policy_iteration".into(),
        alpha: 0.001,
        epsilon: 0.001,
        gamma: 0.9999,
        theta: 1e-4,
        kappa: 0.001,
        max_iter: 1000,
        planning_steps: 10,
        output_dir: "output/secret_env_1/policy_iteration/".into(),
    };

    run_experiment(&config);
}
