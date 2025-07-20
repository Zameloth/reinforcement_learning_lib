use reinforcement_learning_lib::core::utils::{run_experiment, Config};

fn main() {
    let config = Config {
        env_name: "monty_hall_lvl1".into(),
        algorithm: "exp_sarsa".into(),
        alpha: 0.1,
        epsilon: 0.1,
        gamma: 0.99,
        theta: 1e-4,
        kappa: 0.001,
        max_iter: 1_000_000,
        planning_steps: 10,
        output_dir: "output/monty_hall_lvl1/exp_sarsa/".into(),
    };

    run_experiment(&config);
}
