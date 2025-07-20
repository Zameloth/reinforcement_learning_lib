use reinforcement_learning_lib::core::envs::MonteCarloEnvironment;
use reinforcement_learning_lib::core::manual_run::run_policy;
use reinforcement_learning_lib::core::policies::{load_from_file, DeterministicPolicy};
use reinforcement_learning_lib::environments::line_world::LineWorld;

fn main(){
    let env: &mut dyn MonteCarloEnvironment = &mut LineWorld::new();
    let policy:DeterministicPolicy = load_from_file("output/line_world/policy_iteration/policy.json").unwrap();

    run_policy(env, &policy);
}