use reinforcement_learning_lib::core::envs::MonteCarloEnvironment;
use reinforcement_learning_lib::core::manual_run::run_manual;
use reinforcement_learning_lib::environments::grid_world::GridWorld;
use reinforcement_learning_lib::environments::line_world::LineWorld;
use reinforcement_learning_lib::environments::pierre_feuille_ciseaux::PierreFeuilleCiseaux;

fn main(){
    let env: &mut dyn MonteCarloEnvironment = &mut PierreFeuilleCiseaux::new();

    run_manual(env);
}