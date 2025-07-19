use reinforcement_learning_lib::algorithms::dp::policy_iteration::policy_iteration;
use reinforcement_learning_lib::algorithms::dp::value_iteration::value_iteration;
use reinforcement_learning_lib::core::manual_run::run_policy;
use reinforcement_learning_lib::environments::pierre_feuille_ciseaux::{pierre_feuille_ciseaux_dp, PierreFeuilleCiseaux};

fn main(){
    let mut env = pierre_feuille_ciseaux_dp();
    let (policy, value) = policy_iteration(&mut env, 0.0001, 0.99999, 1_000);

    println!("{:?}", policy);
    println!("{:?}", value);

    let (policy, value) = value_iteration(&mut env, 0.0001, 0.99999, 1_000);
    println!("{:?}", policy);
    println!("{:?}", value);

    run_policy(&mut PierreFeuilleCiseaux::new(), &policy);
}