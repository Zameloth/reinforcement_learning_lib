use reinforcement_learning_lib::algorithms::mc::{
    on_policy_first_visit::on_policy_first_visit_mc_control,
    off_policy::off_policy_mc_control,
    mc_es::monte_carlo_es,
};
use reinforcement_learning_lib::environments::line_world::LineWorld;
use reinforcement_learning_lib::core::envs::MonteCarloEnvironment;
use reinforcement_learning_lib::core::policies::Policy;
use reinforcement_learning_lib::core::envs::Environment;



use std::fs::File;
use std::io::{BufWriter, Write};

fn export_q_values(filename: &str, q: &[Vec<f64>]) {
    let mut file = BufWriter::new(File::create(filename).unwrap());
    for (s, actions) in q.iter().enumerate() {
        for (a, &q_val) in actions.iter().enumerate() {
            writeln!(file, "{},{},{}", s, a, q_val).unwrap();
        }
    }
}

fn display_results(q: &[Vec<f64>], policy: &[usize]) {
    for (s, actions) in q.iter().enumerate() {
        for (a, &q_val) in actions.iter().enumerate() {
            println!("État {}, Action {}, Q = {:.3}", s, a, q_val);
        }
    }
    println!("Politique : {:?}", policy);
}

fn main() {
    let mut env = LineWorld { agent_pos: 0 };

    // === ON-POLICY ===
    println!("===== ON-POLICY First Visit MC =====");
    let (policy_on, q_on) = on_policy_first_visit_mc_control(&mut env, 10_000, 0.9, 0.1);
    export_q_values("src/algorithms/mc/q_values_on_policy1.csv", &q_on);
    display_results(&q_on, &policy_on);

    // === OFF-POLICY ===
    println!("\n===== OFF-POLICY MC CONTROL =====");
    let (policy_off, q_off) = off_policy_mc_control(&mut env, 10_000, 0.9, 0.1);
    let policy_vec: Vec<usize> = (0..(&env as &dyn Environment).num_states())

        .map(|s| policy_off.get_action(&s))
        .collect();
    export_q_values("src/algorithms/mc/q_values_off_policy1.csv", &q_off);
    display_results(&q_off, &policy_vec);

    // === MC EXPLORING STARTS ===
println!("\n===== MC Exploring Starts (MC-ES) =====");
let (policy_es, q_es) = monte_carlo_es(&mut env, 10_000, 0.9);
let num_states = (&env as &dyn Environment).num_states();
let policy_vec_es: Vec<usize> = (0..num_states)
    .map(|s| policy_es.get_action(&s))
    .collect();
export_q_values("src/algorithms/mc/q_values_mc_es1.csv", &q_es);
display_results(&q_es, &policy_vec_es);
}