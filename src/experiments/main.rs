use reinforcement_learning_lib::algorithms::planning::{dyna_q::dyna_q, dyna_q_plus::dyna_q_plus};
//use reinforcement_learning_lib::environments::line_world::LineWorld;
//use reinforcement_learning_lib::environments::helpers::{current_state, environment_step, choose_action};

use std::io;
use std::io::Write;

fn main() {
    println!("=== Deep RL Project ===");
    println!("Choisissez un algorithme à tester sur LineWorld :");
    println!("1 - Dyna-Q");
    println!("2 - Dyna-Q+");

    print!("Votre choix : ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let choix = input.trim().parse::<u32>().unwrap_or(0);

    // Hyperparamètres et espaces
    let states = vec![0, 1, 2, 3, 4];
    let actions = vec![0, 1]; // 0 = gauche, 1 = droite

    let alpha = 0.1;
    let gamma = 0.9;
    let epsilon = 0.1;
    let n = 5;
    let kappa = 0.001;

    match choix {
        1 => {
            println!("=== Dyna-Q ===");
            dyna_q(&states, &actions, alpha, gamma, epsilon, n, 10);
        }
        2 => {
            println!("=== Dyna-Q+ ===");
            dyna_q_plus(&states, &actions, alpha, gamma, epsilon, kappa, n, 10);
        }
        _ => {
            println!("Option invalide.");
        }
    }
}