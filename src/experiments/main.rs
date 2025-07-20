use reinforcement_learning_lib::algorithms::dp::policy_iteration::policy_iteration;
use reinforcement_learning_lib::algorithms::planning::{dyna_q::dyna_q, dyna_q_plus::dyna_q_plus};
use reinforcement_learning_lib::algorithms::td::{
    expected_sarsa::expected_sarsa, q_learning::q_learning, sarsa::sarsa,
};
use reinforcement_learning_lib::environments::line_world::{line_world_dp, LineWorld};

use std::io::{self, Write};

fn main() {
    println!("=== Deep RL Project ===");
    println!("Choisissez un algorithme à tester sur LineWorld :");
    println!("1 - Dyna-Q");
    println!("2 - Dyna-Q+");
    println!("3 - SARSA");
    println!("4 - Q-Learning");
    println!("5 - Expected SARSA");
    println!("6 - Test Policy Iteration");

    print!("Votre choix : ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let choix = input.trim().parse::<u32>().unwrap_or(0);

    let alpha = 0.1;
    let gamma = 0.9;
    let epsilon = 0.1;
    let n = 5;
    let kappa = 0.001;
    let episodes = 10;

    match choix {
        1 => {
            println!("=== Dyna-Q ===");
            let mut env = LineWorld::new();
            let (policy, _) = dyna_q(&mut env, alpha, gamma, epsilon, n, episodes);
            println!("{}", policy);
        }
        2 => {
            println!("=== Dyna-Q+ ===");
            let mut env = LineWorld::new();
            let (policy, _) = dyna_q_plus(&mut env, alpha, gamma, epsilon, kappa, n, episodes);
            println!("{}", policy);
        }
        3 => {
            println!("=== SARSA ===");
            let mut env = LineWorld::new();
            let (policy, _) = sarsa(&mut env, alpha, gamma, epsilon, episodes);
            println!("{}", policy);
        }
        4 => {
            println!("=== Q-Learning ===");
            let mut env = LineWorld::new();
            let (policy, _) = q_learning(&mut env, alpha, gamma, epsilon, episodes);
            println!("{}", policy);
        }
        5 => {
            println!("=== Expected SARSA ===");
            let mut env = LineWorld::new();
            let (policy, _) = expected_sarsa(&mut env, alpha, gamma, epsilon, episodes);
            println!("{}", policy);
        }
        6 => {
            println!("=== Test Policy Iteration ===");
            test_policy_iteration();
        }
        _ => {
            println!("Option invalide.");
        }
    }
}

/// Test de Policy Iteration avec assertion (mode "test")
fn test_policy_iteration() {
    let env = line_world_dp();

    let (policy, _) = policy_iteration(&env, 0.0001, 0.99, 1000);
    let expected = vec![0, 1, 1, 1, 0];

    for s in 0..3 {
        assert_eq!(
            expected[s],
            policy.get_action(&s),
            "État {} doit aller à droite",
            s
        );
    }

    println!("✅ Test Policy Iteration passé avec succès !");
}
