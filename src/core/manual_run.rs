use crate::core::envs::MonteCarloEnvironment;
use crate::core::policies::Policy;
use std::io;
use std::io::Write;

pub fn run_policy(env: &mut dyn MonteCarloEnvironment, policy: &dyn Policy) {
    env.reset();

    env.display();

    while !env.is_game_over() {
        println!("Appuyer sur Entrée pour continuer...");
        io::stdout().flush().unwrap();
        let mut buf = String::new();
        io::stdin().read_line(&mut buf).unwrap();

        let state_id = env.state_id();
        let action = policy.get_action(&state_id);
        println!("Action choisie: {}", action);

        env.step(action);
        let reward = env.score();
        println!("Score: {}", reward);

        env.display();
    }

    println!("Fin du jeu");
}

/// Permet à l'utilisateur de choisir les actions manuellement dans un environnement Monte Carlo
pub fn run_manual(env: &mut dyn MonteCarloEnvironment) {
    env.reset();
    env.display();

    while !env.is_game_over() {
        println!("Actions disponibles :");
        let actions = env.available_actions();
        for &action in &actions {
            println!("  - Action {} : {:?}", action, env.action_name(action));
        }

        // Demander à l'utilisateur de choisir une action
        let chosen_action = loop {
            print!("Entrez le numéro de l'action : ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();

            match input.trim().parse::<usize>() {
                Ok(a) if actions.contains(&a) => break a,
                _ => println!("Action invalide, réessayez."),
            }
        };

        env.step(chosen_action);
        let reward = env.score();
        println!("Récompense reçue : {}", reward);
        env.display();
    }

    println!("Fin du jeu.");
}
