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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::dp::value_iteration::value_iteration;
    use crate::environments::line_world::line_world_dp;
    use crate::environments::line_world::LineWorld;

    #[test]
    fn test_run_policy() {

    }
}
