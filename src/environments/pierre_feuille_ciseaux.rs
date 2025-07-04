use crate::core::envs::DPEnvironment;

/// Actions : 
///     Pierre = 0
///     Feuille = 1
///     Ciseau = 2

fn pierre_feuille_ciseaux_dp() -> DPEnvironment {
    let num_states = 13; // état initial + 3 états possibles + 3*3 états finaux
    let num_actions = 3; // pierre, feuille, ciseaux
    let num_rewards = 3;
    let rewards = vec![-1.0, 0.0, 1.0];
    let terminal_states: Vec<usize> = vec![4, 5, 6, 7, 8, 9, 10, 11, 12];

    let mut env = DPEnvironment::new(
        num_states,
        num_actions,
        num_rewards,
        rewards,
        terminal_states,
    );

    // Round 1 : l'adversaire est random
    for a in 0..num_actions {
        for r in 0..num_rewards {
            env.set_transition_prob(0, a, 1 + a, r, 1.0 / num_rewards as f64);
        }
    }


    // Round 2 : L'adversaire refait l'action de l'agent :
    for s in 1..4 {
        let a_adv = s - 1;
        for a in 0..num_actions {
            let s_prime = 4;
            let reward_ind = match (a, a_adv) {
                // égalité
                (0, 0) => 1,
                (1, 1) => 1,
                (2, 2) => 1,

                // Perdu
                (0, 1) => 0,
                (1, 2) => 0,
                (2, 0) => 0,

                // Gagné
                (0, 2) => 2,
                (1, 0) => 2,
                (2, 1) => 2,

                _ => panic!("Invalid action")
            };

            env.set_transition_prob(s, a, s_prime, reward_ind, 1.0);
        }
    }
    env
}