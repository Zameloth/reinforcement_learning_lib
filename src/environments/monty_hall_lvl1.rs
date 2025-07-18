use crate::core::envs::{DPEnvironment, DynamicProgramingEnvironment, Environment};

///## Actions:
///* Porte 1
///* Porte 2
///* Porte 3
///
/// ## States :
/// * Etat Initial
/// * 3 porte ouverte
/// * 3 protes 1er tour * 2 portes second tour
///
/// Au 2ème round, plus que deux actions car plus que 2 portes

pub fn monty_hall_lvl1_dp() -> DPEnvironment {
    let num_states = 0;
    let num_actions = 3;
    let num_rewards = 2;
    let rewards = vec![0.0, 1.0];
    let terminal_states = vec![4];

    let mut env = DPEnvironment::new(
        num_states,
        num_actions,
        num_rewards,
        rewards,
        terminal_states,
    );

    // 1er Round, ouverture d'une porte :
    for a in 0..num_actions {
        env.set_transition_prob(0, a, 1 + a, 0, 1.0);
    }

    // 2ᵉ round, une des deux portes non choisies est supprimée, puis l'agent choisit une des autres portes.
    // Seulement 2 actions à cause de suppression porte
    for a in 0..(num_actions - 1) {
        for s in 1..4 {
            let s_prime = 4 + (s * num_actions) + a;
            for r in 0..num_rewards {
                env.set_transition_prob(s, a, s_prime, r, 0.5);
            }
        }
    }
    env
}

struct MontyHallEnv {
    num_portes: usize,
    porte_gagnante: usize,
    round: usize,
    portes_disponibles: Vec<usize>,
}

impl Environment for MontyHallEnv {
    fn num_states(&self) -> usize {
        let mut total = 0;
        let mut portes_restantes = self.num_portes;

        while portes_restantes > 2 {
            let num_states = self.num_portes * portes_restantes * (portes_restantes - 1);

            total += num_states;
            portes_restantes -= 1;
        }

        total
    }

    fn num_actions(&self) -> usize {
        self.num_portes
    }

    fn num_rewards(&self) -> usize {
        2
    }
}