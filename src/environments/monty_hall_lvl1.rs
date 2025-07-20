use crate::core::envs::{
    DPEnvironment, DynamicProgramingEnvironment, Environment, MonteCarloEnvironment,
};
use rand::random_range;

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
    let num_states = 36;
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

// pub fn monty_hall_lvl2_dp() -> DPEnvironment {
//     let num_states = 72;
//     let num_actions = 5;
//     let num_rewards = 2;
//     let rewards = vec![0.0, 1.0];
//     let terminal_states = vec![12];
//
//
// }

pub struct MontyHallEnv {
    nb_porte: usize,
    porte_choisie: Option<usize>,
    porte_gagnante: usize,
    round: usize,
    portes_disponibles: Vec<usize>,
    lvl: usize,
    state_id: usize,
    nb_porte_initial: usize,
}

impl Environment for MontyHallEnv {
    fn num_states(&self) -> usize {
        if self.lvl == 1 {
            7
        } else if self.lvl == 2 {
            71
        } else {
            panic!("Level not implemented");
        }
    }

    fn num_actions(&self) -> usize {
        self.nb_porte
    }

    fn num_rewards(&self) -> usize {
        2
    }
}

impl MonteCarloEnvironment for MontyHallEnv {
    fn reset(&mut self) {
        match self.lvl {
            1 => {
                self.round = 0;
                self.portes_disponibles = vec![0, 1, 2];
                self.nb_porte = 3;
                self.porte_gagnante = random_range(0..3);
                self.porte_choisie = None;
                self.state_id = self.porte_gagnante;
                self.nb_porte_initial = 3;
            }
            2 => {
                self.round = 0;
                self.nb_porte = 5;
                self.porte_gagnante = random_range(0..5);
                self.porte_choisie = None;
                self.portes_disponibles = vec![0, 1, 2, 3, 4];
                self.state_id = self.porte_gagnante;
                self.nb_porte_initial = 5;
            }
            _ => panic!("Level not implemented"),
        }
    }

    fn step(&mut self, action: usize) -> (usize, f64) {
        if self.portes_disponibles.contains(&action) && self.nb_porte > 2 {
            let old_id = self.state_id;
            let multiplier = self.nb_porte; // 3 au premier tour
            let action_index = self
                .portes_disponibles
                .iter()
                .position(|&p| p == action)
                .unwrap();
            self.state_id = old_id * multiplier + action_index;

            self.porte_choisie = Some(action);
            self.round += 1;
            self.nb_porte -= 1;

            let portes_supprimable: Vec<&usize> = self
                .portes_disponibles
                .iter()
                .filter(|&&p| p != self.porte_choisie.unwrap() && p != self.porte_gagnante)
                .collect();
            let porte_suprimee = portes_supprimable[random_range(0..portes_supprimable.len())];

            let idx_to_remove = self
                .portes_disponibles
                .iter()
                .position(|&p| p == *porte_suprimee)
                .expect("Porte à supprimer non trouvée");
            self.portes_disponibles.remove(idx_to_remove);
        } else {
            panic!("Action not allowed");
        }

        (self.state_id, self.score())
    }

    fn score(&self) -> f64 {
        if self.nb_porte > 2 {
            0.0
        } else {
            if self.porte_choisie.unwrap() == self.porte_gagnante {
                1.0
            } else {
                0.0
            }
        }
    }

    fn is_game_over(&self) -> bool {
        self.nb_porte == 2
    }

    fn available_actions(&self) -> Vec<usize> {
        self.portes_disponibles.clone()
    }

    fn display(&self) {
        todo!()
    }

    fn start_from_random_state(&mut self) {
        self.reset();

        let max_iter = if self.lvl == 1 { 1 } else { 3 };

        for _ in 0..random_range(0..max_iter) {
            let available_actions = self.available_actions();
            let action = available_actions[random_range(0..available_actions.len())];
            self.step(action);
        }
    }

    fn state_id(&self) -> usize {
        self.state_id
    }

    fn is_forbidden(&self, action: usize) -> bool {
        !self.portes_disponibles.contains(&action)
    }

    fn action_name(&self, action: usize) -> String {
        format!("Porte {}", action + 1)
    }
}

pub fn new_monty_hall(lvl: usize) -> MontyHallEnv {
    let mut env = MontyHallEnv {
        nb_porte: 0,
        porte_choisie: None,
        porte_gagnante: 0,
        round: 0,
        portes_disponibles: vec![],
        lvl,
        state_id: 0,
        nb_porte_initial: 0,
    };

    env.reset();
    env
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::envs::MonteCarloEnvironment;

    #[test]
    fn test_reset_initializes_state_id_and_ports() {
        let mut env = new_monty_hall(1);
        env.reset();
        // state_id doit valoir porte_gagnante et dans [0,2]
        let gid = env.porte_gagnante;
        assert!(gid < 3);
        assert_eq!(env.state_id(), gid);

        // portes_disponibles initiales
        assert_eq!(env.available_actions(), vec![0, 1, 2]);
        assert_eq!(env.nb_porte, 3);
        assert_eq!(env.round, 0);
    }

    #[test]
    fn test_step_updates_state_id_correctly() {
        // On construit un état de départ contrôlé :
        let mut env = MontyHallEnv {
            lvl: 1,
            nb_porte: 3,
            porte_gagnante: 2,
            porte_choisie: None,
            round: 0,
            portes_disponibles: vec![0, 1, 2],
            state_id: 2, // on fixe l'état initial = porte gagnante
            nb_porte_initial: 3,
        };

        // On applique l'action 1 (idx = 1 dans [0,1,2])
        env.step(1);
        // new_id = old_id * 3 + idx = 2*3 + 1 = 7
        assert_eq!(env.state_id(), 7);

        // Après un step, nb_porte doit être décrémenté
        assert_eq!(env.nb_porte, 2);
        // round doit avoir été incrémenté
        assert_eq!(env.round, 1);

        // Et is_game_over devient true
        assert!(env.is_game_over());
    }

    #[test]
    #[should_panic(expected = "Action not allowed")]
    fn test_step_invalid_action_panics() {
        let mut env = new_monty_hall(1);
        env.reset();
        // on tente une action hors des portes disponibles
        env.step(99);
    }

    #[test]
    fn test_state_id_within_bounds() {
        for lvl in &[1, 2] {
            let mut env = new_monty_hall(*lvl);
            // On répète plusieurs fois pour couvrir reset + steps
            for _ in 0..500 {
                env.reset();
                // Après reset
                assert!(
                    env.state_id() < env.num_states(),
                    "Après reset, state_id {} doit < num_states {}",
                    env.state_id(),
                    env.num_states()
                );
                // On fait des steps jusqu'à la fin de la partie
                while !env.is_game_over() {
                    let actions = env.available_actions();
                    // Choix arbitraire pour tester
                    let action = actions[0];
                    env.step(action);
                    assert!(
                        env.state_id() < env.num_states(),
                        "Après step, state_id {} doit < num_states {}",
                        env.state_id(),
                        env.num_states()
                    );
                }
            }
        }
    }
}
