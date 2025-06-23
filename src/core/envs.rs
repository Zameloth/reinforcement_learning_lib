pub trait MonteCarloEnvironment {
    fn reset(&mut self); // renvoie l’état initial
    fn step(&mut self, action: usize); // (next_state, reward, done)
    fn score(&self) -> f64;
    fn is_game_over(&self) -> bool;
    fn num_states(&self) -> usize;
    fn num_actions(&self) -> usize;
    //fn render(&self);
}

pub trait Environment {
    fn num_states(&self) -> usize;
    fn num_actions(&self) -> usize;
}


/// Permet de représenter un environnement pour l'utiliser avec les fontions de dynamic programming
#[derive(Debug)]
pub struct DPEnvironment {
    pub states: Vec<usize>,
    pub actions: Vec<usize>,

    // Vecteur listant toutes les rewards possibles
    pub rewards: Vec<f64>,
    pub terminal_states: Vec<usize>,

    // Vecteur représentant les probabilités de transitions entre les
    // états de forme P[state, action, state_prime, reward_index]
    pub transitions: Vec<f64>,

    pub num_states: usize,
    pub num_actions: usize,
    pub num_rewards: usize,
}

impl Environment for DPEnvironment {
    fn num_states(&self) -> usize {
        self.num_states
    }
    fn num_actions(&self) -> usize {
        self.num_actions
    }
}

// impl Environment for MonteCarloEnvironment {
//     fn num_states(&self) -> usize {
//         self.num_states
//     }
//     fn num_actions(&self) -> usize {
//         self.num_actions
//     }
// }

impl DPEnvironment {
    /// Constructeur statique pour initialiser l'environnement avec des transitions nulles
    pub fn new(
        num_states: usize,
        num_actions: usize,
        num_rewards: usize,
        rewards: Vec<f64>,
        terminal_states: Vec<usize>,
    ) -> Self {
        let transitions_size = num_states * num_actions * num_states * num_rewards;
        DPEnvironment {
            states: (0..num_states).collect(),
            actions: (0..num_actions).collect(),
            rewards,
            terminal_states,
            transitions: vec![0.0; transitions_size],
            num_states,
            num_actions,
            num_rewards,
        }
    }

    /// Calcule l'index aplati à partir des indices 4D
    fn get_index(
        &self,
        state: usize,
        action: usize,
        state_prime: usize,
        reward_index: usize,
    ) -> usize {
        assert!(state < self.num_states, "state index out of bounds");
        assert!(action < self.num_actions, "action index out of bounds");
        assert!(
            state_prime < self.num_states,
            "state_prime index out of bounds"
        );
        assert!(
            reward_index < self.num_rewards,
            "reward_index index out of bounds"
        );

        state * (self.num_actions * self.num_states * self.num_rewards)
            + action * (self.num_states * self.num_rewards)
            + state_prime * self.num_rewards
            + reward_index
    }

    /// Retourne la probabilité de transition pour un tuple (state, action, state_prime, reward_index)
    pub fn get_transition_prob(
        &self,
        state: usize,
        action: usize,
        state_prime: usize,
        reward_index: usize,
    ) -> f64 {
        let index = self.get_index(state, action, state_prime, reward_index);
        self.transitions[index]
    }

    /// Permet de modifier la probabilité de transition pour un tuple (state, action, state_prime, reward_index)
    pub fn set_transition_prob(
        &mut self,
        state: usize,
        action: usize,
        state_prime: usize,
        reward_index: usize,
        value: f64,
    ) {
        let index = self.get_index(state, action, state_prime, reward_index);
        self.transitions[index] = value;
    }
}
