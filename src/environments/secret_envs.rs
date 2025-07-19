use crate::core::envs::{DynamicProgramingEnvironment, Environment, MonteCarloEnvironment};
use std::ffi::c_void;

pub struct SecretEnv {
    lib: libloading::Library,
    env: *mut c_void,
    env_id: usize,
}

impl SecretEnv {
    pub fn new(env_id: usize) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            //Selection de la bonne lib selon l'environnement
            #[cfg(target_os = "linux")]
            let path = "./libs/libsecret_envs.so";
            #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
            let path = "./libs/libsecret_envs_intel_macos.dylib";
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let path = "./libs/libsecret_envs.dylib";
            #[cfg(windows)]
            let path = "./libs/secret_envs.dll";

            // Chargement de la lib
            let lib = libloading::Library::new(path).expect("Failed to load library");

            let new_fn: libloading::Symbol<unsafe extern "C" fn() -> *mut c_void> =
                lib.get(format!("secret_env_{}_new", env_id).as_bytes())?;
            let env = new_fn();

            Ok(SecretEnv { lib, env, env_id })
        }
    }
}

impl Environment for SecretEnv {
    fn num_states(&self) -> usize {
        unsafe {
            let num_states_fn: libloading::Symbol<unsafe extern "C" fn() -> usize> = self
                .lib
                .get(format!("secret_env_{}_num_states", self.env_id).as_bytes())
                .expect("Erreur : impossible de trouver la fonction num_states");

            num_states_fn()
        }
    }

    fn num_actions(&self) -> usize {
        unsafe {
            let num_actions_fn: libloading::Symbol<unsafe extern "C" fn() -> usize> = self
                .lib
                .get(format!("secret_env_{}_num_actions", self.env_id).as_bytes())
                .expect("Erreur : impossible de trouver la fonction num_actions");

            num_actions_fn()
        }
    }

    fn num_rewards(&self) -> usize {
        unsafe {
            let num_rewards_fn: libloading::Symbol<unsafe extern "C" fn() -> usize> = self
                .lib
                .get(format!("secret_env_{}_num_rewards", self.env_id).as_bytes())
                .expect("Erreur : impossible de trouver la fonction num_rewards");

            num_rewards_fn()
        }
    }
}

impl DynamicProgramingEnvironment for SecretEnv {
    fn get_transition_prob(
        &self,
        state: usize,
        action: usize,
        state_prime: usize,
        reward_index: usize,
    ) -> f64 {
        unsafe {
            let get_transition_prob_fn: libloading::Symbol<
                unsafe extern "C" fn(usize, usize, usize, usize) -> f32,
            > = self
                .lib
                .get(b"secret_env_0_transition_probability")
                .expect("Failed to load function `secret_env_0_transition_probability`");

            get_transition_prob_fn(state, action, state_prime, reward_index) as f64
        }
    }

    fn set_transition_prob(
        &mut self,
        _state: usize,
        _action: usize,
        _state_prime: usize,
        _reward_index: usize,
        _value: f64,
    ) {
        return;
    }

    fn get_reward(&self, i: usize) -> f64 {
        unsafe {
            let get_reward_fn: libloading::Symbol<unsafe extern "C" fn(usize) -> f32> = self
                .lib
                .get(format!("secret_env_{}_reward", self.env_id).as_bytes())
                .expect("Erreur : impossible de trouver la fonction reward");
            get_reward_fn(i) as f64
        }
    }

    fn get_terminal_states(&self) -> Vec<usize> {
        Vec::new()
    }
}

impl MonteCarloEnvironment for SecretEnv {
    fn reset(&mut self) {
        unsafe {
            let reset_fn: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = self
                .lib
                .get(format!("secret_env_{}_reset", self.env_id).as_bytes())
                .expect("Erreur : impossible de trouver la fonction reset");

            reset_fn(self.env);
        }
    }

    fn step(&mut self, action: usize) -> (usize, f64) {
        unsafe {
            let step_fn: libloading::Symbol<unsafe extern "C" fn(*mut c_void, usize)> = self
                .lib
                .get(format!("secret_env_{}_step", self.env_id).as_bytes())
                .expect("Erreur : impossible de trouver la fonction step");

            step_fn(self.env, action);
        }

        (self.state_id(), self.get_reward(self.state_id()))
    }

    fn score(&self) -> f64 {
        unsafe {
            let score_fn: libloading::Symbol<unsafe extern "C" fn(*const c_void) -> f32> = self
                .lib
                .get(format!("secret_env_{}_score", self.env_id).as_bytes())
                .expect("Erreur : impossible de trouver la fonction score");

            score_fn(self.env) as f64
        }
    }

    fn is_game_over(&self) -> bool {
        unsafe {
            let is_game_over_fn: libloading::Symbol<unsafe extern "C" fn(*const c_void) -> bool> =
                self.lib
                    .get(format!("secret_env_{}_is_game_over", self.env_id).as_bytes())
                    .expect("Erreur : impossible de trouver la fonction is_game_over");

            is_game_over_fn(self.env)
        }
    }

    fn available_actions(&self) -> Vec<usize> {
        unsafe {
            let available_actions_fn: libloading::Symbol<
                unsafe extern "C" fn(*const c_void) -> *const usize,
            > = self
                .lib
                .get(format!("secret_env_{}_available_actions", self.env_id).as_bytes())
                .expect("Erreur : impossible de trouver la fonction available_actions");

            let available_actions_len_fn: libloading::Symbol<
                unsafe extern "C" fn(*const c_void) -> usize,
            > = self
                .lib
                .get(format!("secret_env_{}_available_actions_len", self.env_id).as_bytes())
                .expect("Erreur : impossible de trouver la fonction available_actions_len");

            let available_actions_destroy_fn: libloading::Symbol<
                unsafe extern "C" fn(*const usize, usize),
            > = self
                .lib
                .get(format!("secret_env_{}_available_actions_delete", self.env_id).as_bytes())
                .expect("Erreur : impossible de trouver la fonction available_actions_delete");

            let available_actions = available_actions_fn(self.env);
            let available_actions_len = available_actions_len_fn(self.env);

            let mut actions = Vec::with_capacity(available_actions_len);
            for i in 0..available_actions_len {
                actions.push(*available_actions.add(i));
            }

            available_actions_destroy_fn(available_actions, available_actions_len);

            actions
        }
    }

    fn display(&self) {
        unsafe {
            let display_fn: libloading::Symbol<unsafe extern "C" fn(*const c_void)> = self
                .lib
                .get(format!("secret_env_{}_display", self.env_id).as_bytes())
                .expect("Erreur : impossible de trouver la fonction display");

            display_fn(self.env);
        }
    }

    fn start_from_random_state(&mut self) {
        unsafe {
            let start_from_random_state_fn: libloading::Symbol<
                unsafe extern "C" fn() -> *mut c_void,
            > = self
                .lib
                .get(format!("secret_env_{}_from_random_state", self.env_id).as_bytes())
                .expect("Erreur : impossible de trouver la fonction from_random_state");

            self.env = start_from_random_state_fn();
        }
    }

    fn state_id(&self) -> usize {
        unsafe {
            let state_id_fn: libloading::Symbol<unsafe extern "C" fn(*const c_void) -> usize> =
                self.lib
                    .get(format!("secret_env_{}_state_id", self.env_id).as_bytes())
                    .expect("Erreur : impossible de trouver la fonction state_id");

            state_id_fn(self.env)
        }
    }

    fn is_forbidden(&self, action: usize) -> bool {
        unsafe {
            let is_forbidden_fn: libloading::Symbol<
                unsafe extern "C" fn(*const c_void, usize) -> bool,
            > = self
                .lib
                .get(format!("secret_env_{}_is_forbidden", self.env_id).as_bytes())
                .expect("Erreur : impossible de trouver la fonction is_forbidden");

            is_forbidden_fn(self.env, action)
        }
    }
}

impl Drop for SecretEnv {
    fn drop(&mut self) {
        unsafe {
            let delete_fn: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = self
                .lib
                .get(format!("secret_env_{}_delete", self.env_id).as_bytes())
                .expect("Erreur : impossible de trouver la fonction delete");

            delete_fn(self.env);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vérifie qu’un SecretEnv avec cet `env_id` peut être instancié
    /// et que ses méthodes de base démarrent sans panic.
    fn basic_checks(env_id: usize) {
        println!("--- basic_checks début pour env {} ---", env_id);

        // Construction
        let mut env = SecretEnv::new(env_id)
            .unwrap_or_else(|e| panic!("Échec de création de SecretEnv {}: {}", env_id, e));
        println!("SecretEnv {} instancié avec succès", env_id);

        // Environment trait
        let ns = env.num_states();
        let na = env.num_actions();
        println!("num_states = {}, num_actions = {}", ns, na);
        assert!(ns > 0, "num_states() == 0 pour env {}", env_id);
        assert!(na > 0, "num_actions() == 0 pour env {}", env_id);

        // DynamicProgrammingEnvironment trait
        let nr = env.num_rewards();
        println!("num_rewards = {}", nr);
        assert!(nr > 0, "num_rewards() == 0 pour env {}", env_id);

        // MonteCarloEnvironment trait
        println!("Tests MonteCarloEnvironment...");
        env.reset();
        println!("reset effectué");
        let sid = env.state_id();
        assert!(
            sid < ns,
            "state_id() ({}) hors borne [0, {}) après reset pour env {}",
            sid,
            ns,
            env_id
        );
        println!("state_id après reset = {}", sid);

        let actions = env.available_actions();
        println!("available_actions récupérées, count = {}", actions.len());
        assert!(
            actions.len() <= na,
            "available_actions().len() ({}) != num_actions() ({}) pour env {}",
            actions.len(),
            na,
            env_id
        );
        for &a in &actions {
            println!("Vérification action disponible: {}", a);
            assert!(a < na, "action invalide {} pour env {}", a, env_id);
            let forbidden = env.is_forbidden(a);
            println!("is_forbidden({}) = {}", a, forbidden);
        }
        println!("available_actions valides");

        if !env.is_game_over() {
            let before_score = env.score();
            println!("score avant step = {}", before_score);
            env.step(actions[0]);
            let after_score = env.score();
            println!("score après step = {}", after_score);
            assert!(
                before_score.is_finite() && after_score.is_finite(),
                "score non fini ({}→{}) pour env {}",
                before_score,
                after_score,
                env_id
            );
        } else {
            println!("Jeu déjà terminé pour env {}", env_id);
        }

        // display et start_from_random_state
        println!("Appel de display()...");
        env.display();
        println!("display() terminé");

        println!("Appel de start_from_random_state()...");
        env.start_from_random_state();
        println!("start_from_random_state() terminé");

        println!("--- basic_checks terminé pour env {} ---", env_id);
    }

    #[test]
    fn test_secret_envs_0_to_3() {
        println!("=== Début du test des SecretEnvs 0 à 3 ===");
        for env_id in 0..=3 {
            println!("== Démarrage des tests pour env {} =", env_id);
            basic_checks(env_id);
        }
        println!("=== Fin du test des SecretEnvs 0 à 3 ===");
    }
}
