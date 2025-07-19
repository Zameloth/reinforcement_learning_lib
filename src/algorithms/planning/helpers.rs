use rand::prelude::IndexedRandom;
use rand::Rng;
use std::collections::HashMap;
// pour step(), score(), etc.

/// Choisit une action selon une stratégie ε‑greedy
pub fn choose_action(
    q: &HashMap<(usize, usize), f64>,
    s: usize,
    actions: &[usize],
    epsilon: f64,
) -> usize {
    let mut rng = rand::thread_rng();

    if rng.random::<f64>() < epsilon {
        // exploration
        *actions.choose(&mut rng).unwrap()
    } else {
        // exploitation
        let mut best_action = actions[0];
        let mut best_value = f64::MIN;
        for &a in actions {
            let val = *q.get(&(s, a)).unwrap_or(&0.0);
            if val > best_value {
                best_value = val;
                best_action = a;
            }
        }
        best_action
    }
}
