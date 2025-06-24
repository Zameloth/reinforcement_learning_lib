use crate::core::envs::DPEnvironment;
use crate::core::policies::{Policy, ProbabilisticPolicy};

pub fn policy_evaluation(
    env:    &DPEnvironment,
    policy: ProbabilisticPolicy,
    theta:  f64,
    gamma:  f64,
) -> Vec<f64> {
    let mut values = vec![0.0; env.num_states];

    for ts in &env.terminal_states {
        values[*ts] = 0.0;
    }

    loop {
        let mut delta: f64 = 0.0;
        for s in &env.states {
            let v_old: f64 = values[*s];
            // 1) initialiser total pour CHAQUE état
            let mut total = 0.0;

            // 2) pour chaque action, calculer Q(s,a)
            for a in &env.actions {
                let mut action_value = 0.0;
                for s_prime in &env.states {
                    for r_index in 0..env.num_rewards {
                        let r = env.rewards[r_index];
                        let p = env.get_transition_prob(*s, *a, *s_prime, r_index);
                        // 3) utiliser r + γ * V(s')
                        action_value += p * (r + gamma * values[*s_prime]);
                    }
                }
                // 4) pondérer par π(a|s)
                total += policy.get_proba(*s, *a) * action_value;
            }

            // 5) mettre à jour V(s) **UNE FOIS** par état
            values[*s] = total;
            delta = delta.max((v_old - total).abs());
        }

        if delta < theta {
            break;
        }
    }

    values
}


#[cfg(test)]
mod tests {
    #[test]
    fn test_policy_evaluation() {
        use super::*;
        use crate::core::policies::ProbabilisticPolicy;
        use crate::environments::line_world::line_world_dp;

        let env = line_world_dp();
        let policy: ProbabilisticPolicy = ProbabilisticPolicy::new(&env);

        let result = policy_evaluation(&env, policy, 0.0001, 0.99);

        let expected = vec![0.0, -0.5, 0.0, 0.5, 0.0];
        let rounded: Vec<f64> = result.iter().map(|v| (v * 100.0).round() / 100.0).collect();
        assert_eq!(
            rounded, expected,
            "Comparing {:?} and {:?}",
            result, expected
        );
    }
}
