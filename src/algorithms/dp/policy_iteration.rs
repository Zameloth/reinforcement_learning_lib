use crate::core::envs::DPEnvironment;
use crate::core::policies::{DeterministicPolicy, Policy};

pub fn policy_evaluation(
    env: &DPEnvironment,
    policy: &DeterministicPolicy,
    theta: f64,
    gamma: f64,
) -> Vec<f64> {
    let mut values = vec![0.0; env.num_states];

    for ts in &env.terminal_states {
        values[*ts] = 0.0;
    }

    loop {
        let mut delta: f64 = 0.0;
        for s in &env.states {
            let v_old: f64 = values[*s];
            let mut total = 0.0;

            for s_prime in &env.states {
                for r_index in 0..env.num_rewards {
                    let r = env.rewards[r_index];
                    let a = policy.get_action(s);
                    let p = env.get_transition_prob(*s, a, *s_prime, r_index);
                    total += p * (r + gamma * values[*s_prime]);
                }
            }

            values[*s] = total;
            delta = delta.max((v_old - total).abs());
        }

        if delta < theta {
            break;
        }
    }

    values
}

pub fn policy_improvement(
    env: &DPEnvironment,
    old_policy: &DeterministicPolicy,
    v: &Vec<f64>,
    gamma: f64,
) -> (DeterministicPolicy, bool) {
    let mut new_policy = DeterministicPolicy::new_det_pol(env);
    let mut policy_is_stable = true;

    for s in &env.states {
        let old_actions = old_policy.get_action(s);
        let mut best_action: Option<usize> = None;
        let mut best_value: f64 = f64::NEG_INFINITY;
        for a in 0..env.num_actions {
            let mut score = 0.0;
            for s_prime in &env.states {
                for r_index in 0..env.num_rewards {
                    let r = env.rewards[r_index];
                    let p = env.get_transition_prob(*s, a, *s_prime, r_index);
                    score += p * (r + gamma * v[*s_prime]);
                    if best_action.is_none() || score > best_value {
                        best_action = Some(a);
                        best_value = score;
                    }
                }
                if best_action.unwrap() != old_actions {
                    policy_is_stable = false;
                }
                new_policy.set_action(s, best_action.unwrap())
            }
        }
    }
    (new_policy, policy_is_stable)
}

pub fn policy_iteration(
    env: &DPEnvironment,
    theta: f64,
    gamma: f64,
) -> (DeterministicPolicy, Vec<f64>) {
    let mut policy = DeterministicPolicy::new_det_pol(env);

    let mut cpt = 100;
    loop {
        let v = policy_evaluation(env, &policy, theta, gamma);

        let (new_policy, policy_is_stable) = policy_improvement(env, &policy, &v, gamma);

        if !policy_is_stable {
            return (new_policy, v);
        }
        policy = new_policy;

        cpt -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environments::line_world::line_world_dp;

    #[test]
    fn test_policy_evaluation() {
        let env = line_world_dp();
        let mut policy: DeterministicPolicy = DeterministicPolicy::new_det_pol(&env);
        for s in &env.states {
            policy.set_action(s, 1);
        }

        let result = policy_evaluation(&env, &policy, 0.0001, 0.99);

        let expected = vec![0.0, 1.0, 1.0, 1.0, 0.0];
        let rounded: Vec<f64> = result.iter().map(|v| (v * 100.0).round() / 100.0).collect();
        assert_eq!(
            rounded, expected,
            "Comparing {:?} and {:?}",
            result, expected
        );
    }

    #[test]
    fn test_policy_iteration() {
        let env = line_world_dp();

        let (policy, values) = policy_iteration(&env, 0.0001, 0.99);
        let expected = vec![0, 1, 1, 1, 0];

        for s in 0..3 {
            assert_eq!(
                expected[s],
                policy.get_action(&s),
                "État {} doit aller à droite",
                s
            )
        }
    }
}
