use crate::core::envs::DynamicProgramingEnvironment;
use crate::core::policies::DeterministicPolicy;
use indicatif::ProgressIterator;

pub fn policy_evaluation(
    env: &dyn DynamicProgramingEnvironment,
    policy: &DeterministicPolicy,
    theta: f64,
    gamma: f64,
) -> Vec<f64> {
    let mut values = vec![0.0; env.num_states()];

    for ts in &env.get_terminal_states() {
        values[*ts] = 0.0;
    }

    loop {
        let mut delta: f64 = 0.0;
        for s in 0..env.num_states() {
            let v_old: f64 = values[s];
            let mut total = 0.0;

            for s_prime in 0..env.num_states() {
                for r_index in 0..env.num_rewards() {
                    let r = env.get_reward(r_index);
                    let a = policy.get_action(&s);
                    let p = env.get_transition_prob(s, a, s_prime, r_index);
                    total += p * (r + gamma * values[s_prime]);
                }
            }

            values[s] = total;
            delta = delta.max((v_old - total).abs());
        }

        if delta < theta {
            break;
        }
    }

    values
}

pub fn policy_improvement(
    env: &dyn DynamicProgramingEnvironment,
    old_policy: &DeterministicPolicy,
    v: &Vec<f64>,
    gamma: f64,
) -> (DeterministicPolicy, bool) {
    let mut new_policy = DeterministicPolicy::new_det_pol(env);
    let mut policy_is_stable = true;

    for s in 0..env.num_states() {
        let old_actions = old_policy.get_action(&s);
        let mut best_action: Option<usize> = None;
        let mut best_value: f64 = f64::NEG_INFINITY;

        for a in 0..env.num_actions() {
            let mut score = 0.0;

            for s_prime in 0..env.num_states() {
                for r_index in 0..env.num_rewards() {
                    let r = env.get_reward(r_index);
                    let p = env.get_transition_prob(s, a, s_prime, r_index);
                    score += p * (r + gamma * v[s_prime]);
                }

                if best_action.is_none() || score > best_value {
                    best_action = Some(a);
                    best_value = score;
                }
            }

            let choosen_action = best_action.unwrap();
            if choosen_action != old_actions {
                policy_is_stable = false;
            }
            new_policy.set_action(&s, choosen_action);
        }
    }
    (new_policy, policy_is_stable)
}

pub fn policy_iteration(
    env: &dyn DynamicProgramingEnvironment,
    theta: f64,
    gamma: f64,
    max_iter: usize,
) -> (DeterministicPolicy, Vec<f64>) {
    let mut policy = DeterministicPolicy::new_det_pol(env);
    let mut values = vec![0.0; env.num_states()];

    for ep in (0..max_iter).progress() {
        if ep % 100 == 0 {
            println!("=== Iteration {} ===", ep);
        }
        
        let v = policy_evaluation(env, &policy, theta, gamma);

        let (new_policy, policy_is_stable) = policy_improvement(env, &policy, &v, gamma);

        if policy_is_stable {
            return (new_policy, v);
        }
        policy = new_policy;
        values = v;
    }

    (policy, values)
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

        let result = policy_evaluation(&env, &policy, 0.00001, 0.9999);

        let expected = vec![0.0, 1.0, 1.0, 1.0, 0.0];
        let eps = 1e-2;
        for (got, want) in result.iter().zip(&expected) {
            assert!(
                (got - want).abs() < eps,
                "V computed = {:?}, but expected ≃ {:?}",
                got,
                want
            );
        }
    }

    #[test]
    fn test_policy_iteration() {
        let env = line_world_dp();
        let expected = vec![0, 1, 1, 1, 0];
        for _ in 0..50{
            let (policy, _) = policy_iteration(&env, 0.0001, 0.99, 1000);


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
}
