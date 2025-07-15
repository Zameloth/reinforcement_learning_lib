use crate::core::envs::{DynamicProgramingEnvironment};
use crate::core::policies::DeterministicPolicy;

pub fn value_iteration(
    env: &dyn DynamicProgramingEnvironment,
    theta: f64,
    gamma: f64,
) -> (DeterministicPolicy, Vec<f64>) {
    let mut policy: DeterministicPolicy = DeterministicPolicy::new_det_pol(env);
    let mut values = vec![0.0; env.num_states()];
    
    loop {
        let mut delta: f64 = 0.0;

        for s in 0..env.num_states() {
            let value_old = values[s];
            let mut max_q = f64::NEG_INFINITY;
            let mut best_a: Option<usize> = None;

            for a in 0..env.num_actions() {
                let mut q_s_a = 0.0;
                for s_prime in 0..env.num_states() {
                    for r_index in 0..env.num_rewards() {
                        let r = env.get_reward(r_index);
                        let p = env.get_transition_prob(s, a, s_prime, r_index);
                        q_s_a += p * (r + gamma * values[s_prime]);
                    }
                }

                if q_s_a > max_q {
                    max_q = q_s_a;
                    best_a = Some(a);
                }
            }

            values[s] = max_q;
            delta = delta.max((value_old - max_q).abs());
            policy.set_action(&s, best_a.unwrap());
        }

        if delta < theta {
            return (policy, values);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::algorithms::dp::value_iteration::value_iteration;
    use crate::environments::line_world::line_world_dp;

    #[test]
    fn test_value_iteration() {
        let env = line_world_dp();

        let (policy, values) = value_iteration(&env, 0.0001, 0.99);
        let expected = vec![0, 1, 1, 1, 0];
        println!("Values : {:?}", values);
        println!("Policy : {}", policy);

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
